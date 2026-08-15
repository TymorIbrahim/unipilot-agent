"""Build the agent's data layer: UniPilot's Mongo -> Supabase, plus the wiki artifact.

One committed script rebuilds everything the deployed agent reads. Run it from a
developer machine with a UniPilot checkout; the deployment itself has no Mongo,
no networkx and no wiki corpus, and cannot rebuild any of this.

    ./.venv/bin/python scripts/seed.py --schema

What it produces, and why each part is here rather than computed at runtime:

  courses, course_offerings, degree_programs   the catalog, whole
  student_profiles, completed_courses,         the four demo students only --
  semester_plans                               `/api/execute` takes just a
                                               prompt, so identity comes from a
                                               GUI selector over these
  prerequisite_edges, track_courses            MATERIALISED graph derivations.
                                               Computing them needed networkx
                                               and a 16MB dependency on a cold
                                               path that has 300s to answer
  data/wiki_corpus.json.gz                     4,895 heading-segmented chunks
                                               plus corpus-wide BM25
                                               statistics, replacing 2,753
                                               markdown file opens per cold
                                               start

Idempotent. The catalog and demo tables are truncated and rebuilt inside ONE
transaction, so a failed run leaves the previous data intact rather than a half
-loaded database. `spent_confirmations` and `agent_conversations` are never
touched -- they are runtime state, not seed data.

Connects to Postgres through the SUPAVISOR POOLER, not `SUPABASE_DB_URL`'s host:
that host publishes an AAAA record only and is unroutable from here. See
`pooler_dsn`.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
import time
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.derivations import (  # noqa: E402 -- after sys.path setup
    build_corpus_artifact,
    load_wiki_pages,
    prerequisite_edge_rows,
    track_course_rows,
)

# ---------------------------------------------------------------------------
# The demo roster
# ---------------------------------------------------------------------------
# Verified live against Atlas on 2026-08-14: 31 of 951 profiles are viable (they
# need a `programSlug` AND completed courses). These four were chosen for RANGE
# -- a nearly-finished student, a mid-degree one, and two early ones across three
# different degrees -- so the GUI's selector can demonstrate the agent answering
# the same question very differently.
DEMO_STUDENTS: tuple[dict[str, Any], ...] = (
    {
        "userId": "6a578a2da43a2cfe1bcc791c",
        "label": "ISE, 44 courses done",
        # The student `plan_term` was live-validated against. Kept as the GUI
        # default for that reason.
        "primary": True,
    },
    {"userId": "6a5cc147ff67a48db62d884b", "label": "ISE, 17 courses done"},
    {"userId": "6a557a040edefb30367854ce", "label": "CS general 4-year, 11 done"},
    {"userId": "6a5688319341471497d58c59", "label": "Data & Information Eng, 5 done"},
)

DEMO_USER_IDS = tuple(student["userId"] for student in DEMO_STUDENTS)
PRIMARY_USER_ID = next(s["userId"] for s in DEMO_STUDENTS if s.get("primary"))

MAX_CREDITS_PER_SEMESTER = 18.0
"""Set explicitly on every demo profile, because it is null on all 951 live ones.

The planner falls back to 18 when the field is absent, and the fallback is
INVISIBLE in an answer -- the agent states a credit limit without being able to
say where it came from. Writing the same number down turns an assumption into a
record the answer can be grounded in.
"""

KNOWN_ORPHANED_DEMO_COURSE_IDS = 11
"""Demo transcript rows whose `courseId` matches no catalog course.

11 of the 59 distinct ids across the four students, consistent with the 28%
(155 of 554) measured corpus-wide on 2026-07-19. The writer was never identified
and the rows carry no course number or title, so they cannot be repaired. Loaded
anyway: the credit totals over them are still correct, and only the join to
`courses` fails -- which fails CLOSED. Asserted so that growth is noticed.
"""

DEFAULT_UNIPILOT = Path.home() / "Desktop" / "UniPilot"
WIKI_SUBPATH = Path("services/data-engineering/data/catalog_valut/catalog_valut/wiki")
ARTIFACT_PATH = REPO_ROOT / "data" / "wiki_corpus.json.gz"
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"

# Truncated and rebuilt on every run, in this order (parents before children so
# the FKs hold). `spent_confirmations` and `agent_conversations` are absent on
# purpose -- wiping a confirmation ledger would make every spent confirmation
# replayable.
SEEDED_TABLES = (
    "track_courses",
    "prerequisite_edges",
    "semester_plans",
    "completed_courses",
    "student_profiles",
    "course_offerings",
    "degree_programs",
    "courses",
)

# What `semesters[]` keeps is read off `SEMESTER_PLANS`'s own declaration rather
# than listed here -- see `prune_semesters`. Everything else a live plan carries
# (weeklySchedule, selectedLessonEvents, constraintsSnapshot) is undeclared,
# therefore unreadable by `find`, and would ride into every fetch as dead payload.


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def load_env(path: Path) -> dict[str, str]:
    """`.env` as a plain dict. Not `Settings`: this needs MONGO_URI and
    SUPABASE_DB_URL, which the runtime configuration deliberately does not
    carry."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    # A real environment variable wins, so CI can inject credentials.
    values.update({k: v for k, v in os.environ.items() if k in values or k.startswith(("MONGO_", "SUPABASE_"))})
    return values


def pooler_dsn(env: Mapping[str, str]) -> dict[str, Any]:
    """asyncpg connect kwargs for the Supavisor pooler.

    `SUPABASE_DB_URL` names `db.<ref>.supabase.co`, which resolves to an IPv6
    address and NOTHING ELSE -- unroutable from most networks and from Vercel.
    The pooler is the IPv4 route to the same database, and being a pooler it is
    also what stops a per-invocation serverless connection from exhausting
    Postgres' connection limit.

    Session mode (5432) rather than transaction mode (6543): the seed runs one
    long transaction with COPY, which needs a session to itself.
    """
    raw = env.get("SUPABASE_DB_URL", "")
    if not raw:
        raise SystemExit("SUPABASE_DB_URL is not set; it carries the database password.")
    parsed = urllib.parse.urlparse(raw)
    project_ref = (parsed.hostname or "").removeprefix("db.").split(".")[0]
    if not project_ref:
        raise SystemExit(f"could not read a project ref out of {parsed.hostname!r}")

    return {
        "host": env.get("SUPABASE_POOLER_HOST", "aws-0-us-east-1.pooler.supabase.com"),
        "port": int(env.get("SUPABASE_POOLER_PORT", "5432")),
        "user": f"postgres.{project_ref}",
        "password": urllib.parse.unquote(parsed.password or ""),
        "database": "postgres",
        "ssl": "require",
        "statement_cache_size": 0,
    }


def mongo_database(env: Mapping[str, str]):
    """UniPilot's Atlas cluster, read-only in practice.

    `tlsCAFile` is not optional: this venv has no system CA bundle wired into
    Python's ssl module, so every Atlas connection fails certificate
    verification without it -- with an error that reads like a network problem.
    """
    import certifi
    from pymongo import MongoClient

    uri = env.get("MONGO_URI")
    if not uri:
        raise SystemExit("MONGO_URI is not set; the seed reads its source records from Atlas.")
    client = MongoClient(uri, serverSelectionTimeoutMS=20_000, tlsCAFile=certifi.where())
    return client[env.get("MONGO_DB", "unipilot_python")]


# ---------------------------------------------------------------------------
# Row shaping -- Mongo document -> a tuple in declared column order
# ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    """A stored value as text. ObjectId included -- that is how it reaches
    Postgres, and why no ObjectId binding is needed at query time."""
    if value is None:
        return None
    return str(value)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def course_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (
            _text(d.get("_id")),
            _text(d.get("courseNumber")),
            _text(d.get("title")),
            _text(d.get("titleHebrew")),
            _number(d.get("credits")),
            _text(d.get("faculty")),
            _text(d.get("studyFramework")),
            _integer(d.get("catalogYear")),
            _text(d.get("status")),
        )
        for d in documents
    ]


def offering_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (
            _text(d.get("_id")),
            _text(d.get("courseNumber")),
            _text(d.get("semesterName")),
            _integer(d.get("semesterCode")),
            _integer(d.get("academicYear")),
            _text(d.get("catalogVersion")),
            _text(d.get("status")),
            # Carried verbatim for the term planner, which needs real lesson
            # times to produce a conflict-free plan and exam dates to summarise
            # one. Undeclared in the source registry, so `find` never surfaces
            # them to the model -- see the note in db/schema.sql.
            json.dumps(d.get("scheduleGroups") or [], ensure_ascii=False, default=str),
            json.dumps(d.get("examDates") or {}, ensure_ascii=False, default=str),
        )
        for d in documents
    ]


def degree_program_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (_text(d.get("_id")), _text(d.get("name")), _number(d.get("totalCredits")))
        for d in documents
    ]


def profile_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (
            _text(d.get("userId")),
            _text(d.get("institutionId")),
            _text(d.get("facultyId")),
            _text(d.get("programType")),
            _text(d.get("degreeId")),
            _text(d.get("programSlug")),
            _integer(d.get("catalogYear")),
            _text(d.get("currentSemesterCode")),
            # Written rather than copied -- see MAX_CREDITS_PER_SEMESTER.
            MAX_CREDITS_PER_SEMESTER,
        )
        for d in documents
    ]


def completed_course_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (
            _text(d.get("_id")),
            _text(d.get("courseId")),
            _text(d.get("userId")),
            _text(d.get("semesterCode")),
            _number(d.get("grade")),
            _number(d.get("gradePoints")),
            _number(d.get("creditsEarned")),
            _integer(d.get("attempt")),
            _text(d.get("source")),
        )
        for d in documents
    ]


def prune_semesters(semesters: Any) -> list[dict[str, Any]]:
    """`semesters[]` reduced to what the source registry declares, and TYPED.

    Two jobs, and the second one is not optional.

    **Pruning.** A live plan's `weeklySchedule` alone can outweigh everything
    readable in the document. None of it is declared, so `find` would never read
    a byte of it -- it would just ride through every fetch on a request with a
    240s budget.

    **Coercion.** A plan stores `"credits": "3.5"` -- a STRING -- where the
    registry declares a quantity. At read time `_convert` coerces it and the
    in-memory engine sees 3.5; inside jsonb it stays a string, and a pushed-down
    `credits > 3.0` compares a JSON string to a number and matches nothing. The
    two engines would disagree on real data, silently. Storing what the
    declaration promises is the fix: the value is coerced ON THE WAY IN, using
    `find`'s own `_coerce`, so both engines see the same number.

    Walks the DECLARATION rather than a hand-written field list, so adding a
    field to `SEMESTER_PLANS` is enough to make the seed carry it.
    """
    from app.agent_core.facts.find import ArrayOf, Sub, _coerce
    from app.agent_core.facts.sources import SEMESTER_PLANS
    from app.agent_core.facts.types import ScalarKind

    def prune(value: Any, spec: Any) -> Any:
        if isinstance(spec, ScalarKind):
            # ObjectIds reach here as bson objects; they are identifiers, and
            # `_coerce` already knows how to read one as a string.
            return _coerce(value, spec)
        if isinstance(spec, Sub):
            if not isinstance(value, Mapping):
                return None
            return _prune_fields(value, spec.fields)
        if isinstance(spec, ArrayOf):
            if not isinstance(value, (list, tuple)):
                return None
            elements = [prune(element, spec.element) for element in value]
            return [element for element in elements if element is not None]
        return None

    def _prune_fields(document: Mapping[str, Any], declared: Mapping[str, Any]) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        for name, spec in declared.items():
            if name not in document:
                continue
            pruned = prune(document[name], spec)
            # Omitted rather than stored as null, exactly as `_convert_fields`
            # does at read time: a value that cannot be honoured must not become
            # a 0 or an empty string that an aggregate would silently include.
            if pruned is not None:
                kept[name] = pruned
        return kept

    spec = SEMESTER_PLANS.fields["semesters"]
    return prune(semesters or [], spec) or []


def semester_plan_rows(documents: Iterable[Mapping[str, Any]]) -> list[tuple]:
    return [
        (
            _text(d.get("_id")),
            _text(d.get("userId")),
            _text(d.get("name")),
            _text(d.get("plannerType")),
            _text(d.get("status")),
            _integer(d.get("version")),
            json.dumps(prune_semesters(d.get("semesters")), ensure_ascii=False),
        )
        for d in documents
    ]


def synthetic_plan_for(profile: Mapping[str, Any]) -> dict[str, Any]:
    """An empty-slot plan for the primary demo student, who has none.

    `semester_plans` is the ONLY source that yields `slots`, which is the input
    `optimize` needs -- so without a plan, the student the rest of the demo is
    built around cannot demonstrate the one tool that places courses into terms.

    The slots are EMPTY on purpose. `optimize` places courses INTO capacity; a
    pre-filled plan would be answering the question the tool exists to answer.
    Structure and `goalCredits` mirror the one real demo plan (2025-2, 19.0
    credits) rather than being invented, and both terms have real offering data
    behind them (2025 spring and summer).
    """
    current = str(profile.get("currentSemesterCode") or "2025-2")
    year, _, index = current.partition("-")
    try:
        following = f"{year}-{int(index) + 1}"
    except ValueError:
        following = current

    return {
        "_id": f"demoplan{profile['userId'][-16:]}",
        "userId": profile["userId"],
        "name": f"תוכנית {current}",
        "plannerType": "manual",
        "status": "draft",
        "version": 1,
        "semesters": [
            {
                "semesterCode": current,
                "order": 1,
                "goalCredits": 19.0,
                "notes": "",
                "plannedCourses": [],
            },
            {
                "semesterCode": following,
                "order": 2,
                "goalCredits": 19.0,
                "notes": "",
                "plannedCourses": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

COLUMNS: dict[str, tuple[str, ...]] = {
    "courses": ("_id", "courseNumber", "title", "titleHebrew", "credits", "faculty",
                "studyFramework", "catalogYear", "status"),
    "course_offerings": ("_id", "courseNumber", "semesterName", "semesterCode",
                         "academicYear", "catalogVersion", "status",
                         "scheduleGroups", "examDates"),
    "degree_programs": ("_id", "name", "totalCredits"),
    "student_profiles": ("userId", "institutionId", "facultyId", "programType", "degreeId",
                         "programSlug", "catalogYear", "currentSemesterCode",
                         "maxCreditsPerSemester"),
    "completed_courses": ("_id", "courseId", "userId", "semesterCode", "grade", "gradePoints",
                          "creditsEarned", "attempt", "source"),
    "semester_plans": ("_id", "userId", "name", "plannerType", "status", "version", "semesters"),
    "prerequisite_edges": ("edge", "course", "requires", "group"),
    "track_courses": ("edge", "track", "course"),
}


async def copy_rows(connection, table: str, rows: Sequence[tuple]) -> int:
    if not rows:
        return 0
    await connection.copy_records_to_table(table, records=rows, columns=list(COLUMNS[table]))
    return len(rows)


def dict_rows(table: str, rows: Iterable[Mapping[str, Any]]) -> list[tuple]:
    """Dicts -> tuples in the table's declared column order."""
    columns = COLUMNS[table]
    return [tuple(row.get(column) for column in columns) for row in rows]


async def seed(arguments: argparse.Namespace) -> int:
    import asyncpg

    env = load_env(REPO_ROOT / ".env")
    wiki_root = arguments.unipilot / WIKI_SUBPATH
    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - started:6.1f}s] {message}", flush=True)

    # --- read every source BEFORE opening the write transaction ---------------
    # A long transaction that also waits on Atlas holds Postgres locks for no
    # reason, and a Mongo failure halfway through would abort a load that had
    # already started.
    log("reading UniPilot's Atlas cluster...")
    mongo = mongo_database(env)
    from bson import ObjectId

    demo_object_ids = [ObjectId(user_id) for user_id in DEMO_USER_IDS]

    catalog = list(mongo.courses.find({}))
    offerings = list(mongo.course_offerings.find({}))
    programs = list(mongo.degree_programs.find({}))
    profiles = list(mongo.student_profiles.find({"userId": {"$in": demo_object_ids}}))
    transcripts = list(mongo.completed_courses.find({"userId": {"$in": demo_object_ids}}))
    plans = list(mongo.semester_plans.find({"userId": {"$in": demo_object_ids}}))
    log(f"  courses={len(catalog)} offerings={len(offerings)} programs={len(programs)} "
        f"profiles={len(profiles)} transcript_rows={len(transcripts)} plans={len(plans)}")

    if len(profiles) != len(DEMO_STUDENTS):
        found = {str(p.get("userId")) for p in profiles}
        raise SystemExit(f"missing demo profiles: {sorted(set(DEMO_USER_IDS) - found)}")

    # --- derive what used to be computed at runtime ---------------------------
    log("deriving the prerequisite graph from catalog prose...")
    prerequisite_rows, unparseable = prerequisite_edge_rows(catalog)
    log(f"  prerequisite_edges={len(prerequisite_rows)} unparseable_courses={len(unparseable)}")

    log(f"deriving curriculum membership from {wiki_root}...")
    if not wiki_root.is_dir():
        raise SystemExit(f"no wiki corpus at {wiki_root}; pass --unipilot at a checkout")
    pages = load_wiki_pages(wiki_root)
    track_rows = track_course_rows(pages)
    log(f"  wiki_pages={len(pages)} track_courses={len(track_rows)} "
        f"tracks={len({row['track'] for row in track_rows})}")

    # --- shape ---------------------------------------------------------------
    plan_documents = list(plans)
    if not any(str(p.get("userId")) == PRIMARY_USER_ID for p in plans):
        primary = next(p for p in profiles if str(p.get("userId")) == PRIMARY_USER_ID)
        plan_documents.append(synthetic_plan_for({**primary, "userId": PRIMARY_USER_ID}))
        log("  synthesised an empty-slot plan for the primary demo student (had none)")

    payload = {
        "courses": course_rows(catalog),
        "course_offerings": offering_rows(offerings),
        "degree_programs": degree_program_rows(programs),
        "student_profiles": profile_rows(profiles),
        "completed_courses": completed_course_rows(transcripts),
        "semester_plans": semester_plan_rows(plan_documents),
        "prerequisite_edges": dict_rows("prerequisite_edges", prerequisite_rows),
        "track_courses": dict_rows("track_courses", track_rows),
    }

    if arguments.dry_run:
        log("dry run -- nothing written. Row counts that WOULD be loaded:")
        for table, rows in payload.items():
            log(f"  {table:20s} {len(rows):6d}")
        return 0

    # --- write ---------------------------------------------------------------
    log("connecting to Postgres through the Supavisor pooler...")
    connection = await asyncpg.connect(**pooler_dsn(env), timeout=30)
    try:
        if arguments.schema:
            log(f"applying {SCHEMA_PATH.relative_to(REPO_ROOT)}...")
            await connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

        transaction = connection.transaction()
        await transaction.start()
        try:
            # One transaction for the whole load: a failure leaves the previous
            # data in place rather than a half-seeded database that looks fine.
            await connection.execute(f"truncate {', '.join(SEEDED_TABLES)} cascade")
            for table in reversed(SEEDED_TABLES):
                loaded = await copy_rows(connection, table, payload[table])
                log(f"  loaded {table:20s} {loaded:6d}")
            await transaction.commit()
        except BaseException:
            await transaction.rollback()
            raise

        log("verifying...")
        problems = await verify(connection)
        for problem in problems:
            log(f"  FAIL {problem}")
        if problems:
            return 1
        log("  all checks passed")
    finally:
        await connection.close()

    # --- the wiki artifact ---------------------------------------------------
    if arguments.skip_wiki:
        log("skipping the wiki artifact (--skip-wiki)")
    else:
        log("building the wiki corpus artifact...")
        artifact = build_corpus_artifact(arguments.unipilot, wiki_root)
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(artifact, ensure_ascii=False).encode("utf-8")
        ARTIFACT_PATH.write_bytes(gzip.compress(encoded, 6))
        log(f"  {ARTIFACT_PATH.relative_to(REPO_ROOT)}: {artifact['chunkCount']} chunks, "
            f"{ARTIFACT_PATH.stat().st_size / 1e6:.2f} MB gzipped "
            f"(from {len(encoded) / 1e6:.1f} MB of JSON)")

    log("done")
    return 0


async def verify(connection) -> list[str]:
    """Checks that would each have shipped a confidently wrong answer.

    Not a smoke test. Every one of these corresponds to a failure the fact layer
    cannot detect for itself: a silently empty table reads as "this student has
    no transcript", stated with full confidence.
    """
    problems: list[str] = []

    async def count(sql: str, *args) -> int:
        return await connection.fetchval(sql, *args) or 0

    expectations = {
        "courses": 2613,
        "course_offerings": 6580,
        "degree_programs": 61,
        "student_profiles": len(DEMO_STUDENTS),
    }
    for table, expected in expectations.items():
        actual = await count(f"select count(*) from {table}")
        if actual != expected:
            problems.append(f"{table}: expected {expected} rows, found {actual}")

    for table in ("completed_courses", "semester_plans", "prerequisite_edges", "track_courses"):
        if await count(f"select count(*) from {table}") == 0:
            problems.append(f"{table} is EMPTY -- every question routed through it would answer 'none'")

    # Each demo student must have a transcript, or the agent answers "you have
    # completed nothing" with full confidence.
    for user_id in DEMO_USER_IDS:
        rows = await count('select count(*) from completed_courses where "userId" = $1', user_id)
        if rows == 0:
            problems.append(f"demo student {user_id} has no completed courses")

    # Every profile must reach its curriculum, which is the `programSlug` ->
    # `track_courses.track` join. A slug that matches no track means the student
    # has no reachable degree requirements.
    unmatched = await connection.fetch(
        'select p."userId", p."programSlug" from student_profiles p '
        'where not exists (select 1 from track_courses t where t.track = p."programSlug")'
    )
    for row in unmatched:
        problems.append(f"programSlug {row['programSlug']!r} ({row['userId']}) matches no track_courses")

    # The known, unrepairable orphan defect -- pinned, not tolerated. Growth
    # means the unidentified writer ran again.
    orphans = await count(
        'select count(distinct c."courseId") from completed_courses c '
        'where not exists (select 1 from courses x where x."_id" = c."courseId")'
    )
    if orphans > KNOWN_ORPHANED_DEMO_COURSE_IDS:
        problems.append(
            f"orphaned demo courseIds grew from {KNOWN_ORPHANED_DEMO_COURSE_IDS} to {orphans}"
        )

    # `semester_plans` is the only source of `slots`. If no plan unnests into a
    # slot carrying both an order and a capacity, `optimize` is advertised with
    # nothing able to feed it.
    slots = await count(
        "select count(*) from semester_plans p, jsonb_array_elements(p.semesters) s "
        "where s ? 'order' and s ? 'goalCredits'"
    )
    if slots == 0:
        problems.append("no plan yields slots -- `optimize` would be advertised with no reachable input")

    # A student profile with no maxCreditsPerSemester sends the planner back to
    # its invisible fallback, which is the thing seeding it was meant to fix.
    unset = await count('select count(*) from student_profiles where "maxCreditsPerSemester" is null')
    if unset:
        problems.append(f"{unset} demo profile(s) have no maxCreditsPerSemester")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unipilot", type=Path, default=DEFAULT_UNIPILOT,
                        help=f"path to a UniPilot checkout (default: {DEFAULT_UNIPILOT})")
    parser.add_argument("--schema", action="store_true",
                        help="apply db/schema.sql before loading (idempotent)")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="skip rebuilding data/wiki_corpus.json.gz")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and derive everything, write nothing")
    arguments = parser.parse_args()

    if not arguments.unipilot.is_dir():
        raise SystemExit(f"no UniPilot checkout at {arguments.unipilot}")

    return asyncio.run(seed(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
