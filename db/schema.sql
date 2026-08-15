-- UniPilot Agent -- Supabase (Postgres 17) schema.
--
-- DERIVED FROM `app/agent_core/facts/sources.py`, NOT from the Mongo documents
-- directly. The source registry is what `find` reads: it declares which fields
-- exist and what KIND each one is, and this file is that declaration expressed
-- as tables. Where the two could disagree, the registry wins -- a column the
-- registry does not declare is never read, and a declared field with no column
-- is silently absent on every record, which is the confident-silence failure the
-- whole fact layer exists to prevent.
--
-- Profiled against the live Atlas database on 2026-08-14 (row counts and
-- nullability below are measured, not assumed).
--
-- THREE CONVENTIONS, each load-bearing:
--
-- 1. COLUMN NAMES ARE THE DECLARED FIELD NAMES, VERBATIM -- quoted camelCase.
--    `find` looks up `document["courseNumber"]`. A snake_case column plus a
--    translation layer would be a second place for names to drift apart, and a
--    name that drifts here surfaces as a field that is merely ABSENT rather than
--    as an error. One name, no mapping.
--
-- 2. ObjectIds ARE `text` -- the 24-character hex string.
--    Mongo needed `SourceSchema.object_id_fields` because a string predicate
--    compared against a BSON ObjectId matched nothing and reported the empty
--    result as COMPLETE -- a student with no transcript, stated with full
--    confidence. Postgres has no such type, so a model's string filter compares
--    against a string column and that failure becomes structurally impossible.
--
-- 3. ARRAY-VALUED FIELDS ARE `jsonb`, PRUNED TO THE DECLARATION.
--    `find._convert` already turns a list-of-mappings into a nested Collection,
--    so jsonb lands in exactly the shape it expects. The seed drops every
--    undeclared key on the way in (see `semester_plans.semesters` below).
--
-- Apply with `psql` through the Supavisor pooler, or paste into the Supabase SQL
-- editor. Idempotent: safe to re-run.

begin;

-- ---------------------------------------------------------------------------
-- courses -- the catalog. 2,613 rows, whole catalog seeded.
-- ---------------------------------------------------------------------------
-- Two identities, and the difference is the single most expensive thing to get
-- wrong here: `_id` is what `completed_courses.courseId` joins to, while
-- `courseNumber` is the code a human says out loud and the key `find` sorts on.
-- A live eval filtered `prerequisite_edges.course` by a course's `_id` and got
-- nothing back, because the transcript keys courses by ObjectId and the graph
-- keys them by number.
create table if not exists courses (
    "_id"             text primary key,
    "courseNumber"    text not null unique,
    "title"           text,
    "titleHebrew"     text,
    "credits"         double precision,
    "faculty"         text,
    "studyFramework"  text,
    "catalogYear"     integer,
    "status"          text
    -- NOTE: sources.COURSES also declares `academicYear`, which no `courses`
    -- document carries (0 of 2,613). Deliberately NOT given a column -- see the
    -- schema notes; the declaration should be dropped rather than faked with an
    -- always-null column.
);

create index if not exists courses_faculty_idx on courses ("faculty");

-- ---------------------------------------------------------------------------
-- course_offerings -- when a course actually runs. 6,580 rows, all seeded.
-- ---------------------------------------------------------------------------
-- `forecast` keys on `semesterName` (spring | summer | winter), which is why
-- offering questions are answerable at all. `semesterCode` is an INTEGER here
-- (200/201/202) while `completed_courses.semesterCode` is the text "2025-2" --
-- two different vocabularies that the registry already types apart.
-- `scheduleGroups` and `examDates` are deliberately NOT declared in
-- `sources.COURSE_OFFERINGS`, so `find` never reads them and the model never
-- sees them. They exist for the TERM PLANNER, which reads this table directly:
-- a conflict-free plan needs the actual lesson times, and an exam summary needs
-- the dates. Declaring them would put raw Hebrew schedule blobs (~465 B per
-- offering) into the model's view of the source, inflating a prompt that is
-- already the biggest cost lever, to serve a question the model never asks --
-- it asks `plan_term`, and the planner does this join itself.
create table if not exists course_offerings (
    "_id"             text primary key,
    "courseNumber"    text not null references courses ("courseNumber"),
    "semesterName"    text,
    "semesterCode"    integer,
    "academicYear"    integer,
    "catalogVersion"  text,
    "status"          text,
    "scheduleGroups"  jsonb not null default '[]'::jsonb,
    "examDates"       jsonb not null default '{}'::jsonb
);

-- Idempotent is not the same as MIGRATING. `create table if not exists` skips an
-- existing table wholesale, so a column added to the definition above never
-- reaches a database that was seeded before it -- the seed then fails on a
-- column that this file plainly declares. Additive changes are repeated as
-- `add column if not exists` so re-running converges an existing database on the
-- schema rather than merely declining to break it.
alter table course_offerings add column if not exists "scheduleGroups" jsonb not null default '[]'::jsonb;
alter table course_offerings add column if not exists "examDates" jsonb not null default '{}'::jsonb;

-- The FK is real: 0 of 6,580 offerings reference a course that does not exist.
create index if not exists course_offerings_course_idx on course_offerings ("courseNumber");
create index if not exists course_offerings_period_idx on course_offerings ("academicYear", "semesterName");

-- ---------------------------------------------------------------------------
-- degree_programs -- 61 rows, all seeded.
-- ---------------------------------------------------------------------------
create table if not exists degree_programs (
    "_id"           text primary key,
    "name"          text,
    "totalCredits"  double precision
);

-- ---------------------------------------------------------------------------
-- student_profiles -- the 4 demo students only.
-- ---------------------------------------------------------------------------
-- The registry keys this on `userId`, changed from `institutionId` during the
-- port: `institutionId` is a tenant name, not an identity, and all 951 live
-- profiles share one of its two values ('technion', 'uni-main'). Keyed on it,
-- every profile had the same identity -- unordered under `find` and
-- indistinguishable under `difference`. `userId` is unique and is the column
-- every query filters on, so it is both the declared key and the primary key.
--
-- `programSlug` IS a real top-level field (32 of 951 non-null) and is what makes
-- a profile demoable: it is the filter into `track_courses`.
--
-- `maxCreditsPerSemester` is declared flat by the registry but does NOT exist as
-- a flat field on any live profile (0 of 951; 28 carry it under `preferences`).
-- Given a real column here and set explicitly for the demo students, because the
-- planner otherwise falls back to 18 and the fallback is invisible in an answer.
create table if not exists student_profiles (
    "userId"                 text primary key,
    "institutionId"          text not null,
    "facultyId"              text,
    "programType"            text,
    "degreeId"               text references degree_programs ("_id"),
    "programSlug"            text,
    "catalogYear"            integer,
    "currentSemesterCode"    text,
    "maxCreditsPerSemester"  double precision
);

create index if not exists student_profiles_slug_idx on student_profiles ("programSlug");

-- ---------------------------------------------------------------------------
-- completed_courses -- the transcript. ~77 rows for the 4 demo students.
-- ---------------------------------------------------------------------------
-- NO FOREIGN KEY ON "courseId", and that is the point rather than an omission.
-- 28% of live transcript rows (155 of 554; 11 of the 59 distinct courseIds in
-- the demo set) reference a catalog document that does not exist. The writer was
-- never identified and the records carry no course number or title, so they
-- cannot be repaired. A FK would refuse the load; instead the rows come in and
-- the join to `courses` fails CLOSED, which is why the wrong answer never
-- reaches anyone. The seed asserts the count has not grown.
--
-- "_id" is not declared in the registry, so `find` never reads it -- it exists
-- purely as the deterministic tie-break for ORDER BY, since the declared key
-- `courseId` is not unique (70 duplicate groups live: retakes, and one course
-- completed by several students).
create table if not exists completed_courses (
    "_id"            text primary key,
    "courseId"       text not null,
    "userId"         text not null,
    "semesterCode"   text,
    "grade"          double precision,
    "gradePoints"    double precision,
    "creditsEarned"  double precision,
    "attempt"        integer,
    "source"         text
);

-- `creditsEarned` cannot be summed to get credits toward the degree, and the
-- data is what proves it: course 01040166 is graded 30 -- a fail -- and still
-- carries its full 5.5. The field is otherwise a genuine earned-credit column
-- (03240053 has catalog credits 3.0 and `creditsEarned` 0.0 because it was
-- failed), so that one row contradicts both the regulation and the rest of the
-- transcript. Summing it told a live student 135 credits when the answer is
-- 129.5.
--
-- The rule is not the model's to remember. `concepts/regulations-undergraduate.md`
-- § Grading: "Passing grade: 55 or above / Below 55 = failing", and 3.1.3(d)
-- requires a failed mandatory course to be re-taken -- so it has earned nothing.
-- GENERATED means the arithmetic happens in the database and no caller, model or
-- otherwise, can reach for the raw field by accident and be wrong.
alter table completed_courses
    add column if not exists "creditsCounted" double precision
    generated always as (
        case when "grade" >= 55 then coalesce("creditsEarned", 0) else 0 end
    ) stored;

alter table completed_courses
    add column if not exists "passed" boolean
    generated always as ("grade" >= 55) stored;

create index if not exists completed_courses_user_idx on completed_courses ("userId");
create index if not exists completed_courses_course_idx on completed_courses ("courseId");

-- ---------------------------------------------------------------------------
-- semester_plans -- the demo students' plans.
-- ---------------------------------------------------------------------------
-- `semesters` is the ONLY nested declaration in the registry, and the only
-- stored thing shaped like `optimize`'s slots: an ordered sequence with a
-- per-slot capacity (`order`, `goalCredits`), each holding its own
-- `plannedCourses[]`. The route is find -> unnest -> unnest.
--
-- The seed PRUNES each element to the declared fields only. A live plan document
-- carries `weeklySchedule`, `selectedLessonEvents`, `constraintsSnapshot` and
-- more -- none declared, so none readable, and all of it would otherwise ride
-- into every fetch as dead payload on a request already fighting a 240s budget.
create table if not exists semester_plans (
    "_id"           text primary key,
    "userId"        text not null,
    "name"          text,
    "plannerType"   text,
    "status"        text,
    "version"       integer,
    "semesters"     jsonb not null default '[]'::jsonb
);

create index if not exists semester_plans_user_idx on semester_plans ("userId");

-- ---------------------------------------------------------------------------
-- prerequisite_edges -- MATERIALIZED from the knowledge graph.
-- ---------------------------------------------------------------------------
-- Was a `DerivedSchema` computed at runtime by walking a networkx graph. Now a
-- table, which is what lets `networkx` (and 16MB of bundle) leave the deployment
-- entirely: nothing derives this on the cold path any more.
--
-- `group` is what makes the result honest. Edges sharing a group are
-- ALTERNATIVES -- any one satisfies the requirement -- while different groups
-- are each mandatory. Flattening the AND/OR tree would make "A or B" look like
-- two obligations and double-count a choice.
--
-- `group` is a reserved word; quoted like every other column here.
--
-- The primary key is (edge, "group") rather than `edge` alone: one course may
-- require another in two different OR-branches, which produces the same edge id
-- twice under different groups. `find` still keys on `edge` as declared.
create table if not exists prerequisite_edges (
    "edge"      text not null,
    "course"    text not null,
    "requires"  text not null,
    "group"     text not null,
    primary key ("edge", "group")
);

create index if not exists prerequisite_edges_course_idx on prerequisite_edges ("course");
create index if not exists prerequisite_edges_requires_idx on prerequisite_edges ("requires");

-- ---------------------------------------------------------------------------
-- track_courses -- MATERIALIZED curriculum membership (~2,944 edges).
-- ---------------------------------------------------------------------------
-- The `contains` edges the graph builds from a track page's wikilinks: "which
-- courses belong to my degree", filtered by the student's `programSlug`.
--
-- Membership ONLY, not the required/elective split -- that lives in the credit
-- breakdown table on the track's wiki page and is reached with
-- search_corpus + interpret. The edge records the link, not the section it sat
-- under, and claiming otherwise here would be a schema lie.
create table if not exists track_courses (
    "edge"    text primary key,
    "track"   text not null,
    "course"  text not null
);

create index if not exists track_courses_track_idx on track_courses ("track");

-- ---------------------------------------------------------------------------
-- spent_confirmations -- the confirmation ledger.
-- ---------------------------------------------------------------------------
-- Replaces `MongoLedger`, and keeps the property that made it correct: the token
-- IS the primary key, so spending is a single INSERT ... ON CONFLICT DO NOTHING
-- and the database itself arbitrates. A read-then-write ledger has a window
-- where two concurrent attempts both see "unspent" and both proceed -- which is
-- exactly the shape of a double-submit.
create table if not exists spent_confirmations (
    "token"     text primary key,
    "spent_at"  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- agent_conversations -- durable follow-up context.
-- ---------------------------------------------------------------------------
-- Replaces `MongoConversations`, which appended to one document per
-- conversation. A row per exchange instead: appending is a plain INSERT with no
-- read-modify-write, and `seq` gives the oldest-first ordering `history()`
-- promises without depending on array order.
--
-- Deliberately stores QUESTIONS AND ANSWERS AS TEXT and nothing else. Re-serving
-- a derived fact from an earlier run as though it were still true is the
-- memory-contamination failure the fact layer is built to avoid.
create table if not exists agent_conversations (
    "conversation_id"  text not null,
    "seq"              bigint generated always as identity,
    "question"         text not null,
    "answer"           text not null,
    "created_at"       timestamptz not null default now(),
    primary key ("conversation_id", "seq")
);

-- ---------------------------------------------------------------------------
-- Row-level security.
-- ---------------------------------------------------------------------------
-- Enabled with NO policies, on purpose. The agent connects as `postgres` through
-- the pooler and the seed uses the service role; both bypass RLS. Anything
-- holding only the anon key -- which is the key a browser could reach -- gets
-- nothing. The GUI has no authentication by course requirement, so the anon key
-- must not be a route to student records.
alter table courses              enable row level security;
alter table course_offerings     enable row level security;
alter table degree_programs      enable row level security;
alter table student_profiles     enable row level security;
alter table completed_courses    enable row level security;
alter table semester_plans       enable row level security;
alter table prerequisite_edges   enable row level security;
alter table track_courses        enable row level security;
alter table spent_confirmations  enable row level security;
alter table agent_conversations  enable row level security;

commit;
