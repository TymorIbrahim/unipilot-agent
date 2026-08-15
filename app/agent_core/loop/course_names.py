"""Course code -> human-readable name, for rendering answers a student can read.

The live evals shipped grounded, correct, near-unreadable answers: "The courses
on your record with a final grade above 90 are: 00940704, 00940219, 03240033."
Seven of ten answers in the 2026-07-18 run carried bare codes.

The name cannot come from the model. The grounding backstop checks NUMERALS
only (`answer_boundary._NUM`), so a course name typed into prose is never
validated -- and a plausible fabricated name attached to a real code is worse
than no name, because nothing about it invites doubt. So the name is read from
the catalog here, in code, and slotted at the answer boundary exactly as every
other grounded value is.

Source is the course wiki page's frontmatter title, which reads
`<code> — <English name> (<Hebrew name>)` and covers 2601 of 2611 course pages.
`engine.graph.nodes[code]["name"]` is NOT the source: it is Hebrew-only and
missing outright for some courses (00940704, 01040065 among them).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache


logger = logging.getLogger(__name__)

# A course code as it appears in a fact value: exactly 8 digits. Narrow on
# purpose -- a grade, a credit total and a semester code must never be looked up
# as if they were courses.
_COURSE_CODE = re.compile(r"^\d{8}$")
# The same shape, unanchored, for scanning prose rather than testing one value.
_COURSE_CODE_IN_TEXT = re.compile(r"\b\d{8}\b")
# A course code as the wiki RENDERS it: 8 digits with the leading zero dropped.
# Used by `canonical_course_code` to restore it; see there for why.
_SEVEN_DIGIT_CODE = re.compile(r"^\d{7}$")
_FRONTMATTER_TITLE = re.compile(r"^title:\s*\"?(.+?)\"?\s*$", re.M)
# The title's leading `<code> — ` prefix; the code is already in the answer.
_TITLE_LEAD = re.compile(r"^\s*\d{6,8}\s*[—\-–]\s*")
_TRAILING_PARENS = re.compile(r"\s*\(([^()]*)\)\s*$")
_HEBREW = re.compile(r"[֐-׿]")
# Frontmatter sits at the top; no need to scan whole course pages.
_FRONTMATTER_SCAN_CHARS = 1500


def _english_name(title: str) -> str | None:
    """The English portion of a wiki title, or None if it has none.

    Drops ONLY a trailing parenthesised group that contains Hebrew. Stripping
    every parenthesised group instead would turn "Introduction to Data
    Engineering (Advanced)" into the name of a different course.
    """
    name = _TITLE_LEAD.sub("", title.strip())
    trailing = _TRAILING_PARENS.search(name)
    if trailing and _HEBREW.search(trailing.group(1)):
        name = name[: trailing.start()].strip()
    if not name or _HEBREW.search(name):
        return None
    return name


@lru_cache(maxsize=1)
def _name_index() -> dict[str, str]:
    """code -> English name, built once from the loaded wiki pages (~20ms).

    Degrades to an empty index if the graph is not configured: a missing name
    costs readability, never correctness, so it must not raise into an answer.
    """
    # Imported lazily, and optionally. In UniPilot this index was built from a
    # live in-process graph engine; here the names are precomputed at seed time
    # and injected via `set_catalog_names`, so there is no graph to consult and
    # no reason for its absence to be an import-time error.
    try:
        from app.retrieval.graph_engine.graph_registry import graph_registry

        engine = graph_registry.get_engine()
    except Exception:  # noqa: BLE001 -- an absent graph is the normal case here
        return {}
    index: dict[str, str] = {}
    for slug, code in engine.slug_to_course_code.items():
        content = (engine.wiki_pages.get(slug) or {}).get("content") or ""
        title = _FRONTMATTER_TITLE.search(content[:_FRONTMATTER_SCAN_CHARS])
        if not title:
            continue
        name = _english_name(title.group(1))
        if name:
            index[code] = name
    return index


# code -> catalog title, loaded once at startup by `load_catalog_names`. The wiki
# index above is always preferred because its names are English; this covers what
# the wiki does not. The ISE wiki holds 2601 courses, but a student's record also
# carries general electives and humanities that were never wiki'd -- 03240305
# ("היסטוריה של המדע") shipped as a bare code in a live 2026-07-19 answer, sitting
# among eight correctly-named courses. A Hebrew title is not ideal inside an
# English sentence, but a student can read it; an 8-digit number tells them
# nothing at all.
_catalog_names: dict[str, str] = {}


async def load_catalog_names() -> int:
    """Load code -> catalog title from the catalog table, returning how many.

    Degrades to an empty map on ANY failure, for the same reason `_name_index`
    does: a missing name costs readability, never correctness. It must never
    raise into an answer, and must never block service startup.

    That tolerance hid a real break through the port. This read
    `get_settings().courses_collection`, a setting this configuration does not
    define, so every call raised `AttributeError`, was swallowed here, and left
    the map empty -- every course rendered as a bare 8-digit code with nothing
    in the logs but one warning. Fail-soft and fail-silent are not the same
    thing, so the failure is now logged with the count it managed to load.
    """
    global _catalog_names
    try:
        from app.db.postgres import get_database

        database = await get_database()
        rows = await database.fetch(
            'select "courseNumber", "title" from courses where "title" is not null'
        )
        loaded: dict[str, str] = {}
        for row in rows:
            code, title = row.get("courseNumber"), row.get("title")
            if isinstance(code, str) and isinstance(title, str) and title.strip():
                loaded[code] = title.strip()
        _catalog_names = loaded
    except Exception:  # noqa: BLE001 -- readability fallback, never fatal
        logger.warning("catalog course names unavailable; falling back to bare codes", exc_info=True)
        return 0
    logger.info("catalog course names loaded: %d", len(_catalog_names))
    return len(_catalog_names)


def course_display_name(value: str) -> str | None:
    """The course's display name, or None if `value` is not a known course code.

    Wiki first (English), catalog second (usually Hebrew), bare code last.
    """
    if not _COURSE_CODE.match(value or ""):
        return None
    return _name_index().get(value) or _catalog_names.get(value)


def canonical_course_code(value: str) -> str:
    """Restore the leading zero the wiki drops from a course code.

    Technion course codes are 8 digits (`_COURSE_CODE`), but the wiki RENDERS them
    one digit short: the wikilink `[[00960600-organizational-behavior|0960600]]`
    displays `0960600`, and the extractor reads that label. A 7-digit code will
    not join to the catalog's 8-digit `courseNumber`, so a whole elective list
    extracted from the wiki silently matches nothing. Left-pad the one missing
    zero; leave anything that is not a bare 7-digit run untouched -- 8-digit codes,
    6-digit program codes, slugs -- so this is safe over any extracted identifier.
    """
    return f"0{value}" if _SEVEN_DIGIT_CODE.match(value or "") else value


def course_codes_in(text: str) -> set[str]:
    """Every course code appearing in free text.

    The unanchored twin of `_COURSE_CODE`: that one asks "is this value a course
    code", this one asks "which course codes does this prose mention".
    """
    return set(_COURSE_CODE_IN_TEXT.findall(text or ""))


def set_catalog_names(names: dict[str, str]) -> None:
    """Test hook -- seeds the fallback without a database."""
    global _catalog_names
    _catalog_names = dict(names)


def reset_course_name_index() -> None:
    """Test hook -- the index is built from whichever graph engine is loaded."""
    _name_index.cache_clear()
    set_catalog_names({})
