"""Render the architecture diagram served by `GET /api/model_architecture`.

    ./.venv/bin/python scripts/render_architecture.py

**Generated from `app/tracing/modules.py`, never drawn by hand.** The spec
requires sub-module names to be identical across three surfaces -- the diagram,
the `steps` log, and every description we publish -- and three hand-maintained
copies drift the first time one is renamed, invisibly, until a grader compares
them. Reading `MODULES` here means the diagram cannot disagree with the trace: a
module renamed in one place is renamed in the picture on the next run, and a
module added without a box is a `KeyError` rather than an omission.

Run at DEV time and the PNG is committed. Pillow never reaches the deployment --
`/api/model_architecture` serves a static file, so the endpoint costs nothing at
runtime and cannot fail for want of a drawing library.

Written to `data/`, NOT `public/`. Vercel serves `public/` as static assets
handled by its edge, which means those files are not in the serverless function's
filesystem at all: the endpoint read them fine locally and returned 503 in
production. `data/` ships inside the bundle, which is where a file the FUNCTION
must open has to live.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.tracing.modules import MODULES  # noqa: E402

OUTPUT = REPO_ROOT / "data" / "architecture.png"

WIDTH, HEIGHT = 1680, 1350
BACKGROUND = (255, 255, 255)
INK = (28, 28, 31)
MUTED = (110, 110, 122)
LINE = (198, 198, 208)
# LLM modules are filled, deterministic ones outlined. That distinction is the
# single most useful thing the picture can carry: `steps` contains exactly the
# filled boxes, so a reader can predict the trace from the diagram.
LLM_FILL = (222, 232, 254)
LLM_EDGE = (47, 91, 216)
PLAIN_FILL = (245, 245, 248)
PLAIN_EDGE = (150, 150, 162)
STORE_FILL = (240, 249, 244)
STORE_EDGE = (18, 121, 79)

FONT_DIR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD_DIR = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in ([BOLD_DIR, FONT_DIR] if bold else [FONT_DIR, BOLD_DIR]):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_F = _font(30, bold=True)
NAME_F = _font(19, bold=True)
BODY_F = _font(14)
SMALL_F = _font(13)
TAG_F = _font(12, bold=True)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    name: str,
    subtitle: str,
    *,
    fill,
    edge,
    tag: str | None = None,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=edge, width=2)
    draw.text((x0 + 16, y0 + 13), name, font=NAME_F, fill=INK)
    if tag:
        width = draw.textlength(tag, font=TAG_F)
        draw.rounded_rectangle(
            (x1 - width - 26, y0 + 13, x1 - 10, y0 + 33), radius=8, fill=edge
        )
        draw.text((x1 - width - 18, y0 + 16), tag, font=TAG_F, fill=(255, 255, 255))
    y = y0 + 42
    for line in wrap(draw, subtitle, BODY_F, (x1 - x0) - 32):
        draw.text((x0 + 16, y), line, font=BODY_F, fill=MUTED)
        y += 18


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    label: str = "",
    colour=LINE,
    both: bool = False,
) -> None:
    draw.line([start, end], fill=colour, width=2)
    for tip, tail in ((end, start),) + (((start, end),) if both else ()):
        dx, dy = tip[0] - tail[0], tip[1] - tail[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        draw.polygon(
            [
                tip,
                (tip[0] - 11 * ux + 5 * px, tip[1] - 11 * uy + 5 * py),
                (tip[0] - 11 * ux - 5 * px, tip[1] - 11 * uy - 5 * py),
            ],
            fill=colour,
        )
    if label:
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        width = draw.textlength(label, font=SMALL_F)
        draw.rectangle(
            (mid[0] - width / 2 - 6, mid[1] - 10, mid[0] + width / 2 + 6, mid[1] + 10),
            fill=BACKGROUND,
        )
        draw.text((mid[0] - width / 2, mid[1] - 8), label, font=SMALL_F, fill=MUTED)


def render() -> Path:
    by_name = {module.name: module for module in MODULES}
    # Every module gets a box, and every box names a real module. A module added
    # to the registry without a place here fails loudly rather than going
    # unpictured while still appearing in `steps`.
    placed = {
        "FrontDoor", "ReasoningLoop", "FactDispatch", "Interpreter",
        "ListInterpreter", "AnswerBoundary", "AnswerVerify",
    }
    missing = set(by_name) - placed
    if missing:
        raise SystemExit(f"modules with no box in the diagram: {sorted(missing)}")
    unknown = placed - set(by_name)
    if unknown:
        raise SystemExit(f"diagram draws modules that do not exist: {sorted(unknown)}")

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((60, 44), "UniPilot Agent - model architecture", font=TITLE_F, fill=INK)
    draw.text(
        (60, 84),
        "Every filled box is an LLM call and appears in the `steps` trace of POST /api/execute, in order. "
        "Outlined boxes are deterministic code.",
        font=BODY_F,
        fill=MUTED,
    )

    def style(name: str):
        module = by_name[name]
        return (
            (LLM_FILL, LLM_EDGE, "LLM") if module.calls_llm else (PLAIN_FILL, PLAIN_EDGE, None)
        )

    def draw_module(name: str, xy, subtitle: str) -> None:
        fill, edge, tag = style(name)
        box(draw, xy, name, subtitle, fill=fill, edge=edge, tag=tag)

    # --- the request spine, left column ------------------------------------
    draw.text((62, 140), "REQUEST", font=TAG_F, fill=MUTED)
    box(
        draw, (60, 162, 400, 254), "POST /api/execute",
        '{ "prompt": "..." }  +  demo student from the GUI selector',
        fill=(250, 250, 252), edge=LINE,
    )
    draw_module(
        "FrontDoor", (60, 290, 400, 384),
        "Resolves which student the question concerns and seeds the loop's opening facts.",
    )
    draw_module(
        "ReasoningLoop", (60, 420, 400, 536),
        "Reads the facts derived so far and decides the next move: call tools, or answer. "
        "The only module that chooses what happens next.",
    )
    draw_module(
        "AnswerBoundary", (60, 764, 400, 880),
        "Refuses any answer containing a number no fact produced. The grounding guarantee, "
        "enforced in code rather than requested in a prompt.",
    )
    draw_module(
        "AnswerVerify", (60, 916, 400, 1032),
        "Replays the finished answer's own numbers against post-conditions: no impossible grade, "
        "no out-of-range GPA, a plan's minimums must hold.",
    )
    box(
        draw, (60, 1068, 400, 1160), "response",
        "{ status, error, response, steps } - the same four fields on success and failure.",
        fill=(250, 250, 252), edge=LINE,
    )

    # --- the tool machinery, right ------------------------------------------
    draw.text((622, 140), "TOOL LAYER (one turn of the loop)", font=TAG_F, fill=MUTED)
    draw_module(
        "FactDispatch", (600, 420, 940, 536),
        "Executes the tool calls the loop requests and admits results as typed, "
        "provenance-tagged facts. Ten primitives; no free-form SQL.",
    )
    draw_module(
        "Interpreter", (1000, 290, 1340, 406),
        "Reads ONE value out of a retrieved passage and returns the quote it came from, "
        "so the value can be checked against its source.",
    )
    draw_module(
        "ListInterpreter", (1000, 432, 1340, 548),
        "The plural of Interpreter: every listed value, each with its own quote, "
        "so an invented entry is caught per element.",
    )

    # --- stores --------------------------------------------------------------
    draw.text((622, 594), "GROUNDING SOURCES", font=TAG_F, fill=MUTED)
    box(
        draw, (600, 618, 940, 718), "Supabase (Postgres)",
        "Catalog, offerings, transcripts, plans, and the materialised "
        "prerequisite / curriculum graphs.",
        fill=STORE_FILL, edge=STORE_EDGE, tag="DB",
    )
    box(
        draw, (1000, 618, 1340, 718), "Pinecone + wiki corpus",
        "Hybrid retrieval: BM25 over 4,895 precomputed chunks blended with vector "
        "search, 40/60.",
        fill=STORE_FILL, edge=STORE_EDGE, tag="RAG",
    )

    # --- edges ---------------------------------------------------------------
    arrow(draw, (230, 254), (230, 290))
    arrow(draw, (230, 384), (230, 420))
    arrow(draw, (400, 452), (600, 452), label="tool calls")
    arrow(draw, (600, 504), (400, 504), label="typed facts")
    arrow(draw, (230, 536), (230, 764), label="candidate answer")
    arrow(draw, (230, 880), (230, 916))
    arrow(draw, (230, 1032), (230, 1068))
    arrow(draw, (940, 462), (1000, 368), label="prose")
    arrow(draw, (940, 488), (1000, 488))
    arrow(draw, (770, 536), (770, 618), both=True)
    arrow(draw, (940, 536), (1120, 618), both=True)

    # A refusal is not a dead end -- it re-enters the loop with the reason, which
    # is why the agent recovers from its own ungrounded drafts instead of failing.
    draw.line([(60, 822), (30, 822), (30, 478), (60, 478)], fill=LLM_EDGE, width=2)
    draw.polygon([(60, 478), (49, 473), (49, 483)], fill=LLM_EDGE)
    draw.text((36, 620), "refused:", font=SMALL_F, fill=LLM_EDGE)
    draw.text((36, 636), "retry with", font=SMALL_F, fill=LLM_EDGE)
    draw.text((36, 652), "the reason", font=SMALL_F, fill=LLM_EDGE)

    # --- legend --------------------------------------------------------------
    # Laid out in two passes so the panel FITS ITS CONTENTS. The border used to
    # be a hardcoded y=1296, which is fine until a module description grows: the
    # `AnswerVerify` role gained three post-conditions and its last two lines
    # rendered straight through the border and off the panel. A diagram
    # generated from `MODULES` so it can never disagree with the trace should
    # not need hand-editing when the text it reads gets longer.
    entries = []
    for module in MODULES:
        marker = "appears in steps" if module.calls_llm else "deterministic - no model call"
        # The role's own closing sentence says what the tag beside the name
        # already says. The strip was written with a HYPHEN while `modules.py`
        # uses an em-dash, so it never matched and every deterministic module
        # printed the redundant line -- six wasted lines across four boxes, which
        # is most of the overflow.
        body = module.role
        for tail in (" Deterministic — no model call.", " Deterministic - no model call."):
            body = body.replace(tail, "")
        entries.append((module, marker, wrap(draw, body, BODY_F, 660)))

    top = 754
    height = 46 + sum(26 + 17 * len(lines) + 8 for _module, _marker, lines in entries)
    if top + height > HEIGHT - 20:
        # Fitting the panel to its text moved the failure rather than removing
        # it: the border no longer cuts through a description, but the CANVAS
        # still can. Loud, because the last one was silent -- the overflow only
        # showed up by looking at the picture.
        raise SystemExit(
            f"the legend needs {top + height}px but the canvas is {HEIGHT}px. "
            f"Raise HEIGHT to at least {top + height + 20}, or shorten a module role."
        )
    draw.rounded_rectangle((600, top, 1340, top + height), radius=12, outline=LINE, width=1)
    draw.text((620, top + 16), "HOW TO READ THE TRACE", font=TAG_F, fill=MUTED)
    y = top + 46
    for module, marker, lines in entries:
        fill, edge, _ = style(module.name)
        draw.rounded_rectangle((622, y + 2, 646, y + 20), radius=5, fill=fill, outline=edge, width=2)
        draw.text((658, y), module.name, font=NAME_F, fill=INK)
        draw.text((658 + draw.textlength(module.name, font=NAME_F) + 12, y + 3), marker,
                  font=SMALL_F, fill=MUTED)
        y += 26
        for line in lines:
            draw.text((658, y), line, font=BODY_F, fill=MUTED)
            y += 17
        y += 8

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True)
    return OUTPUT


if __name__ == "__main__":
    path = render()
    print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
