#!/usr/bin/env python3
"""Recover the publication list from the cached Wix publications page.

Why a line stream and not a tag walk
------------------------------------
The whole list lives in ONE Wix rich-text element, and that element uses three
different shapes for what is conceptually the same record::

    # (a) one <h5>, lines separated by <br>
    <h5><b>Erez Weisberg</b>, ... ,<br>Discovery of ...,<br>
        Current Opinion in Microbiology Volume 92 (2026),<br>
        https://doi.org/10.1016/j.mib.2026.102781.</h5>

    # (b) citation in one <h5>, its link in the NEXT <h5>
    <h5><b>Nachmias, N.</b>, ... Nat Commun 17, 1939 (2026).</h5>
    <h5><a href="https://doi.org/10.1038/s41467-026-68710-y">...</a></h5>

    # (c) authors / title / journal each in their own <h5>
    <h5><b>Avi Bograd, Yaara Oppenheimer-Shaanan, Asaf Levy</b></h5>
    <h5>Plasmids, Prophages and Defense Systems are Depleted from ...</h5>

and the older half of the list repeats all three with <p> instead of <h5>.
So the tag tells us nothing about where a record starts.

What DOES mark the boundaries is the blank spacer paragraph Wix inserts between
entries: an element whose only content is the zero-width ``wixGuard`` space. So
this script flattens the block to a list of visible lines (<br>, </p> and </h5>
all end a line), marks the blank ones, and treats a run of non-blank lines as one
entry. That rule holds for all three shapes above.

Bold runs are kept per line because bold is how the page marks lab members - but
only on the recent entries, so a missing bold run is not an error.

Stage 1 (this file) is OFFLINE: it reads the page already cached by
``wix_asset_rescue.py`` and writes the grouped raw entries for inspection.
Nothing is interpreted into authors/title/journal yet - that is stage 2, where
the DOI is resolved against Crossref.

Usage
-----
    python wix_publications_extract.py \
        --page <cache>/publications.html \
        --out <out>/publications.raw.tsv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

LOG = logging.getLogger("wix_pubs")

# The rich-text element that holds the list. The page has four of these; the one
# we want is simply the longest that contains paragraph-ish markup (the biggest
# of the others is 379 KB of inlined JavaScript, which has no <p> or <h5> at all).
MARKER = 'data-testid="richTextElement"'

# Characters Wix uses for spacing that must not count as content. U+200B is the
# wixGuard space that separates entries; U+FEFF and NBSP show up inside lines.
INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"), None)

# Any element that ends a visible line of text.
LINE_BREAKERS = {"br", "p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li"}

RE_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>)\]]+")
RE_YEAR = re.compile(r"\b(19[5-9]\d|20[0-5]\d)\b")
RE_WS = re.compile(r"\s+")


@dataclass
class Line:
    """One visible line of the rich-text block."""

    text: str
    bold: list[str] = field(default_factory=list)  # bold runs, in order
    urls: list[str] = field(default_factory=list)  # href targets on this line

    @property
    def blank(self) -> bool:
        """True for the wixGuard spacer lines that separate entries."""
        return not self.text.strip()


@dataclass
class RawEntry:
    """A run of non-blank lines = one publication, before interpretation."""

    lines: list[Line]

    @property
    def text(self) -> str:
        return " ".join(ln.text for ln in self.lines).strip()

    @property
    def bold_names(self) -> list[str]:
        return [b for ln in self.lines for b in ln.bold]

    @property
    def urls(self) -> list[str]:
        return [u for ln in self.lines for u in ln.urls]

    @property
    def doi(self) -> str:
        """DOI if the entry states one outright (in text or in a link target)."""
        for source in (self.text, *self.urls):
            m = RE_DOI.search(source)
            if m:
                return m.group(0).rstrip(".,;")
        return ""

    @property
    def year(self) -> str:
        """Last 4-digit year in the entry - the publication year sits at the end
        of the journal line, while a title can contain a year too."""
        years = RE_YEAR.findall(self.text)
        return years[-1] if years else ""

    @property
    def co_first(self) -> bool:
        """Asterisks mark co-first authors and must survive into the new site."""
        return "*" in self.text


class RichTextFlattener(HTMLParser):
    """Flatten a Wix rich-text block into :class:`Line` objects.

    Bold is detected two ways because Wix uses both: a real ``<b>``/``<strong>``
    tag, and - far more often - a ``<span>`` carrying ``font-weight:bold``. Note
    that spans also carry ``font-weight:normal`` to *cancel* an outer bold, so the
    depth counter has to be pushed per element, not toggled.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[Line] = []
        self._buf: list[str] = []
        self._bold_runs: list[str] = []
        self._bold_buf: list[str] = []
        self._urls: list[str] = []
        self._weight_stack: list[tuple[str, bool]] = []  # (tag, is_bold)

    # -- line handling ---------------------------------------------------
    def _flush_line(self) -> None:
        text = RE_WS.sub(" ", "".join(self._buf)).strip()
        bold = [RE_WS.sub(" ", b).strip() for b in self._bold_runs]
        self.lines.append(Line(text=text, bold=[b for b in bold if b], urls=list(self._urls)))
        self._buf.clear()
        self._bold_runs.clear()
        self._urls.clear()

    @property
    def _in_bold(self) -> bool:
        return bool(self._weight_stack) and self._weight_stack[-1][1]

    def _close_bold_run(self) -> None:
        run = "".join(self._bold_buf).strip()
        if run:
            self._bold_runs.append(run)
        self._bold_buf.clear()

    # -- HTMLParser hooks ------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag in ("b", "strong"):
            self._weight_stack.append((tag, True))
        elif tag == "span":
            style = (a.get("style") or "").lower()
            if "font-weight:bold" in style.replace(" ", ""):
                self._weight_stack.append((tag, True))
            elif "font-weight:normal" in style.replace(" ", ""):
                self._weight_stack.append((tag, False))
            else:
                # inherit whatever the parent says
                self._weight_stack.append((tag, self._in_bold))
        elif tag == "a" and a.get("href"):
            self._urls.append(a["href"])

        if tag == "br":
            self._close_bold_run()
            self._flush_line()

    def handle_endtag(self, tag: str) -> None:
        if tag in LINE_BREAKERS:
            self._close_bold_run()
            self._flush_line()
        # pop the most recent matching weight frame
        for i in range(len(self._weight_stack) - 1, -1, -1):
            if self._weight_stack[i][0] == tag:
                if self._weight_stack[i][1]:
                    self._close_bold_run()
                del self._weight_stack[i]
                break

    def handle_data(self, data: str) -> None:
        clean = unicodedata.normalize("NFC", data).translate(INVISIBLE)
        clean = clean.replace("\xa0", " ")
        if not clean:
            return
        self._buf.append(clean)
        if self._in_bold:
            self._bold_buf.append(clean)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._close_bold_run()
        if self._buf or self._bold_runs:
            self._flush_line()


def select_block(html: str) -> str:
    """Return the rich-text block that holds the publication list.

    Picked by "longest block that actually contains <p> or <h5>" rather than by
    position, so a Wix layout reshuffle does not silently select the wrong one.
    """
    starts = [m.start() for m in re.finditer(re.escape(MARKER), html)]
    if not starts:
        raise ValueError(f"no {MARKER} in the page - wrong file, or Wix changed the markup")
    bounds = list(zip(starts, starts[1:] + [len(html)]))
    candidates = []
    for s, e in bounds:
        chunk = html[s:e]
        n_para = len(re.findall(r"<(?:p|h5)[^>]*>", chunk))
        if n_para:
            candidates.append((n_para, len(chunk), chunk))
    if not candidates:
        raise ValueError("found rich-text blocks but none contains <p> or <h5>")
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    LOG.info("rich-text blocks: %d total, %d with paragraphs; picked the one with %d paragraphs (%d chars)",
             len(bounds), len(candidates), candidates[0][0], candidates[0][1])
    # The marker sits INSIDE the opening tag, so the slice starts mid-tag. Cut past
    # that tag's '>' or the leftover attributes are parsed as visible text and end
    # up glued to the front of the first citation.
    block = candidates[0][2]
    gt = block.find(">")
    if gt == -1:
        raise ValueError("rich-text block has no closing '>' on its opening tag")
    return block[gt + 1:]


def group_entries(lines: list[Line], min_chars: int = 25) -> list[RawEntry]:
    """Group runs of non-blank lines into entries, split on the blank spacers.

    ``min_chars`` drops the page's own furniture - the "Selected Publications"
    heading and the "(Lab members are in bold)" note are short runs that no
    citation can be confused with.
    """
    entries: list[RawEntry] = []
    run: list[Line] = []
    for ln in lines:
        if ln.blank:
            if run:
                entries.append(RawEntry(run))
                run = []
        else:
            run.append(ln)
    if run:
        entries.append(RawEntry(run))

    kept = [e for e in entries if len(e.text) >= min_chars]
    dropped = len(entries) - len(kept)
    LOG.info("grouped %d runs -> %d entries (%d short runs dropped as page furniture)",
             len(entries), len(kept), dropped)
    return kept


def write_raw_tsv(entries: list[RawEntry], out: Path) -> None:
    """Write one row per entry for eyeball review before anything is interpreted."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["n", "n_lines", "year", "doi", "co_first", "n_bold", "bold_names", "urls", "raw_text"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for i, e in enumerate(entries, 1):
            row = [
                str(i),
                str(len(e.lines)),
                e.year,
                e.doi,
                "1" if e.co_first else "",
                str(len(e.bold_names)),
                " | ".join(e.bold_names),
                " | ".join(dict.fromkeys(e.urls)),
                e.text.replace("\t", " "),
            ]
            fh.write("\t".join(row) + "\n")
    LOG.info("wrote %s (%d entries)", out, len(entries))


def report(entries: list[RawEntry], expect: int) -> int:
    """Print the sanity check. Returns the number of problems worth your eyes."""
    n = len(entries)
    with_doi = sum(1 for e in entries if e.doi)
    with_url = sum(1 for e in entries if e.urls)
    with_year = sum(1 for e in entries if e.year)
    with_bold = sum(1 for e in entries if e.bold_names)
    no_id = [i for i, e in enumerate(entries, 1) if not e.doi and not e.urls]

    LOG.info("--- sanity check ---")
    LOG.info("entries              : %d (expected %d)", n, expect)
    LOG.info("with a DOI           : %d", with_doi)
    LOG.info("with any link        : %d", with_url)
    LOG.info("with a year          : %d", with_year)
    LOG.info("with bold lab members: %d  (bold is only used on the recent entries)", with_bold)
    if no_id:
        LOG.warning("entries with neither DOI nor link (stage 2 cannot resolve these): %s",
                    ", ".join(map(str, no_id)))
    if n != expect:
        LOG.warning("entry count is %d, not the expected %d - check the TSV before stage 2", n, expect)

    problems = len(no_id) + (1 if n != expect else 0)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--page", type=Path, required=True, help="cached publications.html")
    ap.add_argument("--out", type=Path, required=True, help="publications.raw.tsv to write")
    ap.add_argument("--expect-entries", type=int, default=28,
                    help="expected number of publications (default 28, from the site inventory)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)

    if not args.page.is_file():
        LOG.error("page not found: %s", args.page)
        return 2

    html = args.page.read_text(encoding="utf-8")
    LOG.info("read %s (%d bytes)", args.page, len(html))

    block = select_block(html)
    flat = RichTextFlattener()
    flat.feed(block)
    flat.close()
    LOG.info("flattened to %d lines (%d blank spacers)",
             len(flat.lines), sum(1 for ln in flat.lines if ln.blank))

    entries = group_entries(flat.lines)
    write_raw_tsv(entries, args.out)
    report(entries, args.expect_entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
