#!/usr/bin/env python3
"""Dump the visible text of a cached Wix page, in document order.

Why this exists
---------------
The page copy is the last thing still only on Wix. It cannot be scraped by
regex: Wix wraps every run in nested spans and inlines ~380 KB of JavaScript
into the same document, so a naive tag strip returns mostly minified code.

This walks the markup instead and keeps only what a reader sees:

* headings (h1-h6) tagged ``H1:`` .. ``H6:`` so the document structure survives
  into the markdown by hand rather than being guessed;
* paragraphs and list items;
* image references, as ``IMG: <src>``, so a figure is not silently dropped;
* link targets, as ``LINK: <text> -> <href>``.

``<script>``, ``<style>`` and ``<noscript>`` contents are discarded outright,
which is what makes the output readable at all.

The output is for a human to turn into markdown. It is deliberately NOT a
markdown converter: heading levels, what counts as a caption and which of the
Wix furniture lines to drop are editorial calls, and a converter would make
them silently.

Usage
-----
    python wix_page_text.py --page <cache>/index.html
    python wix_page_text.py --page <cache>/contact.html --min-chars 3
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

LOG = logging.getLogger("wix_text")

DROP_CONTENT = {"script", "style", "noscript", "template", "svg"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCKS = HEADINGS | {"p", "li", "blockquote", "figcaption", "td", "th", "dd", "dt"}

INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"), None)
RE_WS = re.compile(r"\s+")

# Wix boilerplate that appears on every page and is never lab content.
FURNITURE = (
    "use tab to navigate through the menu items",
    "this site was designed with the",
    "website builder",
    "create your website today",
    "start now",
    "top of page",
    "skip to main content",
)


class VisibleText(HTMLParser):
    """Collect visible blocks of a page, in order, with their element names."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []  # (kind, text)
        self._stack: list[str] = []
        self._buf: list[str] = []
        self._current: str = ""

    @property
    def _suppressed(self) -> bool:
        return any(t in DROP_CONTENT for t in self._stack)

    def _flush(self) -> None:
        text = RE_WS.sub(" ", "".join(self._buf)).strip()
        self._buf.clear()
        if text and self._current:
            self.blocks.append((self._current, text))
        self._current = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self._stack.append(tag)
        if self._suppressed:
            return
        if tag == "img":
            src = a.get("src") or a.get("data-src") or ""
            if src:
                self.blocks.append(("IMG", src))
        elif tag == "br":
            self._buf.append(" ")
        elif tag in BLOCKS:
            self._flush()
            self._current = tag

    def handle_endtag(self, tag):
        if tag in BLOCKS and not self._suppressed:
            self._flush()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] == tag:
                del self._stack[i]
                break

    def handle_data(self, data):
        if self._suppressed or not self._current:
            return
        clean = unicodedata.normalize("NFC", data).translate(INVISIBLE).replace("\xa0", " ")
        if clean.strip():
            self._buf.append(clean)

    def close(self):  # type: ignore[override]
        super().close()
        self._flush()


def is_furniture(text: str) -> bool:
    """True for Wix's own chrome, which is on every page and is never content."""
    low = text.lower()
    return any(f in low for f in FURNITURE)


def dump(page: Path, min_chars: int) -> int:
    """Print the page's visible blocks. Returns the number printed."""
    html = page.read_text(encoding="utf-8", errors="replace")
    parser = VisibleText()
    parser.feed(html)
    parser.close()

    seen: set[str] = set()
    printed = 0
    for kind, text in parser.blocks:
        if kind == "IMG":
            if text not in seen:
                seen.add(text)
                print(f"IMG: {text}")
                printed += 1
            continue
        if len(text) < min_chars or is_furniture(text):
            continue
        # Wix duplicates whole blocks between its mobile and desktop layouts.
        key = text[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        label = kind.upper() + ":" if kind in HEADINGS else "   "
        print(f"{label} {text}")
        printed += 1
    return printed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--page", type=Path, required=True, help="a cached Wix .html page")
    ap.add_argument("--min-chars", type=int, default=2,
                    help="skip blocks shorter than this (default 2)")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    if not args.page.is_file():
        LOG.error("page not found: %s", args.page)
        return 2

    n = dump(args.page, args.min_chars)
    LOG.info("%s: %d visible blocks", args.page.name, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
