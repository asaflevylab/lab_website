#!/usr/bin/env python3
"""Rebuild the lab roster from the Wix people page: photo -> name -> role.

Why this is not a guessing game
-------------------------------
Wix renders the two member galleries as *repeaters*. Every cell of a repeater gets a
component id of the form::

    comp-<widget_id>__item-<item_key>

The photo and the name live in two different widgets but in the **same repeater cell**,
so they share the ``item_key``. That key is the join:

    image  comp-mhnd97iq__item-kf51ytd5   ->  nimrod_edited.jpg
    name   comp-mhnd97jf__item-kf51ytd5   ->  "Nimrod Nachmias"

This matters because the uploads are not reliably named: roughly a third are
``2.jpg``, ``IMG_1437.jpeg``, ``unnamed_edited.jpg`` or a WhatsApp export, so matching
by filename would mislabel real people. ``alt`` text is no help either - on this site
every alt attribute is just the filename again.

Widgets belonging to one repeater share the leading characters of their widget id
(``mhnd97iq`` / ``mhnd97jf``; ``ltn1d8ky7`` / ``ltn1d8l06``), which is what separates
the current-members repeater from the alumni one. Item keys ARE reused between the two
repeaters, so that scoping is required, not cosmetic.

Input is the page already cached by ``wix_asset_rescue.py`` - this script makes no
network requests at all.

Usage
-----
    python wix_people_extract.py \
        --page <cache>/lab-members-1.html \
        --manifest <out>/asset_manifest.tsv \
        --out <out>/people.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

LOG = logging.getLogger("wix_people")

# The hyphen before the key is optional: Wix names the repeater's *template* cell
# ``__item1`` while every cloned cell gets ``__item-<key>``. Requiring the hyphen
# silently drops the first person in each gallery - which cost us Asaf and Miriam.
RE_COMP_ITEM = re.compile(r"comp-(?P<widget>[A-Za-z0-9]+)__item-?(?P<key>[A-Za-z0-9-]+)")
RE_MEDIA = re.compile(r"static\.wixstatic\.com/media/(?P<media_id>[^/\"\\\s)]+)")

# A role line, as written on this site. Used to tell the name cell from the role cell
# inside one repeater item, rather than relying on which comes first in the markup.
RE_ROLE = re.compile(
    r"Principal Investigator|Investigator|Staff Scientist|Scientist|Lab Manager|"
    r"Manager|Candidate|Postdoc|Technician|Student|M\.?Sc|P\.?h\.?\.?D|Ph\.D|Alumni",
    re.IGNORECASE,
)

# How many leading characters of a widget id identify the repeater it belongs to.
# Wix generates sibling widget ids from a common base, so the photo widget and the
# name widget of one gallery agree on this prefix while the two galleries do not.
REPEATER_PREFIX_LEN = 6


@dataclass
class Cell:
    """One repeater cell: a person, as the page stores them."""

    repeater: str
    key: str
    texts: list[str] = field(default_factory=list)
    media_ids: list[str] = field(default_factory=list)


class RepeaterParser(HTMLParser):
    """Collect text and images per repeater cell, keyed by ``item_key``.

    Text is attributed to the innermost enclosing element that carries a
    ``comp-...__item-...`` id, which is how a name ends up on the right person even
    though the markup nests it several spans deep.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: dict[tuple[str, str], Cell] = {}
        self._stack: list[tuple[str, str] | None] = []

    # -- helpers ---------------------------------------------------------
    def _cell(self, repeater: str, key: str) -> Cell:
        ident = (repeater, key)
        if ident not in self.cells:
            self.cells[ident] = Cell(repeater=repeater, key=key)
        return self.cells[ident]

    @staticmethod
    def _ident(attrs: dict[str, str]) -> tuple[str, str] | None:
        """Return (repeater_prefix, item_key) for an element inside a repeater cell."""
        m = RE_COMP_ITEM.search(attrs.get("id", ""))
        if not m:
            return None
        return m.group("widget")[:REPEATER_PREFIX_LEN], m.group("key")

    def _current(self) -> tuple[str, str] | None:
        for ident in reversed(self._stack):
            if ident is not None:
                return ident
        return None

    # -- HTMLParser hooks ------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        ident = self._ident(a)

        if tag == "img":
            # An <img> is void, so handle it before any stack bookkeeping.
            owner = ident or self._current()
            if owner is not None:
                blob = " ".join([a.get("src", ""), a.get("srcset", "")])
                m = RE_MEDIA.search(blob)
                if m:
                    cell = self._cell(*owner)
                    if m.group("media_id") not in cell.media_ids:
                        cell.media_ids.append(m.group("media_id"))
            return

        self._stack.append(ident)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("img", "br", "meta", "link", "input"):
            return
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", data).strip()
        if not text or len(text) < 2:
            return
        owner = self._current()
        if owner is None:
            return
        cell = self._cell(*owner)
        if text not in cell.texts:
            cell.texts.append(text)


def classify_cell(cell: Cell) -> tuple[str, str]:
    """Split a cell's text into (name, role).

    The name is the text that does NOT read as a role. Doing it this way round means a
    missing role yields an empty role rather than shifting the name into it.
    """
    names = [t for t in cell.texts if not RE_ROLE.search(t)]
    roles = [t for t in cell.texts if RE_ROLE.search(t)]

    # "Dr. Keren Shemesh" and "Associate Prof. Asaf Levy" are names, not roles, even
    # though an honorific can collide with the role vocabulary.
    name = next((n for n in names if len(n) > 3), "")
    if not name and roles:
        # A cell whose only text matched the role pattern: the longest line that looks
        # like a person (2+ capitalised words) is the name.
        person = [r for r in roles if len(re.findall(r"\b[A-Z][a-z]+", r)) >= 2]
        if person:
            name = person[0]
            roles = [r for r in roles if r != name]
    role = "; ".join(roles)
    return name, role


def load_media_names(manifest: Path) -> dict[str, str]:
    """Map Wix media id -> the filename the rescue script saved it as."""
    if not manifest.exists():
        LOG.warning("manifest not found at %s - photo filenames will be left blank", manifest)
        return {}
    mapping: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            media_id = row["original_url"].rsplit("/", 1)[-1]
            mapping[media_id] = row["saved_as"]
    LOG.info("manifest: %d media ids -> saved filenames", len(mapping))
    return mapping


def yaml_escape(value: str) -> str:
    """Quote a scalar for YAML without pulling in a YAML dependency."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--page", type=Path, required=True, help="cached lab-members-1.html")
    ap.add_argument("--manifest", type=Path, required=True, help="asset_manifest.tsv from the rescue script")
    ap.add_argument("--out", type=Path, required=True, help="people.yaml to write")
    ap.add_argument("--expect-current", type=int, default=13, help="expected current members (default 13)")
    ap.add_argument("--expect-alumni", type=int, default=18, help="expected alumni (default 18)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)

    if not args.page.exists():
        LOG.error("page not found: %s - run wix_asset_rescue.py first", args.page)
        return 1

    # --- 1. parse -----------------------------------------------------------
    parser = RepeaterParser()
    parser.feed(args.page.read_text(encoding="utf-8", errors="replace"))

    by_repeater: dict[str, list[Cell]] = defaultdict(list)
    for cell in parser.cells.values():
        if cell.media_ids and any(t for t in cell.texts):
            by_repeater[cell.repeater].append(cell)

    LOG.info("repeaters with photo+text cells:")
    for rep, cells in sorted(by_repeater.items(), key=lambda kv: -len(kv[1])):
        LOG.info("  %-10s %2d cells", rep, len(cells))

    # The two biggest such repeaters are the member galleries; anything smaller is
    # page furniture (logo blocks, the department strip).
    ranked = sorted(by_repeater.items(), key=lambda kv: -len(kv[1]))
    if len(ranked) < 2:
        LOG.error("expected 2 member galleries, found %d - the page structure changed", len(ranked))
        return 1

    groups: list[tuple[str, list[Cell]]] = []
    for rep, cells in ranked[:2]:
        groups.append((rep, cells))

    # Larger gallery = alumni (18 > 13). Verified against the expected counts below.
    groups.sort(key=lambda g: -len(g[1]))
    alumni_rep, alumni_cells = groups[0]
    current_rep, current_cells = groups[1]

    # --- 2. validate --------------------------------------------------------
    ok = True
    if len(current_cells) != args.expect_current:
        LOG.error("current members: got %d, expected %d", len(current_cells), args.expect_current)
        ok = False
    if len(alumni_cells) != args.expect_alumni:
        LOG.error("alumni: got %d, expected %d", len(alumni_cells), args.expect_alumni)
        ok = False
    LOG.info("current repeater %s -> %d, alumni repeater %s -> %d",
             current_rep, len(current_cells), alumni_rep, len(alumni_cells))

    media_names = load_media_names(args.manifest)

    # --- 3. build the roster ------------------------------------------------
    records: list[dict] = []
    for group, cells in (("current", current_cells), ("alumni", alumni_cells)):
        for cell in cells:
            name, role = classify_cell(cell)
            photo = media_names.get(cell.media_ids[0], "")
            records.append({
                "group": group,
                "name": name,
                "role": role,
                "photo": photo,
                "wix_media_id": cell.media_ids[0],
                "wix_item_key": cell.key,
                "raw_texts": " | ".join(cell.texts),
            })
            if not name:
                LOG.warning("no name recovered for item %s (texts: %s)", cell.key, cell.texts)
                ok = False
            if not photo:
                LOG.warning("no saved photo for %s (media %s) - run the rescue script first",
                            name or cell.key, cell.media_ids[0])

    dupes = {r["name"] for r in records if [x["name"] for x in records].count(r["name"]) > 1}
    if dupes:
        LOG.error("duplicate names - the join is wrong somewhere: %s", sorted(dupes))
        ok = False

    # --- 4. write -----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("# Lab roster recovered from the Wix site by wix_people_extract.py\n")
        fh.write("# photo = path under the rescued-assets directory\n")
        fh.write("# wix_* fields are provenance; drop them once the rebuild is verified\n")
        for group in ("current", "alumni"):
            fh.write("\n{}:\n".format(group))
            for r in [x for x in records if x["group"] == group]:
                fh.write("  - name: {}\n".format(yaml_escape(r["name"])))
                fh.write("    role: {}\n".format(yaml_escape(r["role"])))
                fh.write("    photo: {}\n".format(yaml_escape(r["photo"])))
                fh.write("    wix_media_id: {}\n".format(yaml_escape(r["wix_media_id"])))
                fh.write("    wix_item_key: {}\n".format(yaml_escape(r["wix_item_key"])))
    LOG.info("wrote %s (%d people: %d current, %d alumni)",
             args.out, len(records), len(current_cells), len(alumni_cells))

    # A side-by-side table for eyeballing the join before anyone trusts it.
    audit = args.out.with_suffix(".audit.tsv")
    with audit.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["group", "name", "role", "photo", "wix_media_id", "wix_item_key", "raw_texts"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(records)
    LOG.info("wrote audit table %s", audit)

    if not ok:
        LOG.error("finished WITH PROBLEMS - read the warnings above before using people.yaml")
        return 2
    LOG.info("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
