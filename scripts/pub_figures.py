#!/usr/bin/env python3
"""Build the publication thumbnails from what data/publications.yaml says.

Each record may carry:

    figure:         publications/figures/<doi>.jpg   (path under assets/ - the output)
    figure_source:  https://...                      (the image as the publisher serves it)
    figure_crop:    "x0,y0,x1,y1"                    (OPTIONAL, pixels of the source image)

The yaml is the only record of which figure was chosen and which part of it is
shown, so a thumbnail can always be rebuilt or re-cut from it - nothing on the
site is a hand-made file whose origin has to be remembered.

Why the sources are not in the repo
-----------------------------------
The repo is public and about half the papers are under copyright. Only the
chosen crop is published; the full downloaded figure stays in ``--cache``
(gitignored), keyed by a hash of its URL so a re-run makes no requests.

Where the sources come from (all observed working 2026-11-05)
-------------------------------------------------------------
* PMC Open Data bucket   https://pmc-oa-opendata.s3.amazonaws.com/<PMCID>.<v>/<file>
* Elsevier image CDN     https://ars.els-cdn.com/content/image/1-s2.0-<PII>-gr<N>.jpg
                         (-fx1 = graphical abstract). cell.com and ScienceDirect
                         pages return 403 to scripts; this CDN does not.
* Springer Nature        media.springernature.com/lw1200/... (``/full/`` fails for ESM)
* bioRxiv                .../F<N>.large.jpg from the full-text page

Usage
-----
    python scripts/pub_figures.py --yaml data/publications.yaml \\
        --assets assets --cache _pick/sources/cache [--prune]

Needs Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger("pub_figures")

# A browser UA: the publisher CDNs refuse the default urllib one.
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
# Below this a download is an error page, not a figure (Springer's /full/ ESM
# path returned a 2 kB body with HTTP 200).
MIN_BYTES = 5_000


@dataclass
class FigureSpec:
    doi: str
    figure: str
    source: str
    crop: tuple[int, int, int, int] | None


def field(block: str, name: str) -> str | None:
    """Return one quoted scalar field of a record block, or None if absent."""
    m = re.search(rf'^    {name}: "(.*)"$', block, re.M)
    return m.group(1) if m else None


def read_specs(yaml_path: Path) -> list[FigureSpec]:
    """Parse the figure fields out of publications.yaml.

    The file is machine-written with fixed indentation (records start with
    ``  - year:``, fields are 4-space ``key: "value"``), so a regex reader is
    enough and avoids a PyYAML dependency. A record with a ``figure`` but no
    ``figure_source`` is an error - its thumbnail could not be rebuilt.
    """
    text = yaml_path.read_text(encoding="utf-8")
    specs, problems = [], []
    for block in re.split(r"^  - year:", text, flags=re.M)[1:]:
        doi = field(block, "doi") or "?"
        fig = field(block, "figure")
        if not fig:
            continue
        src = field(block, "figure_source")
        if not src or not src.startswith("http"):
            problems.append(f"{doi}: figure set but no http figure_source")
            continue
        crop = None
        if (c := field(block, "figure_crop")):
            parts = [int(p) for p in c.split(",")]
            if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
                problems.append(f"{doi}: figure_crop must be x0,y0,x1,y1 with x0<x1, y0<y1: {c}")
                continue
            crop = tuple(parts)
        specs.append(FigureSpec(doi, fig, src, crop))
    if problems:
        raise SystemExit("publications.yaml:\n  " + "\n  ".join(problems))
    return specs


def fetch(url: str, cache: Path) -> Path:
    """Download ``url`` once into the cache and return the cached path."""
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    path = cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + suffix)
    if path.exists():
        return path
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
    if len(body) < MIN_BYTES:
        raise RuntimeError(f"{len(body)} bytes - an error page, not an image: {url}")
    path.write_bytes(body)
    log.info("fetched %6d kB  %s", len(body) // 1024, url)
    return path


def build(spec: FigureSpec, cache: Path, assets: Path) -> str:
    """Write one thumbnail source under assets/; return a one-line report."""
    raw = fetch(spec.source, cache)
    out = assets / spec.figure
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw) as im:
        w, h = im.size
        if spec.crop is None:
            out.write_bytes(raw.read_bytes())      # untouched: no re-encode loss
            return f"{spec.doi:<34} whole    {w}x{h}"
        x0, y0, x1, y1 = spec.crop
        if x1 > w or y1 > h:
            raise RuntimeError(f"{spec.doi}: crop {spec.crop} exceeds the {w}x{h} source")
        part = im.crop(spec.crop)
        # Hugo re-encodes to webp later; save near-lossless so that is the only loss.
        if out.suffix.lower() in (".jpg", ".jpeg"):
            part.convert("RGB").save(out, quality=95)
        else:
            part.save(out)
        return f"{spec.doi:<34} cropped  {x1 - x0}x{y1 - y0} of {w}x{h}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", type=Path, required=True, help="data/publications.yaml")
    ap.add_argument("--assets", type=Path, required=True, help="the site's assets/ directory")
    ap.add_argument("--cache", type=Path, required=True, help="download cache (gitignored)")
    ap.add_argument("--prune", action="store_true",
                    help="delete files in assets/publications/figures/ that no record uses")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.cache.mkdir(parents=True, exist_ok=True)

    specs = read_specs(args.yaml)
    failed = 0
    for spec in specs:
        try:
            log.info(build(spec, args.cache, args.assets))
        except Exception as exc:                   # report every failure, then exit non-zero
            failed += 1
            log.error("FAILED %s: %s", spec.doi, exc)

    used = {(args.assets / s.figure).resolve() for s in specs}
    fig_dir = args.assets / "publications" / "figures"
    orphans = [p for p in fig_dir.glob("*") if p.is_file() and p.resolve() not in used]
    for p in orphans:
        if args.prune:
            p.unlink()
            log.info("pruned   %s", p.name)
        else:
            log.warning("unused   %s (re-run with --prune to delete)", p.name)

    log.info("%d thumbnails built, %d failed, %d unused file(s)%s",
             len(specs) - failed, failed, len(orphans), " pruned" if args.prune else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
