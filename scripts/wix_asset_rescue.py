#!/usr/bin/env python3
"""Rescue every image asset from the Wix-hosted Levy lab site at full resolution.

Why this exists
---------------
Wix serves images through a transform pipeline:

    https://static.wixstatic.com/media/<media_id>/v1/fill/w_287,h_323,.../<original_name>

Everything from ``/v1/`` onward is a *downscaled, re-encoded* derivative. The
untouched upload sits at ``https://static.wixstatic.com/media/<media_id>``, so
stripping the transform recovers the original file. The final path segment of the
transform URL is the original upload filename, which is how most headshots keep a
usable name.

Running this makes the site content independent of the Wix subscription: once the
assets are on disk the plan can lapse without losing anything.

Nothing here does more than ordinary GET requests, and pages are cached to disk so
re-runs never hit the server again.

Usage
-----
    python wix_asset_rescue.py --out <dir> --dry-run    # inventory only, no downloads
    python wix_asset_rescue.py --out <dir>              # fetch + download
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("wix_rescue")

SITE = "https://www.asaflevylab.com"

# Wix slug -> the folder its assets land in. The slugs are Wix's own (note the
# copy-of- and -1 leftovers); the folder names are the clean ones for the rebuild.
PAGES: dict[str, str] = {
    "/": "home",
    "/copy-of-research": "research",
    "/lab-members-1": "people",
    "/publications": "publications",
    "/lab-life-1": "lab_life",
    "/in-the-media": "media",
    "/open-positions": "join_us",
    "/contact": "contact",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) wix-asset-rescue/1.0"

# A transformed Wix media URL. The trailing path segment is the original filename.
RE_TRANSFORM = re.compile(
    r"https://static\.wixstatic\.com/media/"
    r"(?P<media_id>[^/\"\\\s)]+)"
    r"/v1/(?:fill|crop|fit)/(?P<params>[^/\"\\\s)]+)"
    r"/(?P<filename>[^\"\\\s)?]+)"
)

RE_W = re.compile(r"\bw_(\d+)")
RE_H = re.compile(r"\bh_(\d+)")
RE_TAG = re.compile(r"<[^>]+>")
RE_WS = re.compile(r"\s+")

# Wix editor chrome (social icons, arrows) carries no site-specific media prefix and
# is only ever drawn at icon size. Every lab upload carries the site prefix.
SITE_MEDIA_PREFIX = "f312f1_"
ICON_MAX_PX = 64


@dataclass
class Asset:
    """One unique image on the site, plus where it was seen and how to name it."""

    media_id: str
    filename: str
    pages: set[str] = field(default_factory=set)
    max_w: int = 0
    max_h: int = 0
    context: str = ""

    @property
    def original_url(self) -> str:
        """The untransformed upload - i.e. the full-resolution file."""
        return "https://static.wixstatic.com/media/" + self.media_id

    @property
    def is_site_upload(self) -> bool:
        """True for the lab's own uploads, False for Wix editor chrome."""
        if self.media_id.startswith(SITE_MEDIA_PREFIX):
            return True
        # Unprefixed ids are Wix system assets when never drawn above icon size.
        return max(self.max_w, self.max_h) > ICON_MAX_PX


def fetch_page(url: str, cache_dir: Path, delay: float) -> str:
    """Return a page's HTML, caching it to disk so re-runs do not re-hit the server.

    Raises on a failed fetch instead of returning empty text: an empty string would
    look exactly like a page that legitimately has no images.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = urllib.parse.quote(url.replace(SITE, "").strip("/") or "index", safe="")
    cached = cache_dir / (slug + ".html")

    if cached.exists() and cached.stat().st_size > 0:
        LOG.info("page cached: %s (%d bytes)", url, cached.stat().st_size)
        return cached.read_text(encoding="utf-8", errors="replace")

    LOG.info("fetching page: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if not raw:
        raise RuntimeError("empty response body for " + url)
    text = raw.decode("utf-8", errors="replace")
    cached.write_text(text, encoding="utf-8")
    LOG.info("  -> %d bytes cached at %s", len(raw), cached)
    time.sleep(delay)
    return text


def context_near(page_html: str, pos: int, window: int = 9000) -> str:
    """Strip tags from the HTML just after an image and return the first real words.

    On the people page this is what ties a headshot to a name. Two kinds of noise sit
    between the image and the caption and both have to go first, or the context is
    just the image repeating itself:

    * the ``srcset`` - the same media URL again at a dozen widths, with ``1x``/``2x``
      density marks;
    * Wix transform parameters (``w_538``, ``q_85``, ``enc_avif``, ...).

    What survives is page prose. The result is advisory - the manifest is meant to be
    eyeballed, not trusted blindly.
    """
    chunk = page_html[pos : pos + window]
    text = html_mod.unescape(RE_TAG.sub(" ", chunk))
    text = re.sub(r"\bhttps?://\S+", " ", text)                  # srcset URLs
    text = re.sub(r"\b\d+(?:\.\d+)?x\b", " ", text)              # density marks
    text = re.sub(r"\b[a-z]{1,8}_[A-Za-z0-9.]+", " ", text)      # w_538, enc_avif, ...
    text = re.sub(r"[{}\[\]\"\\<>]+", " ", text)
    text = re.sub(r"[,;|]{2,}", " ", text)
    text = RE_WS.sub(" ", text).strip()
    # Require a letter run: a residue of punctuation and digits is not a caption.
    return text[:120] if re.search(r"[A-Za-z֐-׿]{3}", text) else ""


def extract_assets(page_html: str, page_label: str, assets: dict[str, Asset]) -> int:
    """Merge one page's image references into ``assets``, keyed by Wix media id.

    The same upload appears many times per page at different requested sizes. We keep
    the largest, because that is what separates a photo from an icon.
    """
    seen_here = 0
    for m in RE_TRANSFORM.finditer(page_html):
        media_id = m.group("media_id")
        params = m.group("params")
        filename = urllib.parse.unquote(m.group("filename"))

        mw, mh = RE_W.search(params), RE_H.search(params)
        w = int(mw.group(1)) if mw else 0
        h = int(mh.group(1)) if mh else 0

        asset = assets.get(media_id)
        if asset is None:
            asset = Asset(media_id=media_id, filename=filename)
            assets[media_id] = asset
            seen_here += 1

        asset.pages.add(page_label)
        if w * h > asset.max_w * asset.max_h:
            asset.max_w, asset.max_h = w, h
        if not asset.context:
            asset.context = context_near(page_html, m.end())

    LOG.info("page %-13s -> %3d new unique assets", page_label, seen_here)
    return seen_here


def safe_name(asset: Asset) -> str:
    """A filesystem-safe name that keeps the original upload name where it is useful.

    Wix filenames contain spaces, Hebrew and percent-escapes. The media id is appended
    so that two different uploads both called ``2.jpg`` cannot collide.
    """
    stem = Path(asset.filename).stem
    ext = Path(asset.filename).suffix or Path(asset.media_id).suffix or ".bin"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "image"
    short_id = asset.media_id.replace(SITE_MEDIA_PREFIX, "")[:8]
    return "{}__{}{}".format(stem, short_id, ext)


def download(asset: Asset, dest: Path, delay: float) -> tuple[bool, str]:
    """Download one original. Returns (ok, note); never raises on a single failure.

    Validates that the response really is an image of non-trivial size, so that a Wix
    error page cannot land on disk pretending to be a photo.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skipped, exists ({} bytes)".format(dest.stat().st_size)

    req = urllib.request.Request(asset.original_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return False, "FETCH FAILED: {}".format(exc)

    if not ctype.startswith("image/"):
        return False, "NOT AN IMAGE: content-type={!r}".format(ctype)
    if len(data) < 1024:
        return False, "SUSPICIOUSLY SMALL: {} bytes".format(len(data))

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    time.sleep(delay)
    return True, "{} bytes, {}".format(len(data), ctype)


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write the audit trail: every asset, where it came from, where it landed.

    This is the file to eyeball for the headshots whose upload name is not a person's
    name - the ``context`` column holds the page text that followed the image.
    """
    cols = [
        "page", "saved_as", "orig_filename", "max_w", "max_h",
        "bytes", "status", "note", "context", "original_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("manifest written: %s (%d rows)", path, len(rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory for assets + manifest")
    ap.add_argument("--dry-run", action="store_true", help="inventory and manifest only, download nothing")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests (default 0.3)")
    ap.add_argument("--include-chrome", action="store_true", help="also save Wix editor icons (skipped by default)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)

    out = args.out.resolve()
    cache_dir = out / "_page_cache"
    LOG.info("output root: %s", out)
    LOG.info("mode: %s", "DRY RUN (no downloads)" if args.dry_run else "download")

    # --- 1. fetch the pages -------------------------------------------------
    assets: dict[str, Asset] = {}
    page_failures: list[str] = []
    for slug, label in PAGES.items():
        url = SITE + slug
        try:
            page_html = fetch_page(url, cache_dir, args.delay)
        except Exception as exc:  # one bad page must not cost us the other seven
            LOG.error("page FAILED %s: %s", url, exc)
            page_failures.append("{}: {}".format(url, exc))
            continue
        extract_assets(page_html, label, assets)

    if not assets:
        LOG.error("no assets found on any page - the site changed, or it is blocking us")
        return 1

    # --- 2. validate --------------------------------------------------------
    uploads = [a for a in assets.values() if a.is_site_upload or args.include_chrome]
    chrome = len(assets) - len(uploads)
    LOG.info(
        "unique media ids: %d  (lab uploads: %d, Wix chrome skipped: %d)",
        len(assets), len(uploads), chrome,
    )
    no_dims = [a for a in uploads if a.max_w == 0]
    if no_dims:
        LOG.warning("%d assets carried no w_/h_ in any transform - check them in the manifest", len(no_dims))

    # --- 3. download --------------------------------------------------------
    rows: list[dict] = []
    ok_count = fail_count = 0
    for asset in sorted(uploads, key=lambda a: (sorted(a.pages)[0], a.filename.lower())):
        page_folder = sorted(asset.pages)[0]
        saved_as = safe_name(asset)
        dest = out / page_folder / saved_as

        if args.dry_run:
            status, note, nbytes = "DRY_RUN", "", ""
        else:
            ok, note = download(asset, dest, args.delay)
            status = "OK" if ok else "FAIL"
            nbytes = dest.stat().st_size if ok and dest.exists() else ""
            ok_count += int(ok)
            fail_count += int(not ok)
            LOG.info("%-5s %-45s %s", status, saved_as, note)

        rows.append({
            "page": "|".join(sorted(asset.pages)),
            "saved_as": "{}/{}".format(page_folder, saved_as),
            "orig_filename": asset.filename,
            "max_w": asset.max_w,
            "max_h": asset.max_h,
            "bytes": nbytes,
            "status": status,
            "note": note,
            "context": asset.context,
            "original_url": asset.original_url,
        })

    write_manifest(rows, out / "asset_manifest.tsv")

    # --- 4. report ----------------------------------------------------------
    LOG.info("-" * 60)
    LOG.info("pages fetched : %d of %d", len(PAGES) - len(page_failures), len(PAGES))
    LOG.info("assets listed : %d", len(rows))
    if not args.dry_run:
        LOG.info("downloaded    : %d ok, %d failed", ok_count, fail_count)
    for f in page_failures:
        LOG.error("PAGE FAILURE: %s", f)
    if fail_count:
        LOG.error("%d downloads failed - see the status=FAIL rows in the manifest", fail_count)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
