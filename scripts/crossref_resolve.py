#!/usr/bin/env python3
"""Turn the raw Wix publication lines into structured records, via Crossref.

Why resolve at all, instead of parsing the Wix text
---------------------------------------------------
The Wix list is not just unstructured, it is *wrong* in places. Two errors found
in the first two entries alone:

* the 2026 Nat Commun paper's title is missing its leading "A" on the live site -
  the character was swallowed into the bold author run ``Levy A.``;
* that same entry lists 4 authors where the paper has 5 (Feng Jiang is absent).

So the Wix text is treated as a *lookup key*, not as the record. Crossref supplies
the authoritative title, journal, year and author list; the Wix line is kept
verbatim in ``raw:`` so every record can be audited against what the old site said.

How each entry is resolved, in order
------------------------------------
1. **DOI stated** in the entry text or in a link target - used directly.
2. **DOI recoverable from the publisher URL** - Nature, Frontiers, ASM, APS,
   Microbiology Society and bioRxiv all embed the DOI in the path, so no guessing
   is involved. (ScienceDirect ``pii`` and OUP article paths do not, and fall to 3.)
3. **Bibliographic query** - the whole Wix citation line is sent to Crossref's
   ``query.bibliographic``, which is built for exactly this: free-text citations.
   The top hit is taken, and then *scored*, never trusted.

Every resolved record carries ``match_score``: the fraction of the Crossref title's
words that appear in the Wix line. A DOI-resolved record with a low score means the
DOI and the citation disagree, which is worth an eye; a query-resolved record with a
low score probably means Crossref found the wrong paper. The script does not decide
what is good enough - it writes the score and sorts the audit file by it.

Responses are cached as JSON under ``--cache``, so a second run makes no requests.

Usage
-----
    python crossref_resolve.py \
        --raw <out>/publications.raw.tsv \
        --out <out>/publications.yaml \
        --audit <out>/publications.audit.tsv \
        --cache <cache>/crossref \
        --mailto you@example.org
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("crossref")

CROSSREF_WORK = "https://api.crossref.org/works/{doi}"
CROSSREF_QUERY = "https://api.crossref.org/works"

RE_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>)\]]+")
RE_WS = re.compile(r"\s+")

# Suffixes publishers hang off a DOI inside a URL, which Crossref will not match:
# bioRxiv's preprint version (``...465003v2``) and reader-view paths that the DOI
# regex swallows (``10.3389/fpls.2020.589416/full``). Both looked like a missing
# paper until they were stripped.
RE_DOI_SUFFIX = re.compile(r"(?:/(?:full|abstract|pdf|meta|epdf|html))+$|v\d+$", re.I)

# Publishers that put the DOI straight into the URL path. Nature is the one case
# where the DOI must be *rebuilt*: /articles/s41467-021-23777-7 -> 10.1038/<id>.
RE_URL_DOI = re.compile(r"(?:doi\.org/|/doi/(?:full/)?|/content/(?:journal/\w+/)?)(10\.\d{4,9}/[^\s?&#]+)")
RE_URL_NATURE = re.compile(r"nature\.com/articles/(s\d{5}-\d{3}-\d{4,5}-[\dxy]+)", re.I)
RE_URL_FRONTIERS = re.compile(r"(10\.3389/[^/\s?&#]+)")

# Words that carry no identifying weight when scoring a title against a citation.
STOPWORDS = frozenset(
    "a an the of in on for and or to with within from by at as is are be that this".split()
)


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation - for word-overlap scoring only."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9\s-]", " ", text.lower())


def content_words(text: str) -> set[str]:
    """Identifying words of a string: no stopwords, nothing shorter than 3 chars."""
    return {w for w in normalise(text).split() if len(w) >= 3 and w not in STOPWORDS}


def match_score(crossref_title: str, wix_line: str) -> float:
    """Fraction of the Crossref title's content words present in the Wix citation.

    Containment, not Jaccard: the Wix line holds authors and journal as well, so a
    symmetric measure would punish a perfect title match for the surrounding text.
    """
    want = content_words(crossref_title)
    if not want:
        return 0.0
    have = content_words(wix_line)
    return len(want & have) / len(want)


@dataclass
class RawEntry:
    """One row of publications.raw.tsv."""

    n: int
    year: str
    doi: str
    co_first: bool
    bold_names: list[str]
    urls: list[str]
    raw_text: str


@dataclass
class Record:
    """A resolved publication, plus the provenance needed to audit it."""

    n: int
    title: str = ""
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    url: str = ""
    lab_members: list[str] = field(default_factory=list)
    co_first: bool = False
    is_preprint: bool = False
    raw: str = ""
    resolved_by: str = "unresolved"
    match_score: float = 0.0
    note: str = ""


def load_raw(path: Path) -> list[RawEntry]:
    """Read the stage-1 TSV. Fails loudly on a missing column rather than later."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    required = {"n", "year", "doi", "co_first", "bold_names", "urls", "raw_text"}
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    entries = []
    for r in rows:
        entries.append(
            RawEntry(
                n=int(r["n"]),
                year=r["year"].strip(),
                doi=r["doi"].strip(),
                co_first=bool(r["co_first"].strip()),
                bold_names=[b.strip() for b in r["bold_names"].split("|") if b.strip()],
                urls=[u.strip() for u in r["urls"].split("|") if u.strip()],
                raw_text=r["raw_text"].strip(),
            )
        )
    LOG.info("loaded %d raw entries from %s", len(entries), path)
    return entries


def doi_from_urls(urls: list[str]) -> tuple[str, str]:
    """Recover a DOI from a publisher URL. Returns (doi, which_pattern)."""
    for url in urls:
        m = RE_URL_DOI.search(url)
        if m:
            return m.group(1).rstrip(".,;/"), "url-path"
        m = RE_URL_NATURE.search(url)
        if m:
            # Nature article ids map 1:1 onto Springer Nature DOIs under 10.1038.
            return f"10.1038/{m.group(1)}", "url-nature"
        m = RE_URL_FRONTIERS.search(url)
        if m:
            return m.group(1).rstrip(".,;/"), "url-frontiers"
    return "", ""


def fetch_json(url: str, cache_dir: Path, mailto: str, pause: float) -> dict | None:
    """GET a Crossref URL, with an on-disk cache keyed by the URL.

    Crossref's polite pool wants a contact address in the User-Agent; sending one
    gets more predictable rate limits than going anonymous.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    cached = cache_dir / f"{key}.json"
    if cached.is_file() and cached.stat().st_size:
        return json.loads(cached.read_text(encoding="utf-8"))

    req = urllib.request.Request(
        url, headers={"User-Agent": f"LevyLabSiteBuild/1.0 (mailto:{mailto})"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        LOG.warning("HTTP %s for %s", exc.code, url)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        LOG.warning("request failed for %s: %s", url, exc)
        return None

    cached.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(pause)  # politeness, not correctness - the cache makes reruns free
    return payload


def crossref_by_doi(doi: str, cache_dir: Path, mailto: str, pause: float) -> dict | None:
    url = CROSSREF_WORK.format(doi=urllib.parse.quote(doi, safe="/"))
    payload = fetch_json(url, cache_dir, mailto, pause)
    return payload.get("message") if payload else None


def crossref_by_citation(citation: str, cache_dir: Path, mailto: str, pause: float) -> dict | None:
    """Ask Crossref to identify a free-text citation. Top hit only."""
    query = urllib.parse.urlencode(
        {"query.bibliographic": citation[:400], "rows": "1", "select":
         "DOI,title,container-title,issued,author,URL,type"}
    )
    payload = fetch_json(f"{CROSSREF_QUERY}?{query}", cache_dir, mailto, pause)
    if not payload:
        return None
    items = payload.get("message", {}).get("items") or []
    return items[0] if items else None


def author_names(message: dict) -> list[str]:
    """Crossref authors as 'Given Family', keeping the publication order."""
    out = []
    for a in message.get("author", []) or []:
        name = " ".join(p for p in (a.get("given"), a.get("family")) if p).strip()
        if not name:
            name = (a.get("name") or "").strip()
        if name:
            out.append(name)
    return out


def clean_doi(doi: str) -> str:
    """Strip the version and reader-view suffixes publishers add inside URLs."""
    return RE_DOI_SUFFIX.sub("", doi.strip().rstrip(".,;/"))


def crossref_year(message: dict) -> str:
    """The year the paper is cited by: the print issue's, where one exists.

    ``issued`` is Crossref's earliest known date, so for anything published
    online-first it returns the online year - which is why Crossref said 2021 for
    the Deeplasmid paper that NAR printed in its 22 February 2022 issue. A
    publications page follows the citation, so print wins when it exists, and
    online-first only stands in for papers that have no printed issue.
    """
    for key in ("published-print", "issued", "published-online", "created"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return ""


def citation_year(wix_year: str, cr_year: str) -> tuple[str, str]:
    """Pick the citation year from the two sources: whichever is LATER.

    Both sources can only ever under-report, for opposite reasons, so the later
    one is the printed-issue year in every case here:

    * Crossref under-reports when a journal registers no ``published-print``
      date - two Nucleic Acids Research papers came back 2021 and 2009 against
      printed issues the Wix page cites as 2022 and 2010;
    * Wix under-reports when whoever typed the page used the epub-ahead-of-print
      year - two 2020 issues are listed there as 2019.

    Returns (year, note) - the note is non-empty exactly when the two disagree,
    so every one of them lands in the audit file rather than passing silently.
    """
    if not cr_year:
        return wix_year, ""
    if not wix_year:
        return cr_year, ""
    if wix_year == cr_year:
        return cr_year, ""
    chosen = max(wix_year, cr_year)
    other = "wix" if chosen == cr_year else "crossref"
    return chosen, f"year differs (wix {wix_year}, crossref {cr_year}) - kept {chosen}, {other} is earlier"


def journal_name(message: dict) -> tuple[str, bool]:
    """Journal title, and whether this is a preprint.

    Crossref gives preprints no ``container-title`` at all, which left the two
    bioRxiv entries with a blank journal. For ``posted-content`` the server name
    lives in ``institution`` or ``group-title`` instead, so it is read from there
    rather than inferred from the 10.1101 DOI prefix.
    """
    container = (message.get("container-title") or [""])[0].strip()
    is_preprint = message.get("type") == "posted-content"
    if container:
        return container, is_preprint
    if is_preprint:
        institutions = message.get("institution") or []
        if isinstance(institutions, dict):
            institutions = [institutions]
        for inst in institutions:
            name = (inst.get("name") or "").strip()
            if name:
                return name, True
        group = (message.get("group-title") or "").strip()
        return group or "Preprint", True
    return "", is_preprint


def build_record(entry: RawEntry, cache_dir: Path, mailto: str, pause: float) -> Record:
    """Resolve one entry through the three routes, scoring whatever comes back."""
    rec = Record(
        n=entry.n,
        raw=entry.raw_text,
        lab_members=entry.bold_names,
        co_first=entry.co_first,
        year=entry.year,
        url=entry.urls[0] if entry.urls else "",
    )

    doi, how = entry.doi, "doi-stated"
    if not doi:
        doi, how = doi_from_urls(entry.urls)
    doi = clean_doi(doi)

    message = None
    if doi:
        message = crossref_by_doi(doi, cache_dir, mailto, pause)
        if message is None:
            rec.note = f"DOI {doi} not found in Crossref"
            doi = ""
    if message is None:
        message = crossref_by_citation(entry.raw_text, cache_dir, mailto, pause)
        how = "citation-query"

    if message is None:
        rec.resolved_by = "unresolved"
        rec.note = (rec.note + "; no Crossref match").lstrip("; ")
        return rec

    rec.title = (message.get("title") or [""])[0].strip()
    rec.journal, rec.is_preprint = journal_name(message)
    rec.authors = author_names(message)
    rec.doi = (message.get("DOI") or doi or "").strip()
    rec.resolved_by = how
    rec.match_score = round(match_score(rec.title, entry.raw_text), 3)

    rec.year, year_note = citation_year(entry.year, crossref_year(message))
    if year_note:
        rec.note = (rec.note + "; " + year_note).lstrip("; ")
    if not rec.url:
        rec.url = f"https://doi.org/{rec.doi}" if rec.doi else (message.get("URL") or "")
    return rec


def flatten_ws(value: str) -> str:
    """Collapse every run of whitespace to one space.

    Crossref returns titles with real newlines in them (7 of these 28 records),
    which silently broke the audit TSV into 36 rows and would fold unpredictably
    inside a quoted YAML scalar. Every field written out goes through this.
    """
    return RE_WS.sub(" ", value).strip()


def yaml_escape(value: str) -> str:
    """Double-quoted YAML scalar - the roster file uses the same convention."""
    return '"' + flatten_ws(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_yaml(records: list[Record], out: Path) -> None:
    """Write publications.yaml, newest first."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: (r.year or "0000", r.n), reverse=True)
    lines = [
        "# Publications recovered from the Wix site by wix_publications_extract.py",
        "# and resolved against Crossref by crossref_resolve.py.",
        "#",
        "# To add a paper: append an entry with just its doi, then re-run",
        "# crossref_resolve.py to fill in the rest.",
        "#   raw          = the citation exactly as the old Wix site printed it (audit trail)",
        "#   lab_members  = names the Wix page showed in bold; empty on the older entries",
        "#   co_first     = the citation marked co-first authors with an asterisk",
        "#   is_preprint  = Crossref type posted-content; journal holds the server name",
        "#   match_score  = share of the Crossref title's words found in the Wix line",
        "#",
        "# A title may contain <i>...</i> - Crossref italicises species names, which is",
        "# correct typography for us. The template must therefore render title as HTML",
        "# rather than escaping it.",
        "",
        "publications:",
    ]
    for r in ordered:
        lines.append(f"  - year: {yaml_escape(r.year)}")
        lines.append(f"    title: {yaml_escape(r.title)}")
        lines.append(f"    journal: {yaml_escape(r.journal)}")
        lines.append(f"    doi: {yaml_escape(r.doi)}")
        lines.append(f"    url: {yaml_escape(r.url)}")
        lines.append(f"    co_first: {'true' if r.co_first else 'false'}")
        lines.append(f"    is_preprint: {'true' if r.is_preprint else 'false'}")
        if r.authors:
            lines.append("    authors:")
            lines.extend(f"      - {yaml_escape(a)}" for a in r.authors)
        else:
            lines.append("    authors: []")
        if r.lab_members:
            lines.append("    lab_members:")
            lines.extend(f"      - {yaml_escape(m)}" for m in r.lab_members)
        else:
            lines.append("    lab_members: []")
        lines.append(f"    wix_entry: {r.n}")
        lines.append(f"    resolved_by: {yaml_escape(r.resolved_by)}")
        lines.append(f"    match_score: {r.match_score}")
        if r.note:
            lines.append(f"    note: {yaml_escape(r.note)}")
        lines.append(f"    raw: {yaml_escape(r.raw)}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s (%d records)", out, len(ordered))


def write_audit(records: list[Record], out: Path) -> None:
    """Audit file sorted worst-match-first, so review starts where it matters."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["match_score", "n", "resolved_by", "year", "doi", "n_authors",
            "crossref_title", "wix_line", "note"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in sorted(records, key=lambda r: r.match_score):
            fh.write("\t".join([
                f"{r.match_score:.3f}", str(r.n), r.resolved_by, r.year, r.doi,
                str(len(r.authors)), flatten_ws(r.title),
                flatten_ws(r.raw)[:160], flatten_ws(r.note),
            ]) + "\n")
    LOG.info("wrote %s", out)


def report(records: list[Record]) -> None:
    """Sanity check: where records came from, and the match-score distribution."""
    LOG.info("--- sanity check ---")
    by_route: dict[str, int] = {}
    for r in records:
        by_route[r.resolved_by] = by_route.get(r.resolved_by, 0) + 1
    for route, k in sorted(by_route.items(), key=lambda kv: -kv[1]):
        LOG.info("resolved by %-16s: %d", route, k)

    resolved = [r for r in records if r.resolved_by != "unresolved"]
    LOG.info("records with a title   : %d / %d", sum(1 for r in resolved if r.title), len(records))
    LOG.info("records with authors   : %d / %d", sum(1 for r in resolved if r.authors), len(records))
    LOG.info("records with a journal : %d / %d", sum(1 for r in resolved if r.journal), len(records))

    bands = [(0.9, 1.01), (0.75, 0.9), (0.5, 0.75), (0.0, 0.5)]
    LOG.info("match_score distribution (no threshold applied):")
    for lo, hi in bands:
        k = sum(1 for r in records if lo <= r.match_score < hi)
        LOG.info("  %.2f-%.2f : %d", lo, min(hi, 1.0), k)

    for r in sorted(records, key=lambda r: r.match_score)[:6]:
        LOG.info("  lowest: n=%-2d score=%.3f via %-16s %s", r.n, r.match_score,
                 r.resolved_by, (r.title or "(no title)")[:70])
    for r in records:
        if r.note:
            LOG.warning("n=%d: %s", r.n, r.note)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, required=True, help="publications.raw.tsv from stage 1")
    ap.add_argument("--out", type=Path, required=True, help="publications.yaml to write")
    ap.add_argument("--audit", type=Path, required=True, help="publications.audit.tsv to write")
    ap.add_argument("--cache", type=Path, required=True, help="directory for cached Crossref JSON")
    ap.add_argument("--mailto", required=True, help="contact address for Crossref's polite pool")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between live requests")
    args = ap.parse_args(argv)

    # Titles carry en-dashes and non-breaking hyphens; a Windows console defaults to
    # cp1252 and every such log line dies inside the logging handler.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)

    if not args.raw.is_file():
        LOG.error("raw TSV not found: %s", args.raw)
        return 2

    entries = load_raw(args.raw)
    records = []
    for e in entries:
        rec = build_record(e, args.cache, args.mailto, args.pause)
        LOG.info("n=%-2d %-16s score=%.3f  %s", rec.n, rec.resolved_by, rec.match_score,
                 (rec.title or "(unresolved)")[:72])
        records.append(rec)

    write_yaml(records, args.out)
    write_audit(records, args.audit)
    report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
