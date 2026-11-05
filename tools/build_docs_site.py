#!/usr/bin/env python3
"""Build the static documentation site served from docs/ by GitHub Pages.

Sources:  README.md, IMPLEMENTATION.md, docs/{architecture,configuration,
          benchmarks,troubleshooting}.md
Outputs:  docs/<page>.html + docs/assets/search-index.json

The landing page (docs/index.html) and the search shell (docs/search.html) are
hand-written and left untouched. Requires pandoc: `pandoc -f gfm -t html`.
"""
import html
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

PAGES = [
    ("architecture",    "Architecture",   "How the VRAM / DRAM / SSD tiers work, placement, modes, MoE fast path.", DOCS / "architecture.md"),
    ("configuration",   "Configuration",  "Every environment variable, CMake option and CLI flag, with presets.", DOCS / "configuration.md"),
    ("benchmarks",      "Benchmarks",     "Measured throughput, promotion counters and how to reproduce them.", DOCS / "benchmarks.md"),
    ("implementation",  "Paper vs. code", "What the paper claims, what this code does, and the measured evidence.", ROOT / "IMPLEMENTATION.md"),
    ("troubleshooting", "Troubleshooting","Slow first pass, pin refusals, KV cache types, graph splits.", DOCS / "troubleshooting.md"),
]

NAV = [
    ("index.html",         "Home"),
    ("architecture.html",  "Architecture"),
    ("configuration.html", "Configuration"),
    ("benchmarks.html",    "Benchmarks"),
    ("implementation.html","Paper vs. code"),
    ("troubleshooting.html","Troubleshooting"),
    ("search.html",        "Search"),
]

REPO = "https://github.com/Mr-DS-ML-85/llama.cpp-VUGVA"
PAPER = "https://doi.org/10.5281/zenodo.21549808"


def pandoc_body(path: pathlib.Path) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "gfm+pipe_tables", "-t", "html", "--wrap=none", str(path)],
        check=True, capture_output=True, text=True).stdout
    # external links get target=_blank and rel=noopener
    out = re.sub(r'<a href="(https?://[^"]+)"',
                 r'<a href="\1" target="_blank" rel="noopener"', out)
    return out


def headings(body: str):
    return [(int(m.group(1)), m.group(2), re.sub(r"<[^>]+>", "", m.group(3)))
            for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', body, re.S)]


def toc(body: str) -> str:
    items = headings(body)
    if not items:
        return ""
    out = ['<nav class="toc" aria-label="On this page">', '<p class="toc-title">On this page</p>', '<ul>']
    for level, anchor, text in items:
        cls = "toc-h3" if level == 3 else "toc-h2"
        out.append(f'<li class="{cls}"><a href="#{anchor}">{html.escape(text)}</a></li>')
    out.append("</ul></nav>")
    return "\n".join(out)


def plain(fragment: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def search_chunks(slug: str, title: str, body: str, summary: str):
    """Split a page into heading-scoped chunks for the client-side index."""
    parts = re.split(r'(<h[23] id="[^"]+">.*?</h[23]>)', body, flags=re.S)
    chunks = [{
        "page": slug, "title": title, "heading": title,
        "anchor": "", "text": summary + " " + plain(parts[0])[:900],
    }]
    current = None
    buf = []
    for part in parts[1:]:
        m = re.match(r'<h([23]) id="([^"]+)">(.*?)</h\1>', part, re.S)
        if m:
            if current:
                chunks.append({**current, "text": plain(" ".join(buf))[:1200]})
            current = {"page": slug, "title": title,
                       "heading": plain(m.group(3)), "anchor": m.group(2)}
            buf = []
        else:
            buf.append(part)
    if current:
        chunks.append({**current, "text": plain(" ".join(buf))[:1200]})
    return [c for c in chunks if c["text"] or c["anchor"]]


TEMPLATE = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · VUGVA for llama.cpp</title>
<meta name="description" content="{summary}">
<meta property="og:title" content="{title} · VUGVA for llama.cpp">
<meta property="og:description" content="{summary}">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>&#9679;</text></svg>">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="site-header">
  <a class="brand" href="index.html"><span class="brand-mark">V</span> VUGVA <span class="brand-sub">for llama.cpp</span></a>
  <nav class="site-nav">{nav}</nav>
  <button class="search-btn" id="search-open" type="button" aria-label="Search documentation">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4.3-4.3"/></svg>
    <span>Search</span><kbd>/</kbd>
  </button>
</header>
<main class="layout">
  {toc}
  <article class="doc" id="content">
    <p class="crumb"><a href="index.html">Docs</a> / {title}</p>
    {body}
    <p class="source-link">Source: <a href="{source}">edit this page on GitHub</a></p>
  </article>
</main>
<footer class="site-footer">
  <p><strong>VUGVA</strong> — tiered VRAM → DRAM → SSD memory for llama.cpp.
     Paper DOI <a href="{paper}">10.5281/zenodo.21549808</a> ·
     <a href="{repo}">GitHub</a> · <a href="search.html">Search</a></p>
  <p class="muted">Press <kbd>?</kbd> for keyboard shortcuts.</p>
</footer>
<div class="overlay" id="search-overlay" hidden>
  <div class="overlay-panel" role="dialog" aria-modal="true" aria-label="Search documentation">
    <div class="overlay-bar">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4.3-4.3"/></svg>
      <input id="search-input" type="search" placeholder="Search docs…  e.g. VUGVA_HOT_CACHE_MB, pin, MoE"
             autocomplete="off" spellcheck="false">
      <button class="overlay-close" id="search-close" type="button" aria-label="Close search">esc</button>
    </div>
    <ul id="search-results" class="results"></ul>
    <p class="overlay-hint"><kbd>↑</kbd><kbd>↓</kbd> move · <kbd>enter</kbd> open · <kbd>esc</kbd> close</p>
  </div>
</div>
<div class="overlay" id="help-overlay" hidden>
  <div class="overlay-panel help" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
    <h2>Keyboard shortcuts</h2>
    <dl class="keys">
      <dt><kbd>/</kbd> or <kbd>Ctrl</kbd>+<kbd>K</kbd></dt><dd>open search</dd>
      <dt><kbd>Esc</kbd></dt><dd>close search / help</dd>
      <dt><kbd>↑</kbd> <kbd>↓</kbd></dt><dd>move between results</dd>
      <dt><kbd>Enter</kbd></dt><dd>open selected result</dd>
      <dt><kbd>g</kbd> then <kbd>h</kbd></dt><dd>home</dd>
      <dt><kbd>g</kbd> then <kbd>a</kbd></dt><dd>architecture</dd>
      <dt><kbd>g</kbd> then <kbd>c</kbd></dt><dd>configuration</dd>
      <dt><kbd>g</kbd> then <kbd>b</kbd></dt><dd>benchmarks</dd>
      <dt><kbd>g</kbd> then <kbd>i</kbd></dt><dd>paper vs. code</dd>
      <dt><kbd>g</kbd> then <kbd>t</kbd></dt><dd>troubleshooting</dd>
      <dt><kbd>g</kbd> then <kbd>s</kbd></dt><dd>search page</dd>
      <dt><kbd>t</kbd></dt><dd>toggle light / dark theme</dd>
      <dt><kbd>?</kbd></dt><dd>this dialog</dd>
    </dl>
    <button class="overlay-close wide" id="help-close" type="button">Close</button>
  </div>
</div>
<script src="assets/site.js" defer></script>
</body>
</html>
"""


def main() -> int:
    index = []
    for slug, title, summary, path in PAGES:
        if not path.exists():
            print(f"skip {path} (missing)", file=sys.stderr)
            continue
        body = pandoc_body(path)
        links = []
        for href, label in NAV:
            cls = ' class="active"' if href == slug + ".html" else ""
            links.append(f'<a href="{href}"{cls}>{label}</a>')
        nav = "\n".join(links)
        page = TEMPLATE.format(
            title=title, summary=html.escape(summary), nav=nav, toc=toc(body), body=body,
            source=f"{REPO}/blob/main/{path.relative_to(ROOT)}", paper=PAPER, repo=REPO)
        (DOCS / f"{slug}.html").write_text(page, encoding="utf-8")
        index += search_chunks(slug, title, body, summary)
        print(f"wrote docs/{slug}.html ({len(page)//1024} KiB)")

    (DOCS / "assets" / "search-index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote docs/assets/search-index.json ({len(index)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
