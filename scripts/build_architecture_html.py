#!/usr/bin/env python3
"""
Render ARCHITECTURE.md as ARCHITECTURE.html, for sharing in meetings.

ARCHITECTURE.md stays the single source of truth and the HTML is a build
product. Edit the Markdown, then re-run and commit both:

    python scripts/build_architecture_html.py               # → ARCHITECTURE.html
    python scripts/build_architecture_html.py -o out.html

Output is deterministic (no dates, no commit hashes), so regenerating an
unchanged Markdown file leaves no diff.

Mermaid fences become diagrams, rendered in the browser (Mermaid loads from
a CDN, so viewing the page needs a network connection; the performance
runtime is unaffected). Relative links are pointed at the GitHub repo.

Uses markdown-it-py, which already ships with rich (a core dependency).
"""
import argparse
import html
import re
import sys
from pathlib import Path
from string import Template

from markdown_it import MarkdownIt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/sgl-a/Indira"
MERMAID_JS = "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.0/mermaid.min.js"

# Leading status emoji in a blockquote → callout style
NOTE_STYLES = {"✅": "ok", "🟡": "partial", "🔜": "planned", "⚠": "warn"}

# Any <tag>-looking text in a Mermaid label except <br/>
_MERMAID_TAG = re.compile(r"<(?!br\s*/?>)(/?[A-Za-z][\w-]*)>")


def slugify(text: str, seen: dict[str, int]) -> str:
    """GitHub-style heading anchor, so #links match the rendered .md on GitHub."""
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    n = seen.get(slug, 0)
    seen[slug] = n + 1
    return slug if n == 0 else f"{slug}-{n}"


def prepare_mermaid(src: str) -> str:
    """Adapt a Mermaid fence for the browser build of Mermaid.

    - A literal "\\n" inside a label becomes <br/>, the line break Mermaid
      reliably honours.
    - Tag-like text such as <think> becomes entity codes; otherwise Mermaid's
      sanitizer drops it from the label as an unknown HTML tag.
    """
    src = src.replace("\\n", "<br/>")
    return _MERMAID_TAG.sub(r"#lt;\1#gt;", src)


def _render_fence(self, tokens, idx, options, env):
    token = tokens[idx]
    if token.info.strip() == "mermaid":
        src = html.escape(prepare_mermaid(token.content))
        return f'<figure class="diagram"><pre class="diagram-src">{src}</pre></figure>\n'
    return self.fence(tokens, idx, options, env)


def _render_table_open(self, tokens, idx, options, env):
    return '<div class="table-wrap">' + self.renderToken(tokens, idx, options, env)


def _render_table_close(self, tokens, idx, options, env):
    return self.renderToken(tokens, idx, options, env) + "</div>\n"


def _inline_text(inline) -> str:
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


def render_markdown(text: str) -> tuple[str, str, list[tuple[int, str, str]], int]:
    """Return (h1 text, body html, [(level, text, anchor)] for the TOC, diagram count)."""
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    md.add_render_rule("fence", _render_fence)
    md.add_render_rule("table_open", _render_table_open)
    md.add_render_rule("table_close", _render_table_close)

    env: dict = {}
    tokens = md.parse(text, env)
    seen: dict[str, int] = {}
    toc: list[tuple[int, str, str]] = []
    h1 = ""
    diagrams = 0

    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            heading = _inline_text(tokens[i + 1])
            anchor = slugify(heading, seen)
            token.attrSet("id", anchor)
            level = int(token.tag[1])
            if level == 1 and not h1:
                h1 = heading
            elif level in (2, 3):
                toc.append((level, heading, anchor))

        elif token.type == "blockquote_open":
            first = next((t for t in tokens[i + 1:] if t.type == "inline"), None)
            lead = first.content.lstrip() if first else ""
            style = next((v for k, v in NOTE_STYLES.items() if lead.startswith(k)), None)
            token.attrSet("class", f"note note-{style}" if style else "note")

        elif token.type == "fence" and token.info.strip() == "mermaid":
            diagrams += 1

        elif token.type == "inline":
            for child in token.children or []:
                if child.type != "link_open":
                    continue
                href = str(child.attrGet("href") or "")
                if href.startswith("#"):
                    continue
                if not re.match(r"^[a-z][a-z0-9+.-]*:", href):
                    child.attrSet("href", f"{REPO_URL}/blob/main/{href}")
                child.attrSet("target", "_blank")
                child.attrSet("rel", "noopener")

    body = md.renderer.render(tokens, md.options, env)
    return h1, body, toc, diagrams


def build_toc(toc: list[tuple[int, str, str]]) -> str:
    items = "\n".join(
        f'<li class="toc-l{level}"><a href="#{anchor}">{html.escape(text)}</a></li>'
        for level, text, anchor in toc
    )
    return f"<ol>\n{items}\n</ol>"


def render_page(markdown_text: str, source_name: str = "ARCHITECTURE.md") -> str:
    """The full HTML page for a Markdown document. Deterministic."""
    h1, body, toc, _ = render_markdown(markdown_text)
    # Page/tab name: "Indira — System Architecture" → "Indira System Architecture"
    title = re.sub(r"\s+[—–-]\s+", " ", h1) or Path(source_name).stem
    stamp = f"Generated from {source_name} by scripts/build_architecture_html.py"
    return PAGE.substitute(
        title=html.escape(title),
        toc=build_toc(toc),
        body=body,
        stamp=html.escape(stamp),
        mermaid_js=MERMAID_JS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ARCHITECTURE.md as HTML.")
    parser.add_argument("-i", "--input", type=Path, default=PROJECT_ROOT / "ARCHITECTURE.md")
    parser.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "ARCHITECTURE.html")
    args = parser.parse_args()

    source = args.input.resolve()
    text = source.read_text(encoding="utf-8")
    page = render_page(text, source.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")

    _, _, toc, diagrams = render_markdown(text)
    try:
        shown = args.output.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = args.output
    print(f"Wrote {shown} ({len(toc)} sections, {diagrams} diagrams)")
    return 0


# Page shell. Written as a fragment (no <html>/<head>/<body>): browsers open
# it as-is, and the claude.ai artifact publisher adds its own wrapper.
# string.Template placeholders: $title $toc $body $stamp $mermaid_js
PAGE = Template("""\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>$title</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@800&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
  :root {
    --bg: #ECEEF2;
    --surface: #F8F9FB;
    --surface-2: #E2E6ED;
    --ink: #141822;
    --muted: #566074;
    --line: #C3C9D4;
    --accent: #2B43AE;
    --accent-soft: #DAE0F6;
    --warm: #985100;
    --warm-soft: #F5E4CB;
    --ok: #2A7849;
    --plan: #7C8395;
    --display: "Big Shoulders Display", "Arial Narrow", "Roboto Condensed", sans-serif;
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #0D1018;
      --surface: #151923;
      --surface-2: #1D2230;
      --ink: #E6E9F0;
      --muted: #9BA3B5;
      --line: #2D3445;
      --accent: #93A6FF;
      --accent-soft: #222B50;
      --warm: #EFB050;
      --warm-soft: #382812;
      --ok: #6BCB90;
      --plan: #7B8398;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0D1018;
    --surface: #151923;
    --surface-2: #1D2230;
    --ink: #E6E9F0;
    --muted: #9BA3B5;
    --line: #2D3445;
    --accent: #93A6FF;
    --accent-soft: #222B50;
    --warm: #EFB050;
    --warm-soft: #382812;
    --ok: #6BCB90;
    --plan: #7B8398;
  }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 16px/1.65 var(--sans);
  }
  @media (prefers-reduced-motion: no-preference) {
    html { scroll-behavior: smooth; }
  }

  .layout {
    display: grid;
    grid-template-columns: 230px minmax(0, 1fr);
    gap: 56px;
    max-width: 1240px;
    margin: 0 auto;
    padding-inline: clamp(16px, 4vw, 40px);
    padding-block: 36px 72px;
  }
  @media (max-width: 920px) {
    .layout { grid-template-columns: minmax(0, 1fr); gap: 8px; }
  }

  /* ── Contents ── */
  .toc {
    position: sticky;
    top: calc(env(safe-area-inset-top, 0px) + 24px);
    align-self: start;
    max-height: calc(100vh - 48px);
    overflow-y: auto;
    font-size: 13.5px;
    line-height: 1.4;
  }
  @media (max-width: 920px) {
    .toc { position: static; max-height: none; border-block: 1px solid var(--line); padding-block: 10px; }
  }
  .toc summary {
    font: 500 11.5px/1.4 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    cursor: pointer;
    padding-block: 4px;
  }
  .toc ol { list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 2px; }
  .toc a {
    display: block;
    padding: 4px 8px;
    border-radius: 5px;
    color: var(--ink);
    text-decoration: none;
  }
  .toc a:hover { background: var(--surface-2); }
  .toc .toc-l3 a { padding-left: 22px; color: var(--muted); font-size: 13px; }
  a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  /* ── Document ── */
  .doc { min-width: 0; max-width: 880px; }
  .doc > :first-child { margin-top: 0; }
  .doc h1 {
    font: 800 clamp(48px, 7vw, 76px)/0.95 var(--display);
    text-transform: uppercase;
    letter-spacing: 0.01em;
    margin: 0 0 20px;
    text-wrap: balance;
  }
  .doc h2 {
    font-size: 27px;
    line-height: 1.2;
    font-weight: 600;
    margin: 56px 0 14px;
    padding-top: 22px;
    border-top: 1px solid var(--line);
    text-wrap: balance;
  }
  .doc h3 { font-size: 20px; line-height: 1.3; font-weight: 600; margin: 40px 0 10px; text-wrap: balance; }
  .doc h4 { font-size: 16.5px; font-weight: 600; margin: 28px 0 8px; }
  .doc h2, .doc h3, .doc h4 { scroll-margin-top: 20px; }
  .doc p, .doc li { max-width: 72ch; }
  .doc p { margin: 0 0 14px; }
  .doc ul, .doc ol { padding-left: 1.4em; margin: 0 0 14px; }
  .doc li { margin: 4px 0; }
  .doc a { color: var(--accent); text-underline-offset: 2px; }
  .doc strong { font-weight: 600; }
  .doc hr { border: 0; border-top: 1px solid var(--line); margin: 40px 0; }

  .doc code {
    font: 0.86em var(--mono);
    background: var(--surface-2);
    padding: 1px 5px;
    border-radius: 4px;
  }
  .doc pre {
    font: 13px/1.55 var(--mono);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    margin: 0 0 18px;
  }
  .doc pre code { background: none; padding: 0; font-size: inherit; }

  .note {
    margin: 0 0 18px;
    padding: 12px 16px;
    background: var(--surface);
    border-left: 3px solid var(--line);
    border-radius: 0 8px 8px 0;
  }
  .note > :last-child { margin-bottom: 0; }
  .note p { max-width: none; }
  .note-ok { border-left-color: var(--ok); }
  .note-partial, .note-warn { border-left-color: var(--warm); }
  .note-planned { border-left-color: var(--plan); border-left-style: dashed; }

  .table-wrap { overflow-x: auto; margin: 0 0 20px; }
  .doc table { border-collapse: collapse; width: 100%; font-size: 14.5px; line-height: 1.5; }
  .doc th, .doc td { text-align: left; vertical-align: top; padding: 9px 12px; border-bottom: 1px solid var(--line); }
  .doc thead th {
    font-weight: 600;
    border-bottom: 1.5px solid var(--ink);
    white-space: nowrap;
  }

  figure.diagram {
    margin: 0 0 20px;
    padding: 18px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow-x: auto;
  }
  figure.diagram svg { display: block; margin: 0 auto; height: auto; }
  .diagram-src {
    margin: 0;
    border: 0;
    padding: 0;
    background: none;
    color: var(--muted);
  }

  footer {
    margin-top: 64px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font: 12.5px/1.5 var(--mono);
    color: var(--muted);
  }
</style>

<div class="layout" lang="en">
  <nav class="toc" aria-label="Contents">
    <details open>
      <summary>Contents</summary>
$toc
    </details>
  </nav>
  <main class="doc">
$body
    <footer>$stamp</footer>
  </main>
</div>

<script src="$mermaid_js"></script>
<script>
(function () {
  var details = document.querySelector(".toc details");
  if (details && window.matchMedia("(max-width: 920px)").matches) details.open = false;

  var figures = Array.prototype.slice.call(document.querySelectorAll("figure.diagram"));
  if (!figures.length || !window.mermaid) return;  // no network: the source stays visible

  var sources = figures.map(function (fig) {
    return fig.querySelector(".diagram-src").textContent;
  });
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function token(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function isDark() {
    var theme = root.getAttribute("data-theme");
    return theme ? theme === "dark" : media.matches;
  }

  // Diagrams take their colours from the page tokens, so they follow the
  // light/dark theme. Re-rendered whenever the theme changes.
  var generation = 0;
  function render() {
    var current = ++generation;
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      darkMode: isDark(),
      fontFamily: token("--sans"),
      flowchart: { htmlLabels: true, curve: "basis" },
      themeVariables: {
        fontFamily: token("--sans"),
        fontSize: "14px",
        background: token("--surface"),
        primaryColor: token("--surface"),
        primaryTextColor: token("--ink"),
        primaryBorderColor: token("--muted"),
        secondaryColor: token("--accent-soft"),
        tertiaryColor: token("--surface-2"),
        lineColor: token("--muted"),
        textColor: token("--ink"),
        mainBkg: token("--surface"),
        nodeBorder: token("--muted"),
        clusterBkg: token("--bg"),
        clusterBorder: token("--line"),
        titleColor: token("--ink"),
        edgeLabelBackground: token("--surface"),
        actorBkg: token("--surface"),
        actorBorder: token("--muted"),
        actorTextColor: token("--ink"),
        actorLineColor: token("--line"),
        signalColor: token("--ink"),
        signalTextColor: token("--ink"),
        labelBoxBkgColor: token("--surface-2"),
        labelBoxBorderColor: token("--line"),
        labelTextColor: token("--ink"),
        loopTextColor: token("--ink"),
        noteBkgColor: token("--warm-soft"),
        noteBorderColor: token("--warm"),
        noteTextColor: token("--ink"),
        activationBkgColor: token("--accent-soft"),
        activationBorderColor: token("--accent")
      }
    });
    figures.forEach(function (fig, i) {
      window.mermaid.render("arch-diagram-" + i + "-" + current, sources[i])
        .then(function (out) { if (current === generation) fig.innerHTML = out.svg; })
        .catch(function () { /* leave the previous rendering (or the source) in place */ });
    });
  }

  render();
  if (media.addEventListener) media.addEventListener("change", render);
  new MutationObserver(render).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>
""")


if __name__ == "__main__":
    sys.exit(main())
