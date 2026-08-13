"""Export helpers for deterministic report artifacts."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from app.tools.file_utils import save_json, write_text_atomic


def save_markdown(path: str | Path, content: str) -> Path:
    """Save a Markdown report."""

    markdown_path = Path(path)
    return write_text_atomic(markdown_path, _with_trailing_newline(content))


def save_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a structured JSON report artifact."""

    return save_json(path, payload)


def save_html_from_markdown(path: str | Path, markdown_content: str, title: str) -> Path:
    """Save a lightweight HTML export without adding Markdown dependencies."""

    html_path = Path(path)
    return write_text_atomic(
        html_path,
        _markdown_to_basic_html(markdown_content, title=title),
    )


def _with_trailing_newline(content: str) -> str:
    return content.rstrip() + "\n"


def _markdown_to_basic_html(markdown_content: str, title: str) -> str:
    body_lines: list[str] = []
    in_list = False
    in_pre = False

    for raw_line in markdown_content.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_pre:
                body_lines.append("</pre>")
                in_pre = False
            else:
                _close_list(body_lines, in_list)
                in_list = False
                body_lines.append("<pre>")
                in_pre = True
            continue

        if in_pre:
            body_lines.append(escape(line))
            continue

        if not line:
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            continue

        if line.startswith("# "):
            _close_list(body_lines, in_list)
            in_list = False
            body_lines.append(f"<h1>{escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            _close_list(body_lines, in_list)
            in_list = False
            body_lines.append(f"<h2>{escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            _close_list(body_lines, in_list)
            in_list = False
            body_lines.append(f"<h3>{escape(line[4:].strip())}</h3>")
        elif line.startswith("- "):
            if not in_list:
                body_lines.append("<ul>")
                in_list = True
            body_lines.append(f"<li>{_inline_markdown_to_html(line[2:].strip())}</li>")
        else:
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            body_lines.append(f"<p>{_inline_markdown_to_html(line)}</p>")

    if in_list:
        body_lines.append("</ul>")
    if in_pre:
        body_lines.append("</pre>")

    body = "\n".join(body_lines)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.55;
      margin: 0 auto;
      max-width: 960px;
      padding: 32px 20px;
    }}
    h1, h2, h3 {{ color: #102a43; line-height: 1.25; }}
    code {{ background: #f3f4f6; border-radius: 4px; padding: 2px 4px; }}
    pre {{ background: #f3f4f6; border-radius: 6px; overflow-x: auto; padding: 12px; }}
    p {{ margin: 0 0 12px; }}
    ul {{ margin-top: 0; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _inline_markdown_to_html(text: str) -> str:
    escaped = escape(text)
    pieces = escaped.split("`")
    if len(pieces) == 1:
        return escaped

    html = []
    for index, piece in enumerate(pieces):
        if index % 2:
            html.append(f"<code>{piece}</code>")
        else:
            html.append(piece)
    return "".join(html)


def _close_list(lines: list[str], in_list: bool) -> None:
    if in_list:
        lines.append("</ul>")
