"""Generate Marp weekly progress slides for the GraphRAG thesis project.

Usage:
    uv run python scripts/gen_weekly_slides.py                # last 7 days
    uv run python scripts/gen_weekly_slides.py --days 14
    uv run python scripts/gen_weekly_slides.py --since 2026-05-15 --until 2026-05-21
    uv run python scripts/gen_weekly_slides.py --pdf          # also render PDF via npx marp-cli

Output:
    reports/weekly/<YYYY>-W<WW>.md     (Marp markdown, manually editable)
    reports/weekly/<YYYY>-W<WW>.pdf    (only when --pdf is passed)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "weekly"

CONV_TYPES: dict[str, str] = {
    "feat": "Tính năng mới",
    "fix": "Sửa lỗi",
    "refactor": "Tái cấu trúc",
    "perf": "Cải thiện hiệu năng",
    "test": "Kiểm thử",
    "docs": "Tài liệu",
    "doc": "Tài liệu",
    "chore": "Vận hành & dọn dẹp",
    "ops": "Vận hành & dọn dẹp",
    "ci": "CI/CD",
    "build": "Build & dependencies",
    "style": "Định dạng mã",
}
TYPE_ORDER = ["feat", "fix", "refactor", "perf", "test", "docs", "ops", "ci", "build", "style"]

COMMIT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<bang>!)?:\s*(?P<subject>.+)$"
)


@dataclass
class Commit:
    sha: str
    subject: str
    author: str
    when: str  # YYYY-MM-DD
    type: str | None
    scope: str | None
    breaking: bool


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout


def parse_commits(since: str, until: str) -> list[Commit]:
    raw = run_git(
        [
            "log",
            f"--since={since}",
            f"--until={until} 23:59:59",
            "--no-merges",
            "--pretty=format:%H%x1f%s%x1f%an%x1f%ad",
            "--date=short",
        ]
    )
    commits: list[Commit] = []
    for line in raw.splitlines():
        if not line:
            continue
        sha, subject, author, when = line.split("\x1f")
        m = COMMIT_RE.match(subject)
        if m and m.group("type") in CONV_TYPES:
            commits.append(
                Commit(
                    sha=sha[:8],
                    subject=m.group("subject").strip(),
                    author=author,
                    when=when,
                    type=m.group("type"),
                    scope=m.group("scope"),
                    breaking=bool(m.group("bang")),
                )
            )
        else:
            commits.append(
                Commit(sha[:8], subject, author, when, None, None, False)
            )
    return commits


def shortstat(since: str, until: str) -> tuple[int, int, int]:
    """Return (files_changed, insertions, deletions) for the date range."""
    raw = run_git(
        [
            "log",
            f"--since={since}",
            f"--until={until} 23:59:59",
            "--no-merges",
            "--shortstat",
            "--pretty=format:",
        ]
    )
    files = ins = dels = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(\d+) files? changed", line)
        if m:
            files += int(m.group(1))
        m = re.search(r"(\d+) insertions?\(\+\)", line)
        if m:
            ins += int(m.group(1))
        m = re.search(r"(\d+) deletions?\(-\)", line)
        if m:
            dels += int(m.group(1))
    return files, ins, dels


def group_by_type(commits: list[Commit]) -> dict[str, list[Commit]]:
    groups: dict[str, list[Commit]] = defaultdict(list)
    for c in commits:
        key = c.type if c.type in CONV_TYPES else "other"
        groups[key].append(c)
    return groups


def render(commits: list[Commit], since: str, until: str, week_label: str) -> str:
    groups = group_by_type(commits)
    files, ins, dels = shortstat(since, until)

    lines: list[str] = []
    lines.append("---")
    lines.append("marp: true")
    lines.append("theme: default")
    lines.append("paginate: true")
    lines.append("size: 16:9")
    lines.append("lang: vi")
    lines.append("---")
    lines.append("")
    lines.append(f"# Báo cáo tuần — {week_label}")
    lines.append("")
    lines.append("**Đồ án tốt nghiệp:** Vietnamese GraphRAG over Wikipedia + Neo4j")
    lines.append("")
    lines.append(f"**Khoảng thời gian:** {since} → {until}")
    lines.append("")
    lines.append("<!-- TODO: tên SV, MSSV, GVHD -->")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Tóm tắt")
    lines.append("")
    lines.append("<!-- TODO: 2-3 gạch đầu dòng nêu thành quả lớn nhất tuần này -->")
    lines.append("")
    summary_counts: list[str] = []
    for t in TYPE_ORDER:
        if t in groups and groups[t]:
            summary_counts.append(f"{len(groups[t])} {CONV_TYPES[t].lower()}")
    if "other" in groups:
        summary_counts.append(f"{len(groups['other'])} commit khác")
    if summary_counts:
        lines.append("**Hoạt động git:** " + ", ".join(summary_counts) + ".")
        lines.append("")
    lines.append(
        f"**Quy mô thay đổi:** {files} file, +{ins} / -{dels} dòng."
    )
    lines.append("")

    for t in TYPE_ORDER:
        items = groups.get(t)
        if not items:
            continue
        lines.append("---")
        lines.append("")
        lines.append(f"## {CONV_TYPES[t]}")
        lines.append("")
        for c in items:
            scope = f"`{c.scope}` " if c.scope else ""
            bang = " **(BREAKING)**" if c.breaking else ""
            lines.append(f"- {scope}{c.subject}{bang}  ")
            lines.append(f"  <small>{c.sha} · {c.when}</small>")
        lines.append("")

    if groups.get("other"):
        lines.append("---")
        lines.append("")
        lines.append("## Commit khác")
        lines.append("")
        for c in groups["other"]:
            lines.append(f"- {c.subject}  ")
            lines.append(f"  <small>{c.sha} · {c.when}</small>")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Demo / Kết quả")
    lines.append("")
    lines.append("<!-- TODO: ảnh chụp UI, log truy vấn, ví dụ Cypher do model sinh -->")
    lines.append("")
    lines.append("```")
    lines.append("# ví dụ: ảnh demo lưu ở reports/weekly/assets/<tên-file>.png")
    lines.append("# ![demo](assets/demo-week.png)")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Số liệu")
    lines.append("")
    lines.append("<!-- TODO: Coverage, EM/F1 trên ViWiki-MHR, latency, KG size, ... -->")
    lines.append("")
    lines.append("| Chỉ số | Tuần trước | Tuần này | Mục tiêu |")
    lines.append("|---|---|---|---|")
    lines.append("| Test coverage |  |  | ≥ 75% |")
    lines.append("| Số entity trong Neo4j |  |  |  |")
    lines.append("| Số chunk |  |  |  |")
    lines.append("| EM / F1 |  |  |  |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Khó khăn & hướng xử lý")
    lines.append("")
    lines.append("<!-- TODO: blocker chính, cần GVHD góp ý điểm nào -->")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Kế hoạch tuần tới")
    lines.append("")
    lines.append("<!-- TODO: 3-5 mục, gắn với mốc bảo vệ -->")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("# Cảm ơn thầy cô")
    lines.append("")
    lines.append("Q & A")
    lines.append("")

    return "\n".join(lines)


def render_pdf(md_path: Path) -> Path:
    if shutil.which("npx") is None:
        raise RuntimeError("npx không có trên PATH — không thể render PDF.")
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run(
        ["npx", "--yes", "@marp-team/marp-cli", str(md_path), "--pdf", "--allow-local-files"],
        cwd=ROOT,
        check=True,
    )
    return pdf_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate weekly thesis report slides.")
    p.add_argument("--days", type=int, default=7, help="Số ngày nhìn lại (mặc định 7).")
    p.add_argument("--since", type=str, default=None, help="YYYY-MM-DD, ghi đè --days.")
    p.add_argument("--until", type=str, default=None, help="YYYY-MM-DD, mặc định hôm nay.")
    p.add_argument("--pdf", action="store_true", help="Render PDF qua npx marp-cli.")
    p.add_argument("--force", action="store_true", help="Ghi đè file .md đã tồn tại.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    today = date.today()
    until = (
        datetime.strptime(args.until, "%Y-%m-%d").date() if args.until else today
    )
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").date()
    else:
        since = until - timedelta(days=args.days - 1)

    iso_year, iso_week, _ = until.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{week_label}.md"
    if md_path.exists() and not args.force:
        print(
            f"[!] {md_path.relative_to(ROOT)} đã tồn tại — dùng --force để ghi đè.",
            file=sys.stderr,
        )
        return 1

    commits = parse_commits(since.isoformat(), until.isoformat())
    md = render(commits, since.isoformat(), until.isoformat(), week_label)
    md_path.write_text(md, encoding="utf-8")
    print(f"[+] Đã ghi {md_path.relative_to(ROOT)} ({len(commits)} commit).")

    if args.pdf:
        try:
            pdf_path = render_pdf(md_path)
            print(f"[+] Đã render {pdf_path.relative_to(ROOT)}.")
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[!] Không render được PDF: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
