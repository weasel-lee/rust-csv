#!/usr/bin/env python3

from __future__ import annotations

import fnmatch
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_MAX_FILES_PER_BATCH = 15
DEFAULT_MAX_BYTES_PER_BATCH = 150_000
ROTATION_BATCH_COUNT = 3
ROTATION_ANCHOR = date(1970, 1, 5)
ALLOWLIST_PATH = ".github/metadata/allowlist.txt"
PROMPT_PATH = Path(".github/bug_sweep/rust-bug-sweep.txt")
OVERVIEW_OUTPUT_PATH = Path("codex-overview-comment.md")
TASKS_OUTPUT_PATH = Path("codex-task-comments.json")

FileEntry = tuple[str, int]


@dataclass
class Batch:
    key: str
    crate: str
    index: int
    files: list[FileEntry]
    bytes: int
    oversize: bool
    total: int = 0
    global_index: int = 0


def git_lines(*args: str) -> list[str]:
    return subprocess.check_output(["git", *args], text=True).splitlines()


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise SystemExit("Missing .github/bug_sweep/rust-bug-sweep.txt")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_allowlist(allow_file: Path) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if allow_file.exists():
        for raw in allow_file.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)

    includes = [entry for entry in lines if not entry.startswith("!")]
    excludes = [entry[1:] for entry in lines if entry.startswith("!")]
    return includes, excludes


def is_allowed(path: str, includes: list[str], excludes: list[str]) -> bool:
    if includes and not any(fnmatch.fnmatch(path, pattern) for pattern in includes):
        return False
    if excludes and any(fnmatch.fnmatch(path, pattern) for pattern in excludes):
        return False
    return True


def discover_rust_files(allow_file: Path) -> list[FileEntry]:
    includes, excludes = load_allowlist(allow_file)
    tracked = git_lines("ls-files")
    rust_files = [path for path in tracked if path.endswith(".rs")]

    files: list[FileEntry] = []
    for rel_path in rust_files:
        if not is_allowed(rel_path, includes, excludes):
            continue

        file_path = Path(rel_path)
        if file_path.is_file():
            files.append((rel_path, file_path.stat().st_size))

    return files


def load_crate_roots() -> list[str]:
    cargo_tomls = git_lines("ls-files", ":(glob)**/Cargo.toml")
    return sorted({str(Path(path).parent) for path in cargo_tomls}, key=len, reverse=True)


def crate_for(path: str, crate_roots: list[str]) -> str:
    for root in crate_roots:
        if root == ".":
            return "root"
        if path == root or path.startswith(root + "/"):
            return Path(root).name
    return "unknown"


def group_files_by_crate(
    files: list[FileEntry], crate_roots: list[str]
) -> dict[str, list[FileEntry]]:
    grouped: dict[str, list[FileEntry]] = defaultdict(list)
    for rel_path, size in sorted(files, key=lambda entry: entry[0]):
        grouped[crate_for(rel_path, crate_roots)].append((rel_path, size))
    return grouped


def pack_batches(
    by_crate: dict[str, list[FileEntry]], max_files: int, max_bytes: int
) -> list[Batch]:
    batches: list[Batch] = []

    for crate in sorted(by_crate.keys()):
        crate_files = by_crate[crate]
        crate_batches: list[Batch] = []
        current_files: list[FileEntry] = []
        current_bytes = 0
        batch_index = 1

        def flush(
            files_in_batch: list[FileEntry],
            total_bytes: int,
            next_batch_index: int,
        ) -> tuple[list[FileEntry], int, int]:
            if not files_in_batch:
                return files_in_batch, total_bytes, next_batch_index

            crate_batches.append(
                Batch(
                    key=f"{crate}__{next_batch_index:02d}",
                    crate=crate,
                    index=next_batch_index,
                    files=files_in_batch[:],
                    bytes=total_bytes,
                    oversize=False,
                )
            )
            return [], 0, next_batch_index + 1

        for rel_path, size in crate_files:
            if size > max_bytes:
                if current_files:
                    current_files, current_bytes, batch_index = flush(
                        current_files, current_bytes, batch_index
                    )

                crate_batches.append(
                    Batch(
                        key=f"{crate}__{batch_index:02d}",
                        crate=crate,
                        index=batch_index,
                        files=[(rel_path, size)],
                        bytes=size,
                        oversize=True,
                    )
                )
                batch_index += 1
                continue

            if current_files and (
                len(current_files) >= max_files or (current_bytes + size) > max_bytes
            ):
                current_files, current_bytes, batch_index = flush(
                    current_files, current_bytes, batch_index
                )

            current_files.append((rel_path, size))
            current_bytes += size

        current_files, current_bytes, batch_index = flush(
            current_files, current_bytes, batch_index
        )

        total = len(crate_batches)
        for batch in crate_batches:
            batch.total = total

        batches.extend(crate_batches)

    for global_index, batch in enumerate(batches):
        batch.global_index = global_index

    return batches


def select_weekly_batches(
    batches: list[Batch], week_index: int
) -> tuple[list[int], list[int], list[Batch]]:
    if not batches:
        return [], [], []

    raw_indices = [
        ((week_index * ROTATION_BATCH_COUNT) + offset) % len(batches)
        for offset in range(ROTATION_BATCH_COUNT)
    ]
    selected_indices: list[int] = []
    for index in raw_indices:
        if index not in selected_indices:
            selected_indices.append(index)

    return raw_indices, selected_indices, [batches[index] for index in selected_indices]


def fmt_kib(byte_count: int) -> int:
    return math.ceil(byte_count / 1024)


def render_overview(
    *,
    week_key: str,
    allow_file: Path,
    files: list[FileEntry],
    max_files: int,
    max_bytes: int,
    batches: list[Batch],
    week_index: int,
    selected_indices: list[int],
    selected_batches: list[Batch],
) -> str:
    overview: list[str] = []
    overview.append(f"<!-- CODEX_SWEEP_AGG week={week_key} -->")
    overview.append(f"## Weekly Codex Rust bug sweep ({week_key})")
    overview.append(f"Progress: **0/{len(selected_batches)}**")
    overview.append("")
    overview.append("This comment is updated in-place as Codex finishes each batch.")
    overview.append("")
    overview.append("### Sweep parameters")
    overview.append(
        f"- Allowlist: `{allow_file.as_posix() if allow_file.exists() else ALLOWLIST_PATH}`"
    )
    overview.append(f"- Rust files in scope: **{len(files)}**")
    overview.append(f"- Max files per batch: **{max_files}**")
    overview.append(f"- Max bytes per batch: **{max_bytes}** (oversize => single-file batch)")
    overview.append(f"- Total generated batches (`N`): **{len(batches)}**")
    overview.append(f"- Week index (`i`): **{week_index}**")

    if selected_indices:
        index_line = ", ".join(str(index) for index in selected_indices)
        key_line = ", ".join(f"`{batch.key}`" for batch in selected_batches)
    else:
        index_line = "(none)"
        key_line = "(none)"

    overview.append(f"- Selected batch indices: {index_line}")
    overview.append(f"- Selected batch keys: {key_line}")
    overview.append("")

    if not selected_batches:
        if not files:
            overview.append("_No Rust files matched the allowlist._")
        else:
            overview.append("_No batch selected for this week._")
        return "\n".join(overview).rstrip() + "\n"

    selected_by_crate: dict[str, list[Batch]] = defaultdict(list)
    crate_order: list[str] = []
    for batch in selected_batches:
        crate = batch.crate
        if crate not in selected_by_crate:
            crate_order.append(crate)
        selected_by_crate[crate].append(batch)

    for crate in crate_order:
        overview.append(f"### Crate: `{crate}`")
        for batch in selected_by_crate[crate]:
            kib = fmt_kib(batch.bytes)
            oversize = " (OVERSIZE)" if batch.oversize else ""
            overview.append(
                f"#### Batch `{batch.key}` ({len(batch.files)} files, ~{kib} KiB{oversize})"
            )
            overview.append(f"<!-- BEGIN batch={batch.key} -->")
            overview.append("_(pending)_")
            overview.append(f"<!-- END batch={batch.key} -->")
            overview.append("")

    return "\n".join(overview).rstrip() + "\n"


def render_task_comments(
    *,
    repo: str,
    week_key: str,
    batches: list[Batch],
    selected_batches: list[Batch],
    static_prompt: str,
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []

    for batch in selected_batches:
        task_lines: list[str] = []
        task_lines.append("@codex")
        task_lines.append(f"<!-- CODEX_SWEEP_TASK week={week_key} batch={batch.key} -->")
        task_lines.append("")
        task_lines.append("Run the Rust bug sweep for the assigned batch.")
        task_lines.append("")
        task_lines.append("Reply in a new issue comment.")
        task_lines.append("The first line of your reply must be exactly:")
        task_lines.append(f"`<!-- CODEX_SWEEP_RESULT week={week_key} batch={batch.key} -->`")
        task_lines.append("")
        task_lines.append(f"Repository: `{repo}`")
        task_lines.append(f"Week: `{week_key}`")
        task_lines.append(f"Assigned batch: `{batch.key}`")
        task_lines.append(f"Global batch index: `{batch.global_index + 1}/{len(batches)}`")
        task_lines.append("")
        task_lines.append(static_prompt)
        task_lines.append("")
        task_lines.append("## Assigned batch manifest")
        kib = fmt_kib(batch.bytes)
        oversize = " (OVERSIZE)" if batch.oversize else ""
        task_lines.append(
            f"Crate `{batch.crate}` — Batch {batch.index:02d}/{batch.total:02d} — "
            f"{len(batch.files)} files — ~{kib} KiB{oversize}"
        )
        task_lines.append("")
        task_lines.append("```text")
        for rel_path, size in batch.files:
            task_lines.append(f"{rel_path} ({size} bytes)")
        task_lines.append("```")

        tasks.append(
            {
                "batch_key": batch.key,
                "body": "\n".join(task_lines).rstrip() + "\n",
            }
        )

    return tasks


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    week_key = os.environ["WEEK_KEY"]
    max_files = int(os.getenv("MAX_FILES_PER_BATCH") or DEFAULT_MAX_FILES_PER_BATCH)
    max_bytes = int(os.getenv("MAX_BYTES_PER_BATCH") or DEFAULT_MAX_BYTES_PER_BATCH)

    static_prompt = load_prompt()
    allow_file = Path(ALLOWLIST_PATH)
    files = discover_rust_files(allow_file)
    crate_roots = load_crate_roots()
    batches = pack_batches(group_files_by_crate(files, crate_roots), max_files, max_bytes)

    today_utc = datetime.now(timezone.utc).date()
    week_index = (today_utc - ROTATION_ANCHOR).days // 7
    _, selected_indices, selected_batches = select_weekly_batches(batches, week_index)

    OVERVIEW_OUTPUT_PATH.write_text(
        render_overview(
            week_key=week_key,
            allow_file=allow_file,
            files=files,
            max_files=max_files,
            max_bytes=max_bytes,
            batches=batches,
            week_index=week_index,
            selected_indices=selected_indices,
            selected_batches=selected_batches,
        ),
        encoding="utf-8",
    )
    TASKS_OUTPUT_PATH.write_text(
        json.dumps(
            render_task_comments(
                repo=repo,
                week_key=week_key,
                batches=batches,
                selected_batches=selected_batches,
                static_prompt=static_prompt,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
