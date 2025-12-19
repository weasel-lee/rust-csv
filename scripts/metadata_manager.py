import fnmatch
import hashlib
import os
import pathlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI

# ====== ENV ======
MODE = os.getenv("MODE", "diff").strip()  # "diff" | "bootstrap"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-pro")

MAX_RS_LINES = int(os.getenv("MAX_RS_LINES", "2000"))        # 요청: 약 2천 LOC (truncate 금지, 초과면 skip)
MAX_META_LINES = int(os.getenv("MAX_META_LINES", "6000"))    # 기존 metadata 너무 큰 경우 보호 (원하면 조절)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "5"))

FORCE_UPDATE = os.getenv("FORCE_UPDATE", "false").lower() in ("1", "true", "yes")

BASE_SHA = os.getenv("BASE_SHA", "").strip()
HEAD_SHA = os.getenv("HEAD_SHA", "").strip()

PROMPT_PATH = pathlib.Path(".github/metadata/prompt.txt")
ALLOWLIST_PATH = pathlib.Path(".github/metadata/allowlist.txt")

@dataclass(frozen=True)
class Change:
    status: str  # 'A', 'M', 'D'
    path: str

def run_git(args: List[str]) -> str:
    return subprocess.check_output(["git"] + args, text=True, encoding="utf-8", errors="replace")

def load_allowlist_patterns() -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    if ALLOWLIST_PATH.exists():
        for raw in ALLOWLIST_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    includes = [x for x in lines if not x.startswith("!")]
    excludes = [x[1:] for x in lines if x.startswith("!")]
    return includes, excludes

def matches_allowlist(path: str, includes: List[str], excludes: List[str]) -> bool:
    if includes and not any(fnmatch.fnmatch(path, pat) for pat in includes):
        return False
    if excludes and any(fnmatch.fnmatch(path, pat) for pat in excludes):
        return False
    return True

def parse_changes(base: str, head: str) -> List[Change]:
    out = run_git(["diff", "--name-status", base, head])
    changes: List[Change] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        code = parts[0]

        if code.startswith("R") or code.startswith("C"):
            # rename/copy는 old 삭제 + new 생성으로 단순화
            if len(parts) >= 3:
                changes.append(Change("D", parts[1]))
                changes.append(Change("A", parts[2]))
            continue

        if code in ("A", "M", "D") and len(parts) >= 2:
            changes.append(Change(code, parts[1]))

    # 중복 제거
    return list(dict.fromkeys(changes))

def list_repo_rs_files() -> List[str]:
    # tracked 파일 기준
    out = run_git(["ls-files"])
    files = [l.strip() for l in out.splitlines() if l.strip()]
    return [f for f in files if f.endswith(".rs")]

def metadata_path_for_rs(rs_path: str) -> str:
    return rs_path[:-3] + ".metadata.txt"  # a.rs -> a.metadata.txt

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()

def parse_source_hash(meta_text: str) -> Optional[str]:
    """
    Returns source_sha256 if present in the first few lines.
    """
    for line in meta_text.splitlines()[:20]:
        line = line.strip()
        if line.startswith("# source_sha256:"):
            return line.split(":", 1)[1].strip()
    return None

def strip_hash_header(text: str) -> str:
    """
    If model output accidentally includes our hash header line, drop it to avoid duplication.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip().startswith("# source_sha256:"):
        i += 1
    if i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:]).lstrip()

def read_text_with_line_limit(path: pathlib.Path, max_lines: int) -> Tuple[Optional[str], int, bool]:
    if not path.exists():
        return None, 0, False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    if n > max_lines:
        return None, n, True
    return "\n".join(lines), n, False

def extract_output_text(resp) -> str:
    if getattr(resp, "output_text", None):
        return resp.output_text
    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts).strip() or str(resp)

def call_model(prompt: str) -> str:
    client = OpenAI()
    backoff = 1.0
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = client.responses.create(model=MODEL, input=prompt)
            return extract_output_text(resp)
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg) or ("rate limit" in msg) or ("timeout" in msg) or ("temporarily" in msg)
            if transient and attempt < RETRY_MAX:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            raise

def build_prompt(static_prompt: str, rs_path: str, rs_text: str, old_meta_text: Optional[str]) -> str:
    parts = [static_prompt.strip(), ""]
    parts.append(f"=== TARGET ===\n{rs_path}\n")
    parts.append(f"=== RUST SOURCE ({rs_path}) ===\n{rs_text}\n")
    if old_meta_text is not None:
        parts.append(f"=== EXISTING METADATA ({metadata_path_for_rs(rs_path)}) ===\n{old_meta_text}\n")
    else:
        parts.append("=== EXISTING METADATA ===\n<none>\n")
    parts.append("=== OUTPUT INSTRUCTIONS ===\nWrite the new/updated metadata text only.\n")
    return "\n".join(parts)

def process_one(repo_root: pathlib.Path, static_prompt: str, ch: Change) -> str:
    rs_file = repo_root / ch.path
    meta_file = repo_root / metadata_path_for_rs(ch.path)

    if ch.status == "D":
        if meta_file.exists():
            meta_file.unlink()
            return f"[D] Removed metadata: {meta_file}"
        return f"[D] No metadata to remove for: {ch.path}"

    # A or M
    if not rs_file.exists() or rs_file.is_dir():
        return f"[{ch.status}] SKIP missing/dir rs: {ch.path}"

    rs_text, rs_lines, rs_too_big = read_text_with_line_limit(rs_file, MAX_RS_LINES)
    if rs_too_big:
        return f"[{ch.status}] SKIP too large ({rs_lines} lines > {MAX_RS_LINES}): {ch.path}"

    # --- hash cache input (source only) ---
    source_hash = sha256_text(rs_text or "")

    old_meta_text: Optional[str] = None
    if ch.status == "M":
        if meta_file.exists():
            old_meta_text, meta_lines, meta_too_big = read_text_with_line_limit(meta_file, MAX_META_LINES)
            if meta_too_big:
                return f"[M] SKIP metadata too large ({meta_lines} lines > {MAX_META_LINES}): {meta_file}"
        else:
            old_meta_text = None  # 없으면 created처럼

    # --- hash cache check (skip API call if unchanged) ---
    if meta_file.exists() and not FORCE_UPDATE:
        meta_text_for_hash = old_meta_text
        if meta_text_for_hash is None:
            meta_text_for_hash, _, meta_too_big2 = read_text_with_line_limit(meta_file, MAX_META_LINES)
            if meta_too_big2:
                return f"[{ch.status}] SKIP metadata too large ({meta_file})"
        if meta_text_for_hash:
            prev_src = parse_source_hash(meta_text_for_hash)
            if prev_src == source_hash:
                return f"[{ch.status}] SKIP source hash unchanged: {meta_file}"

    prompt = build_prompt(static_prompt, ch.path, rs_text or "", old_meta_text if ch.status == "M" else None)
    result = call_model(prompt)

    meta_file.parent.mkdir(parents=True, exist_ok=True)
    body = strip_hash_header(result)
    meta_file.write_text(
        f"# source_sha256: {source_hash}\n\n{body}\n",
        encoding="utf-8",
    )

    return f"[{ch.status}] Wrote metadata: {meta_file}"

def main() -> None:
    repo_root = pathlib.Path(".")
    includes, excludes = load_allowlist_patterns()
    static_prompt = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

    if MODE == "diff":
        if not BASE_SHA or not HEAD_SHA:
            raise SystemExit("MODE=diff requires BASE_SHA and HEAD_SHA env vars.")

        changes = parse_changes(BASE_SHA, HEAD_SHA)
        # allowlist에 매칭되는 .rs만
        rs_changes = [
            c for c in changes
            if c.path.endswith(".rs") and matches_allowlist(c.path, includes, excludes)
        ]

        if not rs_changes:
            print("No allowlisted .rs changes found.")
            return

        # Deleted 먼저 순차 처리(간단)
        for c in [x for x in rs_changes if x.status == "D"]:
            print(process_one(repo_root, static_prompt, c))

        # Added/Modified는 병렬
        to_process = [x for x in rs_changes if x.status in ("A", "M")]
        if not to_process:
            return

        print(f"Processing {len(to_process)} files in parallel: workers={MAX_WORKERS}, model={MODEL}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, repo_root, static_prompt, c): c for c in to_process}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    print(fut.result())
                except Exception as e:
                    raise RuntimeError(f"Failed for {c.status} {c.path}: {e}") from e

        return

    if MODE == "bootstrap":
        # repo 내 allowlist에 해당하는 모든 .rs에 대해 "최초 생성"
        rs_files = [p for p in list_repo_rs_files() if matches_allowlist(p, includes, excludes)]
        if not rs_files:
            print("No allowlisted .rs files in repo.")
            return

        changes: List[Change] = []
        for p in rs_files:
            meta = repo_root / metadata_path_for_rs(p)
            if meta.exists():
                if not FORCE_UPDATE:
                    # 최초 생성이 목적이므로, 이미 있으면 스킵
                    continue
                # FORCE_UPDATE일 때는 기존 metadata를 입력에 포함시키기 위해 M으로 처리
                changes.append(Change("M", p))
            else:
                changes.append(Change("A", p))

        if not changes:
            print("Nothing to bootstrap (all metadata already exists).")
            return

        print(f"Bootstrapping {len(changes)} files: workers={MAX_WORKERS}, model={MODEL}, FORCE_UPDATE={FORCE_UPDATE}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, repo_root, static_prompt, c): c for c in changes}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    print(fut.result())
                except Exception as e:
                    raise RuntimeError(f"Failed for {c.path}: {e}") from e

        return

    raise SystemExit(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
