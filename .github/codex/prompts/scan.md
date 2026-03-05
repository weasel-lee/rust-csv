@codex Run the weekly Rust bug sweep for this repository and reply with one consolidated final report comment.

<!-- {{WEEK_MARKER}} -->

Repository: `{{REPO}}`

## Scope
- Review the **entire repository**, but focus on tracked `*.rs` files only.
- There are **{{TOTAL_RS_FILES}}** tracked Rust files in **{{TOTAL_BATCHES}}** batches below.
- Use the repository contents as the source of truth; the batch manifest is only a routing aid for batching.

## Execution rules
- Work through **all** batches before writing the final answer.
- Treat the batches as context-control hints; inspect neighboring files when needed.
- Prefer fewer high-confidence findings over many speculative ones.
- Report only likely, actionable bugs with concrete failure modes.
- Ignore style, formatting, naming, docs, and perf-only concerns unless they cause incorrect behavior.
- Do not open a PR and do not modify code.

## Rust focus
- panic paths (`unwrap`, `expect`, `panic!`, unchecked indexing) in non-test code
- async/concurrency bugs: lock ordering, deadlocks, cancellation safety, lost task errors, channel misuse
- integer truncation / overflow / sign bugs
- `unsafe` invariants and aliasing assumptions
- serde / schema / defaulting mismatches
- file / network I/O partial failure handling
- path handling / symlink / canonicalization mistakes
- tests that mask production-only failures

## Output
- Reply with **one consolidated final report comment** for this weekly sweep.
- For each finding include: `severity`, `confidence`, `file`, `line` or `range`, `why this is likely a bug`, and `minimal fix direction`.
- End with a `Coverage` section listing batches reviewed and any important files skipped.
- If you do not find credible bugs, say so and still include the `Coverage` section.

## Batch manifest