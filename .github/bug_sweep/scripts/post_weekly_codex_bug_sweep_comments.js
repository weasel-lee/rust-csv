"use strict";

const fs = require("fs");
const {
  isTrustedCommenter,
  parseTaskMarker,
  startsWithMarker,
  waitForEyesReaction,
} = require("./codex_weekly_bug_sweep_common");

const CODEX_ACK_POLL_INTERVAL_MS = 5_000;
const CODEX_ACK_TIMEOUT_MS = 2 * 60_000;

async function run({ github, context, core }) {
  const issueNumber = Number(process.env.ISSUE_NUMBER);
  if (!issueNumber) {
    throw new Error("Set repo variable CODEX_WEEKLY_BUG_SWEEP_ISSUE_NUMBER.");
  }

  const weekKey = process.env.WEEK_KEY;
  const overviewMarker = `<!-- CODEX_SWEEP_AGG week=${weekKey} -->`;

  const comments = await github.paginate(
    github.rest.issues.listComments,
    {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
      per_page: 100,
    }
  );

  const overviewBody = fs.readFileSync("codex-overview-comment.md", "utf8");
  const tasks = JSON.parse(fs.readFileSync("codex-task-comments.json", "utf8"));
  const overviewComment = comments.find(
    comment => isTrustedCommenter(comment) && startsWithMarker(comment.body, overviewMarker)
  );
  const existingTaskBatches = new Set(
    comments
      .filter(comment => isTrustedCommenter(comment))
      .map(comment => {
        const marker = parseTaskMarker(comment.body);
        if (!marker || marker.weekKey !== weekKey) {
          return null;
        }
        return marker.batchKey;
      })
      .filter(Boolean)
  );

  if (!overviewComment) {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
      body: overviewBody,
    });
    core.info(`Created overview comment for ${weekKey}.`);
  } else {
    core.info(`Overview comment already exists for ${weekKey}; checking for missing task comments.`);
  }

  const missingTasks = tasks.filter(task => !existingTaskBatches.has(task.batch_key));

  for (const task of missingTasks) {
    const createdComment = await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
      body: task.body,
    });

    await waitForEyesReaction({
      github,
      context,
      core,
      commentId: createdComment.data.id,
      pollIntervalMs: CODEX_ACK_POLL_INTERVAL_MS,
      timeoutMs: CODEX_ACK_TIMEOUT_MS,
    });
  }

  if (missingTasks.length === 0) {
    core.info(`No missing task comments for ${weekKey}.`);
  } else {
    core.info(`Posted ${missingTasks.length} missing task comment(s) for ${weekKey}.`);
  }
}

module.exports = { run };
