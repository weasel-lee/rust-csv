"use strict";

const fs = require("fs");
const {
  findEyesReaction,
  isTrustedCommenter,
  markerInfo,
  parseTaskMarker,
  startsWithMarker,
  waitForEyesReaction,
} = require("./codex_weekly_bug_sweep_common");

const CODEX_ACK_POLL_INTERVAL_MS = 5_000;
const CODEX_ACK_TIMEOUT_MS = 2 * 60_000;
const BATCH_BLOCK_REGEX = /<!-- BEGIN batch=([^\s>]+) -->([\s\S]*?)<!-- END batch=\1 -->/g;

function isOverviewComplete(body) {
  const blocks = [...(body || "").matchAll(BATCH_BLOCK_REGEX)];
  return blocks.length > 0 && blocks.every(([, , blockBody]) => blockBody.includes("_updated: "));
}

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
  const trustedComments = comments.filter(comment => isTrustedCommenter(comment));
  const sameWeekTaskComments = trustedComments
    .map(comment => {
      const marker = parseTaskMarker(comment.body);
      return { comment, marker };
    })
    .filter(({ marker }) => marker && marker.weekKey === weekKey);
  let existingTaskBatches = new Set();
  const staleUnacknowledgedTaskComments = [];

  for (const { comment, marker } of sameWeekTaskComments) {
    const eyesReaction = await findEyesReaction({
      github,
      context,
      commentId: comment.id,
    });
    if (eyesReaction) {
      existingTaskBatches.add(marker.batchKey);
    } else {
      staleUnacknowledgedTaskComments.push(comment);
    }
  }

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

    if (isOverviewComplete(overviewComment.body || "")) {
      core.info(`Overview comment for ${weekKey} is complete; resetting it for a fresh rerun.`);
      await github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: overviewComment.id,
        body: overviewBody,
      });

      const staleIntermediateComments = trustedComments.filter(comment => {
        const marker = markerInfo(comment.body);
        return marker && marker.weekKey === weekKey;
      });

      for (const comment of staleIntermediateComments) {
        try {
          await github.rest.issues.deleteComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            comment_id: comment.id,
          });
        } catch (error) {
          core.warning(`Failed to delete stale comment ${comment.id}: ${error.message}`);
        }
      }
      existingTaskBatches = new Set();
    } else if (staleUnacknowledgedTaskComments.length > 0) {
      core.info(
        `Removing ${staleUnacknowledgedTaskComments.length} unacknowledged task comment(s) for ${weekKey}.`
      );
      for (const comment of staleUnacknowledgedTaskComments) {
        try {
          await github.rest.issues.deleteComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            comment_id: comment.id,
          });
        } catch (error) {
          core.warning(`Failed to delete stale task comment ${comment.id}: ${error.message}`);
        }
      }
    }
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
