"use strict";

const fs = require("fs");
const {
  completedBatchKeysFromOverview,
  deleteCommentsBestEffort,
  findTrustedOverviewComment,
  findEyesReaction,
  isTrustedCommenter,
  isOverviewComplete,
  listIssueComments,
  markerInfo,
  parseResultMarker,
  parseTaskMarker,
  waitForEyesReaction,
} = require("./codex_weekly_bug_sweep_common");

const CODEX_ACK_POLL_INTERVAL_MS = 5_000;
const CODEX_ACK_TIMEOUT_MS = 2 * 60_000;

function completedBatchKeysFromResultComments(comments, weekKey) {
  const completedBatchKeys = new Set();

  for (const comment of comments) {
    const marker = parseResultMarker(comment.body);
    if (marker && marker.weekKey === weekKey) {
      completedBatchKeys.add(marker.batchKey);
    }
  }

  return completedBatchKeys;
}

function sameWeekTaskComments(comments, weekKey) {
  return comments
    .map(comment => {
      const marker = parseTaskMarker(comment.body);
      return { comment, marker };
    })
    .filter(({ marker }) => marker && marker.weekKey === weekKey);
}

async function taskPostingState({ github, context, trustedComments, overviewComment, weekKey }) {
  const completedBatchKeys = new Set(
    overviewComment ? completedBatchKeysFromOverview(overviewComment.body || "") : []
  );
  for (const batchKey of completedBatchKeysFromResultComments(trustedComments, weekKey)) {
    completedBatchKeys.add(batchKey);
  }

  const existingTaskBatches = new Set(completedBatchKeys);
  const staleUnacknowledgedTaskComments = [];

  for (const { comment, marker } of sameWeekTaskComments(trustedComments, weekKey)) {
    if (completedBatchKeys.has(marker.batchKey)) {
      continue;
    }

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

  return { existingTaskBatches, staleUnacknowledgedTaskComments };
}

async function run({ github, context, core }) {
  const issueNumber = Number(process.env.ISSUE_NUMBER);
  if (!issueNumber) {
    throw new Error("Set repo variable CODEX_WEEKLY_BUG_SWEEP_ISSUE_NUMBER.");
  }

  const weekKey = process.env.WEEK_KEY;
  const comments = await listIssueComments({ github, context, issueNumber });

  const overviewBody = fs.readFileSync("codex-overview-comment.md", "utf8");
  const tasks = JSON.parse(fs.readFileSync("codex-task-comments.json", "utf8"));
  const overviewComment = findTrustedOverviewComment(comments, weekKey);
  const trustedComments = comments.filter(comment => isTrustedCommenter(comment));
  let { existingTaskBatches, staleUnacknowledgedTaskComments } = await taskPostingState({
    github,
    context,
    trustedComments,
    overviewComment,
    weekKey,
  });

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

      await deleteCommentsBestEffort({
        github,
        context,
        core,
        comments: staleIntermediateComments,
        warningPrefix: "Failed to delete stale comment",
      });
      existingTaskBatches = new Set();
    } else if (staleUnacknowledgedTaskComments.length > 0) {
      core.info(
        `Removing ${staleUnacknowledgedTaskComments.length} unacknowledged task comment(s) for ${weekKey}.`
      );
      await deleteCommentsBestEffort({
        github,
        context,
        core,
        comments: staleUnacknowledgedTaskComments,
        warningPrefix: "Failed to delete stale task comment",
      });
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
