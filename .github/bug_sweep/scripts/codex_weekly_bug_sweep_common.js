"use strict";

const TRUSTED_ASSOCIATIONS = new Set(["OWNER", "MEMBER", "COLLABORATOR", "BOT"]);
const TASK_MARKER_REGEX =
  /^<!--\s*CODEX_SWEEP_TASK\s+week=([0-9]{4}-W[0-9]{2})\s+batch=([^\s>]+)\s*-->$/i;
const RESULT_MARKER_REGEX =
  /^<!--\s*CODEX_SWEEP_RESULT\s+week=([0-9]{4}-W[0-9]{2})\s+batch=([^\s>]+)\s*-->$/i;
const BATCH_BLOCK_REGEX = /<!-- BEGIN batch=([^\s>]+) -->([\s\S]*?)<!-- END batch=\1 -->/g;

function normalizeBody(body) {
  return (body || "").replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
}

function firstLine(body) {
  return normalizeBody(body).split("\n", 1)[0] || "";
}

function firstTaskMarkerLine(body) {
  const lines = normalizeBody(body).split("\n");
  const first = (lines[0] || "").trim();
  if (first === "@codex") {
    return (lines[1] || "").trim();
  }
  return first;
}

function startsWithMarker(body, marker) {
  const normalized = normalizeBody(body);
  return normalized === marker || normalized.startsWith(`${marker}\n`);
}

function overviewMarker(weekKey) {
  return `<!-- CODEX_SWEEP_AGG week=${weekKey} -->`;
}

function parseTaskMarker(body) {
  const match = firstTaskMarkerLine(body).match(TASK_MARKER_REGEX);
  if (!match) {
    return null;
  }

  return { kind: "task", weekKey: match[1], batchKey: match[2] };
}

function parseResultMarker(body) {
  const match = firstLine(body).match(RESULT_MARKER_REGEX);
  if (!match) {
    return null;
  }

  return { kind: "result", weekKey: match[1], batchKey: match[2] };
}

function markerInfo(body) {
  return parseTaskMarker(body) || parseResultMarker(body);
}

function isTrustedCommenter(comment) {
  const association = comment.author_association || "";
  const isBot = comment.user && comment.user.type === "Bot";
  return isBot || TRUSTED_ASSOCIATIONS.has(association);
}

function commentLogin(comment) {
  return comment.user && comment.user.login ? comment.user.login : "unknown";
}

function batchBlocks(body) {
  return [...normalizeBody(body).matchAll(BATCH_BLOCK_REGEX)].map(([, batchKey, blockBody]) => ({
    batchKey,
    blockBody,
  }));
}

function completedBatchKeysFromOverview(body) {
  return new Set(
    batchBlocks(body)
      .filter(block => block.blockBody.includes("_updated: "))
      .map(block => block.batchKey)
  );
}

function isOverviewComplete(body) {
  const blocks = batchBlocks(body);
  return blocks.length > 0 && blocks.every(block => block.blockBody.includes("_updated: "));
}

async function listIssueComments({ github, context, issueNumber }) {
  return github.paginate(
    github.rest.issues.listComments,
    {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
      per_page: 100,
    }
  );
}

function findTrustedOverviewComment(comments, weekKey) {
  const marker = overviewMarker(weekKey);
  return comments.find(comment => isTrustedCommenter(comment) && startsWithMarker(comment.body, marker));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function deleteCommentsBestEffort({
  github,
  context,
  core,
  comments,
  warningPrefix = "Failed to delete comment",
}) {
  for (const comment of comments) {
    try {
      await github.rest.issues.deleteComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: comment.id,
      });
    } catch (error) {
      core.warning(`${warningPrefix} ${comment.id}: ${error.message}`);
    }
  }
}

async function findEyesReaction({ github, context, commentId }) {
  const reactions = await github.paginate(
    github.rest.reactions.listForIssueComment,
    {
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: commentId,
      per_page: 100,
    }
  );

  return reactions.find(reaction => reaction.content === "eyes") || null;
}

async function waitForEyesReaction({
  github,
  context,
  core,
  commentId,
  pollIntervalMs,
  timeoutMs,
}) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const eyesReaction = await findEyesReaction({ github, context, commentId });
    if (eyesReaction) {
      const reactor =
        eyesReaction.user && eyesReaction.user.login ? ` by @${eyesReaction.user.login}` : "";
      core.info(`Observed :eyes: reaction${reactor} on comment ${commentId}.`);
      return;
    }

    core.info(
      `No :eyes: reaction yet on comment ${commentId}; polling again in ${pollIntervalMs}ms.`
    );
    await sleep(pollIntervalMs);
  }

  throw new Error(
    `Timed out waiting ${timeoutMs}ms for an :eyes: reaction on task comment ${commentId}.`
  );
}

module.exports = {
  BATCH_BLOCK_REGEX,
  RESULT_MARKER_REGEX,
  TASK_MARKER_REGEX,
  batchBlocks,
  commentLogin,
  completedBatchKeysFromOverview,
  deleteCommentsBestEffort,
  findTrustedOverviewComment,
  findEyesReaction,
  firstLine,
  isTrustedCommenter,
  isOverviewComplete,
  listIssueComments,
  markerInfo,
  normalizeBody,
  overviewMarker,
  parseResultMarker,
  parseTaskMarker,
  startsWithMarker,
  waitForEyesReaction,
};
