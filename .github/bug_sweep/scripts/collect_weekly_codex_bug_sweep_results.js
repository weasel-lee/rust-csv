"use strict";

const {
  batchBlocks,
  commentLogin,
  deleteCommentsBestEffort,
  findTrustedOverviewComment,
  firstLine,
  isTrustedCommenter,
  listIssueComments,
  markerInfo,
  parseResultMarker,
} = require("./codex_weekly_bug_sweep_common");

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function run({ github, context, core }) {
  const issueNumber = Number(process.env.ISSUE_NUMBER || "0");
  if (!issueNumber) {
    throw new Error("Set repo variable CODEX_WEEKLY_BUG_SWEEP_ISSUE_NUMBER.");
  }

  const currentIssueNumber = context.payload.issue.number;
  if (currentIssueNumber !== issueNumber) {
    core.info(`Ignoring issue #${currentIssueNumber}; target is #${issueNumber}.`);
    return;
  }

  const incomingComment = context.payload.comment;
  const incomingBody = incomingComment.body || "";
  const resultMarker = parseResultMarker(incomingBody);
  if (!resultMarker) {
    core.info("No CODEX_SWEEP_RESULT marker found at the start of the comment; skipping.");
    return;
  }

  if (!isTrustedCommenter(incomingComment)) {
    const login = commentLogin(incomingComment);
    const association = incomingComment.author_association || "NONE";
    core.warning(
      `Ignoring CODEX_SWEEP_RESULT from untrusted commenter @${login} (association: ${association}).`
    );
    return;
  }

  const weekKey = resultMarker.weekKey;
  const batchKey = resultMarker.batchKey;

  const comments = await listIssueComments({ github, context, issueNumber: currentIssueNumber });
  const overviewComment = findTrustedOverviewComment(comments, weekKey);
  if (!overviewComment) {
    core.warning(`No overview comment found for ${weekKey}; skipping.`);
    return;
  }

  const beginMarker = `<!-- BEGIN batch=${batchKey} -->`;
  const endMarker = `<!-- END batch=${batchKey} -->`;

  let overviewBody = overviewComment.body || "";
  if (!overviewBody.includes(beginMarker) || !overviewBody.includes(endMarker)) {
    core.warning(`Batch ${batchKey} does not exist in overview ${weekKey}; skipping.`);
    return;
  }

  const payload = incomingBody.slice(firstLine(incomingBody).length).replace(/^\n/, "").trim();
  const normalizedPayload = payload || "_(no findings provided)_";
  const sourceLogin = commentLogin(incomingComment);
  const statusLine =
    `_updated: ${incomingComment.created_at} by @${sourceLogin} ` +
    `(source comment id ${incomingComment.id})_`;

  const batchBlockRegex = new RegExp(
    `${escapeRegExp(beginMarker)}[\\s\\S]*?${escapeRegExp(endMarker)}`
  );
  overviewBody = overviewBody.replace(
    batchBlockRegex,
    `${beginMarker}\n${statusLine}\n\n${normalizedPayload}\n${endMarker}`
  );

  const blocks = batchBlocks(overviewBody);
  const totalBatches = blocks.length;
  const completedBatches = blocks.filter(block => block.blockBody.includes("_updated: ")).length;

  const progressLine = `Progress: **${completedBatches}/${totalBatches}**`;
  if (/Progress:\s*\*\*\d+\/\d+\*\*/.test(overviewBody)) {
    overviewBody = overviewBody.replace(/Progress:\s*\*\*\d+\/\d+\*\*/, progressLine);
  } else {
    overviewBody = `${overviewBody.trimEnd()}\n\n${progressLine}\n`;
  }

  await github.rest.issues.updateComment({
    owner: context.repo.owner,
    repo: context.repo.repo,
    comment_id: overviewComment.id,
    body: overviewBody,
  });

  if (totalBatches === 0 || completedBatches !== totalBatches) {
    core.info(`Progress updated: ${completedBatches}/${totalBatches}.`);
    return;
  }

  const refreshedComments = await listIssueComments({ github, context, issueNumber: currentIssueNumber });

  const cleanupTargets = refreshedComments.filter(comment => {
    const marker = markerInfo(comment.body);
    if (!marker || marker.weekKey !== weekKey) {
      return false;
    }

    if (marker.kind === "task") {
      return true;
    }

    return isTrustedCommenter(comment);
  });

  await deleteCommentsBestEffort({ github, context, core, comments: cleanupTargets });

  core.info(`All batches complete for ${weekKey}. Cleaned ${cleanupTargets.length} intermediate comments.`);
}

module.exports = { run };
