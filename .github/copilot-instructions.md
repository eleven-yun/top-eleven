# top-eleven Agent Execution Guidelines

## Default Mode: Autonomous Delivery
- Execute tasks end-to-end without waiting for confirmation between intermediate steps.
- When a goal is defined in a plan markdown file, treat unchecked items as executable tasks and continue until all are complete or blocked.
- Prefer taking action over proposing action.

## Planning and Progress
- Create a short execution plan at the start of substantial work.
- Update progress frequently while running commands, editing files, validating, and pushing changes.
- Avoid stopping after analysis when implementation is feasible.

## Stop Conditions (Pause Only Here)
- Required secret input (passwords, tokens, keys) that cannot be inferred.
- Destructive operations outside normal workflow, such as deleting large data sets or irreversible history rewrites not already requested.
- True blocker after reasonable self-recovery attempts, with a concise blocker summary and next best options.

## Validation Requirements
- Run targeted syntax checks and relevant tests after code changes.
- If tests fail, attempt fixes immediately and re-run validation.
- Report what was validated and what was not run.

## Branch and Delivery Behavior
- Work on the intended branch for the active stream of work.
- Keep commits focused and task-scoped.
- If the user asked to proceed autonomously, continue commit-and-push cycles until the current goal is reached.

## Communication Style
- Keep progress updates brief and action-oriented.
- Summarize outcomes, risks, and next actions at each major checkpoint.
