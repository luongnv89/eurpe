#!/usr/bin/env bash
# CLI end-to-end smoke for the `eurpe` Typer app.
#
# Exercises every top-level command (and one representative subcommand each
# for nested Typer groups) with `--help` plus a real invocation where it
# can run fully offline. The goal is to catch import errors, broken Typer
# wiring, and trivial regressions before push.
#
# Wired into .pre-commit-config.yaml on the pre-push stage.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x .venv/bin/eurpe ]]; then
  EURPE=".venv/bin/eurpe"
else
  EURPE="eurpe"
fi

pass=0
fail=0

run() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  √ ${label}"
    pass=$((pass + 1))
  else
    echo "  × ${label} — command failed: $*"
    fail=$((fail + 1))
  fi
}

echo "◆ eurpe CLI E2E smoke"
echo "··································································"

# Top-level
run "eurpe --help"                $EURPE --help
run "eurpe version"               $EURPE version

# Sub-typers — exercise help so Typer can introspect every group.
for sub in ingest index generate analytics benchmark pilot smoke; do
  run "eurpe ${sub} --help"       $EURPE "$sub" --help
done

# Smoke is itself an offline setup check (see eurpe.cli:smoke docstring).
run "eurpe smoke"                 $EURPE smoke

echo "  ____________________________"
echo "  Result: ${pass} passed, ${fail} failed"

exit $fail
