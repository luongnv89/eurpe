# Convenience targets for the EURPE test pipeline.
#
# Issue #43 acceptance criterion 1: a single command runs the full E2E
# pipeline with no prompts. ``make e2e`` is that command.
#
# The fast / e2e split is enforced by the ``e2e`` pytest marker (see
# pyproject.toml). ``make test`` runs everything *except* e2e tests so
# the inner-loop dev cycle stays fast; ``make test-all`` is the catch-all
# that runs both tiers.

.PHONY: e2e e2e-fixtures-only test test-all

# Run the full E2E suite. Picks up every PDF in tests/e2e/fixtures/ and
# proposals/ (the latter is gitignored — present only on a dev machine).
e2e:
	pytest tests/e2e/ -m e2e -v

# E2E suite restricted to the synthetic fixture PDF — useful on a fresh
# checkout where proposals/ is empty, or in CI where only the synthetic
# PDF is available.
e2e-fixtures-only:
	pytest tests/e2e/ -m e2e -v -k "sample_proposal"

# Fast tests only. Skips the e2e marker AND the docling marker (the
# latter is also slow, since it parses synthetic PDFs through Docling).
test:
	pytest -m 'not e2e and not docling' -q

# All tests (fast + docling integration + e2e). The complete pre-merge gate.
test-all:
	pytest -q
