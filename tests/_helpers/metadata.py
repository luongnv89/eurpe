"""Synthesise / load proposal-metadata YAML for tests.

Two responsibilities:

* :func:`synthesise_proposal_metadata` — produce a minimal but valid
  :class:`ProposalMetadata` dict for a PDF that does *not* have a
  sibling YAML. Defaults are intentionally conservative so a brand-new
  PDF dropped into ``proposals/`` works end-to-end with no extra
  configuration.
* :func:`load_or_synthesise_metadata` — prefer a sibling
  ``<stem>.yml`` / ``<stem>.yaml`` when present (real ProposalMetadata
  YAML maintained by a coordinator), and fall back to synthesised
  defaults otherwise. This is the helper the E2E pipeline uses.

The companion :func:`write_metadata_yaml` writes a YAML sidecar to disk
for a single PDF; the E2E pipeline calls it once per proposal into the
per-run output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Default values applied when no sibling YAML exists. Chosen to be the
# safest possible: ``unspecified-call``-ish call_id, ``funded`` outcome
# (because the retrieval policy treats funded as the primary positive
# signal — synthesised PDFs are exercising the happy path, not the
# rejected-handling path), and the current ``proposal_title`` set to
# the PDF stem so the citation table reads correctly.
_SYNTHESISED_DEFAULTS: dict[str, Any] = {
    "programme": "horizon_europe",
    "call_id": "UNSPECIFIED-CALL",
    "year": 2024,
    "outcome": "funded",
    "language": "en",
}


def synthesise_proposal_metadata(pdf_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a ProposalMetadata dict for a PDF without a sibling YAML.

    Returns a plain dict (not a Pydantic model) so callers that just
    want to write YAML do not pay model-validation cost twice. The
    ``index build`` CLI re-validates via ``ProposalMetadata.model_validate``
    so any error in the synthesised values surfaces with a clean message.

    ``overrides`` win over the defaults but never over the two
    PDF-derived fields (``source_path``, ``proposal_title``); a caller
    that needs to override those should pass them through
    ``write_metadata_yaml`` directly.
    """

    body: dict[str, Any] = dict(_SYNTHESISED_DEFAULTS)
    body.update(overrides)
    body["proposal_title"] = pdf_path.stem
    body["source_path"] = str(pdf_path)
    return body


def load_or_synthesise_metadata(pdf_path: Path) -> dict[str, Any]:
    """Return ProposalMetadata as a dict, preferring a sibling YAML.

    Looks for ``<stem>.yml`` then ``<stem>.yaml`` next to ``pdf_path``.
    If neither exists, falls back to :func:`synthesise_proposal_metadata`.
    When a sibling exists, ``source_path`` is forcibly rewritten to the
    absolute path of ``pdf_path`` so callers can copy / re-stage the YAML
    without breaking the parser's ability to find the PDF.
    """

    for suffix in (".yml", ".yaml"):
        sibling = pdf_path.with_suffix(suffix)
        if sibling.exists():
            raw = yaml.safe_load(sibling.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                # The build CLI would error on this; surface it now with
                # a helpful message instead of a deep-stack Pydantic crash.
                raise ValueError(
                    f"{sibling}: expected a YAML mapping at the document root"
                )
            # Force ``source_path`` to the absolute PDF path. The sibling
            # may carry a relative path that only resolves from the
            # original directory; the E2E pipeline stages the YAML into
            # a different directory.
            raw["source_path"] = str(pdf_path)
            return raw

    return synthesise_proposal_metadata(pdf_path)


def write_metadata_yaml(
    yaml_path: Path,
    pdf_path: Path,
    **overrides: Any,
) -> None:
    """Write a ProposalMetadata YAML sidecar for ``pdf_path`` to ``yaml_path``.

    Prefers a sibling YAML next to ``pdf_path`` if one exists (real
    metadata maintained by a coordinator); otherwise synthesises one.
    ``overrides`` are applied on top of either source so a caller can
    pin specific fields for a test.
    """

    body = load_or_synthesise_metadata(pdf_path)
    body.update(overrides)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(body), encoding="utf-8")
