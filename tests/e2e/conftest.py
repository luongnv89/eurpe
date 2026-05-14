"""Per-suite conftest for the E2E pipeline tests.

Three responsibilities:

1. Override the repo-wide ``no_network`` fixture with a no-op. The
   parent conftest (``tests/conftest.py``) monkeypatches
   ``socket.socket.connect`` to raise, which would also block Chroma's
   local sqlite-over-socket access — fatal for the E2E suite that
   talks to a real local index.

2. Expose :func:`discover_proposals` which returns the merged + sorted
   + de-duplicated list of PDFs to test against. The merge sources are
   ``tests/e2e/fixtures/*.pdf`` and ``proposals/*.pdf``. ``proposals/``
   is in ``.gitignore`` so the merge result is small on CI (just the
   synthetic fixture) and complete on a developer machine.

3. Generate the synthetic fixture PDF at module-load time if it is
   missing and ``reportlab`` is importable. This is what makes CI work
   even without any committed PDFs: a parametrise-time call to
   :func:`discover_proposals` always finds at least the synthetic one.
   If ``reportlab`` is unavailable, the function still returns the
   real ``proposals/*.pdf`` list (which may be empty on CI), and the
   E2E test module handles the empty case with ``pytest.skip``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repo-rooted paths so the suite works regardless of cwd.
_TESTS_E2E_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _TESTS_E2E_DIR / "fixtures"
_REPO_ROOT = _TESTS_E2E_DIR.parent.parent
_PROPOSALS_DIR = _REPO_ROOT / "proposals"

_SYNTHETIC_PDF_NAME = "sample_proposal.pdf"
_SYNTHETIC_PDF_PATH = _FIXTURES_DIR / _SYNTHETIC_PDF_NAME


@pytest.fixture
def no_network() -> None:
    """No-op override of the repo-wide ``no_network`` fixture.

    The parent ``tests/conftest.py`` monkeypatches ``socket.socket.connect``
    to a function that calls ``pytest.fail``. That blanket block also
    catches Chroma's local sqlite-over-socket access, which the E2E
    pipeline relies on. Overriding to a no-op in this sub-package
    makes the E2E tests immune to that blanket block; the offline
    contract is still enforced by the unreachable Ollama URL pinned
    in :func:`tests._helpers.offline.write_offline_config`.
    """

    return None


def _build_synthetic_pdf(path: Path) -> None:
    """Generate a small but proposal-shaped PDF via :mod:`reportlab`.

    Adapted from ``test_docling_parser._build_synthetic_pdf``. The PDF
    is intentionally rich enough that the hierarchical chunker produces
    multiple chunks (a single short paragraph would yield exactly one
    chunk, which is enough for the round-trip but a less interesting
    smoke test).
    """

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=LETTER)

    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, 720, "E2E Synthetic Proposal")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 670, "1. Excellence")
    c.setFont("Helvetica", 12)
    c.drawString(72, 640, "We propose a federated learning architecture for")
    c.drawString(72, 622, "secure cross-organisational threat intelligence sharing.")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 580, "2. Methodology")
    c.setFont("Helvetica", 12)
    c.drawString(72, 550, "Our methodology combines differential privacy with")
    c.drawString(72, 532, "homomorphic encryption to protect partner data while")
    c.drawString(72, 514, "still enabling joint model training across the consortium.")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 470, "3. Impact")
    c.setFont("Helvetica", 12)
    c.drawString(72, 440, "The expected impact spans operational, scientific, and")
    c.drawString(72, 422, "policy dimensions across the European cybersecurity")
    c.drawString(72, 404, "ecosystem.")

    c.showPage()
    c.save()


def _ensure_synthetic_pdf() -> Path | None:
    """Create the synthetic fixture PDF on demand. Return its path or None.

    Returns ``None`` if ``reportlab`` is unavailable so we cannot
    generate the PDF. That branch is benign — :func:`discover_proposals`
    just falls back to whatever PDFs sit in ``proposals/``.
    """

    if _SYNTHETIC_PDF_PATH.exists():
        return _SYNTHETIC_PDF_PATH
    try:
        import reportlab  # noqa: F401 — availability probe
    except ImportError:
        return None
    _build_synthetic_pdf(_SYNTHETIC_PDF_PATH)
    return _SYNTHETIC_PDF_PATH


def discover_proposals() -> list[Path]:
    """Return the sorted, de-duplicated list of PDFs to E2E-test.

    Merges:

    * ``tests/e2e/fixtures/*.pdf`` (the synthetic fixture lives here;
      generated lazily if missing and ``reportlab`` is importable).
    * ``proposals/*.pdf`` (developer-only — ``.gitignore`` excludes
      this directory from version control).

    Called at module-load time by :mod:`tests.e2e.test_full_pipeline`
    so ``@pytest.mark.parametrize`` sees a populated list, then again
    if needed to keep the contract single-sourced.
    """

    _ensure_synthetic_pdf()

    pdfs: set[Path] = set()
    if _FIXTURES_DIR.exists():
        pdfs.update(p.resolve() for p in _FIXTURES_DIR.glob("*.pdf"))
    if _PROPOSALS_DIR.exists():
        pdfs.update(p.resolve() for p in _PROPOSALS_DIR.glob("*.pdf"))

    # Sort by absolute path for deterministic ordering — pytest -k
    # filters and CI log diffs both benefit.
    return sorted(pdfs)
