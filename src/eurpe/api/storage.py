"""On-disk parse-token store for the two-step ingestion API.

The ``POST /api/ingestion/parse`` route stages an uploaded PDF, runs the
Docling parser, and hands the operator a token plus a metadata draft.
``POST /api/ingestion/confirm`` later quotes that token, fetches the
staged PDF + draft, and runs the chunker.

A simple file-backed store is enough for this prototype because:

* The data is short-lived (default TTL 30 minutes) — no migrations, no
  schemas, no SQLite needed.
* Restarting the FastAPI service should not destroy in-flight ingestions
  the operator already started, so RAM-only state would be too fragile.
* Atomic writes mirror the rest of the codebase (``tmp + os.replace``)
  so a crashed write cannot leave a half-formed JSON record visible.

Each record lives in ``<runtime_dir>/upload_tokens/<token>.json`` and the
PDF stays under ``<runtime_dir>/staging/<token>.pdf`` until the confirm
route runs (or ``cleanup_expired`` reaps it).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default TTL: long enough that an operator can review the form for a
# while without losing the staged PDF, short enough that abandoned uploads
# don't accumulate. Overridable per-store for tests.
DEFAULT_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class ParseTokenRecord:
    """One staged-upload record, returned by :meth:`ParseTokenStore.get`."""

    token: str
    pdf_path: Path
    draft: dict
    parsed_at: datetime
    expires_at: datetime


class ParseTokenStore:
    """File-backed store for in-flight parse tokens.

    All on-disk state lives under ``<runtime_dir>/upload_tokens`` (one
    JSON file per token) and ``<runtime_dir>/staging`` (one PDF per
    token). The constructor creates both directories if missing — safe
    because ``ensure_runtime_dirs`` already guarantees ``runtime_dir``
    itself exists.
    """

    def __init__(self, runtime_dir: Path, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._runtime_dir = runtime_dir
        self._tokens_dir = runtime_dir / "upload_tokens"
        self._staging_dir = runtime_dir / "staging"
        self._tokens_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds

    @property
    def staging_dir(self) -> Path:
        """Directory where caller code may write staged PDFs.

        Exposed because ``POST /api/ingestion/parse`` decides the staged
        path (so it can sanity-check the suffix) and then calls
        :meth:`put` once the file is on disk.
        """

        return self._staging_dir

    def new_token(self) -> str:
        """Return a fresh URL-safe parse-token string.

        Uses :func:`uuid.uuid4` for the visible part (familiar shape, no
        cryptographic claims needed) plus a small random suffix from
        :mod:`secrets` so two stores running side-by-side cannot collide
        on the same UUID by accident.
        """

        return f"{uuid.uuid4().hex}-{secrets.token_hex(4)}"

    def put(
        self,
        *,
        token: str,
        pdf_path: Path,
        draft: dict,
    ) -> ParseTokenRecord:
        """Persist a record for ``token`` and return the immutable view.

        The PDF must already be on disk under :attr:`staging_dir`; this
        method does not move files. ``draft`` is anything JSON-serialisable
        (typically a ``ProposalMetadata.model_dump(mode="json")`` dict).
        """

        now = datetime.now(UTC)
        expires = datetime.fromtimestamp(now.timestamp() + self._ttl_seconds, tz=UTC)
        record = {
            "token": token,
            "pdf_path": str(pdf_path),
            "draft": draft,
            "parsed_at": now.isoformat(),
            "expires_at": expires.isoformat(),
        }
        target = self._record_path(token)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return ParseTokenRecord(
            token=token,
            pdf_path=pdf_path,
            draft=dict(draft),
            parsed_at=now,
            expires_at=expires,
        )

    def get(self, token: str) -> ParseTokenRecord | None:
        """Return the record for ``token`` if it exists AND has not expired.

        Returns ``None`` for a missing token. An expired record is removed
        from disk (best-effort) and ``None`` is returned — the caller is
        expected to surface a 404 / 410 to the user. We keep it simple here
        rather than discriminate the two states because the FastAPI route
        already collapses both into a 404 with a clear message.
        """

        path = self._record_path(token)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Defensive: a corrupted record is treated as missing rather
            # than crashing the request handler. Logged so the operator
            # can spot the issue in stderr.
            logger.warning("ParseTokenStore: dropping corrupted record %s: %s", path, exc)
            self._safe_unlink(path)
            return None
        expires_raw = raw.get("expires_at", "")
        try:
            expires = datetime.fromisoformat(expires_raw)
        except (TypeError, ValueError):
            logger.warning("ParseTokenStore: bad expires_at on %s; dropping", path)
            self._safe_unlink(path)
            return None
        if expires.timestamp() <= time.time():
            pdf_path = raw.get("pdf_path")
            self._safe_unlink(path)
            if isinstance(pdf_path, str):
                self._safe_unlink(Path(pdf_path))
            return None
        try:
            parsed_at = datetime.fromisoformat(raw["parsed_at"])
        except (KeyError, TypeError, ValueError):
            parsed_at = expires  # best-effort fallback; never raise here.
        return ParseTokenRecord(
            token=str(raw.get("token", token)),
            pdf_path=Path(raw["pdf_path"]),
            draft=dict(raw.get("draft", {})),
            parsed_at=parsed_at,
            expires_at=expires,
        )

    def delete(self, token: str) -> None:
        """Drop a token + its staged PDF after a successful confirm.

        Both the JSON record and the PDF file are removed. Failures are
        logged but never raised — leaving stale state behind is better
        than failing the request the operator already finished.
        """

        record_path = self._record_path(token)
        record = self.get(token)
        self._safe_unlink(record_path)
        if record is not None:
            self._safe_unlink(record.pdf_path)

    def cleanup_expired(self) -> int:
        """Remove records past their ``expires_at`` and return the count.

        Useful for tests and for an eventual periodic cleanup task. The
        FastAPI service does not call this on every request — that would
        be a surprising side-effect.
        """

        removed = 0
        if not self._tokens_dir.exists():
            return 0
        for path in self._tokens_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                expires = datetime.fromisoformat(raw.get("expires_at", ""))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                # Corrupted records are reaped too — they are dead weight.
                self._safe_unlink(path)
                removed += 1
                continue
            if expires.timestamp() <= time.time():
                pdf_path = raw.get("pdf_path")
                self._safe_unlink(path)
                if isinstance(pdf_path, str):
                    self._safe_unlink(Path(pdf_path))
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _record_path(self, token: str) -> Path:
        # Token shape is enforced by ``new_token`` (hex + dash + hex), but
        # callers can in theory pass anything. Reject anything that would
        # escape the tokens directory just in case.
        if "/" in token or ".." in token or not token:
            raise ValueError(f"invalid parse token: {token!r}")
        return self._tokens_dir / f"{token}.json"

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - filesystem oddity
            logger.warning("ParseTokenStore: failed to unlink %s: %s", path, exc)
