"""HTTP routes for the Settings UI (issue #74).

Two endpoints back the Settings page:

* ``GET /api/config`` — return the effective configuration as JSON.
  Secret values (API keys) are never returned; the UI shows the env var
  name and a masked input for the actual secret.
* ``PUT /api/config`` — merge partial config updates into the active
  ``config.yaml``. The server reads the current file, applies the
  supplied fields, validates via Pydantic, and persists.

The YAML file is the source of truth; edits made directly to
``config.yaml`` are reflected on the next ``GET /api/config`` call.
"""

from __future__ import annotations

import logging

import yaml
from fastapi import APIRouter, Depends, HTTPException

from eurpe.api import dependencies as deps
from eurpe.api.dependencies import get_config, reset_dependency_caches
from eurpe.api.schemas import (
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ModelsConfigResponse,
    ModelsConfigUpdate,
    NetworkAllowlistEntry,
)
from eurpe.config import EXAMPLE_CONFIG_PATH, EurpeConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["settings"])


def _config_to_response(cfg: EurpeConfig) -> ConfigResponse:
    """Convert an internal ``EurpeConfig`` to the API response shape."""

    return ConfigResponse(
        corpus_path=str(cfg.corpus_path),
        index_path=str(cfg.index_path),
        runtime_dir=str(cfg.runtime_dir),
        offline_mode=cfg.offline_mode,
        log_level=cfg.log_level,
        models=ModelsConfigResponse(
            runtime=cfg.models.runtime,
            llm_model=cfg.models.llm_model,
            embedding_model=cfg.models.embedding_model,
            ollama_base_url=cfg.models.ollama_base_url,
            llm_base_url=cfg.models.llm_base_url,
            llm_api_key_env=cfg.models.llm_api_key_env,
        ),
        network_allowlist=[
            NetworkAllowlistEntry(
                host=e.host,
                port=e.port,
                reason=e.reason,
            )
            for e in cfg.network_allowlist
        ],
    )


@router.get("", response_model=ConfigResponse)
def get_config_endpoint(cfg: EurpeConfig = Depends(get_config)) -> ConfigResponse:
    """Return the effective configuration (secrets never returned)."""

    return _config_to_response(cfg)


@router.put("", response_model=ConfigUpdateResponse)
def update_config_endpoint(body: ConfigUpdateRequest) -> ConfigUpdateResponse:
    """Merge partial config updates into ``config.yaml``.

    Only supplied fields are changed; the rest of the file is preserved.
    After validation the file is atomically rewritten so a crash during
    write does not corrupt the config.
    """

    config_path = deps._CONFIG_PATH
    if not config_path.exists():
        if not EXAMPLE_CONFIG_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="No configuration file exists and no example template is available.",
            )
        EXAMPLE_CONFIG_PATH.copy(config_path)

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    changes: dict[str, object] = {}

    if body.corpus_path is not None:
        changes["corpus_path"] = body.corpus_path
    if body.index_path is not None:
        changes["index_path"] = body.index_path
    if body.runtime_dir is not None:
        changes["runtime_dir"] = body.runtime_dir
    if body.offline_mode is not None:
        changes["offline_mode"] = body.offline_mode
    if body.log_level is not None:
        changes["log_level"] = body.log_level.upper()
    if body.network_allowlist is not None:
        changes["network_allowlist"] = [
            {"host": e.host, "port": e.port, "reason": e.reason} for e in body.network_allowlist
        ]
    if body.models is not None:
        models_patch: dict[str, object] = {}
        m: ModelsConfigUpdate = body.models
        if m.runtime is not None:
            models_patch["runtime"] = m.runtime
        if m.llm_model is not None:
            models_patch["llm_model"] = m.llm_model
        if m.embedding_model is not None:
            models_patch["embedding_model"] = m.embedding_model
        if m.ollama_base_url is not None:
            models_patch["ollama_base_url"] = m.ollama_base_url
        if m.llm_base_url is not None:
            models_patch["llm_base_url"] = m.llm_base_url
        if m.llm_api_key_env is not None:
            models_patch["llm_api_key_env"] = m.llm_api_key_env
        if models_patch:
            existing_models: dict = raw.get("models", {}) or {}
            existing_models.update(models_patch)
            changes["models"] = existing_models

    if not changes:
        raise HTTPException(
            status_code=400,
            detail="At least one configuration field must be supplied.",
        )

    for key, value in changes.items():
        raw[key] = value

    try:
        EurpeConfig.model_validate(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration validation failed: {exc}",
        ) from exc

    tmp_path = config_path.with_suffix(".yaml.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# EURPE local configuration — managed via Settings UI and config.yaml.\n"
            "# `eurpe smoke` will create config.yaml from the example if it does not exist.\n\n"
        )
        yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)
    tmp_path.replace(config_path)

    # Clear the dependency cache so the next GET returns the updated config.
    reset_dependency_caches()

    logger.info("Configuration updated via Settings UI: %s", list(changes.keys()))

    updated = EurpeConfig.model_validate(raw)
    return ConfigUpdateResponse(
        ok=True,
        config=_config_to_response(updated),
    )
