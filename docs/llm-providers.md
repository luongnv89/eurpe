# LLM Providers And Engines

EURPE defaults to local Ollama plus `offline_mode: true`. Cloud providers and
non-loopback self-hosted endpoints are available only when an operator chooses
them in `config.yaml`, stores secrets in environment variables, and allowlists
the target host:port.

## Supported Runtimes

| Runtime | Type | Default endpoint | Secret env var |
|---|---|---|---|
| `ollama` | local | `models.ollama_base_url` (`http://localhost:11434`) | none |
| `lmstudio` | local OpenAI-compatible | `http://localhost:1234/v1` | none |
| `vllm` | local/self-hosted OpenAI-compatible | `http://localhost:8000/v1` | `VLLM_API_KEY` optional |
| `llamacpp` | local OpenAI-compatible | `http://localhost:8080/v1` | none |
| `openai` | cloud OpenAI-compatible | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | cloud OpenAI-compatible | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `groq` | cloud OpenAI-compatible | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| `anthropic` | cloud | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `gemini` | cloud | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` |

`models.llm_base_url` overrides the default generation endpoint. Use it for
local servers on non-default ports, reverse proxies, or self-hosted gateways.
`models.llm_api_key_env` overrides the environment variable name. Store only the
variable name in config, never the secret value.

## Offline-First Egress Rules

The network policy gate is still enforced for every backend:

- Loopback hosts (`localhost`, `127.0.0.1`, `::1`) are allowed by default.
- Non-loopback hosts are denied unless listed in `network_allowlist`.
- The audit log records host, port, scheme, redacted path, decision, reason, and
  source. It never records prompts, completions, request bodies, or headers.
- Provider API keys are read from environment variables and are not logged.

For example, OpenAI needs both `OPENAI_API_KEY` and an allowlist entry:

```yaml
models:
  runtime: openai
  llm_model: gpt-4o-mini
  embedding_model: nomic-embed-text
  ollama_base_url: http://localhost:11434
  llm_base_url:
  llm_api_key_env:

offline_mode: true
network_allowlist:
  - host: api.openai.com
    port: 443
    reason: "Operator-approved OpenAI generation for this run"
```

Then run with:

```bash
export OPENAI_API_KEY="<your key>"
eurpe generate section ...
```

## Local Engine Examples

LM Studio:

```yaml
models:
  runtime: lmstudio
  llm_model: local-model
  embedding_model: nomic-embed-text
  ollama_base_url: http://localhost:11434
  llm_base_url: http://localhost:1234/v1
```

vLLM:

```yaml
models:
  runtime: vllm
  llm_model: meta-llama/Llama-3.1-8B-Instruct
  embedding_model: nomic-embed-text
  ollama_base_url: http://localhost:11434
  llm_base_url: http://localhost:8000/v1
  llm_api_key_env: VLLM_API_KEY
```

llama.cpp:

```yaml
models:
  runtime: llamacpp
  llm_model: local-model
  embedding_model: nomic-embed-text
  ollama_base_url: http://localhost:11434
  llm_base_url: http://localhost:8080/v1
```
