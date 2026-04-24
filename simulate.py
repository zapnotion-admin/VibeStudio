"""
engine/ollama_client.py
All communication with Ollama's REST API.
No subprocess. No Aider. Pure HTTP with streaming.

Rules enforced here:
- ensure_model() is defined ONCE and ONLY here. Do not import it from anywhere else.
- is_ollama_running() uses /api/tags — never the root / endpoint.
- resolve_model() guards every pipeline call against missing custom models.
- safe_generate() wraps every POST with a timeout + single retry.
- stream_response() skips malformed JSON lines — never crashes on bad chunks.
"""

import json
import requests
from core.config import OLLAMA_URL, MAX_CTX_CODER, OLLAMA_NUM_THREAD
from engine.logger import log

# ---------------------------------------------------------------------------
# [v3.2] Module-level warmup cache.
# Prevents redundant /api/generate warmup calls when the model is already
# loaded. Cleared to None on process start — each fresh session will warm
# the model exactly once, then skip on subsequent calls.
# ---------------------------------------------------------------------------
_last_warmed_model: str | None = None


def ensure_model(model_name: str) -> None:
    """
    Warms up the target model, forcing any prior model to unload first.

    [v3.2] Skips the API call entirely if model_name matches _last_warmed_model.
    This prevents VRAM churn and latency spikes on rapid sends.

    MUST be called from inside a QThread worker — NEVER from the UI thread.
    Failure is logged but non-fatal: the real call will surface the error.
    """
    global _last_warmed_model
    if _last_warmed_model == model_name:
        log(f"[ensure_model] Already warmed: {model_name} — skipping")
        return
    try:
        log(f"[ensure_model] Warming up: {model_name}")
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": "", "stream": False},
            timeout=60,
        )
        _last_warmed_model = model_name
        log(f"[ensure_model] Ready: {model_name}")
    except Exception as e:
        log(f"[ensure_model] Warning: {e}")
        # Non-fatal — the main generation call will surface the real error


def resolve_model(preferred: str, fallback: str) -> str:
    """
    Returns preferred model if it exists locally, otherwise fallback.

    Prevents first-run silent failures when ollama create did not succeed
    (e.g. model pull failed, Modelfile had a syntax error, etc).
    Applied before every stream_response() or single_response() call.
    """
    models = list_local_models()
    if preferred in models:
        return preferred
    log(f"[resolve_model] '{preferred}' not found locally — falling back to '{fallback}'")
    return fallback


def safe_generate(url: str, payload: dict, stream: bool = False) -> requests.Response:
    """
    Wraps requests.post with a 120-second timeout and a single retry on Timeout.
    Returns the Response object or raises on the second failure.
    All other exceptions (ConnectionError, etc.) propagate immediately.
    """
    try:
        return requests.post(url, json=payload, stream=stream, timeout=120)
    except requests.exceptions.Timeout:
        log("[safe_generate] Timeout on first attempt — retrying once")
        return requests.post(url, json=payload, stream=stream, timeout=120)


def stream_response(
    model: str,
    prompt: str,
    system: str = "",
    num_ctx: int = MAX_CTX_CODER,
):
    """
    Generator — yields decoded text chunks as they arrive from Ollama.

    Malformed NDJSON lines are silently skipped so a bad chunk never
    crashes the UI. The caller (OllamaStreamWorker) owns cancellation:
    it holds a reference to the live response object and calls
    response.close() to terminate the HTTP stream immediately.

    Usage (in OllamaStreamWorker.run — NOT called directly from UI thread):
        self._response = safe_generate(url, payload, stream=True)
        for line in self._response.iter_lines():
            if self._cancelled: break
            ...
    stream_response() is kept for use in tests and non-UI contexts.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "temperature": 0.6,
            "num_thread": OLLAMA_NUM_THREAD,   # [perf] cap CPU threads
        },
    }
    log(f"[stream_response] model={model} ctx={num_ctx} prompt_len={len(prompt)}")
    resp = safe_generate(f"{OLLAMA_URL}/api/generate", payload, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line)
            except Exception:
                continue  # Skip malformed JSON — never crash on bad chunk
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break


def single_response(
    model: str,
    prompt: str,
    system: str = "",
    num_ctx: int = MAX_CTX_CODER,
) -> str:
    """
    Blocking (non-streaming) call. Returns the complete response string.
    Used for pipeline stages (Plan, Code, Review, Retry) where the full
    output is needed before proceeding to the next stage.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_thread": OLLAMA_NUM_THREAD,   # [perf] cap CPU threads
        },
    }
    # Warn if prompt is suspiciously long relative to context window
    # ~4 chars per token estimate. If input > 70% of ctx, output space is tiny.
    estimated_input_tokens = len(prompt) // 4
    if estimated_input_tokens > int(num_ctx * 0.70):
        log(f"[single_response] WARNING: prompt ~{estimated_input_tokens} tokens vs ctx {num_ctx} "
            f"— output budget only ~{num_ctx - estimated_input_tokens} tokens, may stall")

    log(f"[single_response] model={model} ctx={num_ctx} prompt_len={len(prompt)} "
        f"(~{estimated_input_tokens} tokens)")
    resp = safe_generate(f"{OLLAMA_URL}/api/generate", payload, stream=False)
    resp.raise_for_status()
    result = resp.json().get("response", "")
    if not result.strip():
        log(f"[single_response] WARNING: empty response from {model} — possible stall")
    return result


def list_local_models() -> list[str]:
    """
    Returns the names of all models currently available in local Ollama.
    Returns an empty list if Ollama is unreachable — never raises.
    Used by resolve_model() and the sidebar model selector.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def unload_model() -> None:
    """
    Forces Ollama to evict the currently loaded model from VRAM immediately.
    Called after every stream or pipeline completes.
    On a 16GB GPU this is critical — without it the model sits in VRAM
    indefinitely, starving the OS, browser, and compositor.
    Non-fatal: logs failure but never raises.
    """
    try:
        # Setting keep_alive=0 on an empty prompt triggers immediate unload
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": _last_warmed_model or "", "keep_alive": 0},
            timeout=5,
        )
        log("[unload_model] VRAM released")
    except Exception as e:
        log(f"[unload_model] Warning (non-fatal): {e}")


def is_ollama_running() -> bool:
    """
    Returns True if Ollama is reachable and responding.

    [v3.1] Uses /api/tags exclusively — stable across all Ollama versions.
    The root / endpoint returns inconsistent status codes across versions
    and should never be used for health checks.
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False
