# vLLM Troubleshooting

---

## `AttributeError: TokenizerInfo has no attribute 'from_huggingface'`

**Date resolved:** 2026-06-19

### Symptom

Server crashes on any request that uses JSON mode. The engine process dies and
all subsequent requests also get HTTP 500 until the server is manually restarted:

```
AttributeError: Error in model execution: type object 'TokenizerInfo' has no attribute 'from_huggingface'
CRITICAL launcher.py: MQLLMEngine is already dead, terminating server process
```

### Why the server dies instead of returning a clean error

This is a vLLM bug. The exception occurs inside the `MQLLMEngine` worker
subprocess where it is not caught properly, killing the entire engine process.
The HTTP server wrapper then has nothing to talk to and shuts itself down. There
is no way to work around this on the server side short of fixing the underlying
xgrammar issue.

### Trigger

Any request that enables guided decoding — both of these forms crash the server:

```json
{ "response_format": { "type": "json_object" } }
```

```json
{ "response_format": { "type": "json_schema", "json_schema": { ... } } }
```

vLLM routes both through `xgrammar`. Multiple clients sending either form will
all crash the server independently.

### Root cause

vLLM calls `xgrammar.TokenizerInfo.from_huggingface()` which does not exist in
xgrammar 0.2.1 or 0.2.2. Upgrading xgrammar alone is not sufficient — the
method is missing in both versions tested.

### Fix (confirmed working — 2026-06-19)

Switch the guided decoding backend to `outlines`, which bypasses xgrammar
entirely:

```bash
pip install outlines

vllm serve ~/projects/vllm-deployment/vllm/models/3.1-8b-instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --guided-decoding-backend outlines
```

**Impact on output quality: none.** Both backends enforce the same JSON schema
constraint — the model's content and reasoning are unchanged. `outlines` uses a
different internal algorithm (finite automaton vs grammar-based) but produces
identical structured output. The only difference is `outlines` is slightly slower
to compile the grammar on the very first request; subsequent requests use a cache.

### Why not just remove `response_format` on the client side

Clients using `json_object` or `json_schema` mode span multiple machines and
adapters. Chasing each one is fragile. The server-side fix covers all clients at
once with no client code changes required.

### Version matrix

| xgrammar | `from_huggingface` present | Status |
|----------|---------------------------|--------|
| 0.2.1    | No                        | Crashes on all JSON-mode requests |
| 0.2.2    | No                        | Still crashes — upgrade is not sufficient |
| —        | —                         | Use `--guided-decoding-backend outlines` instead |
