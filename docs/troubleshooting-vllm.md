# vLLM Troubleshooting

---

## `AttributeError: TokenizerInfo has no attribute 'from_huggingface'`

**Date resolved:** 2026-06-19

### Symptom

Server crashes on the first request that uses JSON mode. The error appears in
the vLLM log and the client receives HTTP 500:

```
AttributeError: Error in model execution: type object 'TokenizerInfo' has no attribute 'from_huggingface'
CRITICAL launcher.py: MQLLMEngine is already dead, terminating server process
```

### Trigger

Any request that enables guided decoding:

```json
{ "response_format": { "type": "json_object" } }
```

or

```json
{ "response_format": { "type": "json_schema", "json_schema": { ... } } }
```

vLLM routes these through `xgrammar`. If the installed version is too old the
call fails and the engine process dies.

### Root cause

vLLM introduced a call to `xgrammar.TokenizerInfo.from_huggingface()` that only
exists in **xgrammar ≥ 0.2.2**. The `ray-env` conda environment had 0.2.1
installed.

### Fix

Upgrade xgrammar, then restart the vLLM server:

```bash
pip install --upgrade xgrammar
# verify
pip show xgrammar | grep Version   # should be 0.2.2 or higher
```

Restart the server normally — no other flags needed.

### Workarounds (if upgrade is not possible)

**Option A — disable guided decoding on the server side:**

```bash
vllm serve <model-path> --guided-decoding-backend disable ...
```

Clients can still send requests but JSON mode will be ignored (no constrained
output).

**Option B — remove `response_format` from the client request** so xgrammar is
never invoked.

### Version matrix

| xgrammar | Status |
|----------|--------|
| 0.2.1    | Crashes on JSON-mode requests |
| 0.2.2    | Fixed |
