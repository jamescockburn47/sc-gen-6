# Zero Character Response - Root Cause Analysis & Fix Plan

## Problem Statement
Queries sometimes return **0 characters** - the LLM generates text successfully (1700+ tokens) but the final response is empty.

## Diagnostic Evidence from Logs

### Sequence of Events:
1. ✅ Batches complete successfully:
   ```
   [BATCH DEBUG] Batch 1 result: 2467 chars, 616 tokens
   [BATCH DEBUG] Batch 2 result: 1238 chars, 309 tokens
   ```

2. ✅ Synthesis starts with good input:
   ```
   [LLM STATS] Prompt: 135ms | Generation: 25675ms | 1701 tokens @ 66.3 t/s
   [CHANNEL DEBUG] Input text length: 3512 chars  <-- GOOD
   ```

3. ❌ Then suddenly drops to ZERO:
   ```
   [CHANNEL DEBUG] Input text length: 0 chars  <-- PROBLEM!
   ```

## Root Causes Identified

### Primary Issue: `_post_process_output()` Returning Empty String

**Location**: `src/generation/llm_service.py` lines 505-556

**Problem**: The `_extract_llama_final_channel()` method processes channel markers:
```python
def _extract_llama_final_channel(text: str) -> str:
    # Searches for <|channel|>final<|message|> markers
    # Falls back to combining all channels
    # Returns raw text if no channels found
```

**Failure Mode**: When the gpt-oss-20b model generates text WITHOUT channel markers (which is normal), the code returns the raw text. HOWEVER, somewhere in the chain this gets lost or overwritten.

### Secondary Issue: Missing UI Method

**Location**: `src/ui/modern_main_window.py` line 883

**Error**:
```python
AttributeError: 'EnhancedOutputPanel' object has no attribute 'set_full_synthesis'
```

This is a **UI bug** - the method doesn't exist on `EnhancedOutputPanel`.

## Suspected Code Flow Problem

Looking at `chunk_batcher.py` lines 504-512:

```python
synthesized = self.base_service.generate(
    prompt=user_prompt,
    system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    model=model_name,
    temperature=0.15,
    cancel_event=cancel_event,
    max_tokens=synthesis_limit,  # <-- LIMITED TO 2048 or synthesis_max_tokens
)
return synthesized or ""  # <-- If empty, returns ""
```

**Hypothesis**:
1. Synthesis generation completes successfully (1701 tokens generated)
2. `_post_process_output()` is called
3. For gpt-oss-20b (not llama.cpp channel format), it should return raw text
4. BUT something causes `synthesized` to be empty or None

## Fix Plan (DO NOT IMPLEMENT YET)

### Fix 1: Add Defensive Logging in Synthesis Chain

**File**: `src/generation/chunk_batcher.py` lines 500-515

Add detailed logging:
```python
try:
    synthesis_limit = max_tokens if max_tokens is not None else getattr(self.settings.generation, "synthesis_max_tokens", 2048)
    
    print(f"[SYNTHESIS DEBUG] Starting synthesis with {len(combined_notes)} chars, max_tokens={synthesis_limit}")
    
    synthesized = self.base_service.generate(
        prompt=user_prompt,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        model=model_name,
        temperature=0.15,
        cancel_event=cancel_event,
        max_tokens=synthesis_limit,
    )
    
    print(f"[SYNTHESIS DEBUG] Raw generation returned: {len(synthesized) if synthesized else 0} chars")
    print(f"[SYNTHESIS DEBUG] Type: {type(synthesized)}, Is None: {synthesized is None}")
    
    if not synthesized:
        print(f"[SYNTHESIS DEBUG] WARNING: Synthesis returned empty! Query: {query[:100]}")
        # FALLBACK: Return combined notes directly
        return combined_notes
    
    return synthesized
except Exception as exc:
    print(f"[BATCH DEBUG] Synthesis failed: {exc}")
    print(f"[SYNTHESIS DEBUG] Returning combined notes as fallback")
    return combined_notes  # <-- FALLBACK instead of empty string
```

### Fix 2: Fix `_post_process_output()` for gpt-oss-20b

**File**: `src/generation/llm_service.py` lines 505-508

Current code assumes llama.cpp channel format. For gpt-oss-20b:

```python
def _post_process_output(self, text: str) -> str:
    print(f"[POST_PROCESS DEBUG] Input: {len(text) if text else 0} chars, Provider: {self.llm_env_config.provider}")
    
    if self.llm_env_config.provider == "llama_cpp":
        result = self._extract_llama_final_channel(text)
        print(f"[POST_PROCESS DEBUG] After channel extraction: {len(result) if result else 0} chars")
        return result
    
    # For other providers, return as-is
    print(f"[POST_PROCESS DEBUG] Non-llama provider, returning raw text")
    return text
```

### Fix 3: Remove `set_full_synthesis` Call

**File**: `src/ui/modern_main_window.py` line 880-883

Either:
- **Option A**: Remove the call entirely (synthesis_full is still available in metrics)
```python
# Remove these lines:
# synthesis_full = metrics.get("synthesis_full")
# if synthesis_full:
#     self.output_panel.set_full_synthesis(synthesis_full)
```

- **Option B**: Add the method to `EnhancedOutputPanel`:
```python
def set_full_synthesis(self, text: str):
    """Store full synthesis for debugging."""
    self._full_synthesis = text
    # Optionally add a button to view it
```

### Fix 4: Increase Synthesis Token Limit

**File**: `config/config.yaml` line 81

Current: `synthesis_max_tokens: 4096`

Problem: Synthesis may be hitting token limit and getting truncated mid-generation.

Recommendation: Increase to **8192** or **unlimited** (None)

```yaml
generation:
  synthesis_max_tokens: 8192  # Was 4096 - increase for comprehensive answers
```

## Testing Plan

1. Add all debug logging first
2. Run a query that produces 0 chars
3. Examine logs to see exactly where the content disappears
4. Apply targeted fix based on evidence
5. Verify with multiple queries

## Priority

1. **Critical**: Fix synthesis returning empty (Fix 1 + 2)
2. **High**: Remove or implement `set_full_synthesis` (Fix 3)
3. **Medium**: Increase synthesis token limit if needed (Fix 4)

## Expected Outcome

After fixes:
- Synthesis should NEVER return empty string
- If synthesis genuinely fails, fallback to combined batch notes
- All successful generations show complete output
- No more AttributeErrors in UI

---
**Status**: Diagnosis complete, ready for implementation on user approval
