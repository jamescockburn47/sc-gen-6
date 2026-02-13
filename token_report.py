#!/usr/bin/env python3
"""Raw token generation performance report for GLM 4.7 Flash on Vulkan.

Tests prompt processing (prefill) and token generation (decode) speeds
across different configurations.
"""

import json
import time
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"
MODEL = "glm-4.7-flash"

def generate(prompt: str, num_ctx: int = 4096, num_predict: int = 200, temperature: float = 0.7) -> dict:
    """Run a single generation and return timing metrics."""
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }, timeout=300)
    data = resp.json()
    if "error" in data:
        return {"error": data["error"]}

    # Ollama returns durations in nanoseconds
    total_ns = data.get("total_duration", 0)
    load_ns = data.get("load_duration", 0)
    prompt_eval_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    prompt_tokens = data.get("prompt_eval_count", 0)
    eval_tokens = data.get("eval_count", 0)

    prompt_tps = (prompt_tokens / (prompt_eval_ns / 1e9)) if prompt_eval_ns > 0 else 0
    eval_tps = (eval_tokens / (eval_ns / 1e9)) if eval_ns > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": eval_tokens,
        "total_time_s": round(total_ns / 1e9, 3),
        "load_time_s": round(load_ns / 1e9, 3),
        "prompt_eval_time_s": round(prompt_eval_ns / 1e9, 3),
        "generation_time_s": round(eval_ns / 1e9, 3),
        "prompt_processing_tps": round(prompt_tps, 1),
        "token_generation_tps": round(eval_tps, 1),
        "response_preview": data.get("response", "")[:200],
    }


def run_report():
    print("=" * 70)
    print("GLM 4.7 Flash — Raw Token Generation Report")
    print(f"Backend: Ollama + Vulkan | Model: {MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check model is loaded
    ps = requests.get(f"{OLLAMA_URL}/api/ps").json()
    models = ps.get("models", [])
    if models:
        m = models[0]
        print(f"Model loaded: {m.get('name')} | Size: {m.get('size',0)/1e9:.1f}GB | Processor: {m.get('details',{}).get('quantization_level','?')}")
    print()

    tests = []

    # ---- Test 1: Short prompt, short generation (warm cache) ----
    print("[1/6] Short prompt → short generation (warm-up)...")
    r = generate("Hello", num_ctx=2048, num_predict=50)
    tests.append({"test": "warmup_short", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Test 2: Short prompt, medium generation ----
    print("[2/6] Short prompt → 200 tokens...")
    r = generate("Write a detailed analysis of trust law principles in English equity.", num_ctx=4096, num_predict=200)
    tests.append({"test": "short_prompt_200tok", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Test 3: Short prompt, long generation ----
    print("[3/6] Short prompt → 500 tokens...")
    r = generate("Explain the legal principles governing equitable assignments, nominee holdings, and the priority of competing equitable interests under English law.", num_ctx=4096, num_predict=500)
    tests.append({"test": "short_prompt_500tok", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Test 4: Medium prompt (simulating RAG context) ----
    rag_context = """Based on the following case materials from [2021] EWHC 1272 (Comm):

CONTEXT CHUNK 1: The Orb Claimants agreed to adjourn an impending application for security for costs. 
Dr Cochrane sent a side letter promising to repay the £10m once sufficient money had been realised.
The IOM Settlement transferred the Claimed Trust Assets to Dr Cochrane and the shares in the Cooper 
and McNally Companies. Dr Cochrane granted a loan to Mr and Mrs Greenstone for the purpose of 
acquiring or re-mortgaging Walham Court.

CONTEXT CHUNK 2: HPII alleges that Mr Stevens was acting as Mr Ruhan's nominee at the time of the 
Cambulo Transaction and the Geneva Settlement. Ms Aird-Brown is a licensed insolvency practitioner 
and certified fraud examiner appointed as the liquidator of HPII on 9 March 2018.

CONTEXT CHUNK 3: The LICSA effected an equitable assignment. The rights which are subject to clause 
2.4 are capable of assignment in equity. The priority of competing equitable interests depends on 
whether the purchaser was bona fide, for value, of a legal interest, without notice.

QUESTION: What was the role of Dr Cochrane in relation to the trust assets and the IOM Settlement?
"""
    print("[4/6] RAG-style prompt (~300 tokens) → 300 tokens...")
    r = generate(rag_context, num_ctx=4096, num_predict=300)
    tests.append({"test": "rag_context_300tok", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Test 5: Large context window ----
    big_context = rag_context * 5  # ~1500 prompt tokens
    print("[5/6] Large context (~1500 tokens) → 300 tokens...")
    r = generate(big_context, num_ctx=8192, num_predict=300)
    tests.append({"test": "large_context_300tok", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Test 6: Maximum generation ----
    print("[6/6] Short prompt → 1000 tokens (sustained generation)...")
    r = generate("Write a comprehensive legal memorandum on the principles of tracing in equity, including proprietary claims, constructive trusts, and the defence of bona fide purchaser for value.", num_ctx=8192, num_predict=1000)
    tests.append({"test": "sustained_1000tok", **r})
    print(f"      Prompt: {r.get('prompt_processing_tps',0)} t/s | Gen: {r.get('token_generation_tps',0)} t/s | {r.get('generated_tokens',0)} tokens in {r.get('generation_time_s',0)}s")

    # ---- Summary ----
    gen_speeds = [t["token_generation_tps"] for t in tests if t.get("token_generation_tps", 0) > 0]
    prompt_speeds = [t["prompt_processing_tps"] for t in tests if t.get("prompt_processing_tps", 0) > 0]

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Avg prompt processing:  {sum(prompt_speeds)/len(prompt_speeds):.1f} tokens/s")
    print(f"  Avg token generation:   {sum(gen_speeds)/len(gen_speeds):.1f} tokens/s")
    print(f"  Min token generation:   {min(gen_speeds):.1f} tokens/s")
    print(f"  Max token generation:   {max(gen_speeds):.1f} tokens/s")
    print(f"  Tests run:              {len(tests)}")
    print("=" * 70)

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "backend": "Ollama + Vulkan",
        "gpu": "AMD Radeon 8060S (gfx1151, RADV)",
        "tests": tests,
        "summary": {
            "avg_prompt_tps": round(sum(prompt_speeds)/len(prompt_speeds), 1),
            "avg_generation_tps": round(sum(gen_speeds)/len(gen_speeds), 1),
            "min_generation_tps": round(min(gen_speeds), 1),
            "max_generation_tps": round(max(gen_speeds), 1),
        },
    }
    with open("benchmarks/token_report_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to benchmarks/token_report_baseline.json")


if __name__ == "__main__":
    run_report()
