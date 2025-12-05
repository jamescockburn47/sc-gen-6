"""Test if Ollama can handle parallel requests."""

import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
PROMPT = "What is 2+2? Answer in one word."

def make_request(request_id: int) -> tuple[int, float, str]:
    """Make a single request to Ollama and time it."""
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": PROMPT,
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=120
        )
        elapsed = time.time() - start
        result = response.json().get("response", "")[:50]
        return request_id, elapsed, f"OK: {result}"
    except Exception as e:
        elapsed = time.time() - start
        return request_id, elapsed, f"ERROR: {str(e)}"

def test_sequential():
    """Run 4 requests sequentially."""
    print("\n=== SEQUENTIAL TEST (4 requests) ===")
    total_start = time.time()
    for i in range(4):
        req_id, elapsed, result = make_request(i)
        print(f"  Request {req_id}: {elapsed:.2f}s - {result}")
    total = time.time() - total_start
    print(f"  TOTAL: {total:.2f}s")
    return total

def test_parallel():
    """Run 4 requests in parallel."""
    print("\n=== PARALLEL TEST (4 requests) ===")
    total_start = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(make_request, i) for i in range(4)]
        for future in as_completed(futures):
            req_id, elapsed, result = future.result()
            print(f"  Request {req_id}: {elapsed:.2f}s - {result}")
            results.append((req_id, elapsed))
    
    total = time.time() - total_start
    print(f"  TOTAL: {total:.2f}s")
    
    # If parallel worked, total time should be ~= max(individual times)
    # If sequential, total time should be ~= sum(individual times)
    max_individual = max(r[1] for r in results)
    sum_individual = sum(r[1] for r in results)
    
    print(f"\n  Analysis:")
    print(f"    Sum of individual times: {sum_individual:.2f}s")
    print(f"    Max individual time: {max_individual:.2f}s")
    print(f"    Actual total time: {total:.2f}s")
    
    if total < sum_individual * 0.7:
        print(f"    ✓ PARALLEL WORKING! (total < 70% of sum)")
    else:
        print(f"    ✗ SEQUENTIAL! Requests are being queued")
    
    return total

if __name__ == "__main__":
    print("Testing Ollama parallel request handling...")
    print(f"Model: {MODEL}")
    
    # First, ensure model is loaded
    print("\nWarming up model...")
    make_request(-1)
    
    seq_time = test_sequential()
    par_time = test_parallel()
    
    print(f"\n=== SUMMARY ===")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")



