"""Test if Ollama handles concurrent requests."""

import time
import concurrent.futures
from src.llm.client import LLMClient
from src.config.llm_config import load_llm_config

config = load_llm_config()

def test_request(i):
    client = LLMClient(config)  # New client per thread
    messages = [{'role': 'user', 'content': f'Say hello #{i} in one sentence'}]
    start = time.time()
    response = client.generate_chat_completion(messages, model='qwen2.5:7b', temperature=0.0)
    return time.time() - start

# Test 4 concurrent requests
print("Testing 4 concurrent requests to Ollama...")
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(test_request, i) for i in range(4)]
    times = [f.result() for f in concurrent.futures.as_completed(futures)]

total = time.time() - start
print(f'4 requests total time: {total:.1f}s')
print(f'Individual times: {[f"{t:.1f}s" for t in times]}')
print(f'If parallel: should be ~{max(times):.1f}s')
print(f'If sequential: should be ~{sum(times):.1f}s')
print()
if total < sum(times) * 0.7:
    print("✓ Ollama IS processing requests in parallel!")
else:
    print("✗ Ollama is processing requests SEQUENTIALLY")
    print("  To enable parallelism: restart Ollama with OLLAMA_NUM_PARALLEL=8")



