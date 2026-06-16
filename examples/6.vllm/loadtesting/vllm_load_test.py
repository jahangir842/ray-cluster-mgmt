#!/usr/bin/env python3
"""
Load Test Script for vLLM Chat Application
Tests streaming chat completions with concurrent users
"""
import threading
import time
import requests
import json
from datetime import datetime
import random
import sys
from collections import defaultdict

# Configuration — vLLM served via `vllm serve ... --host 0.0.0.0 --port 8000`
SERVER_HOST = "192.168.3.73"
SERVER_PORT = "8000"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MODEL_ID = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"


NUM_USERS = 50
TEST_DURATION = 300  # 5 minutes
MESSAGES_PER_USER = 10
DELAY_BETWEEN_MESSAGES = (3, 8)  # Random seconds between messages

# Sample messages for testing
TEST_MESSAGES = [
    "What is artificial intelligence?",
    "Explain machine learning in simple terms",
    "How does neural network work?",
    "Tell me about natural language processing",
    "What are the benefits of AI?",
    "Describe deep learning",
    "What is the difference between AI and ML?",
    "How can AI help businesses?",
    "Explain computer vision",
    "What is reinforcement learning?",
    "Tell me a short story",
    "What is Python programming?",
    "How do transformers work in AI?",
    "Explain large language models",
    "What is the future of AI?"
]

# Thread-safe statistics
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages_sent = 0
        self.messages_completed = 0
        self.messages_failed = 0
        self.total_response_time = 0
        self.total_tokens_received = 0
        self.response_times = []
        self.streaming_errors = 0
        self.connection_errors = 0
        self.active_users = 0
        self.user_stats = defaultdict(lambda: {'sent': 0, 'completed': 0, 'failed': 0})
    
    def record_sent(self, user_id):
        with self.lock:
            self.messages_sent += 1
            self.user_stats[user_id]['sent'] += 1
    
    def record_completed(self, user_id, response_time, tokens):
        with self.lock:
            self.messages_completed += 1
            self.total_response_time += response_time
            self.response_times.append(response_time)
            self.total_tokens_received += tokens
            self.user_stats[user_id]['completed'] += 1
    
    def record_failed(self, user_id, error_type='general'):
        with self.lock:
            self.messages_failed += 1
            self.user_stats[user_id]['failed'] += 1
            if error_type == 'streaming':
                self.streaming_errors += 1
            elif error_type == 'connection':
                self.connection_errors += 1
    
    def increment_active(self):
        with self.lock:
            self.active_users += 1
    
    def decrement_active(self):
        with self.lock:
            self.active_users -= 1
    
    def get_summary(self):
        with self.lock:
            avg_response = self.total_response_time / self.messages_completed if self.messages_completed > 0 else 0
            avg_tokens = self.total_tokens_received / self.messages_completed if self.messages_completed > 0 else 0
            
            percentiles = {}
            if self.response_times:
                sorted_times = sorted(self.response_times)
                percentiles = {
                    'p50': sorted_times[len(sorted_times) // 2],
                    'p90': sorted_times[int(len(sorted_times) * 0.9)],
                    'p95': sorted_times[int(len(sorted_times) * 0.95)],
                    'p99': sorted_times[int(len(sorted_times) * 0.99)]
                }
            
            return {
                'active_users': self.active_users,
                'messages_sent': self.messages_sent,
                'messages_completed': self.messages_completed,
                'messages_failed': self.messages_failed,
                'success_rate': (self.messages_completed / self.messages_sent * 100) if self.messages_sent > 0 else 0,
                'avg_response_time': avg_response,
                'avg_tokens': avg_tokens,
                'streaming_errors': self.streaming_errors,
                'connection_errors': self.connection_errors,
                'percentiles': percentiles
            }

stats = Stats()

def test_server_connection():
    """Test if server is reachable"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if response.ok:
            print(f"✓ Server is reachable at {BASE_URL}")
            models = response.json()
            print(f"✓ Available models: {json.dumps(models, indent=2)}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return False

def simulate_user(user_id, start_time):
    """Simulate a single user sending messages"""
    username = f"LoadTestUser_{user_id}"
    stats.increment_active()
    
    print(f"[User {user_id}] Started")
    
    message_count = 0
    
    while time.time() - start_time < TEST_DURATION and message_count < MESSAGES_PER_USER:
        try:
            # Select a random message
            message = random.choice(TEST_MESSAGES)
            
            # Prepare payload (matching your frontend)
            payload = {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
                "max_tokens": 512,
                "stream": True
            }
            
            stats.record_sent(user_id)
            request_start = time.time()
            
            # Send request with streaming
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"[User {user_id}] HTTP {response.status_code}")
                stats.record_failed(user_id, 'connection')
                time.sleep(random.uniform(*DELAY_BETWEEN_MESSAGES))
                continue
            
            # Process streaming response
            full_response = ""
            token_count = 0
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        try:
                            parsed = json.loads(data)
                            if parsed.get('choices') and len(parsed['choices']) > 0:
                                choice = parsed['choices'][0]
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    full_response += content
                                    token_count += 1
                        except json.JSONDecodeError:
                            continue
            
            response_time = time.time() - request_start
            
            if full_response:
                stats.record_completed(user_id, response_time, token_count)
                message_count += 1
                print(f"[User {user_id}] Message {message_count}/{MESSAGES_PER_USER} - {response_time:.2f}s - {token_count} tokens")
            else:
                stats.record_failed(user_id, 'streaming')
                print(f"[User {user_id}] Empty response")
            
            # Wait before next message
            time.sleep(random.uniform(*DELAY_BETWEEN_MESSAGES))
            
        except requests.exceptions.Timeout:
            stats.record_failed(user_id, 'connection')
            print(f"[User {user_id}] Request timeout")
            time.sleep(5)
            
        except Exception as e:
            stats.record_failed(user_id, 'general')
            print(f"[User {user_id}] Error: {e}")
            time.sleep(5)
    
    stats.decrement_active()
    print(f"[User {user_id}] Finished - Sent {message_count} messages")

def print_stats_periodic():
    """Print statistics every 10 seconds"""
    start_time = time.time()
    
    while True:
        time.sleep(10)
        elapsed = time.time() - start_time
        summary = stats.get_summary()
        
        print(f"\n{'='*70}")
        print(f"STATISTICS @ {datetime.now().strftime('%H:%M:%S')} (Elapsed: {elapsed:.0f}s)")
        print(f"{'='*70}")
        print(f"Active Users:       {summary['active_users']}")
        print(f"Messages Sent:      {summary['messages_sent']}")
        print(f"Messages Completed: {summary['messages_completed']}")
        print(f"Messages Failed:    {summary['messages_failed']}")
        print(f"Success Rate:       {summary['success_rate']:.1f}%")
        print(f"Avg Response Time:  {summary['avg_response_time']:.2f}s")
        print(f"Avg Tokens/Msg:     {summary['avg_tokens']:.0f}")
        print(f"Connection Errors:  {summary['connection_errors']}")
        print(f"Streaming Errors:   {summary['streaming_errors']}")
        
        if summary['percentiles']:
            print(f"\nResponse Time Percentiles:")
            print(f"  P50 (median): {summary['percentiles']['p50']:.2f}s")
            print(f"  P90:          {summary['percentiles']['p90']:.2f}s")
            print(f"  P95:          {summary['percentiles']['p95']:.2f}s")
            print(f"  P99:          {summary['percentiles']['p99']:.2f}s")
        
        print(f"{'='*70}\n")

def main():
    print(f"\n{'='*70}")
    print(f"vLLM CHAT APPLICATION LOAD TEST")
    print(f"{'='*70}")
    print(f"Server:        {BASE_URL}")
    print(f"Model:         {MODEL_ID}")
    print(f"Users:         {NUM_USERS}")
    print(f"Duration:      {TEST_DURATION}s ({TEST_DURATION//60} minutes)")
    print(f"Msgs/User:     {MESSAGES_PER_USER}")
    print(f"{'='*70}\n")
    
    # Test connection first
    print("Testing server connection...")
    if not test_server_connection():
        print("\n❌ Cannot proceed - server is not accessible")
        sys.exit(1)
    
    print("\n✓ Starting load test in 3 seconds...\n")
    time.sleep(3)
    
    start_time = time.time()
    
    # Start statistics thread
    stats_thread = threading.Thread(target=print_stats_periodic, daemon=True)
    stats_thread.start()
    
    # Create and start user threads
    threads = []
    for i in range(NUM_USERS):
        thread = threading.Thread(target=simulate_user, args=(i, start_time))
        threads.append(thread)
        thread.start()
        time.sleep(0.2)  # Stagger connections
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Final statistics
    summary = stats.get_summary()
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Total Duration:     {time.time() - start_time:.1f}s")
    print(f"Total Users:        {NUM_USERS}")
    print(f"Messages Sent:      {summary['messages_sent']}")
    print(f"Messages Completed: {summary['messages_completed']}")
    print(f"Messages Failed:    {summary['messages_failed']}")
    print(f"Success Rate:       {summary['success_rate']:.1f}%")
    print(f"")
    print(f"Avg Response Time:  {summary['avg_response_time']:.2f}s")
    print(f"Avg Tokens/Message: {summary['avg_tokens']:.0f}")
    print(f"")
    print(f"Connection Errors:  {summary['connection_errors']}")
    print(f"Streaming Errors:   {summary['streaming_errors']}")
    
    if summary['percentiles']:
        print(f"\nResponse Time Percentiles:")
        print(f"  P50 (median): {summary['percentiles']['p50']:.2f}s")
        print(f"  P90:          {summary['percentiles']['p90']:.2f}s")
        print(f"  P95:          {summary['percentiles']['p95']:.2f}s")
        print(f"  P99:          {summary['percentiles']['p99']:.2f}s")
    
    # Throughput calculations
    if summary['messages_completed'] > 0:
        duration = time.time() - start_time
        throughput = summary['messages_completed'] / duration
        print(f"\nThroughput:         {throughput:.2f} messages/second")
        print(f"Tokens/second:      {(summary['avg_tokens'] * throughput):.0f}")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)