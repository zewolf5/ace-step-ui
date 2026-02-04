#!/usr/bin/env python3
"""Call Ollama LLM with a query and get formatted text response.

This script provides a simple interface to Ollama for text generation,
following the same structure as format_sample.py for consistency.

Usage examples:

    Command line:
        # Basic query
        python call_ollama.py --prompt "Explain quantum computing in simple terms"
        
        # With custom model and temperature
        python call_ollama.py --prompt "Write a haiku about coding" --model gemma3:4b --temperature 0.9
        
        # With system prompt and JSON output
        python call_ollama.py --prompt "What is AI?" --system "You are a helpful teacher" --json
        
        # With max tokens and custom URL
        python call_ollama.py --prompt "Tell me a story" --max-tokens 100 --url http://localhost:11434

    Python API:
        from call_ollama import call_ollama
        
        # Simple call
        result = call_ollama("What is the meaning of life?")
        if result["success"]:
            print(result["response"])
        
        # With options
        result = call_ollama(
            prompt="Explain photosynthesis",
            model="gemma3:4b",
            temperature=0.5,
            max_tokens=200,
            system_prompt="You are a biology expert"
        )

"""
import argparse
import json
import sys
import time
import requests
from typing import Optional

# Ollama API configuration
OLLAMA_BASE_URL = "http://192.168.0.170:11434" #"http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"

# Global session for connection pooling
_session = None

def get_session():
    """Get or create a requests session for Ollama API calls."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    base_url: str = None,
) -> dict:
    """Call Ollama API with a query and return the response.
    
    Args:
        prompt: The user prompt/query
        model: Ollama model name (default: llama2)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate (optional)
        system_prompt: Optional system prompt to guide the model
        base_url: Ollama API base URL (optional, defaults to global OLLAMA_BASE_URL)
        
    Returns:
        Dictionary with success status, response text, and metadata
    """
    session = get_session()
    url = base_url or OLLAMA_BASE_URL
    
    # Build the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        }
    }
    
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        start_time = time.time()
        response = session.post(
            f"{url}/api/generate",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {
                "success": False,
                "response": "",
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed_seconds": elapsed,
            }
        
        data = response.json()
        
        return {
            "success": True,
            "response": data.get("response", ""),
            "model": data.get("model", model),
            "total_duration": data.get("total_duration", 0) / 1e9,  # nanoseconds to seconds
            "load_duration": data.get("load_duration", 0) / 1e9,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "elapsed_seconds": elapsed,
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": "",
            "error": "Request timed out",
            "elapsed_seconds": time.time() - start_time,
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "response": "",
            "error": f"Could not connect to Ollama at {url}. Is Ollama running?",
            "elapsed_seconds": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "error": str(e),
            "elapsed_seconds": 0,
        }

def main():
    parser = argparse.ArgumentParser(description="Call Ollama LLM with a query")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt/query to send to Ollama")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--system", type=str, help="System prompt to guide the model")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--url", type=str, default=OLLAMA_BASE_URL, help=f"Ollama API URL (default: {OLLAMA_BASE_URL})")

    args = parser.parse_args()

    try:
        result = call_ollama(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
            base_url=args.url,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(result["response"])
                if "elapsed_seconds" in result:
                    print(f"\n[Completed in {result['elapsed_seconds']:.2f}s]", file=sys.stderr)
            else:
                print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
                sys.exit(1)

    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
