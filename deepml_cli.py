#!/usr/bin/env python3
"""
Deep-ML CLI - Fetch all problems from api.deep-ml.com

Usage:
    python deepml_cli.py [--output OUTPUT_DIR] [--start START_ID] [--delay SECONDS]
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


API_BASE_URL = "https://api.deep-ml.com"
DEFAULT_OUTPUT_DIR = "problems"
DEFAULT_DELAY = 0.5  # Delay between requests to be polite


class DeepMLCLI:
    """CLI tool to fetch problems from deep-ml.com API."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, delay: float = DEFAULT_DELAY):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "DeepML-CLI/1.0"
        })
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_problem(self, problem_id: int) -> Optional[dict]:
        """
        Fetch a single problem by ID.
        
        Returns:
            dict: Problem data if found
            None: If problem doesn't exist or request failed
        """
        url = f"{API_BASE_URL}/fetch-problem"
        params = {"problem_id": problem_id}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                print(f"  [!] Unexpected status code: {response.status_code}")
                return None
            
            data = response.json()
            
            # Empty JSON means problem doesn't exist
            if not data or data == {}:
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  [!] Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"  [!] JSON decode error: {e}")
            return None
    
    def decode_base64_fields(self, problem: dict) -> dict:
        """Decode base64 encoded fields in the problem data."""
        decoded = problem.copy()
        
        # Fields that are typically base64 encoded
        base64_fields = ["description", "learn_section", "solution", "tinygrad_starter_code"]
        
        for field in base64_fields:
            if field in decoded and decoded[field]:
                try:
                    decoded[f"{field}_decoded"] = base64.b64decode(decoded[field]).decode("utf-8")
                except Exception:
                    pass  # Keep original if decoding fails
        
        return decoded
    
    def save_problem(self, problem: dict, problem_id: int) -> None:
        """Save problem to a JSON file."""
        # Decode base64 fields for readability
        decoded_problem = self.decode_base64_fields(problem)
        
        filepath = self.output_dir / f"problem_{problem_id:04d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(decoded_problem, f, indent=2, ensure_ascii=False)
    
    def save_summary(self, problems: list[dict]) -> None:
        """Save a summary of all problems."""
        summary = []
        for p in problems:
            summary.append({
                "id": p.get("id"),
                "title": p.get("title"),
                "category": p.get("category"),
                "difficulty": p.get("difficulty"),
            })
        
        filepath = self.output_dir / "problems_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[+] Saved summary to {filepath}")
    
    def fetch_all_problems(self, start_id: int = 1) -> list[dict]:
        """
        Fetch all problems starting from start_id.
        
        Stops when encountering an empty response (problem doesn't exist).
        """
        problems = []
        current_id = start_id
        consecutive_empty = 0
        max_consecutive_empty = 3  # Stop after 3 consecutive empty responses
        
        print(f"[*] Starting to fetch problems from ID {start_id}...")
        print(f"[*] Output directory: {self.output_dir.absolute()}")
        print("-" * 60)
        
        while True:
            print(f"[{current_id}] Fetching problem...", end=" ", flush=True)
            
            problem = self.fetch_problem(current_id)
            
            if problem is None:
                print("NOT FOUND")
                consecutive_empty += 1
                
                if consecutive_empty >= max_consecutive_empty:
                    print(f"\n[!] {max_consecutive_empty} consecutive empty responses. Stopping.")
                    break
            else:
                consecutive_empty = 0
                title = problem.get("title", "Unknown")
                category = problem.get("category", "Unknown")
                difficulty = problem.get("difficulty", "Unknown")
                
                print(f"OK - {title[:40]}... [{category}] [{difficulty}]")
                
                self.save_problem(problem, current_id)
                problems.append(problem)
            
            current_id += 1
            
            # Be polite to the server
            if self.delay > 0:
                time.sleep(self.delay)
        
        print("-" * 60)
        print(f"[+] Fetched {len(problems)} problems total")
        
        return problems
    
    def display_problem(self, problem: dict) -> None:
        """Display a problem in a readable format."""
        decoded = self.decode_base64_fields(problem)
        
        print("\n" + "=" * 70)
        print(f"Problem #{decoded.get('id', 'N/A')}: {decoded.get('title', 'Unknown')}")
        print("=" * 70)
        print(f"Category: {decoded.get('category', 'N/A')}")
        print(f"Difficulty: {decoded.get('difficulty', 'N/A')}")
        print(f"Likes: {decoded.get('likes', 0)} | Dislikes: {decoded.get('dislikes', 0)}")
        print("-" * 70)
        
        if "description_decoded" in decoded:
            print("\nDescription:")
            print(decoded["description_decoded"])
        
        if "example" in decoded:
            example = decoded["example"]
            print("\nExample:")
            print(f"  Input: {example.get('input', 'N/A')}")
            print(f"  Output: {example.get('output', 'N/A')}")
            if "reasoning" in example:
                print(f"  Reasoning: {example.get('reasoning', 'N/A')}")
        
        if "starter_code" in decoded:
            print("\nStarter Code:")
            print("-" * 40)
            print(decoded["starter_code"])
            print("-" * 40)
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Deep-ML CLI - Fetch problems from api.deep-ml.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch all problems starting from ID 1
    python deepml_cli.py
    
    # Fetch problems starting from ID 50
    python deepml_cli.py --start 50
    
    # Save to custom directory with faster polling
    python deepml_cli.py --output my_problems --delay 0.2
    
    # Fetch a single problem and display it
    python deepml_cli.py --id 101 --display
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for problem files (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=1,
        help="Starting problem ID (default: 1)"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})"
    )
    
    parser.add_argument(
        "--id", "-i",
        type=int,
        help="Fetch a single problem by ID"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display problem details (use with --id)"
    )
    
    args = parser.parse_args()
    
    cli = DeepMLCLI(output_dir=args.output, delay=args.delay)
    
    if args.id:
        # Fetch a single problem
        print(f"[*] Fetching problem #{args.id}...")
        problem = cli.fetch_problem(args.id)
        
        if problem:
            if args.display:
                cli.display_problem(problem)
            else:
                cli.save_problem(problem, args.id)
                print(f"[+] Saved to {cli.output_dir / f'problem_{args.id:04d}.json'}")
        else:
            print(f"[!] Problem #{args.id} not found")
            sys.exit(1)
    else:
        # Fetch all problems
        problems = cli.fetch_all_problems(start_id=args.start)
        
        if problems:
            cli.save_summary(problems)
            print(f"[+] All problems saved to {cli.output_dir.absolute()}")
        else:
            print("[!] No problems fetched")
            sys.exit(1)


if __name__ == "__main__":
    main()
