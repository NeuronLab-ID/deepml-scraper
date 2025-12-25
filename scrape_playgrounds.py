#!/usr/bin/env python3
"""
Scrape problems from deep-ml.com API to get playground data.
Uses concurrent requests for faster scraping.
"""

import json
import os
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

API_URL = "https://api.deep-ml.com/fetch-problem"
OUTPUT_DIR = Path("problems_with_playground")
MAX_WORKERS = 10  # Concurrent requests

# Thread-safe counters
progress_lock = Lock()
progress = {"done": 0, "with_pg": 0}

def fetch_problem(problem_id: int) -> tuple:
    """Fetch a single problem from API. Returns (id, data, error)."""
    try:
        response = requests.get(f"{API_URL}?problem_id={problem_id}", timeout=30)
        if response.status_code == 200:
            data = response.json()
            data["problem_id"] = problem_id
            return (problem_id, data, None)
        return (problem_id, None, f"HTTP {response.status_code}")
    except Exception as e:
        return (problem_id, None, str(e))

def has_playground(data: dict) -> bool:
    """Check if problem has playground enabled."""
    pg = data.get("playground", {})
    return pg.get("enabled", False) and pg.get("files", {})

def extract_playground(data: dict) -> dict:
    """Extract playground info from problem data."""
    pg = data.get("playground", {})
    files = pg.get("files", {})
    app_js = files.get("/App.js", {})
    
    return {
        "problem_id": data.get("problem_id"),
        "title": data.get("title"),
        "category": data.get("category"),
        "enabled": pg.get("enabled", False),
        "type": pg.get("type", "react"),
        "code": app_js.get("code", ""),
        "settings": pg.get("settings", {}),
        "dependencies": pg.get("dependencies", {})
    }

def process_result(result: tuple, output_dir: Path) -> dict:
    """Process a fetch result. Returns summary dict or None."""
    problem_id, data, error = result
    
    with progress_lock:
        progress["done"] += 1
        print(f"\rProgress: {progress['done']}/270 (with playground: {progress['with_pg']})", end="", flush=True)
    
    if error:
        return {"id": problem_id, "error": error}
    
    if has_playground(data):
        pg_data = extract_playground(data)
        
        # Save individual playground file
        output_file = output_dir / f"playground_{problem_id:04d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pg_data, f, indent=2, ensure_ascii=False)
        
        with progress_lock:
            progress["with_pg"] += 1
        
        return {
            "id": problem_id,
            "title": data.get("title"),
            "category": data.get("category"),
            "has_playground": True
        }
    
    return {"id": problem_id, "has_playground": False}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"Scraping problems with {MAX_WORKERS} concurrent workers...")
    print("=" * 50)
    
    problems_with_playground = []
    errors = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_problem, pid): pid for pid in range(1, 271)}
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            summary = process_result(result, OUTPUT_DIR)
            
            if summary.get("error"):
                errors.append(summary)
            elif summary.get("has_playground"):
                problems_with_playground.append(summary)
    
    elapsed = time.time() - start_time
    
    print(f"\n\n{'=' * 50}")
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"\nResults:")
    print(f"  With playground: {len(problems_with_playground)}")
    print(f"  Without playground: {270 - len(problems_with_playground) - len(errors)}")
    print(f"  Errors: {len(errors)}")
    
    # Sort by ID
    problems_with_playground.sort(key=lambda x: x["id"])
    
    # Save summary
    summary = {
        "total_with_playground": len(problems_with_playground),
        "scrape_time_seconds": elapsed,
        "problems": problems_with_playground,
        "errors": errors
    }
    
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPlayground files saved to: {OUTPUT_DIR}/")
    print("\nProblems with playground:")
    for p in problems_with_playground[:15]:
        print(f"  #{p['id']:3d}: {p['title'][:50]}")
    if len(problems_with_playground) > 15:
        print(f"  ... and {len(problems_with_playground) - 15} more")

if __name__ == "__main__":
    main()
