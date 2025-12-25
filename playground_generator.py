#!/usr/bin/env python3
"""
Playground Generator - Regenerate React playground components with NeuronLab terminal style.

Uses GitHub Copilot CLI to recreate each playground visualization
with our custom terminal/hacker aesthetic.

Usage:
  python playground_generator.py --id 26       # Generate and update DB for problem 26
  python playground_generator.py --all         # Generate all playgrounds
  python playground_generator.py --limit 5     # Generate first 5 playgrounds
"""

import argparse
import json
import os
import subprocess
import tempfile
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Directories
SCRAPED_DIR = Path("problems_with_playground")
OUTPUT_DIR = Path("playgrounds_neuronlab")
DATABASE_PATH = Path("d:/localdeepml/backend/deepml.db")

# Thread-safe progress
progress_lock = Lock()
progress = {"done": 0, "success": 0}

# NeuronLab theme definition for prompts
NEURONLAB_THEME = """
NeuronLab Terminal Theme:
- Background: #0d0d14 (deepest), #1a1a2e (surface), #252538 (surface-alt)
- Primary/Accent: #00ffff (cyan) - terminal prompt color
- Secondary colors: #22d3ee (bright cyan), #facc15 (yellow), #ec4899 (pink), #4ade80 (green), #f97316 (orange)
- Text: #ffffff (primary), #9ca3af (secondary), #6b7280 (muted)
- Borders: 2px solid with gray-700 (#374151) or colored accents
- Font: 'JetBrains Mono', monospace throughout
- Style: Terminal/hacker aesthetic with:
  - Square/rectangular shapes (no rounded corners or use minimal rx="4")
  - Monospace text everywhere
  - Terminal-style headers with ">" prefix
  - Scanline effects optional
  - Glow effects on active elements
  - Step indicators as square brackets like [1] [2] [3]
"""

SYSTEM_PROMPT = f"""You are an expert React developer recreating interactive visualizations for NeuronLab, a terminal-styled coding education platform.

{NEURONLAB_THEME}

Your task is to take an existing React playground component and completely recreate it with NeuronLab's terminal/hacker aesthetic.

CRITICAL REQUIREMENTS:
1. Keep the EXACT same functionality and interactivity
2. Replace all colors with NeuronLab theme colors
3. Use square/rectangular styling instead of rounded corners
4. Add terminal-style prefixes ("> ") for labels
5. Use monospace font everywhere
6. Keep SVG visualizations but restyle with theme colors
7. Step indicators should use square brackets: [Step 1] [Step 2]
8. Buttons should have border styling, not rounded backgrounds
9. The component must be a SINGLE App.js file that works standalone
10. Use React.useState for state management (functional component)

OUTPUT FORMAT:
Return ONLY the complete React component code, starting with 'import React' and ending with the export.
Do NOT include markdown code blocks or explanations.
"""


def generate_with_copilot_cli(prompt: str, model: str = "claude-sonnet-4", debug: bool = False) -> str:
    """Generate response using GitHub Copilot CLI."""
    try:
        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(prompt)
            prompt_file = f.name
        
        # Find copilot executable
        copilot_path = os.path.join(os.environ.get('APPDATA', ''), 'npm', 'copilot.cmd')
        if not os.path.exists(copilot_path):
            copilot_path = 'copilot'  # Fallback to PATH
        
        # Use copilot CLI with prompt from file
        result = subprocess.run(
            [copilot_path, "-p", f"Read and respond to the prompt in this file: {prompt_file}", 
             "--model", model, "-s", "--allow-all-paths"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300
        )
        
        os.unlink(prompt_file)
        
        if debug:
            print(f"\n[DEBUG] Copilot return code: {result.returncode}")
            print(f"[DEBUG] Response length: {len(result.stdout)} chars")
            if result.stderr:
                print(f"[DEBUG] Stderr: {result.stderr[:500]}")
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"  [!] Copilot CLI error: {result.stderr[:300]}")
            return ""
    except subprocess.TimeoutExpired:
        print("  [!] Copilot CLI timeout (5 min)")
        return ""
    except FileNotFoundError:
        print("  [!] Copilot CLI not found")
        return ""
    except Exception as e:
        print(f"  [!] Error: {e}")
        return ""


def extract_react_code(response: str) -> str:
    """Extract React code from response, handling markdown blocks and prefixes."""
    # Remove markdown code blocks if present
    if "```" in response:
        # Find code block
        parts = response.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd indices are code blocks
                # Remove language identifier if present
                lines = part.split('\n')
                if lines[0].strip() in ['javascript', 'jsx', 'js', 'react', 'tsx']:
                    code = '\n'.join(lines[1:]).strip()
                else:
                    code = part.strip()
                # Isolate single component
                return isolate_single_component(code)
    
    # Find the actual React code start (import React or export default)
    lines = response.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import React') or line.strip().startswith('export default'):
            start_idx = i
            break
    
    if start_idx > 0:
        code = '\n'.join(lines[start_idx:])
        return isolate_single_component(code.strip())
    
    # If no imports found, return as-is if it looks like React code
    if "import React" in response or "export default" in response:
        return isolate_single_component(response.strip())
    
    return ""


def isolate_single_component(code: str) -> str:
    """Ensure only one complete React component is returned."""
    # Find the first 'import React' and the corresponding 'export default function' closing
    lines = code.split('\n')
    
    # Find start
    start_idx = 0
    for i, line in enumerate(lines):
        if 'import React' in line:
            start_idx = i
            break
    
    # Find the export default function and track braces to find its end
    end_idx = len(lines)
    in_export = False
    brace_count = 0
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if 'export default function' in line or 'export default' in line:
            in_export = True
        
        if in_export:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > start_idx:
                # Check for closing brace on its own line
                if line.strip() == '}':
                    end_idx = i + 1
                    break
    
    result = '\n'.join(lines[start_idx:end_idx])
    
    # Sanity check - if result is too short or has another 'import React', something's wrong
    if 'import React' in result[100:]:  # Check if there's another import after the first
        # Just take up to the second import
        second_import = result.find('import React', 50)
        if second_import > 0:
            result = result[:second_import].rstrip()
    
    return result


def regenerate_playground(problem_id: int, original_code: str, title: str, 
                          model: str = "claude-sonnet-4", debug: bool = False) -> str:
    """Regenerate a single playground with NeuronLab style."""
    
    prompt = f"""{SYSTEM_PROMPT}

ORIGINAL COMPONENT (Problem #{problem_id}: {title}):
```jsx
{original_code}
```

Recreate this component with NeuronLab's terminal aesthetic. 
Keep all functionality, just restyle it completely.
Return ONLY the React code."""
    
    response = generate_with_copilot_cli(prompt, model=model, debug=debug)
    
    if not response:
        return ""
    
    return extract_react_code(response)


def update_database_playground(problem_id: int, code: str) -> bool:
    """Update the playground code in the backend database."""
    try:
        if not DATABASE_PATH.exists():
            print(f"  [!] Database not found: {DATABASE_PATH}")
            return False
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Update the problem's playground fields
        cursor.execute("""
            UPDATE problems 
            SET playground_enabled = 1, playground_code = ?
            WHERE id = ?
        """, (code, problem_id))
        
        if cursor.rowcount == 0:
            print(f"  [!] Problem #{problem_id} not found in database")
            conn.close()
            return False
        
        conn.commit()
        conn.close()
        print(f"       [DB] Updated problem #{problem_id} in database")
        return True
    except Exception as e:
        print(f"  [!] Database error: {e}")
        return False


def process_playground(problem_id: int, input_dir: Path, output_dir: Path, 
                       model: str, debug: bool, update_db: bool = True) -> dict:
    """Process a single playground file."""
    input_file = input_dir / f"playground_{problem_id:04d}.json"
    
    if not input_file.exists():
        return {"id": problem_id, "error": "File not found"}
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_code = data.get("code", "")
    title = data.get("title", f"Problem {problem_id}")
    
    if not original_code:
        return {"id": problem_id, "error": "No code in file"}
    
    print(f"  [{problem_id:3d}] {title[:40]}...")
    
    new_code = regenerate_playground(problem_id, original_code, title, model=model, debug=debug)
    
    if new_code and len(new_code) > 100:
        # Save regenerated playground to JSON
        output_data = {
            **data,
            "code": new_code,
            "original_code": original_code,
            "regenerated": True
        }
        
        output_file = output_dir / f"playground_{problem_id:04d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Update database if requested
        if update_db:
            update_database_playground(problem_id, new_code)
        
        with progress_lock:
            progress["success"] += 1
        
        return {"id": problem_id, "title": title, "success": True, "db_updated": update_db}
    else:
        return {"id": problem_id, "title": title, "error": "Generation failed"}


def main():
    parser = argparse.ArgumentParser(description="Regenerate playgrounds with NeuronLab style")
    parser.add_argument("--id", type=int, help="Regenerate specific problem ID")
    parser.add_argument("--all", action="store_true", help="Regenerate all playgrounds")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N playgrounds")
    parser.add_argument("--model", default="claude-sonnet-4", help="Copilot model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default: 1)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load summary to get list of problems
    summary_file = SCRAPED_DIR / "summary.json"
    if not summary_file.exists():
        print(f"Error: {summary_file} not found. Run scrape_playgrounds.py first.")
        return
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    problem_ids = [p["id"] for p in summary["problems"]]
    
    if args.id:
        if args.id not in problem_ids:
            print(f"Problem #{args.id} does not have a playground")
            return
        problem_ids = [args.id]
    elif args.limit:
        problem_ids = problem_ids[:args.limit]
    elif not args.all:
        print("Usage: python playground_generator.py --id <ID> | --all | --limit <N>")
        print(f"\nAvailable: {len(problem_ids)} playgrounds")
        print(f"IDs: {problem_ids[:20]}{'...' if len(problem_ids) > 20 else ''}")
        return
    
    print(f"Regenerating {len(problem_ids)} playgrounds with NeuronLab style")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    results = []
    errors = []
    
    if args.workers > 1:
        # Concurrent processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_playground, pid, SCRAPED_DIR, OUTPUT_DIR, args.model, args.debug): pid
                for pid in problem_ids
            }
            
            for future in as_completed(futures):
                result = future.result()
                with progress_lock:
                    progress["done"] += 1
                    print(f"\rProgress: {progress['done']}/{len(problem_ids)} ({progress['success']} success)", end="")
                
                if result.get("error"):
                    errors.append(result)
                else:
                    results.append(result)
    else:
        # Sequential processing
        for i, pid in enumerate(problem_ids):
            result = process_playground(pid, SCRAPED_DIR, OUTPUT_DIR, args.model, args.debug)
            progress["done"] += 1
            
            if result.get("error"):
                errors.append(result)
                print(f"       [FAILED] {result.get('error')}")
            else:
                results.append(result)
                print(f"       [OK]")
    
    print("\n" + "=" * 60)
    print(f"Success: {len(results)}/{len(problem_ids)}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {OUTPUT_DIR}/")
    
    if errors:
        print("\nFailed:")
        for e in errors[:10]:
            print(f"  #{e['id']}: {e.get('error', 'Unknown')}")
        
        # Save failed list
        with open(OUTPUT_DIR / "failed.json", "w") as f:
            json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
