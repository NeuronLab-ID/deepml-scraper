#!/usr/bin/env python3
"""
Quest Generator - Generate learning sub-quests with verbose mathematical explanations.

Supports:
- OpenAI API (--backend openai)
- GitHub Models API (--backend github) - uses your Copilot quota
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from openai import OpenAI
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Configuration
PROBLEMS_DIR = Path("problems")
QUESTS_DIR = Path("quests")
FAILED_QUESTS_FILE = Path("failed_quests.json")

# API Keys - Load from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


SYSTEM_PROMPT = """You are a rigorous mathematics tutor creating structured learning content for a web-based educational platform (like deep-ml.com).

Your task is to break down programming problems into 4-6 learning sub-quests. 

**CRITICAL: Each sub-quest must DIRECTLY relate to solving the main problem!**
- Sub-quests are NOT generic math lessons
- Each sub-quest teaches a specific skill/concept NEEDED for the final solution
- The final sub-quest should combine all previous skills
- A user completing all sub-quests should be able to solve the main problem

Each sub-quest must:
1. **Teach a concept REQUIRED for solving the main problem** - Be specific
2. **Include formal mathematical definitions** - Use proper notation and LaTeX
3. **Have a coding exercise** that builds toward the final solution
4. **Build progressively** - earlier sub-quests are prerequisites for later ones

FORMAT: Return a JSON array with this EXACT structure (web-app compatible):
```json
[
  {
    "step": 1,
    "title": "Formal title of concept",
    "relation_to_problem": "How this concept directly helps solve the main problem",
    "prerequisites": ["Concept A", "Concept B"],
    "learning_objectives": ["Objective 1", "Objective 2"],
    "math_content": {
      "definition": "Formal mathematical definition with notation...",
      "notation": "$variable$ = meaning",
      "theorem": "If applicable, state the theorem...",
      "proof_sketch": "Brief proof or derivation...",
      "examples": ["Worked example 1", "Worked example 2"]
    },
    "key_formulas": [
      {"name": "Formula Name", "latex": "$formula$", "description": "When to use this"}
    ],
    "exercise": {
      "description": "Clear task description - a building block for the main problem",
      "function_signature": "def func_name(param: type) -> return_type:",
      "starter_code": "def func_name(...):\\n    # Your code here\\n    pass",
      "test_cases": [
        {"input": "func_name(arg1)", "expected": "result", "explanation": "Why this is correct"}
      ]
    },
    "common_mistakes": ["Mistake 1", "Mistake 2"],
    "hint": "Conceptual hint without giving away the solution",
    "references": ["Topic to research for more depth"]
  }
]
```

CRITICAL REQUIREMENTS:
- **Every sub-quest must be ESSENTIAL for solving the main problem**
- Be EXTREMELY detailed with mathematical explanations
- Include proper LaTeX notation for ALL formulas
- Each sub-quest exercise is a BUILDING BLOCK for the final solution
- DO NOT reveal the final solution code
- Use proper mathematical rigor (definitions before theorems, examples after theory)"""


# Research context to inject (populated by Perplexity research)
RESEARCH_CONTEXT_TEMPLATE = """
## Additional Mathematical Context (from research):
{research}

Use this research to enhance your explanations with:
- Formal definitions from textbooks
- Historical context where relevant
- Connections to real-world applications
- Common pitfalls and edge cases
"""


def research_with_perplexity(topic: str, category: str) -> str:
    """Research mathematical concepts using Perplexity API with reasoning."""
    if not PERPLEXITY_API_KEY:
        return ""
    
    try:
        query = f"""Research the mathematical foundations of "{topic}" in the context of {category}.

Provide:
1. Formal mathematical definitions with proper notation
2. Key theorems and their proofs/derivations
3. Step-by-step worked examples
4. Practical implementation considerations
5. Common mistakes and edge cases
6. Prerequisites needed to understand this topic

Use academic sources and textbooks where possible."""
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-reasoning-pro",  # Advanced reasoning with web search
                "messages": [
                    {"role": "user", "content": query}
                ],
                "max_tokens": 3000,
                "temperature": 0.2,
                "search_recency_filter": "month"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        else:
            print(f"  [!] Perplexity API error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"  [!] Perplexity research error: {e}")
        return ""


def get_github_token() -> str:
    """Get GitHub token from gh CLI or environment."""
    if GITHUB_TOKEN:
        return GITHUB_TOKEN
    
    try:
        gh_path = os.path.join(os.environ.get("ProgramFiles", ""), "GitHub CLI", "gh.exe")
        if os.path.exists(gh_path):
            result = subprocess.run([gh_path, "auth", "token"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    
    return ""


def generate_with_copilot_cli(prompt: str, model: str = "claude-sonnet-4.5", debug: bool = False) -> str:
    """Generate response using GitHub Copilot CLI."""
    import tempfile
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
            [copilot_path, "-p", f"Read and respond to the prompt in this file: {prompt_file}", "--model", model, "-s", "--allow-all-paths"],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Fix Unicode encoding on Windows
            timeout=300
        )
        
        os.unlink(prompt_file)
        
        if debug:
            print(f"\n[DEBUG] Copilot return code: {result.returncode}")
            print(f"[DEBUG] Response length: {len(result.stdout)} chars")
            if result.stderr:
                print(f"[DEBUG] Stderr: {result.stderr[:500]}")
            # Save raw response for inspection
            with open("debug_response.txt", "w", encoding="utf-8") as f:
                f.write(result.stdout)
            print(f"[DEBUG] Full response saved to debug_response.txt")
        
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


def create_client(backend: str = "github") -> OpenAI:
    """Create OpenAI client for the specified backend."""
    if backend == "github":
        token = get_github_token()
        if not token:
            print("[!] No GitHub token found. Run: gh auth login")
            print("[!] Falling back to OpenAI...")
            return OpenAI(api_key=OPENAI_API_KEY)
        
        return OpenAI(
            api_key=token,
            base_url="https://models.inference.ai.azure.com"
        )
    else:
        return OpenAI(api_key=OPENAI_API_KEY)


def decode_base64(encoded: str) -> str:
    """Decode base64 string to text."""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return encoded


def load_problem(problem_id: int) -> Optional[dict]:
    """Load a problem from the problems directory."""
    filepath = PROBLEMS_DIR / f"problem_{problem_id:04d}.json"
    if not filepath.exists():
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_problem_context(problem: dict) -> str:
    """Prepare problem context for the AI."""
    title = problem.get("title", "Unknown")
    category = problem.get("category", "Unknown")
    difficulty = problem.get("difficulty", "Unknown")
    
    description = problem.get("description_decoded") or decode_base64(problem.get("description", ""))
    learn_section = problem.get("learn_section_decoded") or decode_base64(problem.get("learn_section", ""))
    
    example = problem.get("example", {})
    starter_code = problem.get("starter_code", "")
    
    context = f"""
# Problem: {title}
Category: {category}
Difficulty: {difficulty}

## Description
{description}

## Example
Input: {example.get('input', '')}
Output: {example.get('output', '')}
Reasoning: {example.get('reasoning', '')}

## Starter Code
```python
{starter_code}
```

## Background Knowledge
{learn_section}
"""
    return context


def generate_sub_quests(problem: dict, client: OpenAI = None, model: str = "gpt-4o", backend: str = "copilot", use_research: bool = True) -> list[dict]:
    """Generate sub-quests for a problem with optional Perplexity research."""
    title = problem.get("title", "Unknown")
    category = problem.get("category", "Unknown")
    
    # Step 1: Research with Perplexity (if enabled)
    research_context = ""
    if use_research and PERPLEXITY_API_KEY:
        print(f"         Researching...", end=" ", flush=True)
        research = research_with_perplexity(title, category)
        if research:
            research_context = RESEARCH_CONTEXT_TEMPLATE.format(research=research)
            print("OK")
        else:
            print("SKIPPED")
    
    # Step 2: Prepare problem context
    context = prepare_problem_context(problem)
    
    # Step 3: Build full prompt with research
    full_prompt = f"""{SYSTEM_PROMPT}

{research_context}

## Problem to Analyze:
{context}

## Your Task:
Create 4-6 learning sub-quests that progressively teach all mathematical concepts needed to solve this problem.

REQUIREMENTS:
- Be EXTREMELY detailed with mathematical explanations (formal definitions, theorems, proofs)
- Each sub-quest MUST have a coding exercise with test_cases for validation
- Use proper LaTeX notation for ALL formulas
- Build concepts progressively (simpler to complex)
- DO NOT reveal the final solution code
- Include prerequisites, learning objectives, and common mistakes

Return ONLY the JSON array, no other text."""

    try:
        if backend == "copilot":
            # Use Copilot CLI
            content = generate_with_copilot_cli(full_prompt, model)
            if not content:
                return []
        else:
            # Use OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Find JSON array in content
        if not content.startswith("["):
            # Try to find the JSON array
            start_idx = content.find("[")
            if start_idx != -1:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i, c in enumerate(content[start_idx:], start_idx):
                    if c == "[":
                        bracket_count += 1
                    elif c == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                content = content[start_idx:end_idx]
        
        # First, convert literal \n to actual newlines (Copilot CLI output format issue)
        # Check if content starts with "[\" followed by "n" - indicating escaped format
        if content.startswith('[\\n') or (len(content) > 2 and content[0] == '[' and content[1] == '\\' and content[2] == 'n'):
            # This is an escaped JSON string, decode it
            try:
                content = content.encode('utf-8').decode('unicode_escape')
            except Exception:
                pass
        
        # Try to parse JSON with multiple fallback strategies
        import re
        for attempt in range(5):
            try:
                sub_quests = json.loads(content)
                return sub_quests
            except json.JSONDecodeError as e:
                if attempt == 0:
                    # Attempt 1: Try unicode_escape if not already done
                    try:
                        content = content.encode('utf-8').decode('unicode_escape')
                    except Exception:
                        pass
                elif attempt == 1:
                    # Attempt 2: Fix unescaped backslashes (common in LaTeX)
                    content = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', content)
                elif attempt == 2:
                    # Attempt 3: Fix newlines in strings
                    content = re.sub(r'(?<!\\)\n', r'\\n', content)
                elif attempt == 3:
                    # Attempt 4: Remove control characters
                    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
                else:
                    # Save failed content for debugging
                    with open("failed_parse.txt", "w", encoding="utf-8") as f:
                        f.write(f"Error: {e}\n\n")
                        f.write(content)
                    print(f"  [!] JSON parse error: {e}")
                    print(f"  [!] Failed content saved to failed_parse.txt")
                    # Also save to failed_quests.json
                    save_failed_quest(problem.get("id", 0), problem, f"JSON Parse Error: {e}", content)
                    return []
        
        return []
        
    except Exception as e:
        print(f"  [!] Error: {e}")
        return []


def load_failed_quests() -> dict:
    """Load the list of failed quests."""
    if FAILED_QUESTS_FILE.exists():
        with open(FAILED_QUESTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"failed": []}


def save_failed_quest(problem_id: int, problem: dict, error: str = "Unknown", raw_content: str = None) -> None:
    """Save a failed quest to the tracking file."""
    data = load_failed_quests()
    
    # Check if already in list - update if exists
    found = False
    for item in data["failed"]:
        if item["problem_id"] == problem_id:
            item["error"] = error
            item["timestamp"] = datetime.now().isoformat()
            if raw_content:
                item["raw_content"] = raw_content
            found = True
            break
            
    if not found:
        entry = {
            "problem_id": problem_id,
            "title": problem.get("title", "Unknown"),
            "category": problem.get("category", "Unknown"),
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        if raw_content:
            entry["raw_content"] = raw_content
        data["failed"].append(entry)

    with open(FAILED_QUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [!] Saved to {FAILED_QUESTS_FILE}")


def remove_from_failed(problem_id: int) -> None:
    """Remove a problem from the failed list after successful generation."""
    if not FAILED_QUESTS_FILE.exists():
        return
    data = load_failed_quests()
    data["failed"] = [q for q in data["failed"] if q["problem_id"] != problem_id]
    with open(FAILED_QUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_quest(problem_id: int, problem: dict, sub_quests: list[dict]) -> None:
    """Save quest data to file."""
    QUESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    quest_data = {
        "problem_id": problem_id,
        "title": problem.get("title", "Unknown"),
        "category": problem.get("category", "Unknown"),
        "difficulty": problem.get("difficulty", "Unknown"),
        "description": problem.get("description_decoded") or decode_base64(problem.get("description", "")),
        "example": problem.get("example", {}),
        "starter_code": problem.get("starter_code", ""),
        "sub_quests": sub_quests
    }
    
    filepath = QUESTS_DIR / f"quest_{problem_id:04d}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(quest_data, f, indent=2, ensure_ascii=False)


def display_quest(quest_data: dict) -> None:
    """Display a quest in readable format."""
    print("\n" + "=" * 70)
    print(f"Problem #{quest_data['problem_id']}: {quest_data['title']}")
    print(f"{quest_data['category']} | {quest_data['difficulty']}")
    print("=" * 70)
    
    print(f"\nDescription:\n{quest_data['description']}")
    
    if quest_data.get('example'):
        ex = quest_data['example']
        print(f"\nExample:")
        print(f"   Input: {ex.get('input', 'N/A')}")
        print(f"   Output: {ex.get('output', 'N/A')}")
    
    print("\n" + "-" * 70)
    print("LEARNING SUB-QUESTS")
    print("-" * 70)
    
    for quest in quest_data.get("sub_quests", []):
        step = quest.get("step", "?")
        title = quest.get("title", "Untitled")
        print(f"\nStep {step}: {title}")
        print("-" * 40)
        
        if quest.get("math_background"):
            print(f"Math Background:\n{quest['math_background']}")
        
        if quest.get("formula"):
            print(f"\nKey Formula: {quest['formula']}")
        
        if quest.get("hint"):
            print(f"\nHint: {quest['hint']}")
        
        if quest.get("questions_to_consider"):
            print("\nQuestions to Consider:")
            for q in quest["questions_to_consider"]:
                print(f"   - {q}")
    
    print("\n" + "=" * 70)


def generate_all_quests(client: OpenAI, model: str, backend: str = "copilot", start_id: int = 1, limit: int = None, category: str = None) -> None:
    """Generate quests for all problems."""
    problem_files = sorted(PROBLEMS_DIR.glob("problem_*.json"))
    
    if category:
        filtered_files = []
        for filepath in problem_files:
            with open(filepath, "r", encoding="utf-8") as f:
                problem = json.load(f)
                if category.lower() in problem.get("category", "").lower():
                    filtered_files.append(filepath)
        problem_files = filtered_files
        print(f"[*] Found {len(problem_files)} problems in category '{category}'")
    
    if limit:
        problem_files = problem_files[:limit]
    
    print(f"[*] Generating quests for {len(problem_files)} problems...")
    print("-" * 60)
    
    for filepath in problem_files:
        problem_id = int(filepath.stem.split("_")[1])
        
        if problem_id < start_id:
            continue
        
        quest_file = QUESTS_DIR / f"quest_{problem_id:04d}.json"
        if quest_file.exists():
            print(f"[{problem_id}] Quest already exists, skipping...")
            continue
        
        print(f"[{problem_id}] Loading problem...", end=" ", flush=True)
        problem = load_problem(problem_id)
        
        if not problem:
            print("NOT FOUND")
            continue
        
        title = problem.get("title", "Unknown")
        print(f"{title[:40]}...")
        
        print(f"         Generating sub-quests...", end=" ", flush=True)
        sub_quests = generate_sub_quests(problem, client, model, backend)
        
        if sub_quests:
            save_quest(problem_id, problem, sub_quests)
            remove_from_failed(problem_id)  # Remove from failed list if it was there
            print(f"OK ({len(sub_quests)} sub-quests)")
        else:
            print("FAILED")
            save_failed_quest(problem_id, problem, "Generation returned empty result")
    
    print("-" * 60)
    print("[+] Quest generation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate learning sub-quests for deep-ml problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for one problem (GitHub Models - uses Copilot quota)
  python quest_generator.py --id 1
  
  # Generate all Linear Algebra quests
  python quest_generator.py --category "Linear Algebra"
  
  # Use OpenAI instead
  python quest_generator.py --backend openai --id 1
"""
    )
    
    parser.add_argument("--id", "-i", type=int, help="Generate quest for a specific problem ID")
    parser.add_argument("--start", "-s", type=int, default=1, help="Start from this problem ID")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of problems")
    parser.add_argument("--category", "-c", type=str, help="Filter by category")
    parser.add_argument("--display", "-d", action="store_true", help="Display the generated quest")
    parser.add_argument("--all", action="store_true", help="Generate quests for all problems")
    parser.add_argument("--retry-failed", action="store_true", help="Retry all previously failed quests")
    parser.add_argument("--list-failed", action="store_true", help="List all failed quests")
    parser.add_argument("--backend", "-b", choices=["copilot", "github", "openai"], default="copilot",
                        help="Backend: copilot (CLI), github (API), or openai")
    parser.add_argument("--model", "-m", type=str, help="Model name (e.g., claude-sonnet-4.5, gpt-4o)")
    
    args = parser.parse_args()
    
    # Select model and backend
    backend = args.backend
    if args.model:
        model = args.model
    elif backend == "copilot":
        model = "claude-sonnet-4.5"  # Default Copilot CLI model
    elif backend == "github":
        model = "gpt-4o"
    else:
        model = "gpt-4o-mini"
    
    print(f"[*] Using {backend} backend with model: {model}")
    
    client = None
    if backend != "copilot":
        client = create_client(backend)
    
    # Handle --list-failed
    if args.list_failed:
        data = load_failed_quests()
        if not data["failed"]:
            print("[*] No failed quests found!")
        else:
            print(f"[*] Failed quests ({len(data['failed'])}):")
            print("-" * 60)
            for q in data["failed"]:
                print(f"  #{q['problem_id']:04d} | {q['category'][:20]:20} | {q['title'][:30]}")
            print("-" * 60)
            print(f"Run: python quest_generator.py --retry-failed")
        return
    
    # Handle --retry-failed
    if args.retry_failed:
        data = load_failed_quests()
        if not data["failed"]:
            print("[*] No failed quests to retry!")
            return
        
        print(f"[*] Retrying {len(data['failed'])} failed quests...")
        print("-" * 60)
        
        for item in data["failed"]:
            problem_id = item["problem_id"]
            print(f"[{problem_id}] Retrying...", end=" ", flush=True)
            problem = load_problem(problem_id)
            
            if not problem:
                print("NOT FOUND")
                continue
            
            print(f"{problem.get('title', 'Unknown')[:40]}...")
            print(f"         Generating sub-quests...", end=" ", flush=True)
            sub_quests = generate_sub_quests(problem, client, model, backend)
            
            if sub_quests:
                save_quest(problem_id, problem, sub_quests)
                remove_from_failed(problem_id)
                print(f"OK ({len(sub_quests)} sub-quests)")
            else:
                print("FAILED AGAIN")
        
        print("-" * 60)
        print("[+] Retry complete!")
        return
    
    if args.id:
        print(f"[*] Generating quest for problem #{args.id}...")
        problem = load_problem(args.id)
        
        if not problem:
            print(f"[!] Problem #{args.id} not found")
            sys.exit(1)
        
        print(f"[*] Problem: {problem.get('title', 'Unknown')}")
        print(f"[*] Generating sub-quests...")
        
        sub_quests = generate_sub_quests(problem, client, model, backend)
        
        if sub_quests:
            save_quest(args.id, problem, sub_quests)
            print(f"[+] Saved to quests/quest_{args.id:04d}.json")
            
            if args.display:
                quest_file = QUESTS_DIR / f"quest_{args.id:04d}.json"
                with open(quest_file, "r", encoding="utf-8") as f:
                    quest_data = json.load(f)
                display_quest(quest_data)
        else:
            print("[!] Failed to generate sub-quests")
            sys.exit(1)
    
    elif args.all or args.category:
        generate_all_quests(client, model, backend, start_id=args.start, limit=args.limit, category=args.category)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
