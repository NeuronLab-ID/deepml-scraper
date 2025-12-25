#!/usr/bin/env python3
"""
Deep-ML Learning CLI - Interactive problem solver with code validation.

Features:
- Browse problems with math hints
- Validate your code against test cases
- Track progress
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()

PROBLEMS_DIR = Path("problems")
QUESTS_DIR = Path("quests")
SOLUTIONS_DIR = Path("solutions")
SOLUTIONS_DIR.mkdir(exist_ok=True)


def load_quest(problem_id: int) -> Optional[dict]:
    """Load quest (with sub-quests) for a problem."""
    filepath = QUESTS_DIR / f"quest_{problem_id:04d}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_subquest(code: str, test_cases: list[dict]) -> dict:
    """Validate user code against sub-quest test cases."""
    results = []
    passed = 0
    
    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        expected = tc.get("expected", "").strip()
        
        # Create test code
        full_code = f"{code}\n\nprint({test_input})"
        
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(full_code)
                temp_file = f.name
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            actual = result.stdout.strip()
            error = result.stderr.strip()
            os.unlink(temp_file)
            
            if error and not actual:
                test_passed = False
                actual = f"ERROR: {error[:150]}"
            else:
                test_passed = actual == expected
            
            if test_passed:
                passed += 1
            
            results.append({
                "test_num": i + 1,
                "input": test_input,
                "expected": expected,
                "actual": actual,
                "passed": test_passed
            })
            
        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            results.append({
                "test_num": i + 1,
                "input": test_input,
                "expected": expected,
                "actual": "TIMEOUT",
                "passed": False
            })
        except Exception as e:
            results.append({
                "test_num": i + 1,
                "input": test_input,
                "expected": expected,
                "actual": f"ERROR: {str(e)[:100]}",
                "passed": False
            })
    
    return {
        "passed": passed,
        "failed": len(test_cases) - passed,
        "total": len(test_cases),
        "results": results
    }


def display_subquest(subquest: dict, step_num: int, total_steps: int) -> None:
    """Display a single sub-quest."""
    console.print(Panel(
        f"[bold cyan]Step {step_num}/{total_steps}: {subquest.get('title', 'Untitled')}[/bold cyan]",
        border_style="cyan"
    ))
    
    # Math background
    if subquest.get("math_background"):
        console.print("\n[bold]📚 Math Background:[/bold]")
        console.print(subquest["math_background"])
    
    # Formula
    if subquest.get("formula"):
        console.print(f"\n[bold]📐 Formula:[/bold] {subquest['formula']}")
    
    # Exercise
    exercise = subquest.get("exercise", {})
    if exercise:
        console.print(f"\n[bold]✏️ Exercise:[/bold] {exercise.get('description', '')}")
        if exercise.get("starter_code"):
            console.print(Syntax(exercise["starter_code"], "python", theme="monokai", line_numbers=True))
    
    # Hint
    if subquest.get("hint"):
        console.print(f"\n[dim]💡 Hint: {subquest['hint']}[/dim]")



def decode_base64(encoded: str) -> str:
    """Decode base64 string."""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return encoded


def load_problem(problem_id: int) -> Optional[dict]:
    """Load problem from file."""
    filepath = PROBLEMS_DIR / f"problem_{problem_id:04d}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_problems_summary() -> list[dict]:
    """Load problems summary."""
    filepath = PROBLEMS_DIR / "problems_summary.json"
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def display_problem(problem: dict, show_hints: bool = True) -> None:
    """Display a problem with optional hints."""
    title = problem.get("title", "Unknown")
    category = problem.get("category", "Unknown")
    difficulty = problem.get("difficulty", "Unknown")
    description = problem.get("description_decoded") or decode_base64(problem.get("description", ""))
    
    # Header
    console.print(Panel(
        f"[bold cyan]{title}[/bold cyan]\n"
        f"[dim]{category} | {difficulty}[/dim]",
        title="🎯 Problem",
        border_style="cyan"
    ))
    
    # Description
    console.print("\n[bold]📝 Description:[/bold]")
    console.print(description)
    
    # Example
    example = problem.get("example", {})
    if example:
        console.print("\n[bold]📌 Example:[/bold]")
        console.print(f"  Input: [green]{example.get('input', 'N/A')}[/green]")
        console.print(f"  Output: [yellow]{example.get('output', 'N/A')}[/yellow]")
        if example.get("reasoning"):
            console.print(f"  Reasoning: [dim]{example.get('reasoning')}[/dim]")
    
    # Math hints (from learn_section if available)
    if show_hints:
        learn_section = problem.get("learn_section_decoded") or decode_base64(problem.get("learn_section", ""))
        if learn_section:
            console.print("\n[bold]📚 Math Background:[/bold]")
            console.print(Markdown(learn_section))
    
    # Starter code
    starter_code = problem.get("starter_code", "")
    if starter_code:
        console.print("\n[bold]💻 Starter Code:[/bold]")
        console.print(Syntax(starter_code, "python", theme="monokai", line_numbers=True))


def validate_code(problem: dict, code: str) -> dict:
    """
    Validate user code against test cases.
    
    Returns:
        {
            "passed": int,
            "failed": int,
            "total": int,
            "results": [{"test": ..., "expected": ..., "actual": ..., "passed": bool}]
        }
    """
    test_cases = problem.get("test_cases", [])
    results = []
    passed = 0
    
    for i, tc in enumerate(test_cases):
        test_code = tc.get("test", "")
        expected = tc.get("expected_output", "").strip()
        
        # Create full code with user's solution
        full_code = f"{code}\n\n{test_code}"
        
        try:
            # Run in subprocess for safety
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(full_code)
                temp_file = f.name
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            actual = result.stdout.strip()
            error = result.stderr.strip()
            
            os.unlink(temp_file)
            
            if error and not actual:
                test_passed = False
                actual = f"ERROR: {error[:200]}"
            else:
                test_passed = actual == expected
            
            if test_passed:
                passed += 1
            
            results.append({
                "test_num": i + 1,
                "test": test_code[:100] + "..." if len(test_code) > 100 else test_code,
                "expected": expected,
                "actual": actual,
                "passed": test_passed
            })
            
        except subprocess.TimeoutExpired:
            results.append({
                "test_num": i + 1,
                "test": test_code[:100],
                "expected": expected,
                "actual": "TIMEOUT (>10s)",
                "passed": False
            })
            os.unlink(temp_file)
        except Exception as e:
            results.append({
                "test_num": i + 1,
                "test": test_code[:100],
                "expected": expected,
                "actual": f"ERROR: {str(e)[:100]}",
                "passed": False
            })
    
    return {
        "passed": passed,
        "failed": len(test_cases) - passed,
        "total": len(test_cases),
        "results": results
    }


def display_validation_results(results: dict) -> None:
    """Display validation results in a nice table."""
    passed = results["passed"]
    total = results["total"]
    
    if passed == total:
        console.print(Panel(
            f"[bold green]✅ ALL TESTS PASSED! ({passed}/{total})[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️ {passed}/{total} tests passed[/bold yellow]",
            border_style="yellow"
        ))
    
    # Results table
    table = Table(title="Test Results")
    table.add_column("#", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Expected", style="green")
    table.add_column("Actual", style="yellow")
    
    for r in results["results"]:
        status = "[green]✓[/green]" if r["passed"] else "[red]✗[/red]"
        actual_style = "green" if r["passed"] else "red"
        table.add_row(
            str(r["test_num"]),
            status,
            r["expected"][:50],
            f"[{actual_style}]{r['actual'][:50]}[/{actual_style}]"
        )
    
    console.print(table)


def save_solution(problem_id: int, code: str) -> None:
    """Save user's solution."""
    filepath = SOLUTIONS_DIR / f"solution_{problem_id:04d}.py"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    console.print(f"[dim]💾 Solution saved to {filepath}[/dim]")


def load_solution(problem_id: int) -> Optional[str]:
    """Load user's previous solution."""
    filepath = SOLUTIONS_DIR / f"solution_{problem_id:04d}.py"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return None


def interactive_solve(problem_id: int) -> None:
    """Interactive problem solving mode."""
    problem = load_problem(problem_id)
    if not problem:
        console.print(f"[red]Problem #{problem_id} not found![/red]")
        return
    
    console.clear()
    display_problem(problem, show_hints=True)
    
    # Load previous solution if exists
    prev_solution = load_solution(problem_id)
    if prev_solution:
        console.print("\n[dim]📁 Previous solution found![/dim]")
    
    console.print("\n" + "=" * 60)
    console.print("[bold]🔧 SOLVE MODE[/bold]")
    console.print("=" * 60)
    console.print("""
Commands:
  [cyan]validate <file.py>[/cyan] - Validate your solution file
  [cyan]validate[/cyan]           - Validate solution from solutions/ folder
  [cyan]hint[/cyan]               - Show math background again
  [cyan]code[/cyan]               - Show starter code
  [cyan]quit[/cyan]               - Exit
""")
    
    while True:
        try:
            cmd = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip().lower()
            
            if cmd == "quit" or cmd == "q":
                break
            
            elif cmd == "hint":
                learn_section = problem.get("learn_section_decoded") or decode_base64(problem.get("learn_section", ""))
                if learn_section:
                    console.print(Markdown(learn_section))
                else:
                    console.print("[dim]No hints available for this problem[/dim]")
            
            elif cmd == "code":
                console.print(Syntax(problem.get("starter_code", ""), "python", theme="monokai", line_numbers=True))
            
            elif cmd.startswith("validate"):
                parts = cmd.split(maxsplit=1)
                
                if len(parts) > 1:
                    # Validate specific file
                    filepath = Path(parts[1])
                else:
                    # Try solutions folder
                    filepath = SOLUTIONS_DIR / f"solution_{problem_id:04d}.py"
                
                if not filepath.exists():
                    console.print(f"[red]File not found: {filepath}[/red]")
                    console.print(f"[dim]Create your solution at: {SOLUTIONS_DIR / f'solution_{problem_id:04d}.py'}[/dim]")
                    continue
                
                console.print(f"[dim]Validating {filepath}...[/dim]")
                
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                
                results = validate_code(problem, code)
                display_validation_results(results)
                
                if results["passed"] == results["total"]:
                    save_solution(problem_id, code)
            
            else:
                console.print("[dim]Unknown command. Try: validate, hint, code, quit[/dim]")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def list_problems(category: str = None, difficulty: str = None) -> None:
    """List all problems with optional filtering."""
    summary = load_problems_summary()
    
    if category:
        summary = [p for p in summary if category.lower() in p.get("category", "").lower()]
    if difficulty:
        summary = [p for p in summary if p.get("difficulty", "").lower() == difficulty.lower()]
    
    table = Table(title="Deep-ML Problems")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Title", style="white")
    table.add_column("Category", style="green")
    table.add_column("Difficulty", style="yellow")
    
    for p in summary[:50]:  # Limit display
        pid = p.get("id") or "?"
        table.add_row(
            str(pid),
            p.get("title", "Unknown")[:50],
            p.get("category", "N/A"),
            p.get("difficulty", "N/A")
        )
    
    console.print(table)
    if len(summary) > 50:
        console.print(f"[dim]... and {len(summary) - 50} more[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Deep-ML Learning CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List problems")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--difficulty", "-d", help="Filter by difficulty")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a problem")
    solve_parser.add_argument("problem_id", type=int, help="Problem ID")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate solution")
    validate_parser.add_argument("problem_id", type=int, help="Problem ID")
    validate_parser.add_argument("--file", "-f", help="Solution file path")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show problem details")
    show_parser.add_argument("problem_id", type=int, help="Problem ID")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_problems(args.category, args.difficulty)
    
    elif args.command == "solve":
        interactive_solve(args.problem_id)
    
    elif args.command == "validate":
        problem = load_problem(args.problem_id)
        if not problem:
            console.print(f"[red]Problem #{args.problem_id} not found![/red]")
            sys.exit(1)
        
        filepath = Path(args.file) if args.file else SOLUTIONS_DIR / f"solution_{args.problem_id:04d}.py"
        
        if not filepath.exists():
            console.print(f"[red]Solution file not found: {filepath}[/red]")
            sys.exit(1)
        
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        
        console.print(f"[bold]Validating Problem #{args.problem_id}: {problem.get('title')}[/bold]\n")
        results = validate_code(problem, code)
        display_validation_results(results)
    
    elif args.command == "show":
        problem = load_problem(args.problem_id)
        if problem:
            display_problem(problem)
        else:
            console.print(f"[red]Problem #{args.problem_id} not found![/red]")
    
    else:
        parser.print_help()
        console.print("\n[bold]Quick Start:[/bold]")
        console.print("  python learning_cli.py list          # List all problems")
        console.print("  python learning_cli.py solve 1       # Start solving problem #1")
        console.print("  python learning_cli.py validate 1    # Validate your solution")


if __name__ == "__main__":
    main()
