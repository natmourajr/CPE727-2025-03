#!/usr/bin/env python3
"""
Run all Fashion MNIST experiments with state tracking and auto-resume.

Usage:
    python run_all_experiments.py              # Run all experiments
    python run_all_experiments.py --reset      # Reset state and start over
    python run_all_experiments.py --status     # Show completion status
    python run_all_experiments.py --skip gmm   # Skip specific models
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


STATE_FILE = Path("experiments_state.json")


EXPERIMENTS = [
    # Fast models first
    {
        "id": "logistic_ovr",
        "name": "Logistic Regression OVR",
        "command": [".venv/bin/python", "src/train_with_pca.py", "--model", "logistic_ovr"],
        "estimated_time": "10-15 min",
    },
    {
        "id": "logistic_softmax",
        "name": "Logistic Regression Softmax",
        "command": [".venv/bin/python", "src/train_with_pca.py", "--model", "logistic_softmax"],
        "estimated_time": "10-15 min",
    },
    {
        "id": "naive_bayes_multinomial",
        "name": "Multinomial Naive Bayes",
        "command": [".venv/bin/python", "src/train_with_pca.py", "--model", "naive_bayes_multinomial"],
        "estimated_time": "5-10 min",
    },
    {
        "id": "random_forest",
        "name": "Random Forest",
        "command": [".venv/bin/python", "src/train_with_pca.py", "--model", "random_forest"],
        "estimated_time": "30-60 min",
    },
    {
        "id": "hierarchical",
        "name": "Hierarchical Classifiers + Flat Baselines",
        "command": [".venv/bin/python", "src/hierarchical_experiment.py"],
        "estimated_time": "30-60 min",
    },
    # Slow models last
    {
        "id": "gmm",
        "name": "Gaussian Mixture Model (SLOW)",
        "command": [".venv/bin/python", "src/train_with_pca.py", "--model", "gmm", "--pca-config", "pca_10"],
        "estimated_time": "60+ min (baseline will timeout, starting with pca_10)",
    },
]


def load_state():
    """Load experiment state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def save_state(state):
    """Save experiment state to file."""
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def clear_pytorch_cache():
    """Clear PyTorch cache to avoid import errors."""
    print("🔧 Clearing PyTorch cache...")

    # Clear src/ __pycache__ (where Python version mismatches occur)
    src_dir = Path("src")
    if src_dir.exists():
        subprocess.run(
            ["find", str(src_dir), "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["find", str(src_dir), "-name", "*.pyc", "-delete"],
            check=False,
            capture_output=True,
        )

    # Clear torch __pycache__ (auto-detect Python version)
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    cache_path = Path(f".venv/lib/python{py_version}/site-packages/torch/__pycache__")
    if cache_path.exists():
        subprocess.run(["rm", "-rf", str(cache_path)], check=False)

    # Clear torch/cuda __pycache__
    cuda_cache = Path(f".venv/lib/python{py_version}/site-packages/torch/cuda/__pycache__")
    if cuda_cache.exists():
        subprocess.run(["rm", "-rf", str(cuda_cache)], check=False)

    # Clear all .pyc files in torch package
    torch_dir = Path(f".venv/lib/python{py_version}/site-packages/torch")
    if torch_dir.exists():
        subprocess.run(
            ["find", str(torch_dir), "-name", "*.pyc", "-delete"],
            check=False,
            capture_output=True
        )

    print("✓ Cache cleared\n")


def run_experiment(exp, dry_run=False):
    """Run a single experiment."""
    print("=" * 80)
    print(f"📊 {exp['name']}")
    print("=" * 80)
    print(f"Estimated time: {exp['estimated_time']}")
    print(f"Command: {' '.join(exp['command'])}\n")

    if dry_run:
        print("(DRY RUN - not executing)\n")
        return True

    # Clear cache before running
    clear_pytorch_cache()

    # Set environment variables to avoid PyTorch CUDA issues
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["TORCH_USE_CUDA_DSA"] = "0"

    # Unset DYLD variables that can cause Python version conflicts
    # (especially DYLD_LIBRARY_PATH pointing to Homebrew Python 3.13)
    env.pop("DYLD_LIBRARY_PATH", None)
    env.pop("DYLD_FALLBACK_LIBRARY_PATH", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)

    # Run the command
    try:
        result = subprocess.run(
            exp["command"],
            check=True,
            text=True,
            env=env,
        )
        print(f"\n✅ {exp['name']} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {exp['name']} failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {exp['name']} interrupted by user\n")
        raise


def show_status(state, skip_list):
    """Show experiment completion status."""
    print("\n" + "=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80 + "\n")

    completed = state.get("completed", [])
    failed = state.get("failed", [])

    total = len([e for e in EXPERIMENTS if e["id"] not in skip_list])
    completed_count = len([e for e in completed if e not in skip_list])

    print(f"Progress: {completed_count}/{total} experiments completed\n")

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        status_icon = "⏭️ " if exp_id in skip_list else ""

        if exp_id in skip_list:
            status = "SKIPPED"
        elif exp_id in completed:
            status = "✅ COMPLETED"
        elif exp_id in failed:
            status = "❌ FAILED"
        else:
            status = "⏳ PENDING"

        print(f"{status_icon}{status:<15} {exp['name']}")
        print(f"                {exp['estimated_time']}")
        print()

    if state.get("last_updated"):
        print(f"Last updated: {state['last_updated']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all Fashion MNIST experiments with state tracking"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset state and start from beginning",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show experiment status and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Skip specific experiments (e.g., --skip gmm hierarchical)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only specific experiments (e.g., --only logistic_ovr random_forest)",
    )

    args = parser.parse_args()

    # Load state
    if args.reset:
        print("🔄 Resetting experiment state...\n")
        state = {"completed": [], "failed": [], "last_updated": None}
        save_state(state)
    else:
        state = load_state()

    # Filter experiments
    skip_list = set(args.skip)
    if args.only:
        # If --only is specified, skip everything not in the list
        only_set = set(args.only)
        skip_list.update(exp["id"] for exp in EXPERIMENTS if exp["id"] not in only_set)

    # Show status
    if args.status or args.dry_run:
        show_status(state, skip_list)
        if args.status:
            return

    # Run experiments
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENTS")
    print("=" * 80 + "\n")

    if args.dry_run:
        print("DRY RUN MODE - No commands will be executed\n")

    completed = state.get("completed", [])
    failed = state.get("failed", [])

    try:
        for exp in EXPERIMENTS:
            exp_id = exp["id"]

            # Skip if in skip list
            if exp_id in skip_list:
                print(f"⏭️  Skipping {exp['name']}\n")
                continue

            # Skip if already completed
            if exp_id in completed:
                print(f"✓ {exp['name']} already completed, skipping\n")
                continue

            # Run experiment
            success = run_experiment(exp, dry_run=args.dry_run)

            if not args.dry_run:
                if success:
                    if exp_id not in completed:
                        completed.append(exp_id)
                    if exp_id in failed:
                        failed.remove(exp_id)
                else:
                    if exp_id not in failed:
                        failed.append(exp_id)

                state["completed"] = completed
                state["failed"] = failed
                save_state(state)

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment run interrupted by user")
        print("Progress has been saved. Run again to resume.\n")
        save_state(state)
        sys.exit(1)

    # Final status
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80 + "\n")
    show_status(state, skip_list)

    if failed:
        print(f"⚠️  {len(failed)} experiment(s) failed. Review logs above.\n")
        sys.exit(1)
    else:
        print("🎉 All experiments completed successfully!\n")


if __name__ == "__main__":
    main()
