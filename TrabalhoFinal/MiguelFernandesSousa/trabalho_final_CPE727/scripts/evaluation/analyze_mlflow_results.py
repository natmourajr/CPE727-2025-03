#!/usr/bin/env python3
"""
Script to analyze MLflow experiments and generate summary for report
"""
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def read_metric(metric_file):
    """Read metric value from file"""
    try:
        with open(metric_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # MLflow metrics format: timestamp value step
                parts = lines[0].strip().split()
                if len(parts) >= 2:
                    return float(parts[1])
    except:
        pass
    return None

def read_param(param_file):
    """Read parameter value from file"""
    try:
        with open(param_file, 'r') as f:
            return f.read().strip()
    except:
        pass
    return None

def read_run_name(meta_file):
    """Extract run name from meta.yaml"""
    try:
        with open(meta_file, 'r') as f:
            for line in f:
                if line.startswith('run_name:'):
                    return line.split(':', 1)[1].strip()
    except:
        pass
    return None

def analyze_mlflow_experiments(mlruns_path):
    """Analyze all MLflow experiments"""

    results = defaultdict(lambda: defaultdict(list))

    mlruns_dir = Path(mlruns_path)

    # Iterate through all experiment directories
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ['.trash', 'models', '.DS_Store']:
            continue

        # Read experiment metadata
        exp_meta = exp_dir / 'meta.yaml'
        if not exp_meta.exists():
            continue

        exp_name = None
        with open(exp_meta, 'r') as f:
            for line in f:
                if line.startswith('name:'):
                    exp_name = line.split(':', 1)[1].strip()
                    break

        if not exp_name:
            continue

        print(f"\n=== Analyzing experiment: {exp_name} ===")

        # Iterate through all runs in this experiment
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name in ['models', 'tags']:
                continue

            meta_file = run_dir / 'meta.yaml'
            if not meta_file.exists():
                continue

            run_name = read_run_name(meta_file)
            if not run_name:
                continue

            print(f"  - Run: {run_name}")

            # Read all parameters
            params = {}
            params_dir = run_dir / 'params'
            if params_dir.exists():
                for param_file in params_dir.iterdir():
                    param_name = param_file.name
                    param_value = read_param(param_file)
                    if param_value:
                        params[param_name] = param_value

            # Read all metrics
            metrics = {}
            metrics_dir = run_dir / 'metrics'
            if metrics_dir.exists():
                for metric_file in metrics_dir.iterdir():
                    metric_name = metric_file.name
                    metric_value = read_metric(metric_file)
                    if metric_value is not None:
                        metrics[metric_name] = metric_value

            # Store results
            results[exp_name][run_name] = {
                'params': params,
                'metrics': metrics,
                'run_id': run_dir.name
            }

    return results

def print_summary(results):
    """Print summary of results"""

    print("\n" + "="*80)
    print("MLFLOW EXPERIMENTS SUMMARY")
    print("="*80)

    for exp_name, runs in sorted(results.items()):
        print(f"\n\n### Experiment: {exp_name}")
        print(f"Total runs: {len(runs)}")
        print("-"*80)

        # Sort runs by test accuracy if available
        sorted_runs = sorted(
            runs.items(),
            key=lambda x: x[1]['metrics'].get('test_accuracy', 0),
            reverse=True
        )

        for run_name, run_data in sorted_runs:
            print(f"\n  Run: {run_name}")

            # Print key parameters
            if run_data['params']:
                print("    Parameters:")
                for param, value in sorted(run_data['params'].items()):
                    print(f"      {param}: {value}")

            # Print metrics
            if run_data['metrics']:
                print("    Metrics:")
                for metric, value in sorted(run_data['metrics'].items()):
                    print(f"      {metric}: {value:.6f}")

def generate_latex_tables(results):
    """Generate LaTeX tables for the report"""

    print("\n" + "="*80)
    print("LATEX TABLES FOR REPORT")
    print("="*80)

    # Group by experiment type
    for exp_name, runs in sorted(results.items()):
        print(f"\n\n% Table for {exp_name}")
        print("\\begin{table}[h]")
        print("    \\centering")
        print(f"    \\caption{{Resultados do experimento: {exp_name}}}")
        print(f"    \\label{{tab:{exp_name.replace('-', '_')}}}")
        print("    \\small")
        print("    \\begin{tabular}{|l|c|c|c|c|c|}")
        print("        \\hline")
        print("        \\textbf{Modelo} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Tempo (s)} \\\\")
        print("        \\hline")

        # Sort by accuracy
        sorted_runs = sorted(
            runs.items(),
            key=lambda x: x[1]['metrics'].get('test_accuracy', 0),
            reverse=True
        )

        for run_name, run_data in sorted_runs:
            metrics = run_data['metrics']
            acc = metrics.get('test_accuracy', 0) * 100
            prec = metrics.get('test_precision_macro', 0) * 100
            rec = metrics.get('test_recall_macro', 0) * 100
            f1 = metrics.get('test_f1_macro', 0) * 100
            time = metrics.get('training_time_seconds', 0)

            # Format run name for display
            display_name = run_name.replace('_', '\\_')

            print(f"        {display_name} & {acc:.2f}\\% & {prec:.2f}\\% & {rec:.2f}\\% & {f1:.2f}\\% & {time:.2f} \\\\")

        print("        \\hline")
        print("    \\end{tabular}")
        print("\\end{table}")

if __name__ == '__main__':
    mlruns_path = 'results/mlruns'

    results = analyze_mlflow_experiments(mlruns_path)
    print_summary(results)
    generate_latex_tables(results)

    # Save results to JSON for further processing
    output_file = 'results/mlflow_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
