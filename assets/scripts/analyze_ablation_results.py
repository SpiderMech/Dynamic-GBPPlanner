#!/usr/bin/env python3
"""
Analyze ablation study results from summary files.
Usage: 
  python analyze_results.py [results_directory]          # Auto-detect baseline and variants
  python analyze_results.py [results_directory] --variant # Force variant mode (show relative changes)
  python analyze_results.py [results_directory] --baseline # Force baseline mode (show raw values)
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
from collections import defaultdict

def read_summary_files_by_variant(results_dir):
    """Read summary files grouped by variant."""
    pattern = os.path.join(results_dir, "*summary*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No summary files found in {results_dir}")
        return {}
    
    # Group files by variant (extract variant from filename)
    variants = defaultdict(list)
    
    for file in files:
        basename = os.path.basename(file)
        # Extract variant: ab4dof_s1_full_2_summary_timestamp.csv -> ab4dof_s1_full
        parts = basename.split('_')
        if len(parts) >= 6:  # ab4dof, s1, full/nodof, seed, summary, timestamp
            # Find the position of 'summary' to know where to cut
            try:
                summary_idx = parts.index('summary')
                variant = '_'.join(parts[:summary_idx-1])  # Everything before seed number
                variants[variant].append(file)
            except ValueError:
                # Fallback if 'summary' not found
                if len(parts) >= 4:
                    variant = '_'.join(parts[:-3])
                    variants[variant].append(file)
    
    return dict(variants)

def calculate_metrics(files):
    """Calculate metrics from a list of summary files."""
    if not files:
        return None
    
    # Read and combine all files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return None
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Calculate per-seed collision rates for proper statistics
    seed_collision_rates = []
    for file in files:
        try:
            seed_df = pd.read_csv(file)
            seed_collided = seed_df['collided'].sum()
            seed_duration = seed_df['duration_s'].max() / 60.0  # Convert to minutes
            if seed_duration > 0:
                seed_collision_rates.append(seed_collided / seed_duration)
            else:
                seed_collision_rates.append(0)
        except:
            continue
    
    metrics = {}
    
    # 1. Collision rate per minute
    if seed_collision_rates:
        metrics['coll_median'] = np.median(seed_collision_rates)
        metrics['coll_q1'] = np.percentile(seed_collision_rates, 25)
        metrics['coll_q3'] = np.percentile(seed_collision_rates, 75)
    else:
        metrics['coll_median'] = metrics['coll_q1'] = metrics['coll_q3'] = 0
    
    # 2. Success rate
    total_robots = len(df)
    total_collided = df['collided'].sum()
    metrics['success_rate'] = (total_robots - total_collided) / total_robots * 100 if total_robots > 0 else 100
    
    # 3. Minimum clearance P5
    min_clearances = df['min_clearance_m'].dropna()
    min_clearances = min_clearances[np.isfinite(min_clearances)]
    
    if len(min_clearances) > 0:
        metrics['minclr_p5'] = np.percentile(min_clearances, 5)
        metrics['minclr_q1'] = np.percentile(min_clearances, 25)
        metrics['minclr_q3'] = np.percentile(min_clearances, 75)
        metrics['minclr_median'] = np.median(min_clearances)
    else:
        metrics['minclr_p5'] = metrics['minclr_q1'] = metrics['minclr_q3'] = metrics['minclr_median'] = np.nan
    
    # 4. Detour ratio
    detour_ratios = df['detour_ratio'].dropna()
    detour_ratios = detour_ratios[np.isfinite(detour_ratios)]
    detour_ratios = detour_ratios[detour_ratios > 0]
    metrics['detour_median'] = np.median(detour_ratios) if len(detour_ratios) > 0 else np.nan
    
    # 5. LDJ
    ldj_values = df['ldj'].dropna()
    ldj_values = ldj_values[np.isfinite(ldj_values)]
    metrics['ldj_median'] = np.median(ldj_values) if len(ldj_values) > 0 else np.nan
    
    # Additional info
    metrics['total_robots'] = total_robots
    metrics['total_seeds'] = len(files)
    metrics['total_collided'] = total_collided
    
    return metrics

def format_baseline_output(metrics):
    """Format baseline output with raw values."""
    return (f"{metrics['coll_median']:.2f} [{metrics['coll_q1']:.2f},{metrics['coll_q3']:.2f}] & "
           f"{metrics['success_rate']:.0f} & "
           f"{metrics['minclr_p5']:.2f} [{metrics['minclr_q1']:.2f},{metrics['minclr_q3']:.2f}] & "
           f"{metrics['detour_median']:.2f} & "
           f"{metrics['ldj_median']:.1f} \\\\")

def format_variant_output(baseline_metrics, variant_metrics):
    """Format variant output with relative changes."""
    parts = []
    
    # Collision rate change (percentage increase)
    if baseline_metrics['coll_median'] > 0:
        coll_change_pct = ((variant_metrics['coll_median'] - baseline_metrics['coll_median']) / baseline_metrics['coll_median']) * 100
        if coll_change_pct > 0:
            parts.append(f"\\textbf{{+{coll_change_pct:.0f}\\%}}")
        else:
            parts.append(f"{coll_change_pct:.0f}\\%")
    else:
        # Handle case where baseline is 0
        if variant_metrics['coll_median'] > 0:
            parts.append("\\textbf{+∞\\%}")
        else:
            parts.append("0\\%")
    
    # Success rate change (absolute difference)
    success_change = variant_metrics['success_rate'] - baseline_metrics['success_rate']
    if success_change < 0:
        parts.append(f"\\textbf{{{success_change:.0f}}}")
    else:
        parts.append(f"+{success_change:.0f}")
    
    # Min clearance change (absolute difference in P5)
    minclr_change = variant_metrics['minclr_p5'] - baseline_metrics['minclr_p5']
    if minclr_change < 0:
        parts.append(f"\\textbf{{{minclr_change:.2f}}}")
    else:
        parts.append(f"+{minclr_change:.2f}")
    
    # Detour ratio change (absolute difference)
    detour_change = variant_metrics['detour_median'] - baseline_metrics['detour_median']
    if detour_change > 0:
        parts.append(f"+{detour_change:.2f}")
    else:
        parts.append(f"{detour_change:.2f}")
    
    # LDJ change (absolute difference)
    ldj_change = variant_metrics['ldj_median'] - baseline_metrics['ldj_median']
    if ldj_change > 0:
        parts.append(f"+{ldj_change:.1f}")
    else:
        parts.append(f"{ldj_change:.1f}")
    
    return " & ".join(parts) + " \\\\"

def print_detailed_stats(variant_name, metrics):
    """Print detailed statistics for debugging."""
    print(f"\n=== {variant_name} ===")
    print(f"Total robots:     {metrics['total_robots']}")
    print(f"Seeds analyzed:   {metrics['total_seeds']}")
    print(f"Collided robots:  {metrics['total_collided']} ({100*metrics['total_collided']/metrics['total_robots']:.1f}%)")
    print(f"Coll./min:        {metrics['coll_median']:.2f} [{metrics['coll_q1']:.2f},{metrics['coll_q3']:.2f}]")
    print(f"Success %:        {metrics['success_rate']:.0f}")
    print(f"MinClr p5 [m]:    {metrics['minclr_p5']:.2f} [{metrics['minclr_q1']:.2f},{metrics['minclr_q3']:.2f}]")
    print(f"Detour:           {metrics['detour_median']:.2f}")
    print(f"LDJ:              {metrics['ldj_median']:.1f}")

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "build/results"
    
    force_mode = None
    if len(sys.argv) > 2:
        if sys.argv[2] == "--baseline":
            force_mode = "baseline"
        elif sys.argv[2] == "--variant":
            force_mode = "variant"
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # Read all variants
    variants = read_summary_files_by_variant(results_dir)
    
    if not variants:
        print("No valid data found!")
        return
    
    print(f"Found variants: {list(variants.keys())}")
    
    # Calculate metrics for all variants
    all_metrics = {}
    for variant_name, files in variants.items():
        metrics = calculate_metrics(files)
        if metrics:
            all_metrics[variant_name] = metrics
            print_detailed_stats(variant_name, metrics)
    
    # Determine if this is baseline or variant analysis
    baseline_variants = [v for v in all_metrics.keys() if 'full' in v.lower()]
    variant_variants = [v for v in all_metrics.keys() if v not in baseline_variants]
    
    print(f"\n{'='*60}")
    print("TABLE FORMAT (copy-paste ready)")
    print(f"{'='*60}")
    
    if force_mode == "baseline" or (force_mode is None and len(baseline_variants) > 0 and len(variant_variants) == 0):
        # Baseline mode - show raw values
        for variant_name in baseline_variants:
            metrics = all_metrics[variant_name]
            output = format_baseline_output(metrics)
            print(f"\\quad Full & {output}")
    
    elif force_mode == "variant" or (force_mode is None and len(baseline_variants) > 0 and len(variant_variants) > 0):
        # Variant mode - show relative changes
        if len(baseline_variants) == 0:
            print("Error: No baseline variant found (looking for 'full' in name)")
            return
        
        # Use the first baseline found
        baseline_name = baseline_variants[0]
        baseline_metrics = all_metrics[baseline_name]
        
        print(f"\\quad Full & {format_baseline_output(baseline_metrics)}")
        
        for variant_name in variant_variants:
            variant_metrics = all_metrics[variant_name]
            # Create a nice display name for the variant
            display_name = variant_name.replace('_', ' ').replace('nodof', '- Dynamics').replace('noint', '- Inter-robot')
            output = format_variant_output(baseline_metrics, variant_metrics)
            print(f"\\quad {display_name} & {output}")
    
    else:
        print("Could not determine analysis mode. Use --baseline or --variant to force a mode.")

if __name__ == "__main__":
    main()
