#!/usr/bin/env python3
"""
Collision rates and obstacle densities analysis from 4dof collision data.

This script:
1. Only processes 4dof collision files
2. Buckets collision events into time bins
3. Computes collision rates and obstacle densities using medians
4. Creates clear plots showing correlation over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import argparse
from collections import defaultdict

# Set style for clean plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
})

# Define colors
COLLISION_COLOR = '#E74C3C'  # Red
OBSTACLE_COLOR = '#3498DB'  # Blue
LEVEL2_COLOR = '#81B5A2'    # Teal
LEVEL4_COLOR = '#D4A574'    # Orange

class CollisionAnalyzer:
    """Analyzer for 4dof collision data."""
    
    def __init__(self, results_dir: str = "build/results"):
        self.results_dir = Path(results_dir)
        self.collision_data = {}
        
    def load_4dof_collision_data(self):
        """Load only 4dof collision files."""
        print("Loading 4dof collision data files...")
        
        # Find only 4dof collision CSV files
        pattern = "*4dof*collisions*.csv"
        csv_files = list(self.results_dir.glob(pattern))
        print(f"Found {len(csv_files)} 4dof collision CSV files")
        
        for file_path in csv_files:
            file_info = self._parse_filename(file_path.name)
            if file_info:
                try:
                    df = pd.read_csv(file_path)
                    key = (file_info['level'], file_info['seed'])
                    self.collision_data[key] = df
                    print(f"Loaded {len(df)} collision events from {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.collision_data)} datasets")
    
    def _parse_filename(self, filename: str):
        """Parse 4dof collision filename."""
        # Pattern: 4dof_level{level}_{seed}_collisions_{timestamp}.csv
        pattern = r'4dof_level(\d+)_(\d+)_collisions_[\d_]+\.csv'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'level': int(match.group(1)),
                'seed': int(match.group(2))
            }
        return None
    
    def process_data(self, time_bin_size: float = 30.0):
        """Process collision data using binning and medians."""
        print(f"Processing data with {time_bin_size}s time bins...")
        
        # Group by level
        level_data = defaultdict(list)
        for (level, seed), df in self.collision_data.items():
            if len(df) > 0 and 'time_s' in df.columns and 'obstacle_density_per_m2' in df.columns:
                level_data[level].append((seed, df))
        
        processed = {}
        
        for level, experiments in level_data.items():
            print(f"Processing level {level} with {len(experiments)} experiments...")
            
            # Process each experiment separately
            experiment_results = []
            
            for seed, df in experiments:
                # Clean data
                df_clean = df.dropna(subset=['time_s', 'obstacle_density_per_m2'])
                df_clean = df_clean[df_clean['time_s'] != '']
                
                # Convert to numeric
                df_clean['time_s'] = pd.to_numeric(df_clean['time_s'], errors='coerce')
                df_clean['obstacle_density_per_m2'] = pd.to_numeric(df_clean['obstacle_density_per_m2'], errors='coerce')
                df_clean = df_clean.dropna()
                
                if len(df_clean) == 0:
                    continue
                
                # Create time bins
                min_time = df_clean['time_s'].min()
                max_time = df_clean['time_s'].max()
                time_bins = np.arange(min_time, max_time + time_bin_size, time_bin_size)
                
                # Bin the data
                df_clean['time_bin'] = pd.cut(df_clean['time_s'], bins=time_bins, right=False)
                df_clean['time_bin_center'] = df_clean['time_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
                df_clean = df_clean.dropna(subset=['time_bin_center'])
                
                if len(df_clean) == 0:
                    continue
                
                # Group by time bin
                grouped = df_clean.groupby('time_bin_center', observed=True)
                collision_counts = grouped.size()
                collision_rates = collision_counts / time_bin_size  # collisions per second
                obstacle_densities = grouped['obstacle_density_per_m2'].median()  # median density
                
                # Store this experiment's results
                time_points = collision_rates.index.values
                experiment_results.append({
                    'seed': seed,
                    'time_points': time_points,
                    'collision_rates': collision_rates.values,
                    'obstacle_densities': obstacle_densities.values,
                    'total_collisions': len(df_clean)
                })
            
            if not experiment_results:
                continue
            
            # Aggregate across experiments using median
            all_time_points = set()
            for exp in experiment_results:
                all_time_points.update(exp['time_points'])
            
            common_time_points = sorted(all_time_points)
            
            # Align experiments to common time grid
            aligned_collision_rates = []
            aligned_obstacle_densities = []
            
            for exp in experiment_results:
                collision_aligned = np.full(len(common_time_points), np.nan)
                obstacle_aligned = np.full(len(common_time_points), np.nan)
                
                for i, tp in enumerate(common_time_points):
                    if tp in exp['time_points']:
                        idx = list(exp['time_points']).index(tp)
                        collision_aligned[i] = exp['collision_rates'][idx]
                        obstacle_aligned[i] = exp['obstacle_densities'][idx]
                
                aligned_collision_rates.append(collision_aligned)
                aligned_obstacle_densities.append(obstacle_aligned)
            
            # Compute medians across experiments
            aligned_collision_rates = np.array(aligned_collision_rates)
            aligned_obstacle_densities = np.array(aligned_obstacle_densities)
            
            collision_medians = np.nanmedian(aligned_collision_rates, axis=0)
            obstacle_medians = np.nanmedian(aligned_obstacle_densities, axis=0)
            
            # Store results
            processed[level] = {
                'time_points': np.array(common_time_points),
                'collision_rates': collision_medians,
                'obstacle_densities': obstacle_medians,
                'num_experiments': len(experiment_results),
                'total_collisions': sum(exp['total_collisions'] for exp in experiment_results)
            }
            
            print(f"Level {level}: {len(experiment_results)} experiments, "
                  f"{processed[level]['total_collisions']} total collisions, "
                  f"{len(common_time_points)} time bins")
        
        return processed
    
    def plot_results(self, processed_data, time_bin_size):
        """Create plots for each level and comparison."""
        
        # Individual level plots
        for level in sorted(processed_data.keys()):
            data = processed_data[level]
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Collision rates on left axis
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Collision Rate (collisions/s)', color=COLLISION_COLOR)
            ax1.plot(data['time_points'], data['collision_rates'], 
                    color=COLLISION_COLOR, linewidth=2, marker='o', markersize=4,
                    label='Collision Rate')
            ax1.tick_params(axis='y', labelcolor=COLLISION_COLOR)
            ax1.grid(True, alpha=0.3)
            
            # Obstacle density on right axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Obstacle Density (per m²)', color=OBSTACLE_COLOR)
            ax2.plot(data['time_points'], data['obstacle_densities'],
                    color=OBSTACLE_COLOR, linewidth=2, marker='s', markersize=4,
                    label='Obstacle Density')
            ax2.tick_params(axis='y', labelcolor=OBSTACLE_COLOR)
            
            # Title and legends (fix overlap by positioning)
            plt.title(f'Collision Rate vs Obstacle Density - Level {level}')
            
            # Position legends to avoid overlap
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Stats box
            stats_text = (f'Experiments: {data["num_experiments"]}\n'
                         f'Total Collisions: {data["total_collisions"]}\n'
                         f'Time Bins: {len(data["time_points"])}\n'
                         f'Median Collision Rate: {np.nanmedian(data["collision_rates"]):.4f}/s\n'
                         f'Median Obstacle Density: {np.nanmedian(data["obstacle_densities"]):.4f}/m²')
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            filename = f'collision_density_level{level}.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
            plt.close()
        
        # Comparison plot
        if len(processed_data) >= 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            levels = sorted(processed_data.keys())
            colors = [LEVEL2_COLOR if level == 2 else LEVEL4_COLOR for level in levels]
            
            # Plot 1: Collision rates
            for i, level in enumerate(levels):
                data = processed_data[level]
                ax1.plot(data['time_points'], data['collision_rates'], 
                        color=colors[i], linewidth=2, marker='o', markersize=3,
                        label=f'Level {level} (n={data["num_experiments"]})')
            ax1.set_ylabel('Collision Rate (collisions/s)')
            ax1.set_title('Collision Rates Over Time (Medians)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Obstacle densities
            for i, level in enumerate(levels):
                data = processed_data[level]
                ax2.plot(data['time_points'], data['obstacle_densities'],
                        color=colors[i], linewidth=2, marker='s', markersize=3,
                        label=f'Level {level} (n={data["num_experiments"]})')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Obstacle Density (per m²)')
            ax2.set_title('Obstacle Densities Over Time (Medians)')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = 'collision_density_comparison.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
            plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Collision density analysis for 4dof data')
    parser.add_argument('--results-dir', default='build/results',
                       help='Directory containing results (default: build/results)')
    parser.add_argument('--time-bin', type=float, default=60.0,
                       help='Time bin size in seconds (default: 60.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found!")
        return
    
    analyzer = CollisionAnalyzer(args.results_dir)
    analyzer.load_4dof_collision_data()
    processed_data = analyzer.process_data(args.time_bin)
    analyzer.plot_results(processed_data, args.time_bin)
    
    print("Collision density analysis completed!")


if __name__ == "__main__":
    main()
