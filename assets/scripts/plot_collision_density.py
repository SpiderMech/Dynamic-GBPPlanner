#!/usr/bin/env python3
"""
Plot collision rates and obstacle densities over time from GBPPlanner collision event data.

This script analyzes collision event data stored in build/results and creates time series plots for:
1. Collision rates over time for level2 and level4
2. Obstacle densities over time for level2 and level4
3. Both metrics on the same plot with dual y-axes to show correlation

Data files follow the naming pattern: {experiment}_level{level}_{seed}_collisions_{timestamp}.csv
The script pools data from multiple seeds for each level and computes aggregate statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

# Set style for academic plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# Define colors for consistency
LEVEL2_COLOR = '#81B5A2'  # Teal
LEVEL4_COLOR = '#D4A574'  # Orange/tan
COLLISION_COLOR = '#E74C3C'  # Red
OBSTACLE_COLOR = '#3498DB'  # Blue

class CollisionDensityPlotter:
    """Class to handle loading and plotting of collision event data."""
    
    def __init__(self, results_dir: str = "build/results"):
        """Initialize with results directory path."""
        self.results_dir = Path(results_dir)
        self.collision_data = {}
        
    def load_collision_data(self):
        """Load all collision event data files."""
        print("Loading collision data files...")
        
        # Find all collision CSV files
        csv_files = list(self.results_dir.glob("*collisions*.csv"))
        print(f"Found {len(csv_files)} collision CSV files")
        
        for file_path in csv_files:
            file_info = self._parse_collision_filename(file_path.name)
            if file_info:
                try:
                    df = pd.read_csv(file_path)
                    key = (file_info['level'], file_info['seed'])
                    self.collision_data[key] = df
                    print(f"Loaded {len(df)} collision events from {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Failed to parse filename: {file_path.name}")
        
        print(f"Loaded {len(self.collision_data)} collision datasets")
    
    def _parse_collision_filename(self, filename: str) -> Dict:
        """Parse collision filename to extract experiment info."""
        # Pattern: {experiment}_level{level}_{seed}_collisions_{timestamp}.csv
        pattern = r'([^_]+)_level(\d+)_(\d+)_collisions_[\d_]+\.csv'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'experiment': match.group(1),
                'level': int(match.group(2)),
                'seed': int(match.group(3))
            }
        return None
    
    def process_collision_density_data(self, time_bin_size: float = 10.0) -> Dict:
        """Process collision data to compute rates and densities over time across multiple experiments."""
        print(f"Processing collision and density data with {time_bin_size}s time bins...")
        
        # Group data by level
        level_data = defaultdict(dict)
        for (level, seed), df in self.collision_data.items():
            if len(df) > 0:
                level_data[level][seed] = df
        
        processed_data = {}
        
        for level, seed_dfs in level_data.items():
            print(f"Processing level {level} data from {len(seed_dfs)} experiment runs...")
            
            # Process each experiment run separately first
            experiment_results = []
            all_time_points = set()
            
            for seed, df in seed_dfs.items():
                # Check if the required columns exist
                if 'time_s' not in df.columns:
                    print(f"    Warning: Skipping seed {seed}: missing 'time_s' column")
                    continue
                if 'obstacle_density_per_m2' not in df.columns:
                    print(f"    Warning: Skipping seed {seed}: missing 'obstacle_density_per_m2' column")
                    continue
                    
                # Clean and prepare data for this experiment
                df_clean = df.copy()
                df_clean = df_clean.dropna(subset=['time_s'])
                df_clean = df_clean[df_clean['time_s'] != '']
                
                if len(df_clean) == 0:
                    continue
                    
                # Convert to numeric
                df_clean['time_s'] = pd.to_numeric(df_clean['time_s'], errors='coerce')
                df_clean['obstacle_density_per_m2'] = pd.to_numeric(df_clean['obstacle_density_per_m2'], errors='coerce')
                df_clean = df_clean.dropna(subset=['time_s', 'obstacle_density_per_m2'])
                
                if len(df_clean) == 0:
                    continue
                
                # Create time bins for this experiment
                min_time = df_clean['time_s'].min()
                max_time = df_clean['time_s'].max()
                time_bins = np.arange(min_time, max_time + time_bin_size, time_bin_size)
                
                # Bin the data
                df_clean['time_bin'] = pd.cut(df_clean['time_s'], bins=time_bins, right=False)
                df_clean['time_bin_center'] = df_clean['time_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
                df_clean = df_clean.dropna(subset=['time_bin_center'])
                
                if len(df_clean) == 0:
                    continue
                
                # Group by time bin and compute statistics for this experiment
                grouped = df_clean.groupby('time_bin_center', observed=True)
                collision_counts = grouped.size()
                collision_rates = collision_counts / time_bin_size
                obstacle_densities = grouped['obstacle_density_per_m2'].mean()
                
                # Store results for this experiment
                time_points = collision_rates.index.values
                all_time_points.update(time_points)
                
                experiment_results.append({
                    'seed': seed,
                    'time_points': time_points,
                    'collision_rates': collision_rates.values,
                    'obstacle_densities': obstacle_densities.values,
                    'total_collisions': len(df_clean),
                    'time_range': (min_time, max_time)
                })
            
            if not experiment_results:
                print(f"No valid data for level {level}")
                continue
            
            # Create a common time grid for all experiments
            common_time_points = sorted(all_time_points)
            
            # Interpolate/align all experiments to common time points
            aligned_collision_rates = []
            aligned_obstacle_densities = []
            
            for exp_result in experiment_results:
                # Create arrays filled with NaN for missing time points
                collision_rate_interp = np.full(len(common_time_points), np.nan)
                obstacle_density_interp = np.full(len(common_time_points), np.nan)
                
                # Fill in the values we have
                for i, time_point in enumerate(common_time_points):
                    if time_point in exp_result['time_points']:
                        idx = list(exp_result['time_points']).index(time_point)
                        collision_rate_interp[i] = exp_result['collision_rates'][idx]
                        obstacle_density_interp[i] = exp_result['obstacle_densities'][idx]
                
                aligned_collision_rates.append(collision_rate_interp)
                aligned_obstacle_densities.append(obstacle_density_interp)
            
            # Convert to numpy arrays for easier statistical computation
            aligned_collision_rates = np.array(aligned_collision_rates)
            aligned_obstacle_densities = np.array(aligned_obstacle_densities)
            
            # Compute statistics across experiments (ignoring NaN values)
            collision_rate_mean = np.nanmean(aligned_collision_rates, axis=0)
            collision_rate_std = np.nanstd(aligned_collision_rates, axis=0)
            collision_rate_sem = collision_rate_std / np.sqrt(np.sum(~np.isnan(aligned_collision_rates), axis=0))
            
            obstacle_density_mean = np.nanmean(aligned_obstacle_densities, axis=0)
            obstacle_density_std = np.nanstd(aligned_obstacle_densities, axis=0)
            obstacle_density_sem = obstacle_density_std / np.sqrt(np.sum(~np.isnan(aligned_obstacle_densities), axis=0))
            
            # Store processed data
            processed_data[level] = {
                'time_points': np.array(common_time_points),
                'collision_rates_mean': collision_rate_mean,
                'collision_rates_std': collision_rate_std,
                'collision_rates_sem': collision_rate_sem,
                'obstacle_densities_mean': obstacle_density_mean,
                'obstacle_densities_std': obstacle_density_std,
                'obstacle_densities_sem': obstacle_density_sem,
                'num_experiments': len(experiment_results),
                'total_collisions_per_exp': [exp['total_collisions'] for exp in experiment_results],
                'time_ranges': [exp['time_range'] for exp in experiment_results]
            }
            
            total_collisions = sum(exp['total_collisions'] for exp in experiment_results)
            print(f"Level {level}: {len(experiment_results)} experiments, {total_collisions} total collisions, "
                  f"processed into {len(common_time_points)} time bins")
        
        return processed_data
    
    def plot_collision_density_correlation(self, time_bin_size: float = 10.0):
        """Create plots showing collision rates and obstacle densities over time."""
        # Process data
        processed_data = self.process_collision_density_data(time_bin_size)
        
        if not processed_data:
            print("No processed data available for plotting")
            return
        
        # Create one figure for each level
        for level in sorted(processed_data.keys()):
            data = processed_data[level]
            
            # Create figure with dual y-axes
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot collision rates on left y-axis with error bars
            color1 = COLLISION_COLOR
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Collision Rate (collisions/s)', color=color1)
            
            # Plot mean with error bars (using SEM)
            ax1.errorbar(data['time_points'], data['collision_rates_mean'], 
                        yerr=data['collision_rates_sem'], color=color1, 
                        linewidth=2, marker='o', markersize=4, capsize=3,
                        label=f'Collision Rate (n={data["num_experiments"]})')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for obstacle density
            ax2 = ax1.twinx()
            color2 = OBSTACLE_COLOR
            ax2.set_ylabel('Obstacle Density (per m²)', color=color2)
            
            # Plot mean with error bars (using SEM)
            ax2.errorbar(data['time_points'], data['obstacle_densities_mean'],
                        yerr=data['obstacle_densities_sem'], color=color2, 
                        linewidth=2, marker='s', markersize=4, capsize=3,
                        label=f'Obstacle Density (n={data["num_experiments"]})')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Add title and legend
            plt.title(f'Collision Rate vs Obstacle Density Over Time - Level {level}')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add statistics text box
            total_collisions = sum(data['total_collisions_per_exp'])
            min_time_range = min(tr[0] for tr in data['time_ranges'])
            max_time_range = max(tr[1] for tr in data['time_ranges'])
            
            # Calculate data coverage
            non_nan_points = np.sum(~np.isnan(data['collision_rates_mean']))
            coverage_percent = (non_nan_points / len(data['time_points'])) * 100 if len(data['time_points']) > 0 else 0
            
            stats_text = (f'Experiments: {data["num_experiments"]}\n'
                         f'Total Collisions: {total_collisions}\n'
                         f'Time Range: {min_time_range:.1f}-{max_time_range:.1f}s\n'
                         f'Data Coverage: {coverage_percent:.1f}%\n'
                         f'Avg Collision Rate: {np.nanmean(data["collision_rates_mean"]):.3f}±{np.nanmean(data["collision_rates_sem"]):.3f}/s\n'
                         f'Avg Obstacle Density: {np.nanmean(data["obstacle_densities_mean"]):.4f}±{np.nanmean(data["obstacle_densities_sem"]):.4f}/m²')
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            filename = f'collision_density_level{level}.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
            plt.close()
        
        # Create comparison plot if we have both levels
        if len(processed_data) >= 2:
            self._plot_level_comparison(processed_data)
        
        print("All collision/density plots saved successfully!")
    
    def _plot_level_comparison(self, processed_data: Dict):
        """Create a comparison plot showing both levels."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        levels = sorted(processed_data.keys())
        colors = [LEVEL2_COLOR, LEVEL4_COLOR]
        
        # Plot 1: Collision rates comparison
        for i, level in enumerate(levels):
            data = processed_data[level]
            ax1.errorbar(data['time_points'], data['collision_rates_mean'], 
                        yerr=data['collision_rates_sem'], color=colors[i], 
                        linewidth=2, marker='o', markersize=3, capsize=2,
                        label=f'Level {level} (n={data["num_experiments"]})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Collision Rate (collisions/s)')
        ax1.set_title('Collision Rates Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Obstacle densities comparison  
        for i, level in enumerate(levels):
            data = processed_data[level]
            ax2.errorbar(data['time_points'], data['obstacle_densities_mean'],
                        yerr=data['obstacle_densities_sem'], color=colors[i], 
                        linewidth=2, marker='s', markersize=3, capsize=2,
                        label=f'Level {level} (n={data["num_experiments"]})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Obstacle Density (per m²)')
        ax2.set_title('Obstacle Densities Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot - Collision rate vs Obstacle density for level 2
        if levels[0] in processed_data:
            data = processed_data[levels[0]]
            # Use mean values for scatter plot
            ax3.scatter(data['obstacle_densities_mean'], data['collision_rates_mean'],
                       color=colors[0], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            ax3.set_xlabel('Obstacle Density (per m²)')
            ax3.set_ylabel('Collision Rate (collisions/s)')
            ax3.set_title(f'Collision Rate vs Obstacle Density - Level {levels[0]}')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot - Collision rate vs Obstacle density for level 4
        if len(levels) > 1 and levels[1] in processed_data:
            data = processed_data[levels[1]]
            # Use mean values for scatter plot
            ax4.scatter(data['obstacle_densities_mean'], data['collision_rates_mean'],
                       color=colors[1], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Obstacle Density (per m²)')
            ax4.set_ylabel('Collision Rate (collisions/s)')
            ax4.set_title(f'Collision Rate vs Obstacle Density - Level {levels[1]}')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Collision Rate and Obstacle Density Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save comparison plot
        filename = 'collision_density_comparison.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Plot collision rates and obstacle densities over time')
    parser.add_argument('--results-dir', default='build/results',
                       help='Directory containing collision results (default: build/results)')
    parser.add_argument('--time-bin', type=float, default=10.0,
                       help='Time bin size in seconds for aggregating data (default: 10.0)')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found!")
        return
    
    # Create plotter and generate plots
    plotter = CollisionDensityPlotter(args.results_dir)
    plotter.load_collision_data()
    plotter.plot_collision_density_correlation(args.time_bin)


if __name__ == "__main__":
    main()
