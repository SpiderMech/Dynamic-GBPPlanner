#!/usr/bin/env python3
"""
Plot key metrics from GBPPlanner experiment results.

This script analyzes experiment data stored in build/results and creates plots for:
1. LDJ (median and IQR) vs difficulty level
2. Collision rates (median and IQR) vs difficulty level  
3. Detour ratio (median and IQR) vs difficulty level
4. In-flow vs out-flow rates scatter plot

Data files follow the naming pattern: {experiment}_level{difficulty}_{seed}_{type}_{timestamp}.csv
where type is either 'summary' or 'experiment'.
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

# Define teal color for consistency
TEAL_COLOR = '#81B5A2'

class MetricsPlotter:
    """Class to handle loading and plotting of experiment metrics."""
    
    def __init__(self, results_dir: str = "build/results"):
        """Initialize with results directory path."""
        self.results_dir = Path(results_dir)
        self.summary_data = {}
        self.experiment_data = {}
        
    def load_data(self):
        """Load all summary and experiment data files."""
        print("Loading data files...")
        
        # Find all CSV files
        csv_files = list(self.results_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        summary_files = [f for f in csv_files if "summary" in f.name]
        experiment_files = [f for f in csv_files if "experiment" in f.name]
        
        print(f"Summary files: {len(summary_files)}")
        print(f"Experiment files: {len(experiment_files)}")
        
        # Load summary files
        for file_path in summary_files:
            file_info = self._parse_filename(file_path.name)
            if file_info:
                try:
                    df = pd.read_csv(file_path)
                    key = (file_info['experiment'], file_info['comm_level'], file_info['seed'])
                    self.summary_data[key] = df
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Failed to parse filename: {file_path.name}")
        
        # Load experiment files
        for file_path in experiment_files:
            file_info = self._parse_filename(file_path.name)
            if file_info:
                try:
                    df = pd.read_csv(file_path)
                    key = (file_info['experiment'], file_info['comm_level'], file_info['seed'])
                    self.experiment_data[key] = df
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Failed to parse filename: {file_path.name}")
        
        print(f"Loaded {len(self.summary_data)} summary datasets")
        print(f"Loaded {len(self.experiment_data)} experiment datasets")
    
    def _parse_filename(self, filename: str) -> Dict:
        """Parse filename to extract experiment info."""
        # Pattern for comm failure data: comm_level{level}_{seed}_{type}_{timestamp}.csv
        pattern = r'comm_level(\d+)_(\d+)_(summary|experiment)_[\d_]+\.csv'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'experiment': 'comm_failure',
                'comm_level': int(match.group(1)),
                'seed': int(match.group(2)),
                'type': match.group(3)
            }
        return None
    
    def process_summary_metrics(self) -> Dict:
        """Process summary data to compute metrics by communication level."""
        print("Processing summary metrics...")
        
        metrics_by_comm_level = {}
        
        # Get all unique communication levels
        comm_levels = set(key[1] for key in self.summary_data.keys())
        
        for comm_level in comm_levels:
            # Get all data for this communication level
            comm_level_data = []
            collision_data = []
            detour_data = []
            ldj_data = []
            
            for key, df in self.summary_data.items():
                if key[1] == comm_level:  # Same comm level
                    # Calculate collision rate per robot
                    total_robots = len(df)
                    collided_robots = len(df[df['collided'] == 1])
                    collision_rate = collided_robots / total_robots if total_robots > 0 else 0
                    collision_data.append(collision_rate)
                    
                    # Get detour ratios (excluding NaN values)
                    detour_ratios = df['detour_ratio'].dropna()
                    if len(detour_ratios) > 0:
                        detour_data.extend(detour_ratios.tolist())
                    
                    # Get LDJ values (excluding NaN values)
                    ldj_values = df['ldj'].dropna()
                    if len(ldj_values) > 0:
                        ldj_data.extend(ldj_values.tolist())
            
            # Compute statistics
            metrics_by_comm_level[comm_level] = {
                'collision_rates': collision_data,
                'detour_ratios': detour_data,
                'ldj_values': ldj_data,
                'collision_median': np.median(collision_data) if collision_data else 0,
                'collision_iqr': np.percentile(collision_data, 75) - np.percentile(collision_data, 25) if collision_data else 0,
                'detour_median': np.median(detour_data) if detour_data else 0,
                'detour_iqr': np.percentile(detour_data, 75) - np.percentile(detour_data, 25) if detour_data else 0,
                'ldj_median': np.median(ldj_data) if ldj_data else 0,
                'ldj_iqr': np.percentile(ldj_data, 75) - np.percentile(ldj_data, 25) if ldj_data else 0,
            }
        
        return metrics_by_comm_level
    
    def process_experiment_metrics(self) -> Dict:
        """Process experiment data for flow rates."""
        print("Processing experiment metrics...")
        
        flow_data = []
        
        for key, df in self.experiment_data.items():
            if len(df) > 0:
                row = df.iloc[0]  # Should only be one row per experiment file
                flow_data.append({
                    'comm_level': key[1],
                    'seed': key[2],
                    'in_flow_rate': row['total_in_flow_rate_per_s'],
                    'out_flow_rate': row['total_out_flow_rate_per_s'],
                    'normalized_collisions': row['normalized_collisions']
                })
        
        return flow_data
    
    def plot_metrics(self):
        """Create all the requested plots as separate files."""
        # Process data
        summary_metrics = self.process_summary_metrics()
        flow_data = self.process_experiment_metrics()
        
        # Sort comm levels for consistent plotting
        comm_levels = sorted(summary_metrics.keys())
        
        # Create individual plots - using line plots since we have single points per comm level
        plot_configs = [
            ('ldj_values', 'LDJ vs Communication Failure Rate', 'LDJ', 'ldj_vs_comm_failure.pdf'),
            ('collision_rates', 'Collision Rate vs Communication Failure Rate', 'Collision Rate', 'collision_rate_vs_comm_failure.pdf'),
            ('detour_ratios', 'Detour Ratio vs Communication Failure Rate', 'Detour Ratio', 'detour_ratio_vs_comm_failure.pdf')
        ]
        
        for data_key, title, ylabel, filename in plot_configs:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self._plot_metric_vs_comm_level(ax, comm_levels, summary_metrics, data_key, title, ylabel, 'Communication Failure Rate (%)')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
            plt.close()
        
        # Create normalized collisions plot from experiment data
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_normalized_collisions(ax, flow_data)
        plt.tight_layout()
        collision_filename = 'normalized_collisions_vs_comm_failure.pdf'
        plt.savefig(collision_filename, dpi=300, bbox_inches='tight')
        print(f"Saved {collision_filename}")
        plt.close()
        
        # Create flow rates plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_flow_rates(ax, flow_data)
        plt.tight_layout()
        flow_filename = 'inflow_vs_outflow.pdf'
        plt.savefig(flow_filename, dpi=300, bbox_inches='tight')
        print(f"Saved {flow_filename}")
        plt.close()
        
        print("All plots saved successfully!")
    
    def _plot_metric_vs_comm_level(self, ax, comm_levels, metrics, data_key, title, ylabel, xlabel='Communication Failure Rate (%)'):  
        """Plot a metric vs communication level with line plot."""
        # Prepare data for line plot
        x_values = []
        y_values = []
        
        for comm_level in comm_levels:
            if data_key in metrics[comm_level]:
                data = metrics[comm_level][data_key]
                if len(data) > 0:
                    # Use median value for single data point per comm level
                    median_val = np.median(data)
                    x_values.append(comm_level)
                    y_values.append(median_val)
        
        if x_values:
            # Create line plot with markers
            ax.plot(x_values, y_values, 'o-', color=TEAL_COLOR, linewidth=2, markersize=8,
                   markerfacecolor=TEAL_COLOR, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(comm_levels)
        
    def _plot_flow_rates(self, ax, flow_data):
        """Plot in-flow vs out-flow rates scatter plot."""
        if not flow_data:
            ax.text(0.5, 0.5, 'No flow data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        df = pd.DataFrame(flow_data)
        
        # Create scatter plot with different shades of teal for different comm levels
        comm_levels = sorted(df['comm_level'].unique())
        colors = [TEAL_COLOR] * len(comm_levels)  # Use same teal for all
        alphas = np.linspace(0.4, 0.9, len(comm_levels))  # Vary transparency
        
        for i, comm_level in enumerate(comm_levels):
            comm_data = df[df['comm_level'] == comm_level]
            ax.scatter(comm_data['in_flow_rate'], comm_data['out_flow_rate'], 
                      c=colors[i], label=f'{comm_level}% Failure', alpha=alphas[i], s=60, 
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('In-flow Rate (robots/s)')
        ax.set_ylabel('Out-flow Rate (robots/s)')
        # ax.set_title('In-flow vs Out-flow Rates')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add diagonal reference line
        if len(df) > 0:
            min_val = min(df['in_flow_rate'].min(), df['out_flow_rate'].min())
            max_val = max(df['in_flow_rate'].max(), df['out_flow_rate'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
    
    def _plot_normalized_collisions(self, ax, flow_data):
        """Plot normalized collisions vs communication failure rate."""
        if not flow_data:
            ax.text(0.5, 0.5, 'No collision data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        df = pd.DataFrame(flow_data)
        
        # Sort by comm level for line plot
        df_sorted = df.sort_values('comm_level')
        
        # Create line plot
        ax.plot(df_sorted['comm_level'], df_sorted['normalized_collisions'], 
               'o-', color=TEAL_COLOR, linewidth=2, markersize=8,
               markerfacecolor=TEAL_COLOR, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Communication Failure Rate (%)')
        ax.set_ylabel('Normalized Collisions')
        # ax.set_title('Normalized Collisions vs Communication Failure Rate')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df['comm_level'].unique()))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Plot GBPPlanner experiment metrics')
    parser.add_argument('--results-dir', default='build/results', 
                       help='Directory containing experiment results (default: build/results)')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found!")
        return
    
    # Create plotter and generate plots
    plotter = MetricsPlotter(args.results_dir)
    plotter.load_data()
    plotter.plot_metrics()


if __name__ == "__main__":
    main()
