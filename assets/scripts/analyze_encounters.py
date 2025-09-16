#!/usr/bin/env python3
"""
Analyze encounter rates and obstacle density metrics from GBPPlanner experiment results.

This script extracts and displays:
- Robot encounter rates per minute
- Obstacle encounter rates per minute  
- Total obstacle density (per 1000m²)
- Total obstacle crowdedness

Data is organized by difficulty level and seed from experiment CSV files.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import argparse
from typing import Dict, List

class EncounterAnalyzer:
    """Class to analyze encounter rates and obstacle metrics."""
    
    def __init__(self, results_dir: str = "build/results"):
        """Initialize with results directory path."""
        self.results_dir = Path(results_dir)
        self.experiment_data = {}
        
    def load_experiment_data(self):
        """Load all experiment data files."""
        print("Loading experiment files...")
        
        # Find all experiment CSV files
        csv_files = list(self.results_dir.glob("*experiment*.csv"))
        print(f"Found {len(csv_files)} experiment CSV files")
        
        # Load experiment files
        for file_path in csv_files:
            file_info = self._parse_filename(file_path.name)
            if file_info:
                try:
                    df = pd.read_csv(file_path)
                    key = (file_info['experiment'], file_info['difficulty'], file_info['seed'])
                    self.experiment_data[key] = df
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Failed to parse filename: {file_path.name}")
        
        print(f"Loaded {len(self.experiment_data)} experiment datasets")
    
    def _parse_filename(self, filename: str) -> Dict:
        """Parse filename to extract experiment info."""
        # Pattern: {experiment}_level{difficulty}_{seed}_{type}_{timestamp}.csv
        pattern = r'([^_]+)_level(\d+)_(\d+)_(experiment)_[\d_]+\.csv'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'experiment': match.group(1),
                'difficulty': int(match.group(2)),
                'seed': int(match.group(3)),
                'type': match.group(4)
            }
        return None
    
    def analyze_encounters(self):
        """Analyze and print encounter metrics by difficulty level and seed."""
        if not self.experiment_data:
            print("No experiment data loaded!")
            return
        
        # Group data by difficulty level
        difficulties = {}
        for key, df in self.experiment_data.items():
            experiment, difficulty, seed = key
            
            if len(df) > 0:
                row = df.iloc[0]  # Should only be one row per experiment file
                
                if difficulty not in difficulties:
                    difficulties[difficulty] = []
                
                difficulties[difficulty].append({
                    'seed': seed,
                    'robot_encounters_per_min': row['avg_robot_encounters_per_s'] * 60,  # Convert /s to /min
                    'obstacle_encounters_per_min': row['avg_obstacle_encounters_per_s'] * 60,  # Convert /s to /min
                    'obstacle_density_per_1000m2': row['total_obstacle_density'] * 1000,  # Convert to per 1000m²
                    'obstacle_crowdedness': row['total_obstacle_crowdedness']
                })
        
        # Print results organized by difficulty level
        print("\n" + "="*80)
        print("ENCOUNTER RATES AND OBSTACLE DENSITY ANALYSIS")
        print("="*80)
        
        for difficulty in sorted(difficulties.keys()):
            print(f"\n{'─'*60}")
            print(f"DIFFICULTY LEVEL {difficulty}")
            print(f"{'─'*60}")
            
            data = difficulties[difficulty]
            
            # Sort by seed for consistent output
            data.sort(key=lambda x: x['seed'])
            
            # Print header
            print(f"{'Seed':>4} {'Robot Enc/min':>14} {'Obstacle Enc/min':>16} {'Density/1000m²':>14} {'Crowdedness':>12}")
            print("─" * 70)
            
            # Print data for each seed
            for entry in data:
                print(f"{entry['seed']:>4} "
                      f"{entry['robot_encounters_per_min']:>14.3f} "
                      f"{entry['obstacle_encounters_per_min']:>16.3f} "
                      f"{entry['obstacle_density_per_1000m2']:>14.3f} "
                      f"{entry['obstacle_crowdedness']:>12.6f}")
            
            # Calculate and print summary statistics
            robot_enc_rates = [d['robot_encounters_per_min'] for d in data]
            obstacle_enc_rates = [d['obstacle_encounters_per_min'] for d in data]
            densities = [d['obstacle_density_per_1000m2'] for d in data]
            crowdedness = [d['obstacle_crowdedness'] for d in data]
            
            print("─" * 70)
            print(f"{'MEAN':>4} "
                  f"{np.mean(robot_enc_rates):>14.3f} "
                  f"{np.mean(obstacle_enc_rates):>16.3f} "
                  f"{np.mean(densities):>14.3f} "
                  f"{np.mean(crowdedness):>12.6f}")
            
            print(f"{'STD':>4} "
                  f"{np.std(robot_enc_rates):>14.3f} "
                  f"{np.std(obstacle_enc_rates):>16.3f} "
                  f"{np.std(densities):>14.3f} "
                  f"{np.std(crowdedness):>12.6f}")
            
            print(f"{'MIN':>4} "
                  f"{np.min(robot_enc_rates):>14.3f} "
                  f"{np.min(obstacle_enc_rates):>16.3f} "
                  f"{np.min(densities):>14.3f} "
                  f"{np.min(crowdedness):>12.6f}")
            
            print(f"{'MAX':>4} "
                  f"{np.max(robot_enc_rates):>14.3f} "
                  f"{np.max(obstacle_enc_rates):>16.3f} "
                  f"{np.max(densities):>14.3f} "
                  f"{np.max(crowdedness):>12.6f}")
        
        print("\n" + "="*80)
        
        # Create summary table
        self._print_summary_table(difficulties)
    
    def _print_summary_table(self, difficulties):
        """Print a condensed summary table."""
        print("SUMMARY TABLE - MEANS BY DIFFICULTY LEVEL")
        print("="*80)
        print(f"{'Level':>5} {'Robot Enc/min':>14} {'Obstacle Enc/min':>16} {'Density/1000m²':>14} {'Crowdedness':>12} {'N Seeds':>8}")
        print("─" * 90)
        
        for difficulty in sorted(difficulties.keys()):
            data = difficulties[difficulty]
            
            robot_enc_rates = [d['robot_encounters_per_min'] for d in data]
            obstacle_enc_rates = [d['obstacle_encounters_per_min'] for d in data]
            densities = [d['obstacle_density_per_1000m2'] for d in data]
            crowdedness = [d['obstacle_crowdedness'] for d in data]
            
            print(f"{difficulty:>5} "
                  f"{np.mean(robot_enc_rates):>14.3f} "
                  f"{np.mean(obstacle_enc_rates):>16.3f} "
                  f"{np.mean(densities):>14.3f} "
                  f"{np.mean(crowdedness):>12.6f} "
                  f"{len(data):>8}")
        
        print("="*90)
    
    def save_to_csv(self, output_file: str = "encounter_analysis.csv"):
        """Save analysis results to CSV file."""
        if not self.experiment_data:
            print("No data to save!")
            return
        
        # Prepare data for CSV
        rows = []
        for key, df in self.experiment_data.items():
            experiment, difficulty, seed = key
            
            if len(df) > 0:
                row = df.iloc[0]
                rows.append({
                    'experiment': experiment,
                    'difficulty_level': difficulty,
                    'seed': seed,
                    'robot_encounters_per_min': row['avg_robot_encounters_per_s'] * 60,
                    'obstacle_encounters_per_min': row['avg_obstacle_encounters_per_s'] * 60,
                    'obstacle_density_per_1000m2': row['total_obstacle_density'] * 1000,
                    'obstacle_crowdedness': row['total_obstacle_crowdedness']
                })
        
        # Create DataFrame and save
        df_output = pd.DataFrame(rows)
        df_output = df_output.sort_values(['difficulty_level', 'seed'])
        df_output.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze encounter rates and obstacle density from GBPPlanner experiments')
    parser.add_argument('--results-dir', default='build/results', 
                       help='Directory containing experiment results (default: build/results)')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--output', default='encounter_analysis.csv',
                       help='Output CSV filename (default: encounter_analysis.csv)')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found!")
        return
    
    # Create analyzer and run analysis
    analyzer = EncounterAnalyzer(args.results_dir)
    analyzer.load_experiment_data()
    analyzer.analyze_encounters()
    
    if args.save_csv:
        analyzer.save_to_csv(args.output)


if __name__ == "__main__":
    main()
