#!/usr/bin/env python3
"""
Analyze the results of the keyboard layout optimization runs.
"""
import os
import pandas as pd
import glob
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing all layout results
RESULTS_DIR = 'output/layouts'

def parse_result_csv(filepath):
    """Parse a layout results CSV file and extract key metrics."""
    try:
        df = pd.read_csv(filepath, header=None)
        
        # Extract configuration info from header rows
        config_info = {}
        for i in range(6):  # First 6 rows contain configuration info
            if len(df.iloc[i]) >= 2:
                key, value = df.iloc[i, 0], df.iloc[i, 1]
                config_info[key] = value
        
        # Extract the best layout result (first result row)
        layout_row = df.iloc[8]  # Skip the empty row (7) and header row (8)
        
        if len(layout_row) >= 5:
            result = {
                'items': layout_row[0],
                'positions': layout_row[1],
                'total_score': float(layout_row[3]),
                'item_score': float(layout_row[4]),
                'item_pair_score': float(layout_row[5])
            }
            
            return {**config_info, **result}
        
        return None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def load_all_results():
    """Load all result files and return a dataframe."""
    all_results = []
    
    # Find all CSV result files
    for config_dir in glob.glob(f"{RESULTS_DIR}/config_*"):
        config_id = os.path.basename(config_dir)
        
        # Find the latest CSV file in the directory
        csv_files = glob.glob(f"{config_dir}/layout_results_*.csv")
        if not csv_files:
            continue
            
        # Sort by modification time to get the latest
        latest_csv = max(csv_files, key=os.path.getmtime)
        
        # Load the metadata file
        meta_file = f"configs/{config_id}_meta.txt"
        strategy = "unknown"
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                for line in f:
                    if line.startswith("Strategy:"):
                        strategy = line.strip().split(": ")[1]
                        break
        
        # Parse result and add config ID and strategy
        result = parse_result_csv(latest_csv)
        if result:
            result['config_id'] = config_id
            result['strategy'] = strategy
            all_results.append(result)
    
    return pd.DataFrame(all_results)

def analyze_results(df):
    """Analyze results and generate insights."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"Analyzed {len(df)} optimization results")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"Mean total score: {df['total_score'].mean():.6f}")
    print(f"Best total score: {df['total_score'].max():.6f}")
    print(f"Worst total score: {df['total_score'].min():.6f}")
    
    # Analysis by strategy
    print("\nPerformance by Strategy:")
    strategy_stats = df.groupby('strategy').agg({
        'total_score': ['count', 'mean', 'min', 'max']
    })
    print(strategy_stats)
    
    # Find the best layout
    best_idx = df['total_score'].idxmax()
    best_layout = df.iloc[best_idx]
    
    print("\nBest Layout:")
    print(f"Config ID: {best_layout['config_id']}")
    print(f"Strategy: {best_layout['strategy']}")
    print(f"Total Score: {best_layout['total_score']:.6f}")
    print(f"Items: {best_layout['items']}")
    print(f"Positions: {best_layout['positions']}")
    print(f"Items to constrain: {best_layout.get('Items to constrain', 'N/A')}")
    print(f"Positions to constrain: {best_layout.get('Constraint positions', 'N/A')}")
    
    # Save to Excel for further analysis
    df.to_excel("optimization_results_summary.xlsx", index=False)
    print("\nResults summary saved to optimization_results_summary.xlsx")
    
    # Generate some visualizations
    try:
        # Score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['total_score'], bins=20, alpha=0.7)
        plt.title('Distribution of Layout Scores')
        plt.xlabel('Total Score')
        plt.ylabel('Frequency')
        plt.savefig('score_distribution.png')
        
        # Performance by strategy
        plt.figure(figsize=(12, 6))
        strategy_means = df.groupby('strategy')['total_score'].mean().sort_values()
        strategy_means.plot(kind='bar')
        plt.title('Average Score by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Mean Score')
        plt.tight_layout()
        plt.savefig('strategy_performance.png')
        
        print("Visualization charts saved to current directory.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    print("Loading and analyzing keyboard optimization results...")
    results_df = load_all_results()
    analyze_results(results_df)