#!/usr/bin/env python3
"""
Publication-Quality Training Curve Plotter
===========================================
This script generates academic-style plots from TensorBoard CSV exports.

Features:
- Exponential Moving Average (EMA) smoothing with alpha=0.99
- Times New Roman font for publication readiness
- Dual-layer visualization (transparent raw + bold smoothed)
- Professional color schemes
- High-resolution output (PDF + 300 DPI PNG)

Usage:
    python plot_training_curves.py --data_dir <path_to_csv> --output_dir <output_path>
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Font settings for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 10

# Academic color palette
COLORS = {
    'deep_blue': '#1f77b4',
    'deep_orange': '#ff7f0e',
    'deep_green': '#2ca02c',
    'deep_red': '#d62728',
    'deep_purple': '#9467bd'
}

# Plotting parameters
SMOOTHING_ALPHA = 0.99  # TensorBoard default
RAW_ALPHA = 0.2         # Transparency for raw data
SMOOTH_LINEWIDTH = 2.5  # Bold line for smoothed data
RAW_LINEWIDTH = 1.0     # Thin line for raw data

# ============================================================================
# Core Functions
# ============================================================================

def exponential_moving_average(data, alpha=0.99):
    """
    Apply Exponential Moving Average (EMA) smoothing.
    
    Formula: y_t = alpha * y_{t-1} + (1 - alpha) * x_t
    
    Args:
        data: numpy array or list of values
        alpha: smoothing factor (0 < alpha < 1), higher = smoother
    
    Returns:
        numpy array of smoothed values
    """
    data = np.array(data)
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Initialize with first value
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * data[t]
    
    return smoothed


def find_reward_csv(data_dir, csv_filename=None):
    """
    Find CSV files containing reward data in the given directory.
    
    Args:
        data_dir: path to directory containing CSV files
        csv_filename: (optional) specific CSV filename to use. If None, search all .csv files
    
    Returns:
        list of Path objects to reward CSV files
    """
    data_path = Path(data_dir)
    
    # If specific filename provided, use it directly
    if csv_filename:
        csv_file = data_path / csv_filename
        if csv_file.exists():
            return [csv_file]
        else:
            return []
    
    # Otherwise, find all CSV files in directory
    csv_files = list(data_path.glob('*.csv'))
    
    # Remove duplicates
    csv_files = list(set(csv_files))
    
    return csv_files


def plot_reward_curve(csv_file, output_dir, color='deep_blue', label='Reward'):
    """
    Generate publication-quality reward curve plot.
    
    Args:
        csv_file: Path to CSV file
        output_dir: Directory to save output plots
        color: Color key from COLORS dict
        label: Label for the curve
    """
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Identify columns (TensorBoard exports typically have 'Step' and 'Value')
    step_col = 'Step' if 'Step' in df.columns else df.columns[0]
    value_col = 'Value' if 'Value' in df.columns else df.columns[1]
    
    steps = df[step_col].values
    rewards = df[value_col].values
    
    # Convert steps to millions (1e6 scale)
    steps_millions = steps / 1e6
    
    # Apply EMA smoothing
    rewards_smoothed = exponential_moving_average(rewards, alpha=SMOOTHING_ALPHA)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot raw data (background, high transparency)
    ax.plot(steps_millions, rewards, 
            color=COLORS[color], 
            alpha=RAW_ALPHA, 
            linewidth=RAW_LINEWIDTH,
            label=f'{label} (raw)')
    
    # Plot smoothed data (foreground, bold)
    ax.plot(steps_millions, rewards_smoothed, 
            color=COLORS[color], 
            alpha=1.0, 
            linewidth=SMOOTH_LINEWIDTH,
            label=f'{label} (smoothed, α={SMOOTHING_ALPHA})')
    
    # Styling
    ax.set_xlabel('Simulation Steps (1e6)')
    ax.set_ylabel('Reward')
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.legend(loc='best', framealpha=0.9)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines thinner
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = csv_file.stem
    
    # Save as PDF (vector format for publications)
    pdf_path = output_path / f'{base_name}_plot.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f'✓ Saved PDF: {pdf_path}')
    
    # Save as PNG (300 DPI for presentations)
    png_path = output_path / f'{base_name}_plot.png'
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved PNG: {png_path}')
    
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality training curve plots from TensorBoard CSV exports.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Plot specific CSV file
    python plot_training_curves.py \\
        --data_dir /path/to/runs/gmm_noise_run_18-18-04-45 \\
        --csv_file csv.csv \\
        --output_dir ./plots
    
    # Plot all CSV files in directory
    python plot_training_curves.py \\
        --data_dir /path/to/runs/gmm_noise_run_18-18-04-45 \\
        --output_dir ./plots
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing TensorBoard CSV exports'
    )
    
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='Specific CSV filename to plot (e.g., csv.csv). If not specified, all .csv files will be processed.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots',
        help='Directory to save output plots (default: ./plots)'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='deep_blue',
        choices=list(COLORS.keys()),
        help='Color scheme for the plot (default: deep_blue)'
    )
    
    args = parser.parse_args()
    
    # Find reward CSV files
    csv_files = find_reward_csv(args.data_dir, args.csv_file)
    
    if not csv_files:
        if args.csv_file:
            print(f'⚠ CSV file not found: {args.csv_file}')
            print(f'  Searched in: {args.data_dir}')
        else:
            print(f'⚠ No CSV files found in: {args.data_dir}')
            print('  Please ensure CSV files are exported from TensorBoard.')
        return
    
    print(f'Found {len(csv_files)} CSV file(s):')
    for f in csv_files:
        print(f'  - {f.name}')
    
    print(f'\nGenerating plots...')
    
    # Plot each CSV file
    for csv_file in csv_files:
        print(f'\nProcessing: {csv_file.name}')
        plot_reward_curve(csv_file, args.output_dir, color=args.color)
    
    print(f'\n✓ All plots saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
