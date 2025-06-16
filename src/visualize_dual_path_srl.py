#!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('visualize_dual_path_srl')

def plot_pathway_weights(weights_data, save_path='results/enhanced_srl_figures'):
    """
    Plot the pathway weights distribution and time series.
    
    Args:
        weights_data: Dictionary containing pathway weights data
        save_path: Directory to save the figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot text-driven vs price-driven pathway weights distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weights_data['text_weight'].flatten(), bins=20, alpha=0.7, label='Text-Driven Pathway')
    plt.hist(weights_data['price_weight'].flatten(), bins=20, alpha=0.7, label='Price-Driven Pathway')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pathway Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'pathway_weight_distribution.png'), dpi=300)
    plt.close()
    
    # Plot pathway weights over time for a sample
    sample_idx = 0  # First sample
    plt.figure(figsize=(12, 6))
    days = np.arange(len(weights_data['text_weight'][sample_idx]))
    plt.plot(days, weights_data['text_weight'][sample_idx], 'o-', label='Text-Driven Weight')
    plt.plot(days, weights_data['price_weight'][sample_idx], 'o-', label='Price-Driven Weight')
    plt.xlabel('Day')
    plt.ylabel('Pathway Weight')
    plt.title('Pathway Weights Over Time (Sample 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(days)
    plt.savefig(os.path.join(save_path, 'pathway_weights_time_series.png'), dpi=300)
    plt.close()

def plot_volatility_influence(data, save_path='results/enhanced_srl_figures'):
    """
    Plot the relationship between volatility and text influence.
    
    Args:
        data: Dictionary containing volatility and text influence data
        save_path: Directory to save the figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Flatten arrays for scatter plot
    volatility_flat = data['volatility'].flatten()
    text_influence_flat = data['text_influence'].flatten()
    text_weight_flat = data['text_weight'].flatten()
    
    # Plot volatility vs text influence
    plt.figure(figsize=(10, 6))
    plt.scatter(volatility_flat, text_influence_flat, alpha=0.5)
    plt.xlabel('Volatility')
    plt.ylabel('Text Influence Score')
    plt.title('Volatility vs Text Influence')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(volatility_flat, text_influence_flat, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(volatility_flat), p(np.sort(volatility_flat)), "r--", 
             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'volatility_vs_text_influence.png'), dpi=300)
    plt.close()
    
    # Plot volatility vs text pathway weight
    plt.figure(figsize=(10, 6))
    plt.scatter(volatility_flat, text_weight_flat, alpha=0.5)
    plt.xlabel('Volatility')
    plt.ylabel('Text-Driven Pathway Weight')
    plt.title('Volatility vs Text-Driven Pathway Weight')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(volatility_flat, text_weight_flat, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(volatility_flat), p(np.sort(volatility_flat)), "r--", 
             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'volatility_vs_text_weight.png'), dpi=300)
    plt.close()

def plot_message_attention(attention_data, save_path='results/enhanced_srl_figures'):
    """
    Plot message attention heatmaps.
    
    Args:
        attention_data: Dictionary containing message attention data
        save_path: Directory to save the figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot price-driven attention for a sample day
    sample_idx = 0  # First sample
    day_idx = 0  # First day
    
    # Get attention weights for the first day of the first sample
    attention = attention_data['price_attention'][sample_idx, day_idx]
    
    # Only plot for actual messages (not padding)
    n_msgs = min(len(attention), 10)  # Limit to 10 messages for clarity
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(attention[:n_msgs].reshape(1, -1), cmap='hot', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Message Index')
    plt.ylabel('Day 1')
    plt.title('Price-Driven Attention for Messages (Sample 1, Day 1)')
    plt.xticks(np.arange(n_msgs))
    plt.savefig(os.path.join(save_path, 'price_attention_heatmap.png'), dpi=300)
    plt.close()

def plot_accuracy_comparison(metrics, save_path='results/enhanced_srl_figures'):
    """
    Plot accuracy comparison between high and low text influence days.
    
    Args:
        metrics: Dictionary containing accuracy metrics
        save_path: Directory to save the figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extract metrics
    high_influence_acc = metrics.get('high_influence_acc', 0)
    low_influence_acc = metrics.get('low_influence_acc', 0)
    
    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(['High Text Influence Days', 'Low Text Influence Days'], 
            [high_influence_acc, low_influence_acc],
            color=['#2C7BB6', '#D7191C'])
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy: High vs Low Text Influence Days')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy values on top of bars
    plt.text(0, high_influence_acc + 0.01, f'{high_influence_acc:.4f}', 
             ha='center', va='bottom')
    plt.text(1, low_influence_acc + 0.01, f'{low_influence_acc:.4f}', 
             ha='center', va='bottom')
    
    plt.ylim(0, max(high_influence_acc, low_influence_acc) * 1.2)
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize enhanced dual-path SRL results')
    parser.add_argument('--results_file', type=str, default='results/enhanced_srl_results.json',
                        help='Path to the results JSON file')
    parser.add_argument('--metrics_file', type=str, default='results/enhanced_srl_metrics.txt',
                        help='Path to the metrics text file')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_srl_figures',
                        help='Directory to save the figures')
    args = parser.parse_args()
    
    # Load data from file (if it exists)
    if os.path.exists(args.results_file):
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
        
        # Convert lists to numpy arrays
        for key, value in results_data.items():
            if isinstance(value, list):
                results_data[key] = np.array(value)
        
        # Create visualizations
        plot_pathway_weights(results_data, args.output_dir)
        plot_volatility_influence(results_data, args.output_dir)
        plot_message_attention(results_data, args.output_dir)
        
        logger.info(f"Saved pathway visualizations to {args.output_dir}")
    else:
        logger.warning(f"Results file {args.results_file} not found. Cannot create visualizations.")
    
    # Load metrics from file (if it exists)
    metrics = {}
    if os.path.exists(args.metrics_file):
        with open(args.metrics_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metrics[key.strip()] = float(value.strip())
        
        # Create accuracy comparison visualization
        plot_accuracy_comparison(metrics, args.output_dir)
        logger.info(f"Saved accuracy comparison to {args.output_dir}")
    else:
        logger.warning(f"Metrics file {args.metrics_file} not found. Cannot create accuracy comparison.")

if __name__ == "__main__":
    main() 