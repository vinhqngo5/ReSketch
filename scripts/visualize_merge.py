import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization_common import (
    load_results,
    setup_fonts,
    load_material_colors,
    save_figure
)


def aggregate_results(results_data):
    results = results_data['results']
    
    metrics = {
        'a_vs_true_on_da': {'are': [], 'aae': []},
        'b_vs_true_on_db': {'are': [], 'aae': []},
        'c_vs_true_on_all': {'are': [], 'aae': []},
        'd_vs_true_on_all': {'are': [], 'aae': []}
    }
    
    for rep in results:
        if 'accuracy' in rep:
            accuracy = rep['accuracy']
        else:
            accuracy = rep
        
        for key in metrics.keys():
            metrics[key]['are'].append(accuracy[key]['are'])
            metrics[key]['aae'].append(accuracy[key]['aae'])
    
    aggregated = {}
    for key in metrics.keys():
        aggregated[key] = {
            'are_mean': np.mean(metrics[key]['are']),
            'are_std': np.std(metrics[key]['are']),
            'aae_mean': np.mean(metrics[key]['aae']),
            'aae_std': np.std(metrics[key]['aae'])
        }
    
    return aggregated


def plot_results(results_data, output_path):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    
    font_config = setup_fonts(__file__)
    
    config = results_data['config']
    aggregated = aggregate_results(results_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = [
        'A on DA',
        'B on DB', 
        'C (merged)\non All',
        'D (2× width)\non All'
    ]
    
    keys = ['a_vs_true_on_da', 'b_vs_true_on_db', 'c_vs_true_on_all', 'd_vs_true_on_all']
    
    colors = [
        material_colors['blue']['500'],      # A
        material_colors['green']['500'],     # B
        material_colors['purple']['500'],    # C (merged)
        material_colors['orange']['500']
    ]
    
    x_pos = np.arange(len(labels))
    bar_width = 0.6
    
    are_means = [aggregated[key]['are_mean'] for key in keys]
    are_stds = [aggregated[key]['are_std'] for key in keys]
    
    bars1 = ax1.bar(x_pos, are_means, bar_width, yerr=are_stds, 
                    color=colors, alpha=0.8, capsize=5, 
                    error_kw={'linewidth': 1.5})
    
    ax1.set_ylabel('Average Relative Error (ARE)', fontsize=12)
    ax1.set_xlabel('Sketch Configuration', fontsize=12)
    ax1.set_title('Accuracy: Average Relative Error', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    for i, (bar, mean, std) in enumerate(zip(bars1, are_means, are_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    aae_means = [aggregated[key]['aae_mean'] for key in keys]
    aae_stds = [aggregated[key]['aae_std'] for key in keys]
    
    bars2 = ax2.bar(x_pos, aae_means, bar_width, yerr=aae_stds,
                    color=colors, alpha=0.8, capsize=5,
                    error_kw={'linewidth': 1.5})
    
    ax2.set_ylabel('Average Absolute Error (AAE)', fontsize=12)
    ax2.set_xlabel('Sketch Configuration', fontsize=12)
    ax2.set_title('Accuracy: Average Absolute Error', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    for i, (bar, mean, std) in enumerate(zip(bars2, aae_means, aae_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {}).get('resketch', {})
    
    depth = sketch_config.get('depth', config.get('resketch_depth', 4))
    kll_k = sketch_config.get('kll_k', config.get('resketch_kll_k', 10))
    memory_kb = exp_config.get('memory_budget_kb', config.get('memory_budget_kb', 32))
    diversity = exp_config.get('stream_diversity', config.get('stream_diversity', 1000000))
    zipf = exp_config.get('zipf_param', config.get('zipf_param', 1.1))
    
    suptitle = (f"ReSketchV2 Merge Experiment: "
                f"depth={depth}, k={kll_k}, "
                f"memory={memory_kb}KB, "
                f"diversity={diversity}, "
                f"zipf={zipf:.2f}")
    fig.suptitle(suptitle, fontsize=11, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, output_path)

def print_summary(results_data):
    config = results_data['config']
    aggregated = aggregate_results(results_data)
    
    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {}).get('resketch', {})
    
    depth = sketch_config.get('depth', config.get('resketch_depth', 4))
    kll_k = sketch_config.get('kll_k', config.get('resketch_kll_k', 10))
    memory_kb = exp_config.get('memory_budget_kb', config.get('memory_budget_kb', 32))
    dataset = exp_config.get('dataset_type', config.get('dataset_type', 'zipf'))
    diversity = exp_config.get('stream_diversity', config.get('stream_diversity', 1000000))
    stream_size = exp_config.get('stream_size', config.get('stream_size', 10000000))
    repetitions = exp_config.get('repetitions', config.get('repetitions', 1))
    
    print("\n" + "="*60)
    print("MERGE EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  ReSketch: depth={depth}, k={kll_k}")
    print(f"  Memory Budget: {memory_kb} KB per sketch")
    print(f"  Dataset: {dataset}, diversity={diversity}")
    print(f"  Stream Size: {stream_size}")
    print(f"  Repetitions: {repetitions}")
    
    print(f"\nAccuracy Results (mean ± std):")
    print(f"\n  A on DA (items [0, {diversity//2})):")
    print(f"    ARE: {aggregated['a_vs_true_on_da']['are_mean']:.4f} ± {aggregated['a_vs_true_on_da']['are_std']:.4f}")
    print(f"    AAE: {aggregated['a_vs_true_on_da']['aae_mean']:.2f} ± {aggregated['a_vs_true_on_da']['aae_std']:.2f}")
    
    print(f"\n  B on DB (items [{diversity//2}, {diversity})):")
    print(f"    ARE: {aggregated['b_vs_true_on_db']['are_mean']:.4f} ± {aggregated['b_vs_true_on_db']['are_std']:.4f}")
    print(f"    AAE: {aggregated['b_vs_true_on_db']['aae_mean']:.2f} ± {aggregated['b_vs_true_on_db']['aae_std']:.2f}")
    
    print(f"\n  C (merged A+B) on All items:")
    print(f"    ARE: {aggregated['c_vs_true_on_all']['are_mean']:.4f} ± {aggregated['c_vs_true_on_all']['are_std']:.4f}")
    print(f"    AAE: {aggregated['c_vs_true_on_all']['aae_mean']:.2f} ± {aggregated['c_vs_true_on_all']['aae_std']:.2f}")
    
    print(f"\n  D (ground truth, 2× width) on All items:")
    print(f"    ARE: {aggregated['d_vs_true_on_all']['are_mean']:.4f} ± {aggregated['d_vs_true_on_all']['are_std']:.4f}")
    print(f"    AAE: {aggregated['d_vs_true_on_all']['aae_mean']:.2f} ± {aggregated['d_vs_true_on_all']['aae_std']:.2f}")
    

def main():
    parser = argparse.ArgumentParser(
        description='Visualize ReSketchV2 merge experiment results'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to merge results JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path (without extension, will generate .png and .pdf). '
             'Default: output/merge_results'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    results_data = load_results(args.input)
    
    print_summary(results_data)
    
    output_path = args.output if args.output else 'output/merge_results'
    plot_results(results_data, output_path)


if __name__ == '__main__':
    main()
