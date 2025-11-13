import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from visualization_common import (
    load_material_colors, setup_fonts, get_sketch_styles,
    style_axis, plot_line_with_error, save_figure, create_shared_legend, load_results
)

def aggregate_results(results_dict):
    aggregated = {}
    
    for sketch_name, repetitions in results_dict.items():
        items_list = []
        throughput_list = []
        query_throughput_list = []
        are_list = []
        aae_list = []
        memory_list = []
        is_warmup_list = []
        geometric_cannot_shrink_list = []
        
        for rep_data in repetitions:
            checkpoints = rep_data['checkpoints']
            items = [cp['items_processed'] for cp in checkpoints]
            throughput = [cp['throughput_mops'] for cp in checkpoints]
            query_throughput = [cp['query_throughput_mops'] for cp in checkpoints]
            are = [cp['are'] for cp in checkpoints]
            aae = [cp['aae'] for cp in checkpoints]
            memory = [cp.get('memory_bytes', cp.get('memory_kb', 0) * 1024) / 1024 for cp in checkpoints]
            is_warmup = [cp.get('is_warmup', False) for cp in checkpoints]
            geometric_cannot_shrink = [cp.get('geometric_cannot_shrink', False) for cp in checkpoints]
            
            items_list.append(items)
            throughput_list.append(throughput)
            query_throughput_list.append(query_throughput)
            are_list.append(are)
            aae_list.append(aae)
            memory_list.append(memory)
            is_warmup_list.append(is_warmup)
            geometric_cannot_shrink_list.append(geometric_cannot_shrink)
        
        items_array = np.array(items_list[0])
        throughput_array = np.array(throughput_list)
        query_throughput_array = np.array(query_throughput_list)
        are_array = np.array(are_list)
        aae_array = np.array(aae_list)
        memory_array = np.array(memory_list)
        is_warmup_array = np.array(is_warmup_list[0])
        geometric_cannot_shrink_array = np.array(geometric_cannot_shrink_list[0])
        
        aggregated[sketch_name] = {
            'items': items_array,
            'throughput_mean': np.mean(throughput_array, axis=0),
            'throughput_std': np.std(throughput_array, axis=0),
            'query_throughput_mean': np.mean(query_throughput_array, axis=0),
            'query_throughput_std': np.std(query_throughput_array, axis=0),
            'are_mean': np.mean(are_array, axis=0),
            'are_std': np.std(are_array, axis=0),
            'aae_mean': np.mean(aae_array, axis=0),
            'aae_std': np.std(aae_array, axis=0),
            'memory_mean': np.mean(memory_array, axis=0),
            'memory_std': np.std(memory_array, axis=0),
            'is_warmup': is_warmup_array,
            'geometric_cannot_shrink': geometric_cannot_shrink_array,
        }
    
    return aggregated

def plot_results(config, aggregated, output_path, plot_every_n_points=1):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    
    font_config = setup_fonts(__file__)
    
    fig, axes = plt.subplots(5, 1, figsize=(6.45, 10), sharex=True)
    
    styles = get_sketch_styles(material_colors)
    
    exp_config = config.get('experiment', config)
    initial_memory_kb = exp_config.get('initial_memory_kb', 1600)
    
    gs_data = aggregated.get('GeometricSketch', None)
    geometric_limit_idx = None
    if gs_data is not None:
        geometric_cannot_shrink = gs_data['geometric_cannot_shrink']
        for i, cannot_shrink in enumerate(geometric_cannot_shrink):
            if cannot_shrink:
                geometric_limit_idx = i
                break
    
    ax_throughput = axes[0]
    for sketch_name, data in aggregated.items():
        style = styles.get(sketch_name, {})
        items = data['items'] / 1e6
        mean = data['throughput_mean']
        std = data['throughput_std']
        
        indices = np.arange(0, len(items), plot_every_n_points)
        items_sampled = items[indices]
        mean_sampled = mean[indices]
        std_sampled = std[indices]
        
        plot_line_with_error(ax_throughput, items_sampled, mean_sampled, std_sampled, style)
    
    if geometric_limit_idx is not None and gs_data is not None:
        limit_items = gs_data['items'][geometric_limit_idx] / 1e6
        ax_throughput.axvline(x=limit_items, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, 
                             label=f'GS limit ({initial_memory_kb}KB)')
    
    style_axis(ax_throughput, font_config, 
              ylabel='Throughput (Mops/s)',
              title='Shrinking Experiment Results')
    
    ax_query = axes[1]
    for sketch_name, data in aggregated.items():
        style = styles.get(sketch_name, {})
        items = data['items'] / 1e6
        mean = data['query_throughput_mean']
        std = data['query_throughput_std']
        
        indices = np.arange(0, len(items), plot_every_n_points)
        items_sampled = items[indices]
        mean_sampled = mean[indices]
        std_sampled = std[indices]
        
        plot_line_with_error(ax_query, items_sampled, mean_sampled, std_sampled, style)
    
    if geometric_limit_idx is not None and gs_data is not None:
        limit_items = gs_data['items'][geometric_limit_idx] / 1e6
        ax_query.axvline(x=limit_items, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    style_axis(ax_query, font_config, ylabel='Query Throughput (Mops/s)')
    
    ax_are = axes[2]
    for sketch_name, data in aggregated.items():
        style = styles.get(sketch_name, {})
        items = data['items'] / 1e6
        mean = data['are_mean']
        std = data['are_std']
        
        indices = np.arange(0, len(items), plot_every_n_points)
        items_sampled = items[indices]
        mean_sampled = mean[indices]
        std_sampled = std[indices]
        
        plot_line_with_error(ax_are, items_sampled, mean_sampled, std_sampled, style)
    
    if geometric_limit_idx is not None and gs_data is not None:
        limit_items = gs_data['items'][geometric_limit_idx] / 1e6
        ax_are.axvline(x=limit_items, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    style_axis(ax_are, font_config, ylabel='ARE', use_log_scale=True)
    
    ax_aae = axes[3]
    for sketch_name, data in aggregated.items():
        style = styles.get(sketch_name, {})
        items = data['items'] / 1e6
        mean = data['aae_mean']
        std = data['aae_std']
        
        indices = np.arange(0, len(items), plot_every_n_points)
        items_sampled = items[indices]
        mean_sampled = mean[indices]
        std_sampled = std[indices]
        
        plot_line_with_error(ax_aae, items_sampled, mean_sampled, std_sampled, style)
    
    if geometric_limit_idx is not None and gs_data is not None:
        limit_items = gs_data['items'][geometric_limit_idx] / 1e6
        ax_aae.axvline(x=limit_items, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    style_axis(ax_aae, font_config, ylabel='AAE', use_log_scale=True)
    
    ax_memory = axes[4]
    for sketch_name, data in aggregated.items():
        style = styles.get(sketch_name, {})
        items = data['items'] / 1e6
        mean = data['memory_mean']
        std = data['memory_std']
        
        indices = np.arange(0, len(items), plot_every_n_points)
        items_sampled = items[indices]
        mean_sampled = mean[indices]
        std_sampled = std[indices]
        
        plot_line_with_error(ax_memory, items_sampled, mean_sampled, std_sampled, style)
    
    if geometric_limit_idx is not None and gs_data is not None:
        limit_items = gs_data['items'][geometric_limit_idx] / 1e6
        ax_memory.axvline(x=limit_items, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    style_axis(ax_memory, font_config, 
              xlabel='Items Processed (millions)',
              ylabel='Memory (KB)')
    
    create_shared_legend(fig, ax_throughput, ncol=3, font_config=font_config,
                        bbox_to_anchor=(0.5, 1.02), top_adjust=0.96)
    
    plt.tight_layout()
    
    save_figure(fig, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize shrinking experiment results'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='output/shrinking_results.json',
        help='Path to input JSON file (default: output/shrinking_results.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/shrinking_results.png',
        help='Path to output figure (default: output/shrinking_results.png)'
    )
    parser.add_argument(
        '--plot-every',
        type=int,
        default=1,
        help='Plot 1 point for every N points (default: 1, plot all points). Use higher values (e.g., 5) to reduce clutter.'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    
    config = data['config']
    results = data['results']
    
    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {})
    
    resketch_config = sketch_config.get('resketch', {})
    geometric_config = sketch_config.get('geometric', {})
    
    dataset_type = exp_config.get('dataset_type', config.get('dataset_type', 'zipf'))
    total_items = exp_config.get('total_items', config.get('total_items', 10000000))
    initial_memory_kb = exp_config.get('initial_memory_kb', config.get('initial_memory_kb', 1600))
    max_memory_kb = exp_config.get('max_memory_kb', config.get('max_memory_kb', 6400))
    final_memory_kb = exp_config.get('final_memory_kb', config.get('final_memory_kb', 32))
    shrinking_interval = exp_config.get('shrinking_interval', config.get('shrinking_interval', 10000))
    memory_decrement_kb = exp_config.get('memory_decrement_kb', config.get('memory_decrement_kb', 32))
    repetitions = exp_config.get('repetitions', config.get('repetitions', 1))
    
    resketch_depth = resketch_config.get('depth', config.get('resketch_depth'))
    resketch_kll_k = resketch_config.get('kll_k', config.get('resketch_kll_k'))
    geometric_depth = geometric_config.get('depth', config.get('geometric_depth'))
    
    print("\nExperiment Configuration:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total Items: {total_items}")
    print(f"  Initial Memory: {initial_memory_kb} KB")
    print(f"  Max Memory (Warmup): {max_memory_kb} KB")
    print(f"  Final Memory: {final_memory_kb} KB")
    print(f"  Shrinking Interval: {shrinking_interval} items")
    print(f"  Memory Decrement: {memory_decrement_kb} KB")
    print(f"  Repetitions: {repetitions}")
    if resketch_depth is not None:
        print(f"  ReSketch Depth: {resketch_depth}, KLL K: {resketch_kll_k}")
    if geometric_depth is not None:
        print(f"  Geometric Depth: {geometric_depth}")
    
    print("\nAggregating results across repetitions...")
    aggregated = aggregate_results(results)
    
    print("Generating plots...")
    print(f"  Plotting every {args.plot_every} point(s)")
    plot_results(config, aggregated, args.output, plot_every_n_points=args.plot_every)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
