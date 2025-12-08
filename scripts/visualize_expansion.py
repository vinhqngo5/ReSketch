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
        
        for rep_data in repetitions:
            checkpoints = rep_data['checkpoints']
            items = [cp['items_processed'] for cp in checkpoints]
            throughput = [cp['throughput_mops'] for cp in checkpoints]
            query_throughput = [cp['query_throughput_mops'] for cp in checkpoints]
            are = [cp['are'] for cp in checkpoints]
            aae = [cp['aae'] for cp in checkpoints]
            memory = [cp.get('memory_bytes', cp.get('memory_kb', 0) * 1024) / 1024 for cp in checkpoints]
            
            items_list.append(items)
            throughput_list.append(throughput)
            query_throughput_list.append(query_throughput)
            are_list.append(are)
            aae_list.append(aae)
            memory_list.append(memory)
        
        items_array = np.array(items_list[0])
        throughput_array = np.array(throughput_list)
        query_throughput_array = np.array(query_throughput_list)
        are_array = np.array(are_list)
        aae_array = np.array(aae_list)
        memory_array = np.array(memory_list)
        
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
        }
    
    return aggregated

def plot_results(config, aggregated, output_path, plot_every_n_points=1):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    
    font_config = setup_fonts(__file__)
    
    fig, axes = plt.subplots(5, 1, figsize=(6.45, 10), sharex=True)
    
    styles = get_sketch_styles(material_colors)
    
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
    
    style_axis(ax_throughput, font_config, 
              ylabel='Throughput (Mops/s)',
              title='Expansion Experiment Results')
    
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
    
    style_axis(ax_memory, font_config, 
              xlabel='Items Processed (millions)',
              ylabel='Memory (KB)')
    
    create_shared_legend(fig, ax_throughput, ncol=4, font_config=font_config,
                        bbox_to_anchor=(0.5, 1.02), top_adjust=0.96)
    
    plt.tight_layout()
    
    save_figure(fig, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize expansion experiment results'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='output/expansion_results.json',
        help='Path to input JSON file (default: output/expansion_results.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/expansion_results.png',
        help='Path to output figure (default: output/expansion_results.png)'
    )
    parser.add_argument(
        '--plot-every',
        type=int,
        default=4,
        help='Plot 1 point for every N points (default: 1, plot all points). Use higher values (e.g., 5) to reduce clutter.'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    
    config = data['config']
    results = data['results']
    
    
    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {})
    
    countmin_config = sketch_config.get('countmin', {})
    resketch_config = sketch_config.get('resketch', {})
    geometric_config = sketch_config.get('geometric', {})
    dynamic_config = sketch_config.get('dynamic', {})
    
    dataset_type = exp_config.get('dataset_type', config.get('dataset_type', 'zipf'))
    total_items = exp_config.get('total_items', config.get('total_items', 10000000))
    stream_size = exp_config.get('stream_size', config.get('stream_size', 10000000))
    initial_memory_kb = exp_config.get('initial_memory_kb', config.get('initial_memory_kb', 32))
    expansion_interval = exp_config.get('expansion_interval', config.get('expansion_interval', 100000))
    memory_increment_kb = exp_config.get('memory_increment_kb', config.get('memory_increment_kb', 32))
    repetitions = exp_config.get('repetitions', config.get('repetitions', 1))
    
    countmin_depth = countmin_config.get('depth', config.get('countmin_depth'))
    resketch_depth = resketch_config.get('depth', config.get('resketch_depth'))
    resketch_kll_k = resketch_config.get('kll_k', config.get('resketch_kll_k'))
    geometric_depth = geometric_config.get('depth', config.get('geometric_depth'))
    dynamic_depth = dynamic_config.get('depth', config.get('dynamic_depth'))
    
    print("\nExperiment Configuration:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total Items: {total_items}")
    print(f"  Dataset Size: {stream_size}")
    print(f"  Initial Memory: {initial_memory_kb} KB")
    print(f"  Expansion Interval: {expansion_interval} items")
    print(f"  Memory Increment: {memory_increment_kb} KB")
    print(f"  Repetitions: {repetitions}")
    if countmin_depth is not None:
        print(f"  CountMin Depth: {countmin_depth}")
    if resketch_depth is not None:
        print(f"  ReSketch Depth: {resketch_depth}, KLL K: {resketch_kll_k}")
    if geometric_depth is not None:
        print(f"  Geometric Depth: {geometric_depth}")
    if dynamic_depth is not None:
        print(f"  Dynamic Depth: {dynamic_depth}")
    
    print("\nAggregating results across repetitions...")
    aggregated = aggregate_results(results)
    
    print("Generating plots...")
    print(f"  Plotting every {args.plot_every} point(s)")
    plot_results(config, aggregated, args.output, plot_every_n_points=args.plot_every)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
