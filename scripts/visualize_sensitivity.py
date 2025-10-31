import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from visualization_common import (
    load_material_colors, setup_fonts,
    get_resketch_depth_styles, get_countmin_style,
    style_axis, plot_line_with_error, plot_horizontal_baseline,
    save_figure, create_shared_legend, load_results
)

def aggregate_results(results_dict):
    aggregated = {}
    
    for config_name, repetitions in results_dict.items():
        throughput_list = []
        query_throughput_list = []
        are_list = []
        aae_list = []
        memory_list = []
        k_value = None
        depth = None
        
        for rep_data in repetitions:
            results = rep_data['results']
            if len(results) > 0:
                result = results[0]
                throughput_list.append(result['throughput_mops'])
                query_throughput_list.append(result['query_throughput_mops'])
                are_list.append(result['are'])
                aae_list.append(result['aae'])
                memory_list.append(result['memory_bytes'] / 1024)
                if result['algorithm'] == 'ReSketch':
                    k_value = result['k_value']
                    depth = result['depth']
        
        aggregated[config_name] = {
            'k_value': k_value,
            'depth': depth,
            'throughput_mean': np.mean(throughput_list),
            'throughput_std': np.std(throughput_list),
            'query_throughput_mean': np.mean(query_throughput_list),
            'query_throughput_std': np.std(query_throughput_list),
            'are_mean': np.mean(are_list),
            'are_std': np.std(are_list),
            'aae_mean': np.mean(aae_list),
            'aae_std': np.std(aae_list),
            'memory_mean': np.mean(memory_list),
            'memory_std': np.std(memory_list),
        }
    
    return aggregated

def plot_results(config, aggregated, output_path):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    
    font_config = setup_fonts(__file__)
    
    cm_style = get_countmin_style(material_colors)
    depth_styles = get_resketch_depth_styles(material_colors)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    
    cm_data = aggregated.get('CountMin', None)
    
    depth_data = {}
    for config_name, data in aggregated.items():
        if config_name.startswith('ReSketch_d'):
            depth = data['depth']
            k = data['k_value']
            if depth is not None and k is not None:
                if depth not in depth_data:
                    depth_data[depth] = {}
                depth_data[depth][k] = data
    
    depths = sorted(depth_data.keys())
    k_values = sorted(list(depth_data[depths[0]].keys())) if depths else []
    
    print(f"Found depths: {depths}")
    print(f"Found k values: {k_values}")
    
    ax_throughput = axes[0]
    if cm_data:
        plot_horizontal_baseline(ax_throughput, k_values,
                                cm_data['throughput_mean'],
                                cm_data['throughput_std'],
                                cm_style)
    
    for depth in depths:
        style = depth_styles.get(depth, depth_styles[4])
        data_dict = depth_data[depth]
        
        throughput_mean = np.array([data_dict[k]['throughput_mean'] for k in k_values])
        throughput_std = np.array([data_dict[k]['throughput_std'] for k in k_values])
        
        plot_line_with_error(ax_throughput, k_values, throughput_mean, throughput_std, style)
    
    style_axis(ax_throughput, font_config, 
              ylabel='Update Throughput (Mops/s)', 
              xlabel='k value',
              title='Sensitivity Analysis')
    
    ax_query = axes[1]
    if cm_data:
        plot_horizontal_baseline(ax_query, k_values,
                                cm_data['query_throughput_mean'],
                                cm_data['query_throughput_std'],
                                cm_style)
    
    for depth in depths:
        style = depth_styles.get(depth, depth_styles[4])
        data_dict = depth_data[depth]
        
        query_mean = np.array([data_dict[k]['query_throughput_mean'] for k in k_values])
        query_std = np.array([data_dict[k]['query_throughput_std'] for k in k_values])
        
        plot_line_with_error(ax_query, k_values, query_mean, query_std, style)
    
    style_axis(ax_query, font_config, 
              ylabel='Query Throughput (Mops/s)', 
              xlabel='k value')
    
    ax_are = axes[2]
    if cm_data:
        plot_horizontal_baseline(ax_are, k_values,
                                cm_data['are_mean'],
                                cm_data['are_std'],
                                cm_style)
    
    for depth in depths:
        style = depth_styles.get(depth, depth_styles[4])
        data_dict = depth_data[depth]
        
        are_mean = np.array([data_dict[k]['are_mean'] for k in k_values])
        are_std = np.array([data_dict[k]['are_std'] for k in k_values])
        
        plot_line_with_error(ax_are, k_values, are_mean, are_std, style)
    
    style_axis(ax_are, font_config, 
              ylabel='ARE', 
              xlabel='k value',
              use_log_scale=False)
    
    ax_aae = axes[3]
    if cm_data:
        plot_horizontal_baseline(ax_aae, k_values,
                                cm_data['aae_mean'],
                                cm_data['aae_std'],
                                cm_style)
    
    for depth in depths:
        style = depth_styles.get(depth, depth_styles[4])
        data_dict = depth_data[depth]
        
        aae_mean = np.array([data_dict[k]['aae_mean'] for k in k_values])
        aae_std = np.array([data_dict[k]['aae_std'] for k in k_values])
        
        plot_line_with_error(ax_aae, k_values, aae_mean, aae_std, style)
    
    style_axis(ax_aae, font_config, 
              ylabel='AAE', 
              xlabel='k value',
              use_log_scale=False)
    
    create_shared_legend(fig, ax_throughput, ncol=4, font_config=font_config,
                        bbox_to_anchor=(0.5, 1.02), top_adjust=0.96)
    
    plt.tight_layout()
    
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description='Visualize sensitivity analysis results')
    parser.add_argument('-i', '--input', type=str, required=True,
                      help='Path to input JSON file with results')
    parser.add_argument('-o', '--output', type=str, default='output/sensitivity_results',
                      help='Base path for output files (without extension)')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    
    config = data['config']
    results = data['results']
    
    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {})
    sensitivity_params = config.get('sensitivity_params', {})
    
    countmin_config = sketch_config.get('countmin', {})
    resketch_config = sketch_config.get('resketch', {})
    
    dataset_type = exp_config.get('dataset_type', config.get('dataset_type', 'zipf'))
    total_items = exp_config.get('total_items', config.get('total_items', 10000000))
    memory_budget_kb = exp_config.get('memory_budget_kb', config.get('memory_budget_kb', 32))
    repetitions = exp_config.get('repetitions', config.get('repetitions', 1))
    countmin_depth = countmin_config.get('depth', config.get('countmin_depth', 8))
    resketch_depth = resketch_config.get('depth', config.get('resketch_depth', 4))
    k_values = sensitivity_params.get('k_values', config.get('k_values', []))
    depth_values = sensitivity_params.get('depth_values', config.get('depth_values', []))
    
    print("\nExperiment Configuration:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total Items: {total_items}")
    print(f"  Memory Budget: {memory_budget_kb} KB")
    print(f"  Repetitions: {repetitions}")
    print(f"  Count-Min Depth: {countmin_depth}")
    print(f"  ReSketch Depth: {resketch_depth}")
    print(f"  K values tested: {k_values}")
    print(f"  Depth values tested: {depth_values}")
    
    print("\nAggregating results across repetitions...")
    aggregated = aggregate_results(results)
    
    print("Generating plots...")
    plot_results(config, aggregated, args.output)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
