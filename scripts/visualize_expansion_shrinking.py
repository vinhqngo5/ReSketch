import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path

from visualization_common import (
    load_material_colors, setup_fonts, get_sketch_styles,
    style_axis, plot_line_with_error, save_figure, create_shared_legend, load_results
)

def aggregate_expansion_results(results_dict):
    aggregated = {}
    
    for sketch_name in ['ReSketch', 'GeometricSketch', 'StaticReSketch', 'CountMin', 'DynamicSketch']:
        if sketch_name not in results_dict:
            continue
            
        repetitions = results_dict[sketch_name]
        items_list = []
        throughput_list = []
        query_throughput_list = []
        are_list = []
        aae_list = []
        are_var_list = []
        aae_var_list = []
        memory_list = []
        
        for rep_data in repetitions:
            checkpoints = rep_data['checkpoints']
            items = [cp['items_processed'] for cp in checkpoints]
            throughput = [cp['throughput_mops'] for cp in checkpoints]
            query_throughput = [cp['query_throughput_mops'] for cp in checkpoints]
            are = [cp['are'] for cp in checkpoints]
            aae = [cp['aae'] for cp in checkpoints]
            are_var = [np.log(cp.get('are_variance', 0.0)) for cp in checkpoints]
            aae_var = [np.log(cp.get('aae_variance', 0.0)) for cp in checkpoints]
            memory = [cp['memory_kb'] for cp in checkpoints]
            
            items_list.append(items)
            throughput_list.append(throughput)
            query_throughput_list.append(query_throughput)
            are_list.append(are)
            aae_list.append(aae)
            are_var_list.append(are_var)
            aae_var_list.append(aae_var)
            memory_list.append(memory)
        
        items_array = np.array(items_list[0])
        throughput_array = np.array(throughput_list)
        query_throughput_array = np.array(query_throughput_list)
        are_array = np.array(are_list)
        aae_array = np.array(aae_list)
        are_var_array = np.array(are_var_list)
        aae_var_array = np.array(aae_var_list)
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
            'are_var_mean': np.mean(are_var_array, axis=0),
            'are_var_std': np.std(are_var_array, axis=0),
            'aae_var_mean': np.mean(aae_var_array, axis=0),
            'aae_var_std': np.std(aae_var_array, axis=0),
            'memory_mean': np.mean(memory_array, axis=0),
            'memory_std': np.std(memory_array, axis=0),
        }
    
    return aggregated

def aggregate_shrinking_results(results_dict, phase_suffix):
    aggregated = {}
    
    for sketch_name in ['ReSketch', 'GeometricSketch']:
        full_name = f"{sketch_name}_{phase_suffix}"
        if full_name not in results_dict:
            continue
            
        repetitions = results_dict[full_name]
        items_list = []
        throughput_list = []
        query_throughput_list = []
        are_list = []
        aae_list = []
        are_var_list = []
        aae_var_list = []
        memory_list = []
        geometric_cannot_shrink_list = []
        
        for rep_data in repetitions:
            checkpoints = rep_data['checkpoints']
            items = [cp['items_in_phase'] if phase_suffix == 'ShrinkWithData' else cp['items_processed'] for cp in checkpoints]
            throughput = [cp['throughput_mops'] for cp in checkpoints]
            query_throughput = [cp['query_throughput_mops'] for cp in checkpoints]
            are = [cp['are'] for cp in checkpoints]
            aae = [cp['aae'] for cp in checkpoints]
            are_var = [cp.get('are_variance', 0.0) for cp in checkpoints]
            aae_var = [cp.get('aae_variance', 0.0) for cp in checkpoints]
            memory = [cp['memory_kb'] for cp in checkpoints]
            geometric_cannot_shrink = [cp.get('geometric_cannot_shrink', False) for cp in checkpoints]
            
            items_list.append(items)
            throughput_list.append(throughput)
            query_throughput_list.append(query_throughput)
            are_list.append(are)
            aae_list.append(aae)
            are_var_list.append(are_var)
            aae_var_list.append(aae_var)
            memory_list.append(memory)
            geometric_cannot_shrink_list.append(geometric_cannot_shrink)
        
        items_array = np.array(items_list[0])
        throughput_array = np.array(throughput_list)
        query_throughput_array = np.array(query_throughput_list)
        are_array = np.array(are_list)
        aae_array = np.array(aae_list)
        are_var_array = np.array(are_var_list)
        aae_var_array = np.array(aae_var_list)
        memory_array = np.array(memory_list)
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
            'are_var_mean': np.mean(are_var_array, axis=0),
            'are_var_std': np.std(are_var_array, axis=0),
            'aae_var_mean': np.mean(aae_var_array, axis=0),
            'aae_var_std': np.std(aae_var_array, axis=0),
            'memory_mean': np.mean(memory_array, axis=0),
            'memory_std': np.std(memory_array, axis=0),
            'geometric_cannot_shrink': geometric_cannot_shrink_array,
        }
    
    return aggregated

def plot_expansion_results(config, aggregated, output_path, plot_every_n_points=1, show_within_variance=True):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    font_config = setup_fonts(__file__)
    
    num_plots = 7 if show_within_variance else 5
    fig_height = (9 if show_within_variance else 7) * 0.56
    fig, axes = plt.subplots(num_plots, 1, figsize=(1.665, fig_height), sharex=True)
    
    styles = get_sketch_styles(material_colors)
    
    # Throughput plot
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
              ylabel='Update',)
            #   title='Expansion Experiment Results')
    ax_throughput.text(0.12, 1.00, 'Mops/s', transform=ax_throughput.transAxes,
                      fontsize=font_config['tick_size'], va='bottom', ha='right',
                      fontfamily=font_config['family'])
    ax_throughput.text(0.12, 1.00, 'Mops/s', transform=ax_throughput.transAxes,
                      fontsize=font_config['tick_size'], va='bottom', ha='right',
                      fontfamily=font_config['family'])
    
    # Query throughput plot
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
    
    style_axis(ax_query, font_config, ylabel='Query')
    ax_query.text(0.12, 1.00, 'Mops/s', transform=ax_query.transAxes,
                 fontsize=font_config['tick_size'], va='bottom', ha='right',
                 fontfamily=font_config['family'])
    ax_query.text(0.12, 1.00, 'Mops/s', transform=ax_query.transAxes,
                 fontsize=font_config['tick_size'], va='bottom', ha='right',
                 fontfamily=font_config['family'])
    
    # ARE plot
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
    
    # AAE plot
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
    
    if show_within_variance:
        # ARE within-run variance
        ax_are_var = axes[4]
        for sketch_name, data in aggregated.items():
            style = styles.get(sketch_name, {})
            items = data['items'] / 1e6
            mean = data['are_var_mean']
            std = data['are_var_std']
            
            indices = np.arange(0, len(items), plot_every_n_points)
            items_sampled = items[indices]
            mean_sampled = mean[indices]
            std_sampled = std[indices]
            
            plot_line_with_error(ax_are_var, items_sampled, mean_sampled, std_sampled, style)
        
        style_axis(ax_are_var, font_config, ylabel='ARE Variance')
        
        # AAE within-run variance
        ax_aae_var = axes[5]
        for sketch_name, data in aggregated.items():
            style = styles.get(sketch_name, {})
            items = data['items'] / 1e6
            mean = data['aae_var_mean']
            std = data['aae_var_std']
            
            indices = np.arange(0, len(items), plot_every_n_points)
            items_sampled = items[indices]
            mean_sampled = mean[indices]
            std_sampled = std[indices]
            
            plot_line_with_error(ax_aae_var, items_sampled, mean_sampled, std_sampled, style)
        
        style_axis(ax_aae_var, font_config, ylabel='AAE Variance')
    
    # Memory plot is always at the bottom
    memory_idx = 6 if show_within_variance else 4
    ax_memory = axes[memory_idx]
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
              ylabel='Memory')
    ax_memory.text(0.05, 1.00, 'KiB', transform=ax_memory.transAxes,
                  fontsize=font_config['tick_size'], va='bottom', ha='right',
                  fontfamily=font_config['family'])
    
    top_adjust = 0.89 if show_within_variance else 0.91
    create_shared_legend(fig, ax_throughput, ncol=2, font_config=font_config,
                        bbox_to_anchor=(0.4, 1.00), top_adjust=top_adjust)
    
    plt.subplots_adjust(left=-0.1, right=1, top=top_adjust, bottom=0, hspace=0.18)
    
    save_figure(fig, output_path)

def plot_shrinking_results(config, no_data_aggregated, with_data_aggregated, expansion_aggregated, output_path, plot_every_n_points=1, show_within_variance=True):
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    font_config = setup_fonts(__file__)
    
    num_plots = 7 if show_within_variance else 5
    fig_height = (9 if show_within_variance else 7) * 0.56
    fig, axes = plt.subplots(num_plots, 1, figsize=(1.665, fig_height), sharex=True)
    
    styles = get_sketch_styles(material_colors)
    
    exp_config = config.get('experiment', config)
    m0_kb = exp_config.get('m0_kb', 32)
    m2_kb = exp_config.get('m2_kb', 16)
    
    # Get M1 from last expansion checkpoint
    print("  Extracting M1 (final expansion memory) from last expansion checkpoint...")
    m1_data = {}
    
    for sketch_name in ['ReSketch', 'GeometricSketch']:
        if sketch_name in expansion_aggregated:
            exp_data = expansion_aggregated[sketch_name]
            # Get last expansion checkpoint
            last_idx = -1
            m1_data[sketch_name] = {
                'memory_kb': exp_data['memory_mean'][last_idx],
                'items_processed': exp_data['items'][last_idx],
                'throughput_mean': exp_data['throughput_mean'][last_idx],
                'throughput_std': exp_data['throughput_std'][last_idx],
                'query_throughput_mean': exp_data['query_throughput_mean'][last_idx],
                'query_throughput_std': exp_data['query_throughput_std'][last_idx],
                'are_mean': exp_data['are_mean'][last_idx],
                'are_std': exp_data['are_std'][last_idx],
                'aae_mean': exp_data['aae_mean'][last_idx],
                'aae_std': exp_data['aae_std'][last_idx],
                'are_var_mean': exp_data['are_var_mean'][last_idx],
                'are_var_std': exp_data['are_var_std'][last_idx],
                'aae_var_mean': exp_data['aae_var_mean'][last_idx],
                'aae_var_std': exp_data['aae_var_std'][last_idx],
            }
            print(f"    {sketch_name} M1: {m1_data[sketch_name]['memory_kb']:.1f} KiB at {m1_data[sketch_name]['items_processed']:.0f} items")
    
    # Prepend M1 data to WithData variants
    for sketch_name in ['ReSketch', 'GeometricSketch']:
        if sketch_name in with_data_aggregated and sketch_name in m1_data:
            shrink_data = with_data_aggregated[sketch_name]
            m1_point = m1_data[sketch_name]
            
            shrink_data['memory_mean'] = np.concatenate([[m1_point['memory_kb']], shrink_data['memory_mean']])
            shrink_data['memory_std'] = np.concatenate([[0.0], shrink_data['memory_std']])
            shrink_data['items'] = np.concatenate([[0.0], shrink_data['items']])
            shrink_data['throughput_mean'] = np.concatenate([[0.0], shrink_data['throughput_mean']])
            shrink_data['throughput_std'] = np.concatenate([[0.0], shrink_data['throughput_std']])
            shrink_data['query_throughput_mean'] = np.concatenate([[m1_point['query_throughput_mean']], shrink_data['query_throughput_mean']])
            shrink_data['query_throughput_std'] = np.concatenate([[m1_point['query_throughput_std']], shrink_data['query_throughput_std']])
            shrink_data['are_mean'] = np.concatenate([[m1_point['are_mean']], shrink_data['are_mean']])
            shrink_data['are_std'] = np.concatenate([[m1_point['are_std']], shrink_data['are_std']])
            shrink_data['aae_mean'] = np.concatenate([[m1_point['aae_mean']], shrink_data['aae_mean']])
            shrink_data['aae_std'] = np.concatenate([[m1_point['aae_std']], shrink_data['aae_std']])
            shrink_data['are_var_mean'] = np.concatenate([[m1_point['are_var_mean']], shrink_data['are_var_mean']])
            shrink_data['are_var_std'] = np.concatenate([[m1_point['are_var_std']], shrink_data['are_var_std']])
            shrink_data['aae_var_mean'] = np.concatenate([[m1_point['aae_var_mean']], shrink_data['aae_var_mean']])
            shrink_data['aae_var_std'] = np.concatenate([[m1_point['aae_var_std']], shrink_data['aae_var_std']])
            shrink_data['geometric_cannot_shrink'] = np.concatenate([[False], shrink_data['geometric_cannot_shrink']])
    
    # Prepend M1 data to NoData variants
    for sketch_name in ['ReSketch', 'GeometricSketch']:
        if sketch_name in no_data_aggregated and sketch_name in m1_data:
            shrink_data = no_data_aggregated[sketch_name]
            m1_point = m1_data[sketch_name]
            
            shrink_data['memory_mean'] = np.concatenate([[m1_point['memory_kb']], shrink_data['memory_mean']])
            shrink_data['memory_std'] = np.concatenate([[0.0], shrink_data['memory_std']])
            shrink_data['items'] = np.concatenate([[0.0], shrink_data['items']])
            shrink_data['throughput_mean'] = np.concatenate([[0.0], shrink_data['throughput_mean']])
            shrink_data['throughput_std'] = np.concatenate([[0.0], shrink_data['throughput_std']])
            shrink_data['query_throughput_mean'] = np.concatenate([[m1_point['query_throughput_mean']], shrink_data['query_throughput_mean']])
            shrink_data['query_throughput_std'] = np.concatenate([[m1_point['query_throughput_std']], shrink_data['query_throughput_std']])
            shrink_data['are_mean'] = np.concatenate([[m1_point['are_mean']], shrink_data['are_mean']])
            shrink_data['are_std'] = np.concatenate([[m1_point['are_std']], shrink_data['are_std']])
            shrink_data['aae_mean'] = np.concatenate([[m1_point['aae_mean']], shrink_data['aae_mean']])
            shrink_data['aae_std'] = np.concatenate([[m1_point['aae_std']], shrink_data['aae_std']])
            shrink_data['are_var_mean'] = np.concatenate([[m1_point['are_var_mean']], shrink_data['are_var_mean']])
            shrink_data['are_var_std'] = np.concatenate([[m1_point['are_var_std']], shrink_data['are_var_std']])
            shrink_data['aae_var_mean'] = np.concatenate([[m1_point['aae_var_mean']], shrink_data['aae_var_mean']])
            shrink_data['aae_var_std'] = np.concatenate([[m1_point['aae_var_std']], shrink_data['aae_var_std']])
            shrink_data['geometric_cannot_shrink'] = np.concatenate([[False], shrink_data['geometric_cannot_shrink']])
    
    # Calculate M1 from first data point
    m1_kb = 0
    reference_withdata = with_data_aggregated.get('ReSketch')
    if reference_withdata is not None and len(reference_withdata['memory_mean']) > 0:
        m1_kb = reference_withdata['memory_mean'][0]
    elif no_data_aggregated.get('ReSketch') is not None and len(no_data_aggregated['ReSketch']['memory_mean']) > 0:
        m1_kb = no_data_aggregated['ReSketch']['memory_mean'][0]
    
    m1_kb_rounded = 2 ** int(np.floor(np.log2(m1_kb)))
    
    print(f"  Calculated M1 from data: {m1_kb:.1f} KiB (next power-of-2 below: {m1_kb_rounded} KiB)")
    print(f"  M2 target: {m2_kb} KiB")
    
    # Create custom styles for shrinking variants
    shrinking_styles = {}
    for base_name in ['ReSketch', 'GeometricSketch']:
        base_style = styles.get(base_name, {})
        base_color = base_style.get('color', 'blue')
        base_marker = base_style.get('marker', 'o')
        
        shrinking_styles[base_name + '_ShrinkNoData'] = {
            'color': base_color,
            'linestyle': '--',
            'marker': 'o',
            'label': f'{base_name} (∅)',
            # 'linewidth': 2,
            # 'markersize': 6
        }
        
        shrinking_styles[base_name + '_ShrinkWithData'] = {
            'color': base_color,
            'linestyle': '-' if base_name == 'ReSketch' else ':',
            'marker': 'x' if base_name == 'ReSketch' else 'D',
            'label': f'{base_name}',
            # 'linewidth': 2,
            # 'markersize': 6
        }
    
    # Find geometric limit (M0)
    gs_data = with_data_aggregated.get('GeometricSketch', None)
    geometric_limit_idx = None
    geometric_limit_memory = m0_kb
    if gs_data is not None:
        memory_values = gs_data['memory_mean']
        # Find first checkpoint where memory reaches M0 (within tolerance)
        for i in range(len(memory_values)):
            if abs(memory_values[i] - m0_kb) < 5.0:  # Within 5KiB of M0
                geometric_limit_idx = i
                break
    
    # secondary x-axis: checkpoint indices
    num_checkpoints = len(reference_withdata['memory_mean']) if reference_withdata else 0
    checkpoint_indices = np.arange(num_checkpoints)
    
    memory_checkpoints = []
    if reference_withdata is not None:
        memory_checkpoints.append(int(m1_kb))
        current_mem = m1_kb_rounded
        for i in range(1, num_checkpoints):
            memory_checkpoints.append(current_mem)
            current_mem //= 2
    
    print(f"  Actual memory values from data: {[int(m) for m in reference_withdata['memory_mean']] if reference_withdata else []}")
    print(f"  Memory checkpoint labels (power-of-2): {memory_checkpoints}")
    
    print(f"  Number of checkpoints: {num_checkpoints}")
    print(f"  Using equal spacing: checkpoint indices {list(checkpoint_indices)}")
    
    ax2_list = []
    
    metrics = [
        ('throughput_mean', 'throughput_std', 'Update', False, 0, None, False),
        ('query_throughput_mean', 'query_throughput_std', 'Query', False, 1, None, False),
        ('are_mean', 'are_std', 'ARE', True, 2, None, True),
        ('aae_mean', 'aae_std', 'AAE', True, 3, None, True),
    ]
    
    if show_within_variance:
        metrics.extend([
            ('are_var_mean', 'are_var_std', 'ARE Variance', False, 4, None, True),
            ('aae_var_mean', 'aae_var_std', 'AAE Variance', False, 5, None, True),
        ])
    
    memory_idx = 6 if show_within_variance else 4
    metrics.append(('memory_mean', 'memory_std', 'Memory', False, memory_idx, None, True))
    
    for mean_key, std_key, ylabel, use_log, ax_idx, title, plot_nodata in metrics:
        ax = axes[ax_idx]
        
        is_throughput_plot = ax_idx in [0, 1]
        
        if not is_throughput_plot and plot_nodata:
            if ax_idx == 2:
                ax2 = ax.twiny()
                ax2_list.append(ax2)
            else:
                ax2 = ax.twiny()
                ax2.sharex(ax2_list[0])
                ax2_list.append(ax2)
            
            for spine in ax2.spines.values():
                spine.set_visible(False)
        
        # Plot WithData variants
        for sketch_name, data in with_data_aggregated.items():
            full_name = sketch_name + '_ShrinkWithData'
            style = shrinking_styles.get(full_name, styles.get(sketch_name, {}))
            mean = data[mean_key]
            std = data[std_key]
            
            x_positions = checkpoint_indices[:len(mean)]
            
            plot_line_with_error(ax, x_positions, mean, std, style)
        
        # Plot NoData variants
        if plot_nodata and not is_throughput_plot and ax_idx != memory_idx:
            for sketch_name, data in no_data_aggregated.items():
                full_name = sketch_name + '_ShrinkNoData'
                style = shrinking_styles.get(full_name, {})
                mean = data[mean_key]
                std = data[std_key]
                
                x_positions = checkpoint_indices[:len(mean)]
                
                target_ax = ax2 if not is_throughput_plot and plot_nodata else ax
                plot_line_with_error(target_ax, x_positions, mean, std, style)
        
        if geometric_limit_idx is not None and gs_data is not None:
            limit_label = f'GS limit ({int(geometric_limit_memory)}KiB)' if geometric_limit_memory is not None else f'GS limit ({m0_kb}KiB)'
            ax.axvline(x=geometric_limit_idx, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=limit_label if ax_idx == 0 else '')
        
        xlabel_bottom = 'Items Processed (millions)' if ax_idx == num_plots - 1 else None
        style_axis(ax, font_config, ylabel=ylabel, xlabel=xlabel_bottom, title=title, use_log_scale=use_log)
        
        # Add unit labels on top of y-axis
        if ax_idx in [0, 1]:  # Throughput and Query plots
            ax.text(0.12, 1.00, 'Mops/s', transform=ax.transAxes,
                   fontsize=font_config['tick_size'], va='bottom', ha='right',
                   fontfamily=font_config['family'])
        elif ax_idx == memory_idx:  # Memory plot
            ax.text(0.05, 1.00, 'KiB', transform=ax.transAxes,
                   fontsize=font_config['tick_size'], va='bottom', ha='right',
                   fontfamily=font_config['family'])
        
        if not use_log:
            formatter = ScalarFormatter(useOffset=False)
            if ax_idx in [4, 5]:  # Variance plots
                formatter.set_powerlimits((5, 5))
                ax.yaxis.get_offset_text().set_visible(False)
                ax.text(0.05, 1.00, '×10⁵', transform=ax.transAxes,
                       fontsize=font_config['tick_size'], va='bottom', ha='right',
                       fontfamily=font_config['family'])
            else:
                formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, min_n_ticks=3))
        else:
            from matplotlib.ticker import LogFormatterSciNotation
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        
        # Set x-axis limits and ticks for primary axis
        ax.set_xlim(-0.5, num_checkpoints - 0.5)
        ax.set_xticks(checkpoint_indices)
        
        if xlabel_bottom and reference_withdata is not None:
            items_labels = reference_withdata['items'] / 1e6
            ax.set_xticklabels([f'{items:.1f}' for items in items_labels])
        
        # Memory plot zoom inset
        if ax_idx == memory_idx:
            
            axins = inset_axes(ax, width="25%", height="30%", loc='upper right',
                             borderpad=1.5)
            
            # Plot last 2-3 checkpoints in inset for both sketches
            zoom_start = max(0, num_checkpoints - 3)
            zoom_indices = checkpoint_indices[zoom_start:]
            
            for sketch_name, data in with_data_aggregated.items():
                full_name = sketch_name + '_ShrinkWithData'
                style = shrinking_styles.get(full_name, styles.get(sketch_name, {}))
                mean = data['memory_mean']
                std = data['memory_std']
                
                zoom_mean = mean[zoom_start:]
                zoom_std = std[zoom_start:]
                
                plot_line_with_error(axins, zoom_indices, zoom_mean, zoom_std, style)
            
            axins.set_xlim(zoom_start - 0.3, num_checkpoints - 0.7)
            axins.set_xticks(zoom_indices)
            
            tick_labels = []
            for i in zoom_indices:
                mem = int(memory_checkpoints[i])
                tick_labels.append(f'{mem}')
            axins.set_xticklabels(tick_labels, fontsize=font_config['tick_size'] - 1)
            axins.tick_params(labelsize=font_config['tick_size'] - 1)
            axins.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            # axins.set_ylabel('Memory (KiB)', fontsize=font_config['label_size'] - 1)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axins.yaxis.set_major_formatter(formatter)
            axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, min_n_ticks=3))
            
            for spine in axins.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.1)
        
        # ARE/AAE zoom insets
        if ax_idx in [2, 3, 4, 5] and plot_nodata and ax_idx != memory_idx:
            zoom_start = max(0, num_checkpoints - 4)
            zoom_indices = checkpoint_indices[zoom_start:]
            
            # Create 2 insets: one for ReSketch pair, one for GeometricSketch pair
            for inset_idx, sketch_name in enumerate(['GeometricSketch', 'ReSketch']):
                if sketch_name not in with_data_aggregated or sketch_name not in no_data_aggregated:
                    continue
                
                is_upper = (inset_idx == 0)
                
                if ax_idx in [4, 5]:  # Variance plots
                    base_x = 0.28
                else:
                    base_x = 0.72
                
                if is_upper:
                    base_y = 0.65
                else:
                    base_y = 0.30

                
                bbox_tuple = (base_x, base_y, 0.25, 0.25)
                
                axins = inset_axes(ax, width="100%", height="100%", 
                                 bbox_to_anchor=bbox_tuple, bbox_transform=ax.transAxes,
                                 loc='lower left',
                                 borderpad=0)
                
                with_data = with_data_aggregated[sketch_name]
                no_data = no_data_aggregated[sketch_name]
                
                full_name = sketch_name + '_ShrinkWithData'
                style = shrinking_styles.get(full_name, styles.get(sketch_name, {}))
                mean = with_data[mean_key]
                std = with_data[std_key]
                zoom_mean = mean[zoom_start:]
                zoom_std = std[zoom_start:]
                plot_line_with_error(axins, zoom_indices[:len(zoom_mean)], zoom_mean, zoom_std, style)
                
                full_name = sketch_name + '_ShrinkNoData'
                style = shrinking_styles.get(full_name, {})  
                mean = no_data[mean_key]
                std = no_data[std_key]
                zoom_mean = mean[zoom_start:]
                zoom_std = std[zoom_start:]
                plot_line_with_error(axins, zoom_indices[:len(zoom_mean)], zoom_mean, zoom_std, style)
                
                axins.set_xlim(zoom_start - 0.3, num_checkpoints - 0.7)
                
                def format_tick_inset(x, pos=None):
                    if x == 0: return "0"
                    if abs(x) >= 100:
                        return f"{int(x)}"
                    if abs(x - round(x)) < 0.05:
                        return f"{int(round(x))}"
                    return f"{x:.2g}"

                
                current_ylim = axins.get_ylim()
                ymin, ymax = current_ylim
                
                if use_log:
                    axins.set_yscale('log')
                    import math
                    # Safety check for log
                    if ymin <= 0: ymin = 1e-10
                    if ymax <= 0: ymax = 1e-9
                    
                    log_min = math.log10(ymin)
                    log_max = math.log10(ymax)
                    log_mid = (log_min + log_max) / 2.0
                    
                    forced_ticks = [10**log_min, 10**log_mid, 10**log_max]
                    axins.set_yticks(forced_ticks)
                    axins.set_yticklabels([format_tick_inset(t) for t in forced_ticks], fontsize=font_config['tick_size'], fontfamily=font_config['family'])
                    axins.minorticks_off()
                    axins.yaxis.set_minor_locator(plt.NullLocator())
                else:
                    forced_ticks = [ymin, (ymin + ymax) / 2.0, ymax]
                    axins.set_yticks(forced_ticks)
                    axins.set_yticklabels([format_tick_inset(t) for t in forced_ticks], fontsize=font_config['tick_size'], fontfamily=font_config['family'])
                    axins.yaxis.set_minor_locator(plt.NullLocator())
                
                axins.tick_params(labelsize=font_config['tick_size'] - 2)
                axins.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                axins.set_xticklabels([])
                
                
                for spine in axins.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.1)
        
        # Style secondary axis
        if not is_throughput_plot and plot_nodata:
            if ax_idx == 2:
                # ax2.set_xlabel('Memory (KiB)', fontsize=font_config['label_size'])
                
                # Set secondary axis to match primary
                ax2.set_xlim(-0.5, num_checkpoints - 0.5)
                
                # Label with power-of-2 memory values at checkpoint positions
                # make the xticks label closer to the ticks
                ax2.set_xticks(checkpoint_indices)
                labels = [f'{int(m)}' for m in memory_checkpoints]
                if labels:
                    labels[-1] += ' KiB'
                ax2.set_xticklabels(labels)
                ax2.tick_params(labelsize=font_config['tick_size'], pad=-2)
                # make xticks shorter
                ax2.tick_params(length=2)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
            else:
                ax2.tick_params(labeltop=False)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.1)
            spine.set_color('black')
    
    # Create shared legend
    handles_list = []
    labels_list = []
    seen_labels = set()
    
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                handles_list.append(handle)
                labels_list.append(label)
                seen_labels.add(label)
    
    for ax2 in ax2_list:
        h, l = ax2.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                handles_list.append(handle)
                labels_list.append(label)
                seen_labels.add(label)
    
    top_adjust = 0.89 if show_within_variance else 0.91
    if handles_list:
        fig.legend(handles_list, labels_list,
                  loc='upper center',
                  bbox_to_anchor=(0.38, 1.00),
                  ncol=2,
                  frameon=False,
                  handlelength=1.5,
                  handletextpad=0.2,
                  columnspacing=0.3,
                  prop={'family': font_config['family'], 'size': font_config['legend_size']})
    
    plt.subplots_adjust(left=-0.1, right=1, top=top_adjust, bottom=0, hspace=0.18)
    save_figure(fig, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize expansion-shrinking experiment results'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='output/expansion_shrinking_results.json',
        help='Path to input JSON file (default: output/expansion_shrinking_results.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/expansion_shrinking_results',
        help='Base path for output figures (without extension), will create _expansion.png and _shrinking.png'
    )
    parser.add_argument(
        '--plot-every',
        type=int,
        default=1,
        help='Plot 1 point for every N points (default: 1)'
    )
    parser.add_argument(
        '--show-within-variance',
        action='store_true',
        help='Show within-run variance plots'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    
    config = data['config']
    results = data['results']
    
    exp_config = config.get('experiment', config)
    
    m0_kb = exp_config.get('m0_kb', 32)
    m2_kb = exp_config.get('m2_kb', 0)
    expansion_items = exp_config.get('expansion_items', 0)
    shrinking_items = exp_config.get('shrinking_items', 0)
    repetitions = exp_config.get('repetitions', 1)
    
    # Plot expansion phase
    print("\nAggregating expansion results...")
    expansion_aggregated = aggregate_expansion_results(results)
    
    # Calculate M1 from actual expansion data (last checkpoint)
    m1_kb_resketch = 0
    m1_kb_geometric = 0
    if 'ReSketch' in expansion_aggregated:
        m1_kb_resketch = expansion_aggregated['ReSketch']['memory_mean'][-1]
    if 'GeometricSketch' in expansion_aggregated:
        m1_kb_geometric = expansion_aggregated['GeometricSketch']['memory_mean'][-1]
    
    print("\nExperiment Configuration:")
    print(f"  M0 (Initial): {m0_kb} KiB")
    print(f"  M1 (After Expansion): ReSketch={m1_kb_resketch:.0f} KiB, GeometricSketch={m1_kb_geometric:.0f} KiB")
    print(f"  M2 (Target): {m2_kb} KiB")
    print(f"  Expansion Items: {expansion_items}")
    print(f"  Shrinking Items: {shrinking_items}")
    print(f"  Repetitions: {repetitions}")
    
    expansion_output = args.output + '_expansion.png'
    print(f"Generating expansion plot: {expansion_output}")
    plot_expansion_results(config, expansion_aggregated, expansion_output, 
                          plot_every_n_points=args.plot_every,
                          show_within_variance=args.show_within_variance)
    
    # Plot shrinking phases (combined - both NoData and WithData)
    print("\nAggregating shrinking results...")
    shrink_no_data_aggregated = aggregate_shrinking_results(results, 'ShrinkNoData')
    shrink_with_data_aggregated = aggregate_shrinking_results(results, 'ShrinkWithData')
    
    shrinking_output = args.output + '_shrinking.png'
    print(f"Generating combined shrinking plot: {shrinking_output}")
    plot_shrinking_results(config, shrink_no_data_aggregated, shrink_with_data_aggregated, 
                          expansion_aggregated,  # Pass expansion data to get M1
                          shrinking_output,
                          plot_every_n_points=args.plot_every,
                          show_within_variance=args.show_within_variance)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
