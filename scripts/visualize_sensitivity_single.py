import argparse
from collections import defaultdict
from typing import NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization_common import (
    load_material_colors, setup_fonts,
    get_resketch_depth_styles, get_countmin_style,
    style_axis, plot_line_with_error, plot_horizontal_baseline,
    save_figure, load_results
)

def aggregate_results(results_dict):
    aggregated = {}

    for config_name, repetitions in results_dict.items():
        throughput_list = []
        query_throughput_list = []
        are_list = []
        aae_list = []
        are_within_var_list = []
        aae_within_var_list = []
        memory_used_list = []
        memory_budget = None
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
                if 'are_within_var' in result:
                    are_within_var_list.append(result.get('are_within_var', 0.0))
                if 'aae_within_var' in result:
                    aae_within_var_list.append(result.get('aae_within_var', 0.0))
                memory_used_list.append(result['memory_used_bytes'])
                memory_budget = result['memory_budget_bytes']

                if result['algorithm'] == 'ReSketch':
                    k_value = result['k_value']
                    depth = result['depth']

        aggregated[config_name] = {
            'k_value': k_value,
            'depth': depth,
            'memory_budget': memory_budget,
            'throughput_mean': np.mean(throughput_list),
            'throughput_std': np.std(throughput_list),
            'query_throughput_mean': np.mean(query_throughput_list),
            'query_throughput_std': np.std(query_throughput_list),
            'are_mean': np.mean(are_list),
            'are_std': np.std(are_list),
            'aae_mean': np.mean(aae_list),
            'aae_std': np.std(aae_list),
            'are_within_var_mean': (np.mean(are_within_var_list) if len(are_within_var_list) > 0 else 0.0),
            'are_within_var_std': (np.std(are_within_var_list) if len(are_within_var_list) > 0 else 0.0),
            'aae_within_var_mean': (np.mean(aae_within_var_list) if len(aae_within_var_list) > 0 else 0.0),
            'aae_within_var_std': (np.std(aae_within_var_list) if len(aae_within_var_list) > 0 else 0.0),
            'memory_used_mean': np.mean(memory_used_list),
            'memory_used_std': np.std(memory_used_list),
        }

    return aggregated

def plot_results(aggregated, output_path, memory_budget_kb, show_within_variance=False):
    material_colors = load_material_colors("scripts/colors/material-colors.json")

    font_config = setup_fonts(__file__)

    cm_style = get_countmin_style(material_colors)
    depth_styles = get_resketch_depth_styles(material_colors)

    cm_data = aggregated.get('CountMin', None)
    cm_data = None

    plot_data = defaultdict(lambda: defaultdict(dict))
    for config_name, data in aggregated.items():
        if config_name.startswith('ReSketch_'):
            depth = data['depth']
            k = data['k_value']
            memory_budget = data['memory_budget']
            plot_data[memory_budget][depth][k] = data

    # Filter to selected memory budget
    memory_budget_bytes = memory_budget_kb * 1024
    if memory_budget_bytes not in plot_data:
        print(f"Error: Memory budget {memory_budget_kb} KiB not found in data")
        available = [mb // 1024 for mb in plot_data.keys()]
        print(f"Available budgets (KiB): {available}")
        return

    memory_data = plot_data[memory_budget_bytes]

    # Remove certain depths from plots
    unwanted_depths = (1, 2)
    depths = sorted(d for d in memory_data if d not in unwanted_depths)

    k_values = sorted(memory_data[depths[0]])

    print(f"Visualizing memory budget: {memory_budget_kb} KiB")
    print(f"Found depths: {depths}")
    print(f"Found k values: {k_values}")

    class Plot(NamedTuple):
        ylabel: str
        mean_col_name: str
        std_col_name: str
        title: Optional[str] = None
        xlabel: str = "k value"

    plots: list[Plot] = [
        Plot("Throughput", "throughput_mean", "throughput_std"),
        Plot("Query", "query_throughput_mean", "query_throughput_std"),
        Plot("ARE", "are_mean", "are_std"),
        Plot("AAE", "aae_mean", "aae_std"),
    ]
    # If requested, add two more plots for within-run variance of ARE and AAE
    if show_within_variance:
        plots.extend([
            Plot("ARE within-var", "are_within_var_mean", "are_within_var_std"),
            Plot("AAE within-var", "aae_within_var_mean", "aae_within_var_std"),
        ])
    fig_width = 3.33 * 1.38
    fig_height = 4.1
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for plot_idx, plot in enumerate(plots):
        ax = axes[plot_idx]
        
        if cm_data:
            plot_horizontal_baseline(ax, k_values,
                                    cm_data[plot.mean_col_name],
                                    cm_data[plot.std_col_name],
                                    cm_style)

        for depth in depths:
            style = depth_styles.get(depth, depth_styles[3])
            depth_data = memory_data[depth]

            data_mean = np.array([depth_data[k][plot.mean_col_name] for k in k_values])
            data_std = np.array([depth_data[k][plot.std_col_name] for k in k_values])

            if plot.mean_col_name == 'aae_within_var_mean':
                data_mean = data_mean / 1e4
                data_std = data_std / 1e4

            plot_line_with_error(ax, k_values, data_mean, data_std, style)

        style_axis(
            ax,
            font_config,
            ylabel=plot.ylabel,
            xlabel=plot.xlabel if plot_idx >= 4 else None,  # Only show xlabel on bottom row
            title=plot.title,
        )
        
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)

        if plot.mean_col_name in ['throughput_mean', 'query_throughput_mean']:
            ax.text(0.25, 1.02, 'Mops/s', transform=ax.transAxes,
                   fontsize=font_config['tick_size'], va='bottom', ha='right',
                   fontfamily=font_config['family'])

        if plot.mean_col_name == 'aae_within_var_mean':
            ax.text(0.12, 1.02, '×10⁴', transform=ax.transAxes,
                   fontsize=font_config['tick_size'], va='bottom', ha='right',
                   fontfamily=font_config['family'])

    # Create legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 0.98),
              ncol=2,
              fontsize=font_config['legend_size'],
              frameon=False,
              handlelength=1.5,
              handletextpad=0.5,
              columnspacing=1.0,
              prop={'family': font_config['family']})

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    save_figure(fig, output_path)


def print_summary_table(agg, memory_budget_kb):
    # Filter to only show configs for the selected memory budget
    names = [name for name in agg.keys() 
             if not name.startswith('ReSketch_') or 
             f'_M{memory_budget_kb}_' in name or name == 'CountMin']

    if not names:
        print(f"No data found for memory budget {memory_budget_kb} KiB")
        return

    thr = [(name, agg[name]['throughput_mean']) for name in names]
    are = [(name, agg[name]['are_mean']) for name in names]
    aae = [(name, agg[name]['aae_mean']) for name in names]

    has_within_var = any(agg[name]['are_within_var_mean'] > 0 for name in names)

    are_var = [(name, agg[name]['are_within_var_mean']) for name in names] if has_within_var else []
    aae_var = [(name, agg[name]['aae_within_var_mean']) for name in names] if has_within_var else []

    thr_sorted = sorted(thr, key=lambda x: x[1], reverse=True)
    are_sorted = sorted(are, key=lambda x: x[1])
    aae_sorted = sorted(aae, key=lambda x: x[1])

    thr_best = thr_sorted[0][0] if thr_sorted else None
    thr_second = thr_sorted[1][0] if len(thr_sorted) > 1 else None
    are_best = are_sorted[0][0] if are_sorted else None
    are_second = are_sorted[1][0] if len(are_sorted) > 1 else None
    aae_best = aae_sorted[0][0] if aae_sorted else None
    aae_second = aae_sorted[1][0] if len(aae_sorted) > 1 else None

    if has_within_var:
        are_var_sorted = sorted(are_var, key=lambda x: x[1])
        aae_var_sorted = sorted(aae_var, key=lambda x: x[1])
        are_var_best = are_var_sorted[0][0] if are_var_sorted else None
        are_var_second = are_var_sorted[1][0] if len(are_var_sorted) > 1 else None
        aae_var_best = aae_var_sorted[0][0] if aae_var_sorted else None
        aae_var_second = aae_var_sorted[1][0] if len(aae_var_sorted) > 1 else None

    print(f'\nSummary Table (Memory Budget: {memory_budget_kb} KiB):')
    if has_within_var:
        hdr = '{:30s} {:>12s} {:>12s} {:>12s} {:>12s} {:>14s} {:>14s}'.format(
            'Config', 'Throughput', 'QueryThroughput', 'ARE', 'AAE', 'ARE_var', 'AAE_var')
    else:
        hdr = '{:30s} {:>12s} {:>12s} {:>12s} {:>10s}'.format('Config', 'Throughput', 'QueryThroughput', 'ARE', 'AAE')
    print(hdr)
    print('-' * len(hdr))

    for name in names:
        row_thr = agg[name]['throughput_mean']
        row_q = agg[name]['query_throughput_mean']
        row_are = agg[name]['are_mean']
        row_aae = agg[name]['aae_mean']

        thr_mark = ''
        are_mark = ''
        aae_mark = ''
        if name == thr_best: thr_mark = '*'
        elif name == thr_second: thr_mark = '**'
        if name == are_best: are_mark = '*'
        elif name == are_second: are_mark = '**'
        if name == aae_best: aae_mark = '*'
        elif name == aae_second: aae_mark = '**'

        if has_within_var:
            row_are_var = agg[name]['are_within_var_mean']
            row_aae_var = agg[name]['aae_within_var_mean']
            are_var_mark = ''
            aae_var_mark = ''
            if name == are_var_best: are_var_mark = '*'
            elif name == are_var_second: are_var_mark = '**'
            if name == aae_var_best: aae_var_mark = '*'
            elif name == aae_var_second: aae_var_mark = '**'

            print('{:30s} {:12.3f}{:<2s} {:12.3f} {:12.6f}{:<2s} {:12.6f}{:<2s} {:14.6f}{:<2s} {:14.6f}{:<2s}'.format(
                name, row_thr, thr_mark, row_q, row_are, are_mark, row_aae, aae_mark,
                row_are_var, are_var_mark, row_aae_var, aae_var_mark
            ))
        else:
            print('{:30s} {:12.3f}{} {:12.3f} {:12.6f}{} {:10.6f}{}'.format(
                name, row_thr, thr_mark, row_q, row_are, are_mark, row_aae, aae_mark
            ))


def main():
    parser = argparse.ArgumentParser(description='Visualize sensitivity analysis results for a single memory budget (3x2 layout)')
    parser.add_argument('-i', '--input', type=str, required=True,
                      help='Path to input JSON file with results')
    parser.add_argument('-o', '--output', type=str, default='output/sensitivity_results_single',
                      help='Base path for output files (without extension)')
    parser.add_argument('--memory-budget', type=int, required=True,
                      help='Memory budget in KiB to visualize (e.g., 64)')
    parser.add_argument('--show-within-variance', action='store_true', help='Also visualize within-run variance (ARE/AAE)')

    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    print(f"Memory budget: {args.memory_budget} KiB")
    print("Showing within-run variance:", args.show_within_variance)
    
    data = load_results(args.input)

    config = data['config']
    results = data['results']

    exp_config = config.get('experiment', config)
    sketch_config = config.get('base_sketch_config', {})
    sensitivity_params = config.get('sensitivity_params', {})

    countmin_config = sketch_config.get('countmin', {})
    resketch_config = sketch_config.get('resketch', {})

    dataset_type = exp_config.get('dataset_type', config.get('dataset_type', 'zipf'))
    total_items = exp_config.get('total_items', config.get('total_items', 10_000_000))
    repetitions = exp_config.get('repetitions', config.get('repetitions', 1))
    countmin_depth = countmin_config.get('depth', config.get('countmin_depth', 8))
    resketch_depth = resketch_config.get('depth', config.get('resketch_depth', 4))
    k_values = sensitivity_params.get('k_values', config.get('k_values', []))
    depth_values = sensitivity_params.get('depth_values', config.get('depth_values', []))
    memory_budgets = sensitivity_params.get('memory_budgets_kb', config.get('memory_budgets_kb', []))

    print("\nExperiment Configuration:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total Items: {total_items}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Count-Min Depth: {countmin_depth}")
    print(f"  ReSketch Depth: {resketch_depth}")
    print(f"  K values tested: {k_values}")
    print(f"  Depth values tested: {depth_values}")
    print(f"  Memory budgets tested (KiB): {memory_budgets}")

    print("\nAggregating results across repetitions...")
    aggregated = aggregate_results(results)

    print_summary_table(aggregated, args.memory_budget)

    print("\nGenerating plots...")
    plot_results(aggregated, output_path=args.output, 
                memory_budget_kb=args.memory_budget,
                show_within_variance=args.show_within_variance)

    print("\nDone!")

if __name__ == '__main__':
    main()
