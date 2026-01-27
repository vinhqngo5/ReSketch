import argparse
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from visualization_common import (
    create_shared_legend,
    load_material_colors,
    load_results,
    save_figure,
    setup_fonts,
    style_axis,
)


def aggregate_results(results_data):
    results = results_data['results']

    metrics = {
        'a_vs_true_on_da': {'are': [], 'aae': [], 'are_var': [], 'aae_var': []},
        'b_vs_true_on_db': {'are': [], 'aae': [], 'are_var': [], 'aae_var': []},
        'c_vs_true_on_all': {'are': [], 'aae': [], 'are_var': [], 'aae_var': []},
        'd_vs_true_on_all': {'are': [], 'aae': [], 'are_var': [], 'aae_var': []}
    }

    for rep in results:
        if 'accuracy' in rep:
            accuracy = rep['accuracy']
        else:
            accuracy = rep

        for key in metrics.keys():
            metrics[key]['are'].append(accuracy[key]['are'])
            metrics[key]['aae'].append(accuracy[key]['aae'])
            metrics[key]['are_var'].append(accuracy[key].get('are_variance', 0.0))
            metrics[key]['aae_var'].append(accuracy[key].get('aae_variance', 0.0))

    aggregated = {}
    for key in metrics.keys():
        aggregated[key] = {
            'are_mean': np.mean(metrics[key]['are']),
            'are_std': np.std(metrics[key]['are']),
            'aae_mean': np.mean(metrics[key]['aae']),
            'aae_std': np.std(metrics[key]['aae']),
            'are_var_mean': np.mean(metrics[key]['are_var']),
            'are_var_std': np.std(metrics[key]['are_var']),
            'aae_var_mean': np.mean(metrics[key]['aae_var']),
            'aae_var_std': np.std(metrics[key]['aae_var'])
        }

    return aggregated


def calculate_accuracy_data(results_data, sketch_accuracy_data_name: str):
    """Calculate absolute and relative error for each item and order by rank."""
    sketch_item_acc_data = results_data[sketch_accuracy_data_name]

    # Order by rank (true frequency)
    sketch_item_acc_data.sort(key=lambda item_data: item_data["freq"], reverse=True)

    # Compute errors for each item
    for item in sketch_item_acc_data:
        item["abs_err"] = abs(item["freq"] - item["est"])
        item["rel_err"] = abs(item["freq"] - item["est"]) / item["freq"]


def plot_results(results_data, output_path, show_within_variance=True):
    material_colors = load_material_colors("scripts/colors/material-colors.json")

    font_config = setup_fonts(__file__)

    config = results_data['config']
    aggregated = aggregate_results(results_data)

    num_cols = 4 if show_within_variance else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 5))

    if num_cols == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3, ax4 = axes

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

    are_var_means = [aggregated[key]['are_var_mean'] for key in keys]

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
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}',
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
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9)

    # ARE variance plots
    if show_within_variance:
        are_var_means = [aggregated[key]['are_var_mean'] for key in keys]
        are_var_stds = [aggregated[key]['are_var_std'] for key in keys]

        bars3 = ax3.bar(x_pos, are_var_means, bar_width, yerr=are_var_stds,
                        color=colors, alpha=0.8, capsize=5,
                        error_kw={'linewidth': 1.5})

        ax3.set_ylabel('ARE Within-Run Variance', fontsize=12)
        ax3.set_xlabel('Sketch Configuration', fontsize=12)
        ax3.set_title('Within-Run Variance: ARE', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, fontsize=11)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)

        for i, (bar, mean, std) in enumerate(zip(bars3, are_var_means, are_var_stds)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=9)

        # AAE variance plots
        aae_var_means = [aggregated[key]['aae_var_mean'] for key in keys]
        aae_var_stds = [aggregated[key]['aae_var_std'] for key in keys]

        bars4 = ax4.bar(x_pos, aae_var_means, bar_width, yerr=aae_var_stds,
                        color=colors, alpha=0.8, capsize=5,
                        error_kw={'linewidth': 1.5})

        ax4.set_ylabel('AAE Within-Run Variance', fontsize=12)
        ax4.set_xlabel('Sketch Configuration', fontsize=12)
        ax4.set_title('Within-Run Variance: AAE', fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, fontsize=11)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        ax4.set_axisbelow(True)

        for i, (bar, mean, std) in enumerate(zip(bars4, aae_var_means, aae_var_stds)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{mean:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=9)

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


def plot_accuracy_per_key(results_data: dict, output_path):
    material_colors = load_material_colors("scripts/colors/material-colors.json")

    font_config = setup_fonts(__file__)

    class TraceConfig(NamedTuple):
        dataset_name: str
        label: str
        color: str
        linestyle: str = "-"
        linewidth: float = 1.5
        alpha: float = 0.7

    class PlotConfig(NamedTuple):
        xlabel: str
        ylabel: str
        result_data_key: str
        traces: list[TraceConfig]

    plots = [
        PlotConfig("Item Rank", "Relative Error", "rel_err", traces=[
            TraceConfig("c_frequencies", "C (merged) on All", material_colors["purple"]["500"]),
            TraceConfig("d_frequencies", "D (2× width) on All", material_colors["orange"]["500"])
        ]),
        PlotConfig("Item Rank", "Absolute Error", "abs_err", traces=[
            TraceConfig("c_frequencies", "C (merged) on All", material_colors["purple"]["500"]),
            TraceConfig("d_frequencies", "D (2× width) on All", material_colors["orange"]["500"])
        ]),
    ]

    def moving_average(x, window_size: int = 1000):
        return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

    fig, axes = plt.subplots(nrows=1, ncols=len(plots), figsize=(3.33*1.182, 0.79))

    for row_idx, plot in enumerate(plots):
        ax = axes[row_idx]
        for trace in plot.traces:
            trace_data = moving_average([d[plot.result_data_key] for d in results_data[trace.dataset_name]])
            ax.plot(
                trace_data,
                color=trace.color,
                linestyle=trace.linestyle,
                linewidth=trace.linewidth,
                label=trace.label,
                alpha=trace.alpha,
            )
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto', min_n_ticks=3))
        style_axis(ax, font_config, plot.ylabel, plot.xlabel)

    top_adjust = 0.71
    create_shared_legend(fig, axes[0], ncol=2, font_config=font_config,
                         bbox_to_anchor=(0.5, 1.02), top_adjust=top_adjust)

    plt.subplots_adjust(left=0, right=1, top=top_adjust, bottom=0, hspace=0.0, wspace=0.18)

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
    print(f"    ARE: {aggregated['a_vs_true_on_da']['are_mean']:.4f} ± {aggregated['a_vs_true_on_da']['are_std']:.4f} (across runs)")
    print(f"    AAE: {aggregated['a_vs_true_on_da']['aae_mean']:.2f} ± {aggregated['a_vs_true_on_da']['aae_std']:.2f} (across runs)")
    print(f"    ARE within-run variance: {aggregated['a_vs_true_on_da']['are_var_mean']:.4f} ± {aggregated['a_vs_true_on_da']['are_var_std']:.4f}")
    print(f"    AAE within-run variance: {aggregated['a_vs_true_on_da']['aae_var_mean']:.2f} ± {aggregated['a_vs_true_on_da']['aae_var_std']:.2f}")

    print(f"\n  B on DB (items [{diversity//2}, {diversity})):")
    print(f"    ARE: {aggregated['b_vs_true_on_db']['are_mean']:.4f} ± {aggregated['b_vs_true_on_db']['are_std']:.4f} (across runs)")
    print(f"    AAE: {aggregated['b_vs_true_on_db']['aae_mean']:.2f} ± {aggregated['b_vs_true_on_db']['aae_std']:.2f} (across runs)")
    print(f"    ARE within-run variance: {aggregated['b_vs_true_on_db']['are_var_mean']:.4f} ± {aggregated['b_vs_true_on_db']['are_var_std']:.4f}")
    print(f"    AAE within-run variance: {aggregated['b_vs_true_on_db']['aae_var_mean']:.2f} ± {aggregated['b_vs_true_on_db']['aae_var_std']:.2f}")

    print(f"\n  C (merged A+B) on All items:")
    print(f"    ARE: {aggregated['c_vs_true_on_all']['are_mean']:.4f} ± {aggregated['c_vs_true_on_all']['are_std']:.4f} (across runs)")
    print(f"    AAE: {aggregated['c_vs_true_on_all']['aae_mean']:.2f} ± {aggregated['c_vs_true_on_all']['aae_std']:.2f} (across runs)")
    print(f"    ARE within-run variance: {aggregated['c_vs_true_on_all']['are_var_mean']:.4f} ± {aggregated['c_vs_true_on_all']['are_var_std']:.4f}")
    print(f"    AAE within-run variance: {aggregated['c_vs_true_on_all']['aae_var_mean']:.2f} ± {aggregated['c_vs_true_on_all']['aae_var_std']:.2f}")

    print(f"\n  D (ground truth, 2× width) on All items:")
    print(f"    ARE: {aggregated['d_vs_true_on_all']['are_mean']:.4f} ± {aggregated['d_vs_true_on_all']['are_std']:.4f} (across runs)")
    print(f"    AAE: {aggregated['d_vs_true_on_all']['aae_mean']:.2f} ± {aggregated['d_vs_true_on_all']['aae_std']:.2f} (across runs)")
    print(f"    ARE within-run variance: {aggregated['d_vs_true_on_all']['are_var_mean']:.4f} ± {aggregated['d_vs_true_on_all']['are_var_std']:.4f}")
    print(f"    AAE within-run variance: {aggregated['d_vs_true_on_all']['aae_var_mean']:.2f} ± {aggregated['d_vs_true_on_all']['aae_var_std']:.2f}")


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
        default="output/merge_results",
        help='Output path (without extension, will generate .png and .pdf). '
             'Default: output/merge_results'
    )
    parser.add_argument(
        '--show-within-variance',
        action='store_true',
        help='Show within-run variance on bar charts'
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    results_data = load_results(args.input)

    print_summary(results_data)

    # output_path = args.output if args.output else 'output/merge_results'
    # print(f"Show within-run variance: {args.show_within_variance}")
    # plot_results(results_data, output_path, show_within_variance=args.show_within_variance)

    # Plot accuracy of merged vs directly sketched
    final_repetition_data = results_data["results"][-1]
    calculate_accuracy_data(final_repetition_data, "c_frequencies")
    calculate_accuracy_data(final_repetition_data, "d_frequencies")
    plot_accuracy_per_key(final_repetition_data, args.output)


if __name__ == '__main__':
    main()
