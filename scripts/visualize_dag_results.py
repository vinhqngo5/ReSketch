import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import argparse
import sys
import numpy as np
from pathlib import Path

from visualization_common import (
    load_material_colors, setup_fonts, style_axis, save_figure, load_results
)

def format_memory(kb):
    if kb >= 1024:
        return f"{kb/1024:.1f}MB"
    else:
        return f"{kb}KB"

def aggregate_checkpoint_metrics(checkpoints, sketch_name):
    sketch_checkpoints = [cp for cp in checkpoints if cp['sketch_name'] == sketch_name]
    
    if not sketch_checkpoints:
        return None
    
    throughput = np.mean([cp['throughput_mops'] for cp in sketch_checkpoints])
    query_throughput = np.mean([cp['query_throughput_mops'] for cp in sketch_checkpoints])
    are = np.mean([cp['are'] for cp in sketch_checkpoints])
    aae = np.mean([cp['aae'] for cp in sketch_checkpoints])
    memory_kb = sketch_checkpoints[-1]['memory_kb']
    
    return {
        'throughput_mops': throughput,
        'query_throughput_mops': query_throughput,
        'are': are,
        'aae': aae,
        'memory_kb': memory_kb,
        'num_checkpoints': len(sketch_checkpoints)
    }

def get_structural_op_metrics(structural_ops, sketch_name):
    for op in structural_ops:
        if op['sketch_name'] == sketch_name:
            return {
                'operation': op['operation'],
                'latency_s': op['latency_s'],
                'are': op['are'],
                'aae': op['aae'],
                'memory_kb': op['memory_kb']
            }
    return None

def build_dag_structure(sketches):
    edges = []
    sketch_to_family = {}
    
    color_family_counter = 0
    
    # Assign families to create nodes
    for sketch_id, sketch_info in sketches.items():
        op = sketch_info['operation']
        if op == 'create':
            sketch_to_family[sketch_id] = color_family_counter
            color_family_counter += 1
    
    # Build edges and assign families
    for sketch_id, sketch_info in sketches.items():
        op = sketch_info['operation']
        
        if op == 'expand' or op == 'shrink':
            source = sketch_info.get('source')
            if not source:
                sources = sketch_info.get('sources', [])
                source = sources[0] if sources else None
            
            if source:
                edges.append((source, sketch_id, op))
                sketch_to_family[sketch_id] = sketch_to_family[source]
            
        elif op == 'merge':
            sources = sketch_info.get('sources', [])
            for src in sources:
                edges.append((src, sketch_id, 'merge'))
            sketch_to_family[sketch_id] = color_family_counter
            color_family_counter += 1
            
        elif op == 'split':
            source = sketch_info.get('source')
            if not source:
                sources = sketch_info.get('sources', [])
                source = sources[0] if sources else None
            
            if source:
                edges.append((source, sketch_id, 'split'))
                sketch_to_family[sketch_id] = color_family_counter
                color_family_counter += 1
    
    return edges, sketch_to_family

def compute_layout(sketches, edges):
    """Compute layout positions for DAG nodes using topological sort."""
    in_degree = {sid: 0 for sid in sketches.keys()}
    adjacency = {sid: [] for sid in sketches.keys()}
    reverse_adjacency = {sid: [] for sid in sketches.keys()}
    
    for src, dst, _ in edges:
        adjacency[src].append(dst)
        reverse_adjacency[dst].append(src)
        in_degree[dst] += 1
    
    layers = []
    queue = [sid for sid in sketches.keys() if in_degree[sid] == 0]
    node_to_layer = {}
    
    current_layer = 0
    while queue:
        layers.append(queue[:])
        for node in queue:
            node_to_layer[node] = current_layer
        
        next_queue = []
        for node in queue:
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        queue = next_queue
        current_layer += 1
    
    positions = {}
    x_spacing = 4.5
    y_spacing = 3.0
    
    for layer_idx, layer_nodes in enumerate(layers):
        num_nodes = len(layer_nodes)
        
        if layer_idx > 0:
            positioned_nodes = []
            for node_id in layer_nodes:
                parents = reverse_adjacency[node_id]
                if parents and all(p in positions for p in parents):
                    parent_y = sum(positions[p][1] for p in parents) / len(parents)
                    positioned_nodes.append((node_id, parent_y))
                else:
                    positioned_nodes.append((node_id, 0))
            
            positioned_nodes.sort(key=lambda x: x[1])
            layer_nodes = [n[0] for n in positioned_nodes]
        
        for node_idx, node_id in enumerate(layer_nodes):
            x = layer_idx * x_spacing
            y_offset = (num_nodes - 1) * y_spacing / 2
            y = node_idx * y_spacing - y_offset
            positions[node_id] = (x, y)
    
    return positions

def plot_results(config, results, metadata, output_path, 
                          repetition_id=0, show_structural_ops=True):
    sketches = config['sketches']
    sketch_config = config['sketch_config']
    
    # Get results
    if repetition_id >= len(results):
        print(f"Warning: Repetition {repetition_id} not found. Using repetition 0.")
        repetition_id = 0
    
    rep_result = results[repetition_id]
    checkpoints = rep_result['checkpoints']
    structural_ops = rep_result['structural_operations']
    
    edges, sketch_families = build_dag_structure(sketches)
    positions = compute_layout(sketches, edges)
    
    print ("edges:", edges)
    print ("positions:", positions)
    print ("sketch_families:", sketch_families)
    
    # Setup visualization
    material_colors = load_material_colors("scripts/colors/material-colors.json")
    font_config = setup_fonts(__file__)
    
    font_config['title_size'] = 14
    font_config['label_size'] = 11
    font_config['tick_size'] = 9
    font_config['legend_size'] = 11
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    base_fill_colors = [
        material_colors['orange']['50'],
        material_colors['blue']['50'],
        material_colors['purple']['50'],
        material_colors['green']['50'],
        material_colors['red']['50'],
        material_colors['yellow']['50'],
        material_colors['teal']['50'],
        material_colors['pink']['50'],
    ]
    
    base_border_colors = [
        material_colors['orange']['600'],
        material_colors['blue']['600'],
        material_colors['purple']['600'],
        material_colors['green']['600'],
        material_colors['red']['600'],
        material_colors['yellow']['600'],
        material_colors['teal']['600'],
        material_colors['pink']['600'],
    ]
    
    family_fill_colors = {}
    family_border_colors = {}
    for family_id in set(sketch_families.values()):
        idx = family_id % len(base_fill_colors)
        family_fill_colors[family_id] = base_fill_colors[idx]
        family_border_colors[family_id] = base_border_colors[idx]
    
    # Draw edges
    for src, dst, op_type in edges:
        src_pos = positions[src]
        dst_pos = positions[dst]
        
        if op_type == 'expand':
            style = 'solid'
            label = 'Expand'
        elif op_type == 'shrink':
            style = 'solid'
            label = 'Shrink'
        elif op_type == 'merge':
            style = (0, (5, 5))
            label = 'Merge'
        elif op_type == 'split':
            style = (0, (5, 5))
            label = 'Split'
        else:
            style = 'solid'
            label = ''
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (src_pos[0] + 1.3, src_pos[1]), 
            (dst_pos[0] - 1.3, dst_pos[1]),
            arrowstyle='->', 
            mutation_scale=20, 
            linewidth=1.5,
            linestyle=style,
            color='#424242',
            zorder=1
        )
        ax.add_patch(arrow)
        
        # Draw edge labeled with structural operation metrics
        mid_x = (src_pos[0] + dst_pos[0]) / 2
        mid_y = (src_pos[1] + dst_pos[1]) / 2
        
        if show_structural_ops and op_type in ['expand', 'shrink', 'merge', 'split']:
            struct_metrics = get_structural_op_metrics(structural_ops, dst)
            if struct_metrics:
                latency_ms = struct_metrics['latency_s'] * 1000
                edge_label = f"{label}\n{latency_ms:.1f}ms"
            else:
                edge_label = label
        else:
            edge_label = label
        
        ax.text(mid_x, mid_y + 0.2, edge_label, 
                fontsize=font_config['tick_size'], ha='center',
                fontfamily=font_config['family'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', linewidth=0.8))
    
    # Draw nodes with results
    box_width = 2.4
    box_height = 1.8
    
    for sketch_id, sketch_info in sketches.items():
        pos = positions[sketch_id]
        operation = sketch_info['operation']
        
        family_id = sketch_families[sketch_id]
        fill_color = family_fill_colors[family_id]
        border_color = family_border_colors[family_id]
        
        # Draw node box
        fancy_box = FancyBboxPatch(
            (pos[0] - box_width/2, pos[1] - box_height/2),
            box_width, box_height,
            boxstyle="square,pad=0.08",
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=2.0,
            zorder=2
        )
        ax.add_patch(fancy_box)
        
        # Get metrics
        checkpoint_metrics = aggregate_checkpoint_metrics(checkpoints, sketch_id)
        struct_metrics = get_structural_op_metrics(structural_ops, sketch_id)
        # if sketch_id == 'F':
        #   breakpoint()
        # Node title
        y_offset = box_height/2 - 0.2
        ax.text(pos[0], pos[1] + y_offset, f"{sketch_id}", 
                fontsize=font_config['label_size'] + 3, fontweight='bold', 
                ha='center', va='top', zorder=3, 
                fontfamily=font_config['family'])
        
        # Draw metrics
        if checkpoint_metrics:
            throughput = checkpoint_metrics['throughput_mops']
            are = checkpoint_metrics['are']
            aae = checkpoint_metrics['aae']
            memory_kb = checkpoint_metrics['memory_kb']
            
            metrics_y_start = pos[1] + 0.15
            line_spacing = 0.22
            
            ax.text(pos[0], metrics_y_start, f"{throughput:.2f} MOps/s",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
            
            ax.text(pos[0], metrics_y_start - line_spacing, f"ARE: {are:.3f}",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
            
            ax.text(pos[0], metrics_y_start - 2*line_spacing, f"AAE: {aae:.1f}",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
            
        elif struct_metrics:
            # If no checkpoint data, show structural op results
            memory_kb = struct_metrics['memory_kb']
            are = struct_metrics['are']
            aae = struct_metrics['aae']
            
            metrics_y_start = pos[1] + 0.05
            line_spacing = 0.22
            
            ax.text(pos[0], metrics_y_start, f"ARE: {are:.3f}",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
            
            ax.text(pos[0], metrics_y_start - line_spacing, f"AAE: {aae:.1f}",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
        else:
            ax.text(pos[0], pos[1], "No data",
                    fontsize=font_config['tick_size'], ha='center', va='center', 
                    zorder=3, fontfamily=font_config['family'])
        
        # Draw operation type label at bottom
        if operation == 'create':
            config_text = f"CREATE\n(d={sketch_config['depth']}, k={sketch_config['kll_k']})\nmemory_budget_kb={format_memory(sketch_info.get('memory_budget_kb', 0))}\nused={format_memory(memory_kb)}"
        else:
            memory_used_kb = checkpoint_metrics['memory_kb']
            config_text = f"memory_budget_kb={format_memory(sketch_info.get('memory_budget_kb', 0))}\nused={format_memory(memory_used_kb)}"
        
        ax.text(pos[0], pos[1] - box_height/2 - 0.15, config_text,
                fontsize=font_config['tick_size'] - 1, ha='center', va='top', 
                style='italic', color='#616161', zorder=3,
                fontfamily=font_config['family'])
    
    # Set axis limits, title, legend
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    
    x_margin = 2.0
    y_margin = 2.0
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    dag_name = metadata.get('dag_name', 'Unknown')
    timestamp = metadata.get('timestamp', 'Unknown')
    num_reps = config.get('experiment', {}).get('repetitions', 1)
    
    title = f"ReSketch DAG Results: {dag_name}"
    if num_reps > 1:
        title += f" (Repetition {repetition_id + 1}/{num_reps})"
    
    plt.title(title, 
              fontsize=font_config['title_size'] + 2, fontweight='bold', 
              pad=20, fontfamily=font_config['family'])
    
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, 
                      label='Sketch node with results'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='solid', 
                   label='Expand/Shrink (same thread)'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='dashed', 
                   label='Merge/Split (new thread)'),
        mpatches.Patch(facecolor='lightgray', edgecolor='gray', linewidth=2, 
                      label='Color = thread/device family'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              fontsize=font_config['legend_size'],
              prop={'family': font_config['family']})
    
    plt.tight_layout()
    save_figure(fig, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize ReSketch DAG experimental results'
    )
    parser.add_argument('result_file', type=str, 
                       help='Path to JSON result file')
    parser.add_argument('-o', '--output', default="output/dag_results.png", type=str, 
                       help='Output image path (default: <result_name>_visualization.png)')
    parser.add_argument('-r', '--repetition', type=int, default=0,
                       help='Repetition ID to visualize (default: 0)')
    parser.add_argument('--no-structural-ops', action='store_true',
                       help='Hide structural operation metrics on edges')
    
    args = parser.parse_args()
    
 
    data = load_results(args.result_file)
    config = data.get('config', {})
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    plot_results(
        config, 
        results, 
        metadata,
        args.output,
        repetition_id=args.repetition,
        show_structural_ops=not args.no_structural_ops
    )
        
if __name__ == '__main__':
    main()
