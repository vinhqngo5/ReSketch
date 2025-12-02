import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

def load_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
  
def load_material_colors(filepath='scripts/colors/material-colors.json'):
    with open(filepath, 'r') as f:
        colors = json.load(f)
    return colors

def setup_fonts(script_file):
    script_dir = Path(script_file).parent
    font_path = script_dir / "fonts" / "LinLibertine_R.ttf"
    bold_font_path = script_dir / "fonts" / "LinLibertine_RB.ttf"
    
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
    if bold_font_path.exists():
        fm.fontManager.addfont(str(bold_font_path))
    
    font_config = {
        'family': 'Linux Libertine',
        'title_size': 12,
        'label_size': 12,
        'tick_size': 10,
        'legend_size': 10
    }
    
    plt.rcParams['font.family'] = font_config['family']
    plt.rcParams['font.serif'] = ['Linux Libertine']
    plt.rcParams['pdf.fonttype'] = 42
    
    return font_config

def get_sketch_styles(material_colors):
    return {
        'CountMin': {
            'color': material_colors['green']['500'], 
            'marker': 'o', 
            'linestyle': '-', 
            'label': 'Count-Min Sketch'
        },
        'ReSketch': {
            'color': material_colors['purple']['500'], 
            'marker': 'x', 
            'linestyle': '-', 
            'label': 'ReSketch'
        },
        'DynamicSketch': {
            'color': material_colors['orange']['500'], 
            'marker': 's', 
            'linestyle': '--', 
            'label': 'DynamicSketch'
        },
        'GeometricSketch': {
            'color': material_colors['red']['500'], 
            'marker': 'D', 
            'linestyle': ':', 
            'label': 'GeometricSketch'
        },
    }

def get_resketch_k_styles(material_colors):
    colors_palette = [
        material_colors['purple']['300'],
        material_colors['purple']['500'],
        material_colors['purple']['700'],
        material_colors['deeppurple']['500'],
        material_colors['indigo']['500'],
    ]
    
    markers = ['o', 's', 'D', '^', 'v']
    
    k_values = [5, 10, 30, 50, 100]
    styles = {}
    
    for i, k in enumerate(k_values):
        config_name = f"ReSketch_k{k}"
        styles[config_name] = {
            'color': colors_palette[i % len(colors_palette)],
            'marker': markers[i % len(markers)],
            'linestyle': '-',
            'label': f'ReSketch (k={k})'
        }
    
    return styles

# def get_resketch_depth_styles(material_colors):
#     return {
#         2: {
#             'color': material_colors['purple']['300'],
#             'marker': 'o',
#             'linestyle': '-',
#             'label': 'ReSketch (depth=2)'
#         },
#         4: {
#             'color': material_colors['purple']['500'],
#             'marker': 's',
#             'linestyle': '-',
#             'label': 'ReSketch (depth=4)'
#         },
#         6: {
#             'color': material_colors['purple']['700'],
#             'marker': 'D',  
#             'linestyle': '-',
#             'label': 'ReSketch (depth=6)'
#         },
#         8: {
#             'color': material_colors['deeppurple']['500'],
#             'marker': 'D',
#             'linestyle': '-',
#             'label': 'ReSketch (depth=8)'
#         }
#     }
    
def get_resketch_depth_styles(material_colors):
    return {
        1: {
            'color': material_colors['purple']['300'],
            'marker': 'o',
            'linestyle': '-',
            'label': 'ReSketch (depth=1)'
        },
        3: {
            'color': material_colors['purple']['500'],
            'marker': 's',
            'linestyle': '-',
            'label': 'ReSketch (depth=3)'
        },
        5: {
            'color': material_colors['purple']['700'],
            'marker': 'D',
            'linestyle': '-',
            'label': 'ReSketch (depth=5)'
        },
        7: {
            'color': material_colors['deeppurple']['500'],
            'marker': '^',
            'linestyle': '-',
            'label': 'ReSketch (depth=7)'
        },
        2: {
            'color': material_colors['orange']['300'],
            'marker': 'o',
            'linestyle': '-',
            'label': 'ReSketch (depth=2)'
        },
        4: {
            'color': material_colors['orange']['500'],
            'marker': 's',
            'linestyle': '-',
            'label': 'ReSketch (depth=4)'
        },
        6: {
            'color': material_colors['orange']['700'],
            'marker': 'D',  
            'linestyle': '-',
            'label': 'ReSketch (depth=6)'
        },
        8: {
            'color': material_colors['red']['500'],
            'marker': 'D',
            'linestyle': '-',
            'label': 'ReSketch (depth=8)'
        }
    }

def get_countmin_style(material_colors):
    return {
        'color': material_colors['green']['500'],
        'marker': 'o',
        'linestyle': '--',
        'label': 'Count-Min Sketch'
    }

def style_axis(ax, font_config, ylabel, xlabel=None, title=None, use_log_scale=False):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=font_config['label_size'], 
                     fontfamily=font_config['family'], labelpad=1)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_config['label_size'],
                     fontfamily=font_config['family'], labelpad=1)
    
    if title:
        ax.set_title(title, fontsize=font_config['title_size'], 
                    fontfamily=font_config['family'], pad=2)
    
    if use_log_scale:
        ax.set_yscale('log')
    
    ax.grid(True, color='gray', alpha=0.2, linestyle='-', linewidth=0.1, axis='y')
    ax.tick_params(axis='both', which='both', direction='in', 
                  pad=2, labelsize=font_config['tick_size'])
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.1)
        spine.set_color('black')

def plot_line_with_error(ax, x_data, y_mean, y_std, style, markerfacecolor='none'):
    ax.plot(x_data, y_mean, 
           color=style.get('color', 'black'),
           marker=style.get('marker', 'o'),
           linestyle=style.get('linestyle', '-'),
           label=style.get('label', 'Unknown'),
           linewidth=1.5, markersize=6,
           markerfacecolor=markerfacecolor)
    
    ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, 
                   alpha=0.2, color=style.get('color', 'black'))

def plot_horizontal_baseline(ax, x_range, y_mean, y_std, style):
    ax.axhline(y=y_mean, 
              color=style.get('color', 'black'),
              linestyle=style.get('linestyle', '--'),
              linewidth=1.5,
               label=style.get('label', 'Baseline'))
    
    ax.fill_between(x_range,
                   y_mean - y_std,
                   y_mean + y_std,
                   alpha=0.2, 
                   color=style.get('color', 'black'))

def save_figure(fig, output_path, tight=True):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    pdf_file = output_file.with_suffix('.pdf')
    fig.savefig(str(pdf_file), format='pdf', bbox_inches='tight' if tight else None, 
                pad_inches=0.022, dpi=300)
    print(f"Figure saved to: {pdf_file}")
    
    png_file = output_file.with_suffix('.png')
    fig.savefig(str(png_file), dpi=300, bbox_inches='tight' if tight else None)
    print(f"Figure saved to: {png_file}")
    
    plt.close(fig)

def create_shared_legend(fig, ax, ncol=2, bbox_to_anchor=(0.5, 1.0), 
                        font_config=None, top_adjust=0.93):
    if font_config is None:
        font_config = {'legend_size': 10, 'family': 'Linux Libertine'}
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=bbox_to_anchor,
              ncol=ncol,
              fontsize=font_config['legend_size'],
              frameon=False,
              handlelength=1.5,
              handletextpad=0.5,
              columnspacing=1.0,
              prop={'family': font_config['family']})
    
    fig.subplots_adjust(top=top_adjust)
