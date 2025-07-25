import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker

def plot_linechart(df):
    df['Matrix Size'] = pd.to_numeric(df['Matrix Size'])
    df['Cache Config'] = "L2: " + df['l2_config'] + " | L1: " + df['dcache_config']

    # Get unique cache configs and split into chunks of 2 for subplot rows
    unique_configs = df['Cache Config'].unique()
    n_configs = len(unique_configs)
    n_cols = 2
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Plot each cache config
    for i, config in enumerate(unique_configs):
        ax = axes[i]
        subset = df[df['Cache Config'] == config]
        
        sns.lineplot(
            data=subset,
            x="Matrix Size",
            y="Execution Time (s)",
            hue="Implementation",
            style="Implementation",
            markers=True,
            dashes=False,
            markersize=8,
            palette="tab10",
            linewidth=2,
            ax=ax
        )
        
        ax.set_title(f"Cache Config: {config}")
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Execution Time (s)")
        ax.grid(True, alpha=0.3)
        

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Implementation', 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle("Execution Time Scaling (Legends Sorted by Time at 1024)", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_linechart_mean(df):
    mean_df = df.groupby(['Matrix Size', 'Implementation'], as_index=False).agg({
        'Execution Time (s)': 'mean',
        'L1 Miss %': 'mean',
        'L2 Miss %': 'mean'
    })

    # Line chart (mean across all cache configs)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=mean_df,
        x="Matrix Size",
        y="Execution Time (s)",
        hue="Implementation",
        style="Implementation",  # Different line styles
        markers=True,           # Force markers
        dashes=False,           # Solid lines but vary markers
        markersize=10,
        palette="tab10",        # High-contrast colors
        linewidth=2.5,
    )
    plt.title('Mean Execution Time by Matrix Size (Averaged Across All Cache Configs)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.grid(True)
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    df = df[(df['Matrix Size'] == 512)].copy()

    # Pivot the data for heatmap (example: Execution Time)
    heatmap_data = df.pivot_table(
        index="Implementation",
        columns=["l2_config", "dcache_config"],  # Multi-level columns
        values="Execution Time (s)",  # Can swap to "L2 Miss %"
        aggfunc="mean",  # In case of duplicates
    )

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,  # Show values in cells
        fmt=".1f",   # Float formatting
        cmap="viridis_r",  # Reverse colormap (darker = slower/higher misses)
        linewidths=0.5,
    )
    plt.title("Execution Time (s) by Implementation and Cache Config")
    plt.xlabel("Cache Config (L2 | L1)")
    plt.ylabel("Implementation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_top_performing(df):
    df = df[(df['Matrix Size'] == 1024)].copy()

    samples = 10
    performance_df = df.groupby(['Implementation', 'dcache_config', 'l2_config'], as_index=False)\
                      .agg({'Execution Time (s)': 'mean'})\
                      .sort_values('Execution Time (s)').head(10)  # Get top 10 fastest
    
    performance_df['Config_Label'] = (
        performance_df['Implementation'] + "\n" +
        "L1:" + performance_df['dcache_config'].str.replace('dc=', '') + " | " +
        "L2:" + performance_df['l2_config'].str.replace('l2=', '')
    )
    

    plt.figure(figsize=(14, 7))  
    
    barplot = sns.barplot(
        data=performance_df,
        x='Config_Label',
        y='Execution Time (s)',
        palette='viridis_r',
        hue='Implementation',
        dodge=False,
        linewidth=1,
        edgecolor='black'
    )
    
    for p in barplot.patches[:samples]:
        barplot.annotate(
            f"{p.get_height():.1f}s", 
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='center', xytext=(0, 5), 
            textcoords='offset points',
            fontsize=9
        )
    
    plt.title(f"Top {samples} Fastest Implementations + Cache Configurations", pad=20)
    plt.xlabel("Implementation + Cache Configuration")
    plt.ylabel("Mean Execution Time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.legend(
        title='Implementation',
        bbox_to_anchor=(1.05, 0.5),  # Anchor to right-center
        loc='center left',
        frameon=True,
        framealpha=0.8
    )
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make space for legend
    plt.show()

def plot_miss_by_config(df, cache_miss = 'L2 Miss %'):
    # Combine L1/L2 configs into one column
    df['Cache Config'] = "L1:" + df['dcache_config'].str.replace('dc=', '') + "\nL2:" + df['l2_config'].str.replace('l2=', '')
    
    # Melt for L1/L2 side-by-side bars
    melt_df = df.melt(
        id_vars=['Cache Config', 'Implementation'],
        value_vars=[cache_miss],
        var_name='Cache Level', 
        value_name='Miss Rate %'
    )
    
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=melt_df,
        x='Cache Config',
        y='Miss Rate %',
        hue='Implementation',
        palette='tab10',
        edgecolor='w',
        linewidth=0.5,
        alpha=0.9
    )
    
    plt.title(f'{cache_miss} by Cache Configuration and Implementation')
    plt.xlabel('Cache Configuration (L1 | L2)')
    plt.ylabel('Miss Rate (%)')
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_miss_heatmaps(df):
    # Extract cache sizes
    df = df[(df['Matrix Size'] == 256)].copy()

    df['L1_Size'] = df['dcache_config'].str.extract(r'dc=(\d+):').astype(int)
    df['L2_Size'] = df['l2_config'].str.extract(r'l2=(\d+):').astype(int)
    
    # Create composite config label
    df['Full Config']= "L1:" + df['L1_Size'].astype(str) + "\nL2:" + df['L2_Size'].astype(str)
    # Create composite config column
    df['Full Config'] = (
        "L1:" + df['dcache_config'].str.replace('dc=', '') + 
        "\nL2:" + df['l2_config'].str.replace('l2=', '')
    )

    config_order = [
        'L1:4:8:32\nL2:64:8:32',
        'L1:4:8:32\nL2:128:8:64',
        'L1:16:8:32\nL2:64:8:32',
        'L1:16:8:32\nL2:128:8:64',
        'L1:32:8:64\nL2:256:8:64',
        'L1:32:8:64\nL2:512:16:128',
        'L1:128:16:128\nL2:256:8:64',
        'L1:128:16:128\nL2:512:16:128'
    ]

    df['Full Config'] = pd.Categorical(df['Full Config'], categories=config_order, ordered=True)

    # Pivot for heatmap
    pivot_df = df.pivot_table(
        index=['Full Config'],
        columns=['Implementation'],
        values=['L1 Miss %', 'L2 Miss %'],
        aggfunc='mean'
    ).sort_index() 
    # Plot L1 and L2 heatmaps side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(
        pivot_df['L1 Miss %'],
        annot=True, fmt=".1f", cmap='Blues',
        ax=ax1, cbar_kws={'label': 'Miss Rate %'}
    )
    ax1.set_title('L1 Cache Miss Rates')
    ax1.set_ylabel('Cache Configuration')
    
    sns.heatmap(
        pivot_df['L2 Miss %'],
        annot=True, fmt=".1f", cmap='Reds',
        ax=ax2, cbar_kws={'label': 'Miss Rate %'}
    )
    ax2.set_title('L2 Cache Miss Rates')
    
    plt.suptitle('Cache Miss Rates by Configuration and Implementation', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_miss_time_trend(df, cache_miss='L2 Miss %'):
    # Filter data
    plot_df = df#df[(df['Matrix Size'] == 256) | (df['Matrix Size'] == 256) |  (df['Matrix Size'] == 256)].copy()
    
    # Calculate point density
    x = plot_df['Execution Time (s)']
    y = plot_df[cache_miss]
    xy = np.vstack([x, y])  # Log-transform x for density estimation
    kde = gaussian_kde(xy)
    density = kde(xy)  # Density values for each point
    
    # --- Create plot ---
    plt.figure(figsize=(12, 7))
    plt.xscale('log')

    # Scatter plot with density coloring
    scatter = plt.scatter(
        x, y, 
        s=50,
        c=density,
        cmap='viridis',  # Use a colormap (e.g., 'viridis', 'plasma')
        alpha=0.3,
        edgecolors='none'
    )

      # Add colorbar for density
    cbar = plt.colorbar(scatter)
    cbar.set_label('Point Density')
    # Add trend line (LOESS smoothing)
    sns.regplot(
        x=x, y=y,
        scatter=False,
        logx=True,
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )
    
    # Add quadrant lines at medians
    plt.axhline(y.median(), color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x.median(), color='gray', linestyle=':', alpha=0.7)
    
    # Label quadrants
    plt.text(x.max()*0.9, y.max()*0.85, "High Miss\nSlow", ha='center', color='darkred')
    plt.text(x.max()*0.9, y.min()*1.2, "Low Miss\nSlow", ha='center', color='darkgreen') 
    plt.text(x.min()*1, y.max()*0.85, "High Miss\nFast", ha='left', color='darkred')
    plt.text(x.min()*1, y.min()*1.2, "Low Miss\nFast", ha='left', color='darkgreen')
    
    plt.title(f"Execution Time Trend vs Cache Miss")
    plt.ylabel(f"{cache_miss} Percentage")
    plt.xlabel("Execution Time (seconds)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_cache_trends_log_scale(df):
    # Extract cache sizes
    df['L1_Size'] = df['dcache_config'].str.extract(r'dc=(\d+):').astype(int)
    df['L2_Size'] = df['l2_config'].str.extract(r'l2=(\d+):').astype(int)
    
    # Create composite config label
    df['Cache Config'] = "L1:" + df['L1_Size'].astype(str) + "\nL2:" + df['L2_Size'].astype(str)
    
    # Create numerical ordering for cache configurations
    # config_order = sorted(df['Cache Config'].unique())

    config_order = [
        'L1:4\nL2:64',
        'L1:4\nL2:128', 
        'L1:16\nL2:64',
        'L1:16\nL2:128',
        'L1:32\nL2:256',
        'L1:32\nL2:512',
        'L1:128\nL2:256',
        'L1:128\nL2:512'
    ]

    print('config_order', config_order)
    config_map = {config: i for i, config in enumerate(config_order)}
    df['Config_Order'] = df['Cache Config'].map(config_map)
    
    # Create plot with log scale y-axis
    plt.figure(figsize=(16, 9))
    plt.yscale('log')  # Set logarithmic y-axis
    
    # Plot ALL individual points (all matrix sizes)
    scatter = sns.stripplot(
        data=df,
        x='Cache Config',
        y='Execution Time (s)',
        hue='Matrix Size',
        dodge=True,
        palette='viridis',
        alpha=0.6,
        jitter=0.2,
        order=config_order,
        size=6
    )
    
    # Calculate and plot colored average trend lines for EACH matrix size
    matrix_sizes = sorted(df['Matrix Size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(matrix_sizes)))
    
    for i, size in enumerate(matrix_sizes):
        size_data = df[df['Matrix Size'] == size]
        avg_times = size_data.groupby('Cache Config')['Execution Time (s)'].mean().loc[config_order]
        
        # Darker version of the point color
        line_color = colors[i] * 0.7  # Reduce brightness by 30%
        line_color[3] = 1.0  # Keep alpha at 1.0
        
        plt.plot(
            [config_map[config] for config in config_order],
            avg_times,
            color=line_color,
            linestyle='-',
            linewidth=2.5,  # Slightly thicker than before
            marker='o',
            markersize=9,
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=f'Size {size} Avg'
        )
    
    # Formatting
    plt.title("Execution Time by Cache Configuration (Log Scale)", pad=20)
    plt.xlabel("Cache Configuration")
    plt.ylabel("Execution Time (s) - Log Scale")
    plt.grid(True, which='both', alpha=0.3)
    
    # Add legends
    plt.legend(
        title='Matrix Size',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True
    )
    
    plt.tight_layout()
    plt.show()

def plot_matrix_size_comparison(df, matrix_size):
    """
    Compare mmult vs naive for a specific matrix size
    :param df: Input DataFrame with benchmark data
    :param matrix_size: Specific matrix size to analyze (e.g., 256)
    """
    # Filter data for specified matrix size
    size_data = df[df['Matrix Size'] == matrix_size]
    
    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Comparison for Matrix Size = {matrix_size}', fontsize=14)
    
    # Execution Time comparison
    for impl in ['naive', 'mmult']:
        impl_data = size_data[size_data['Implementation'].str.contains(impl)]
        ax1.bar(impl_data['Test'], impl_data['Execution Time (s)'], 
                label=impl, alpha=0.7)
    ax1.set_title('Execution Time (s)')
    ax1.set_ylabel('Seconds')
    ax1.legend()
    
    # L1 Miss comparison
    for impl in ['naive', 'mmult']:
        impl_data = size_data[size_data['Implementation'].str.contains(impl)]
        ax2.bar(impl_data['Test'], impl_data['L1 Miss %'], 
                label=impl, alpha=0.7)
    ax2.set_title('L1 Miss Percentage')
    ax2.set_ylabel('Miss %')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_overall_averages(df):
    """Show average comparison across all matrix sizes"""
    # Calculate means grouped by implementation type
    avg_results = df.groupby(
        df['Implementation'].str.contains('mmult').map({True: 'mmult', False: 'naive'})
    ).agg({
        'Execution Time (s)': 'mean',
        'L1 Miss %': 'mean'
    }).reset_index()
    
    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Overall Average Comparison', fontsize=14)
    
    # Execution Time comparison
    ax1.bar(avg_results['Implementation'], avg_results['Execution Time (s)'],
            color=['skyblue', 'salmon'], alpha=0.7)
    ax1.set_title('Average Execution Time')
    ax1.set_ylabel('Seconds')
    
    # L1 Miss comparison
    ax2.bar(avg_results['Implementation'], avg_results['L1 Miss %'],
            color=['skyblue', 'salmon'], alpha=0.7)
    ax2.set_title('Average L1 Miss Percentage')
    ax2.set_ylabel('Miss %')
    
    plt.tight_layout()
    plt.show()

def compare_mmult_with_naive():
    path = './gemm/results/compare_vector_allocation.json'

    with open(path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    plot_matrix_size_comparison(df, 512)

def main():
    path = './gemm/results/standard_metrics.json'

    with open(path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    plot_linechart(df)
    plot_linechart_mean(df)

    plot_heatmap(df)
    plot_top_performing(df)

    plot_miss_by_config(df)
    plot_miss_by_config(df, 'L1 Miss %')
    plot_miss_heatmaps(df)

    plot_miss_time_trend(df)
    plot_miss_time_trend(df, 'L1 Miss %')

    plot_cache_trends_log_scale(df)
    compare_mmult_with_naive()


main()