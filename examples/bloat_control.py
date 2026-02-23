#!/usr/bin/env python3
"""Visualize bloat control in genetic programming."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def create_bloat_control():
    """Show program size vs fitness over generations with bloat control."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#0d1117')
    fig.suptitle('⚖️ Bloat Control — Managing Program Size', 
                 fontsize=18, fontweight='bold', color='white', y=0.95)
    
    for row in axes:
        for ax in row:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
    
    n_generations = 100
    
    # Simulate evolution WITHOUT bloat control
    generations_no_control = []
    sizes_no_control = []
    fitness_no_control = []
    
    for gen in range(n_generations):
        # Size grows exponentially without control
        base_size = 5 + gen * 0.5 + np.random.exponential(gen * 0.1)
        # Fitness plateaus while size explodes
        base_fitness = max(0.1, 1.0 - gen * 0.008 + np.random.normal(0, 0.05))
        
        sizes_no_control.append(base_size)
        fitness_no_control.append(base_fitness)
    
    # Simulate evolution WITH bloat control (parsimony pressure)
    generations_with_control = []
    sizes_with_control = []
    fitness_with_control = []
    
    for gen in range(n_generations):
        # Size controlled by parsimony pressure
        if gen < 30:
            # Initial growth phase
            base_size = 5 + gen * 0.2 + np.random.normal(0, 1)
        else:
            # Size stabilizes due to parsimony pressure
            base_size = 12 + np.random.normal(0, 2) + np.sin(gen * 0.1) * 1
            
        base_size = max(3, base_size)  # Minimum viable size
        
        # Fitness continues improving
        base_fitness = max(0.05, 1.0 - gen * 0.009 + np.random.normal(0, 0.03))
        
        sizes_with_control.append(base_size)  
        fitness_with_control.append(base_fitness)
    
    # Plot 1: Without bloat control
    ax = axes[0, 0]
    
    # Dual axis for size and fitness
    ax2 = ax.twinx()
    
    # Size (left axis)
    line1 = ax.plot(range(n_generations), sizes_no_control, color='#ff6b6b', 
                   linewidth=3, alpha=0.9, label='Program Size')
    ax.fill_between(range(n_generations), sizes_no_control, alpha=0.3, color='#ff6b6b')
    ax.set_ylabel('Average Program Size (nodes)', color='#ff6b6b', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#ff6b6b')
    
    # Fitness (right axis)
    line2 = ax2.plot(range(n_generations), fitness_no_control, color='#4ecdc4', 
                    linewidth=3, alpha=0.9, label='Fitness (1-error)')
    ax2.set_ylabel('Fitness', color='#4ecdc4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#4ecdc4')
    
    ax.set_title('❌ Without Bloat Control', color='#ff6b6b', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Annotations
    ax.annotate('Size explodes!', xy=(80, sizes_no_control[80]), xytext=(60, sizes_no_control[80]+10),
               color='#ff6b6b', fontweight='bold', fontsize=11,
               arrowprops=dict(arrowstyle='->', color='#ff6b6b'))
    ax2.annotate('Fitness plateaus', xy=(70, fitness_with_control[70]), xytext=(50, 0.8),
                color='#4ecdc4', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#4ecdc4'))
    
    # Plot 2: With bloat control
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    # Size (left axis)
    line1 = ax.plot(range(n_generations), sizes_with_control, color='#7ee787', 
                   linewidth=3, alpha=0.9, label='Program Size')
    ax.fill_between(range(n_generations), sizes_with_control, alpha=0.3, color='#7ee787')
    ax.set_ylabel('Average Program Size (nodes)', color='#7ee787', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#7ee787')
    
    # Fitness (right axis)
    line2 = ax2.plot(range(n_generations), fitness_with_control, color='#58a6ff', 
                    linewidth=3, alpha=0.9, label='Fitness (1-error)')
    ax2.set_ylabel('Fitness', color='#58a6ff', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#58a6ff')
    
    ax.set_title('✅ With Bloat Control', color='#7ee787', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Annotations
    ax.annotate('Size controlled', xy=(70, sizes_with_control[70]), xytext=(50, sizes_with_control[70]+5),
               color='#7ee787', fontweight='bold', fontsize=11,
               arrowprops=dict(arrowstyle='->', color='#7ee787'))
    ax2.annotate('Fitness improves', xy=(80, fitness_with_control[80]), xytext=(60, 0.3),
                color='#58a6ff', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#58a6ff'))
    
    # Plot 3: Comparison of final program sizes
    ax = axes[1, 0]
    
    # Create histograms of final program sizes
    final_sizes_no_control = [sizes_no_control[-1] + np.random.normal(0, 5) for _ in range(50)]
    final_sizes_with_control = [sizes_with_control[-1] + np.random.normal(0, 2) for _ in range(50)]
    
    bins = np.linspace(0, 80, 20)
    ax.hist(final_sizes_no_control, bins=bins, alpha=0.7, color='#ff6b6b', 
           label='Without Control', edgecolor='white', linewidth=1)
    ax.hist(final_sizes_with_control, bins=bins, alpha=0.7, color='#7ee787', 
           label='With Control', edgecolor='white', linewidth=1)
    
    ax.set_title('Final Program Size Distribution', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Program Size (nodes)', color='#8b949e')
    ax.set_ylabel('Count', color='#8b949e') 
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add mean lines
    mean_no_control = np.mean(final_sizes_no_control)
    mean_with_control = np.mean(final_sizes_with_control)
    ax.axvline(mean_no_control, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_with_control, color='#7ee787', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(mean_no_control, ax.get_ylim()[1]*0.8, f'Mean: {mean_no_control:.1f}', 
           color='#ff6b6b', fontweight='bold', ha='center')
    ax.text(mean_with_control, ax.get_ylim()[1]*0.6, f'Mean: {mean_with_control:.1f}', 
           color='#7ee787', fontweight='bold', ha='center')
    
    # Plot 4: Parsimony pressure methods
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('Bloat Control Methods', color='white', fontsize=14, fontweight='bold')
    
    methods = [
        ('🎯 Parsimony Pressure', 'Add size penalty to fitness:\nfitness = accuracy - α × size'),
        ('✂️ Size Limits', 'Maximum tree depth/nodes:\nReject programs exceeding limits'),
        ('⚖️ Multi-objective', 'Optimize accuracy AND size:\nPareto front trade-offs'),
        ('🔄 Bloat-aware Operators', 'Crossover/mutation favors\nsmaller, simpler programs')
    ]
    
    y_positions = [3.5, 2.5, 1.5, 0.5]
    colors = ['#f97583', '#58a6ff', '#d2a8ff', '#ffa657']
    
    for (title, desc), y, color in zip(methods, y_positions, colors):
        # Method box
        rect = Rectangle((0, y-0.3), 8, 0.6, facecolor=color, alpha=0.2, 
                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        ax.text(0.2, y+0.15, title, fontsize=12, fontweight='bold', color=color, va='center')
        ax.text(0.2, y-0.1, desc, fontsize=10, color='#8b949e', va='center')
    
    # Key insight box
    ax.text(4, -0.3, 
           '💡 Key Insight: Bloat hurts both performance and interpretability.\n'
           'Smaller programs are often more generalizable and efficient.',
           ha='center', va='top', fontsize=11, color='#8b949e',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/bloat_control.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_bloat_control()