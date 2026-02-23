#!/usr/bin/env python3
"""Operator frequency across generations."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_operator_frequency():
    """Stacked bar chart showing operator usage per generation."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    # Define operators and their colors
    operators = ['x', '+', '×', 'sin', 'cos', 'exp', 'log', 'if', 'constants']
    colors = ['#7ee787', '#f97583', '#58a6ff', '#d2a8ff', '#ffa657', 
              '#ff6b9d', '#79c0ff', '#f0e68c', '#8b949e']
    
    generations = np.arange(0, 51, 5)  # Every 5 generations
    np.random.seed(42)
    
    # Generate operator usage data
    operator_data = {}
    
    for i, gen in enumerate(generations):
        progress = i / (len(generations) - 1)
        
        # Initial distribution (more random)
        if gen == 0:
            # Random initial distribution
            usage = np.random.dirichlet([1] * len(operators)) * 100
        else:
            # Evolution trends
            usage = []
            
            # Variables become more common (fundamental)
            usage.append(15 + progress * 10 + np.random.normal(0, 2))  # x
            
            # Basic arithmetic stays important
            usage.append(20 + np.random.normal(0, 3))  # +
            usage.append(15 + progress * 5 + np.random.normal(0, 2))  # ×
            
            # Trigonometric functions - depends on problem type
            trig_importance = 10 * np.sin(progress * np.pi) + 5
            usage.append(trig_importance + np.random.normal(0, 2))  # sin
            usage.append(trig_importance * 0.7 + np.random.normal(0, 2))  # cos
            
            # Complex functions decrease (bloat control)
            usage.append(max(1, 8 - progress * 6 + np.random.normal(0, 1)))  # exp
            usage.append(max(1, 6 - progress * 4 + np.random.normal(0, 1)))  # log
            usage.append(max(1, 7 - progress * 5 + np.random.normal(0, 1)))  # if
            
            # Constants adjust based on needs
            usage.append(12 + progress * 3 + np.random.normal(0, 2))  # constants
            
            # Normalize to 100%
            usage = np.array(usage)
            usage = np.maximum(usage, 0.5)  # Minimum usage
            usage = (usage / usage.sum()) * 100
        
        operator_data[gen] = usage
    
    # Create stacked bar chart
    bottom = np.zeros(len(generations))
    
    for i, (operator, color) in enumerate(zip(operators, colors)):
        values = [operator_data[gen][i] for gen in generations]
        bars = ax.bar(generations, values, bottom=bottom, label=operator, 
                     color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values
        
        # Add percentage labels on larger segments
        for j, (gen, value, bar_bottom) in enumerate(zip(generations, values, bottom - values)):
            if value > 5:  # Only label if segment is large enough
                ax.text(gen, bar_bottom + value/2, f'{value:.0f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white' if value > 10 else 'black')
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Operator Usage (%)', color='white', fontsize=14, fontweight='bold')
    ax.set_title('📊 Operator Frequency Evolution — GP Component Usage', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3, color='#30363d')
    
    # Legend with better positioning
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      facecolor='#21262d', edgecolor='#30363d', 
                      labelcolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Add trend annotations
    annotations = [
        (10, 85, 'Random\nExploration', '#f97583'),
        (25, 90, 'Function\nSpecialization', '#ffa657'),
        (40, 85, 'Simplified\nSolutions', '#7ee787'),
    ]
    
    for x, y, label, color in annotations:
        ax.annotate(label, xy=(x, y), xytext=(x, y+8),
                   color=color, fontsize=11, fontweight='bold', ha='center',
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.8))
    
    # Evolution insights
    insights = ('Evolution Patterns:\n'
               '• Variables (x) increase - more direct solutions\n'
               '• Complex functions (exp, log) decrease - bloat control\n'
               '• Trigonometric usage varies by problem type\n'
               '• Constants stabilize around optimal values')
    
    ax.text(0.02, 0.35, insights, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#8b949e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                     edgecolor='#30363d', alpha=0.9))
    
    # Statistics for key operators
    x_start = operator_data[0][0]
    x_end = operator_data[50][0]
    complex_start = sum(operator_data[0][5:8])  # exp + log + if
    complex_end = sum(operator_data[50][5:8])
    
    stats = (f'Key Changes (Gen 0→50):\n'
            f'• Variables: {x_start:.1f}% → {x_end:.1f}%\n'
            f'• Complex Ops: {complex_start:.1f}% → {complex_end:.1f}%\n'
            f'• Simplification: {((complex_start-complex_end)/complex_start)*100:.0f}%')
    
    ax.text(0.98, 0.35, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                     edgecolor='#30363d', alpha=0.9))
    
    # Add diversity trend line (entropy calculation)
    diversities = []
    for gen in generations:
        usage = operator_data[gen]
        # Calculate Shannon entropy (diversity measure)
        entropy = -np.sum((usage/100) * np.log2(usage/100 + 1e-10))
        diversities.append(entropy)
    
    # Plot diversity on secondary axis
    ax2 = ax.twinx()
    ax2.plot(generations, diversities, color='white', linewidth=3, 
            marker='o', markersize=6, alpha=0.9, zorder=10, 
            label='Operator Diversity')
    ax2.set_ylabel('Diversity (Shannon Entropy)', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.spines['right'].set_color('white')
    for spine in ['top', 'bottom', 'left']:
        ax2.spines[spine].set_color('#30363d')
    
    # Diversity legend
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9),
              facecolor='#21262d', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/operator_frequency.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_operator_frequency()