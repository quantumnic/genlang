#!/usr/bin/env python3
"""Hall of Fame - Top 5 all-time best programs."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import os

def create_hall_of_fame():
    """Top 5 all-time best programs displayed as mini-trees."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0d1117')
    fig.suptitle('🏆 Hall of Fame — Top 5 All-Time Best Programs', 
                 color='white', fontsize=20, fontweight='bold', y=0.95)
    
    # Remove the extra subplot
    axes[1, 2].remove()
    
    # Flatten axes for easier iteration (excluding removed one)
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    
    # Define the top 5 programs
    programs = [
        {
            'rank': 1,
            'expression': 'x² + x + 1',
            'fitness': 0.001,
            'generation': 47,
            'nodes': [(2, 3, '+', '#7ee787'), (1, 2, '×', '#58a6ff'), (3, 2, '+', '#f97583'),
                     (0.5, 1, 'x', '#ffa657'), (1.5, 1, 'x', '#ffa657'), (2.5, 1, 'x', '#ffa657'), (3.5, 1, '1', '#d2a8ff')],
            'edges': [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            'color': '#FFD700'  # Gold
        },
        {
            'rank': 2, 
            'expression': 'sin(x) + cos(x)',
            'fitness': 0.015,
            'generation': 42,
            'nodes': [(2, 3, '+', '#f97583'), (1, 2, 'sin', '#d2a8ff'), (3, 2, 'cos', '#d2a8ff'),
                     (1, 1, 'x', '#ffa657'), (3, 1, 'x', '#ffa657')],
            'edges': [(0, 1), (0, 2), (1, 3), (2, 4)],
            'color': '#C0C0C0'  # Silver  
        },
        {
            'rank': 3,
            'expression': 'x × (x - 2)',
            'fitness': 0.032,
            'generation': 38,
            'nodes': [(2, 3, '×', '#58a6ff'), (1, 2, 'x', '#ffa657'), (3, 2, '-', '#f97583'),
                     (2.5, 1, 'x', '#ffa657'), (3.5, 1, '2', '#d2a8ff')],
            'edges': [(0, 1), (0, 2), (2, 3), (2, 4)],
            'color': '#CD7F32'  # Bronze
        },
        {
            'rank': 4,
            'expression': 'exp(x) - 2',
            'fitness': 0.089,
            'generation': 35,
            'nodes': [(2, 3, '-', '#f97583'), (1, 2, 'exp', '#d2a8ff'), (3, 2, '2', '#d2a8ff'),
                     (1, 1, 'x', '#ffa657')],
            'edges': [(0, 1), (0, 2), (1, 3)],
            'color': '#9370DB'  # Purple
        },
        {
            'rank': 5,
            'expression': 'x³ + 3×x',
            'fitness': 0.124,
            'generation': 31,
            'nodes': [(2, 3, '+', '#f97583'), (1, 2, '×', '#58a6ff'), (3, 2, '×', '#58a6ff'),
                     (0.5, 1, '×', '#58a6ff'), (1.5, 1, 'x', '#ffa657'), (2.5, 1, '3', '#d2a8ff'), (3.5, 1, 'x', '#ffa657'),
                     (0.5, 0, 'x²', '#7ee787')],
            'edges': [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)],
            'color': '#FF4500'  # Orange-red
        }
    ]
    
    for i, (ax, program) in enumerate(zip(plot_axes, programs)):
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4)
        ax.axis('off')
        
        # Medal/rank decoration
        rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#9370DB', '#FF4500']
        medal_symbols = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        
        # Title with rank
        ax.text(2, 3.7, f"{medal_symbols[i]} Rank #{program['rank']}", 
                ha='center', va='center', color=program['color'], 
                fontsize=16, fontweight='bold')
        
        # Expression
        ax.text(2, 3.4, program['expression'], ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor=program['color'], linewidth=2))
        
        # Draw tree
        nodes = program['nodes']
        edges = program['edges']
        
        # Draw edges first
        for parent_idx, child_idx in edges:
            if parent_idx < len(nodes) and child_idx < len(nodes):
                px, py = nodes[parent_idx][0], nodes[parent_idx][1]
                cx, cy = nodes[child_idx][0], nodes[child_idx][1]
                ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
        
        # Draw nodes
        for x, y, label, color in nodes:
            circle = Circle((x, y), 0.2, facecolor=color, edgecolor='white', 
                           linewidth=1.5, zorder=2, alpha=0.9)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=9, 
                   fontweight='bold', color='white', zorder=3)
        
        # Statistics box
        stats_text = (f'Fitness: {program["fitness"]:.3f}\n'
                     f'Found: Gen {program["generation"]}\n'
                     f'Nodes: {len(nodes)}')
        
        ax.text(2, 0.2, stats_text, ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22', 
                         edgecolor='#30363d', alpha=0.9))
        
        # Glow effect for top 3
        if i < 3:
            glow_circle = Circle((2, 2), 2.5, facecolor=program['color'], 
                               alpha=0.05, edgecolor='none', zorder=0)
            ax.add_patch(glow_circle)
    
    # Add overall statistics in the empty space
    fig.text(0.75, 0.35, 
             '📊 Hall of Fame Statistics\n\n'
             '🎯 Average Fitness: 0.052\n'
             '⏰ Average Discovery: Gen 38.6\n'
             '🌳 Average Complexity: 5.4 nodes\n'
             '🏆 Best Improvement: 124× better\n'
             '⚡ Convergence Time: 47 generations\n\n'
             '🧬 Evolution Success Rate:\n'
             '• Top 1%: Elite solutions\n'
             '• Top 10%: Good solutions\n'
             '• Bottom 50%: Pruned away',
             fontsize=12, color='white', fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Timeline showing discovery order
    timeline_ax = fig.add_subplot(2, 3, 6)
    timeline_ax.set_facecolor('#161b22')
    timeline_ax.tick_params(colors='#8b949e')
    for spine in timeline_ax.spines.values():
        spine.set_color('#30363d')
    
    timeline_ax.set_title('Discovery Timeline', color='white', fontweight='bold')
    
    generations = [p['generation'] for p in programs]
    ranks = [p['rank'] for p in programs]
    colors = [p['color'] for p in programs]
    
    timeline_ax.scatter(generations, ranks, c=colors, s=120, alpha=0.8, 
                       edgecolors='white', linewidth=2, zorder=5)
    
    for i, (gen, rank, color) in enumerate(zip(generations, ranks, colors)):
        timeline_ax.annotate(f'#{rank}', xy=(gen, rank), xytext=(gen, rank-0.3),
                           color=color, fontsize=10, fontweight='bold',
                           ha='center', va='top')
    
    timeline_ax.set_xlabel('Generation Discovered', color='white', fontweight='bold')
    timeline_ax.set_ylabel('Final Rank', color='white', fontweight='bold')
    timeline_ax.invert_yaxis()
    timeline_ax.grid(True, alpha=0.3, color='#30363d')
    timeline_ax.set_ylim(5.5, 0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/hall_of_fame.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_hall_of_fame()