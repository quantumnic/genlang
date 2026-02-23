#!/usr/bin/env python3
"""Visualize genetic programming crossover operation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import os

def create_crossover_demo():
    """Show how crossover works with two parent ASTs."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='#0d1117')
    fig.suptitle('🧬 Crossover Operation — Subtree Exchange', 
                 fontsize=20, fontweight='bold', color='white', y=0.95)
    
    for row in axes:
        for ax in row:
            ax.set_facecolor('#0d1117')
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 4)
            ax.axis('off')
    
    # Parent 1: (x + y) * 2
    ax = axes[0, 0]
    ax.set_title('Parent 1: (x + y) × 2', color='#58a6ff', fontsize=14, fontweight='bold')
    
    # Draw tree structure
    nodes_p1 = [(2, 3, '×', '#f97583'), (1, 2, '+', '#7ee787'), (3, 2, '2', '#58a6ff'), 
                (0.5, 1, 'x', '#7ee787'), (1.5, 1, 'y', '#7ee787')]
    edges_p1 = [(0,1), (0,2), (1,3), (1,4)]
    
    for parent, child in edges_p1:
        px, py = nodes_p1[parent][0], nodes_p1[parent][1]
        cx, cy = nodes_p1[child][0], nodes_p1[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes_p1):
        # Highlight crossover subtree (node 1: '+')
        if i == 1 or i == 3 or i == 4:  # + node and its children
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    # Crossover annotation
    ax.annotate('Crossover\nPoint ✂️', xy=(1, 2), xytext=(-0.5, 3.5),
                color='#ffa657', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2))
    
    # Parent 2: z / (3 - w)  
    ax = axes[0, 1]
    ax.set_title('Parent 2: z ÷ (3 - w)', color='#58a6ff', fontsize=14, fontweight='bold')
    
    nodes_p2 = [(2, 3, '÷', '#d2a8ff'), (1, 2, 'z', '#7ee787'), (3, 2, '-', '#f97583'), 
                (2.5, 1, '3', '#58a6ff'), (3.5, 1, 'w', '#7ee787')]
    edges_p2 = [(0,1), (0,2), (2,3), (2,4)]
    
    for parent, child in edges_p2:
        px, py = nodes_p2[parent][0], nodes_p2[parent][1]
        cx, cy = nodes_p2[child][0], nodes_p2[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes_p2):
        # Highlight crossover subtree (nodes 2,3,4: '- 3 w')
        if i == 2 or i == 3 or i == 4:
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.annotate('Crossover\nPoint ✂️', xy=(3, 2), xytext=(4.2, 3.5),
                color='#ffa657', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2))
    
    # Arrow showing swap
    ax = axes[0, 2]
    ax.set_title('Exchange Subtrees', color='#ffa657', fontsize=14, fontweight='bold')
    ax.text(2, 2, '⟷', fontsize=80, ha='center', va='center', color='#ffa657', 
            fontweight='bold')
    ax.text(2, 1, 'Swap highlighted\nsubtrees', ha='center', va='center', 
            color='#8b949e', fontsize=12)
    
    # Child 1: (3 - w) * 2
    ax = axes[1, 0]
    ax.set_title('Child 1: (3 - w) × 2', color='#7ee787', fontsize=14, fontweight='bold')
    
    nodes_c1 = [(2, 3, '×', '#f97583'), (1, 2, '-', '#ffa657'), (3, 2, '2', '#58a6ff'), 
                (0.5, 1, '3', '#ffa657'), (1.5, 1, 'w', '#ffa657')]
    edges_c1 = [(0,1), (0,2), (1,3), (1,4)]
    
    for parent, child in edges_c1:
        px, py = nodes_c1[parent][0], nodes_c1[parent][1]
        cx, cy = nodes_c1[child][0], nodes_c1[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_c1:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    # Child 2: z / (x + y)
    ax = axes[1, 1]
    ax.set_title('Child 2: z ÷ (x + y)', color='#7ee787', fontsize=14, fontweight='bold')
    
    nodes_c2 = [(2, 3, '÷', '#d2a8ff'), (1, 2, 'z', '#7ee787'), (3, 2, '+', '#ffa657'), 
                (2.5, 1, 'x', '#ffa657'), (3.5, 1, 'y', '#ffa657')]
    edges_c2 = [(0,1), (0,2), (2,3), (2,4)]
    
    for parent, child in edges_c2:
        px, py = nodes_c2[parent][0], nodes_c2[parent][1]
        cx, cy = nodes_c2[child][0], nodes_c2[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_c2:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    # Result description
    ax = axes[1, 2]
    ax.text(2, 2.5, '✅ Crossover Complete!', ha='center', va='center', 
            color='#7ee787', fontsize=16, fontweight='bold')
    ax.text(2, 1.5, '• Genetic diversity increased\n• New program variants\n• Preserve good building blocks', 
            ha='center', va='center', color='#8b949e', fontsize=11, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/crossover_demo.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_crossover_demo()