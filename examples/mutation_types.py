#!/usr/bin/env python3
"""Visualize all genetic programming mutation types."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import os

def create_mutation_types():
    """Show all mutation types: point, subtree, hoist, shrink."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), facecolor='#0d1117')
    fig.suptitle('🎲 Mutation Types in Genetic Programming', 
                 fontsize=20, fontweight='bold', color='white', y=0.95)
    
    for row in axes:
        for ax in row:
            ax.set_facecolor('#0d1117')
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 4)
            ax.axis('off')
    
    # === Point Mutation ===
    # Original: x + 3
    ax = axes[0, 0]
    ax.set_title('Point Mutation (Before)', color='#58a6ff', fontsize=12, fontweight='bold')
    
    nodes = [(2, 3, '+', '#f97583'), (1, 2, 'x', '#7ee787'), (3, 2, '3', '#58a6ff')]
    edges = [(0,1), (0,2)]
    
    for parent, child in edges:
        px, py = nodes[parent][0], nodes[parent][1]
        cx, cy = nodes[child][0], nodes[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes):
        if i == 2:  # Highlight node to mutate
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, 'x + 3', ha='center', va='center', color='#8b949e', fontsize=11)
    
    ax = axes[0, 1] 
    ax.text(2, 2, '→', fontsize=40, ha='center', va='center', color='#ffa657', fontweight='bold')
    ax.text(2, 1, 'Change\nnode value', ha='center', va='center', color='#8b949e', fontsize=10)
    
    # After: x + 7
    ax = axes[0, 2]
    ax.set_title('Point Mutation (After)', color='#7ee787', fontsize=12, fontweight='bold')
    
    nodes_after = [(2, 3, '+', '#f97583'), (1, 2, 'x', '#7ee787'), (3, 2, '7', '#ffa657')]
    for parent, child in edges:
        px, py = nodes_after[parent][0], nodes_after[parent][1]
        cx, cy = nodes_after[child][0], nodes_after[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_after:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, 'x + 7', ha='center', va='center', color='#8b949e', fontsize=11)
    
    # === Subtree Mutation ===
    # Original: (x + y) * 2
    ax = axes[1, 0]
    ax.set_title('Subtree Mutation (Before)', color='#58a6ff', fontsize=12, fontweight='bold')
    
    nodes = [(2, 3, '×', '#f97583'), (1, 2, '+', '#7ee787'), (3, 2, '2', '#58a6ff'), 
             (0.5, 1, 'x', '#7ee787'), (1.5, 1, 'y', '#7ee787')]
    edges = [(0,1), (0,2), (1,3), (1,4)]
    
    for parent, child in edges:
        px, py = nodes[parent][0], nodes[parent][1]
        cx, cy = nodes[child][0], nodes[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes):
        if i == 1 or i == 3 or i == 4:  # Highlight subtree to replace
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, '(x + y) × 2', ha='center', va='center', color='#8b949e', fontsize=11)
    
    ax = axes[1, 1]
    ax.text(2, 2, '→', fontsize=40, ha='center', va='center', color='#ffa657', fontweight='bold')
    ax.text(2, 1, 'Replace\nsubtree', ha='center', va='center', color='#8b949e', fontsize=10)
    
    # After: z * 2
    ax = axes[1, 2]
    ax.set_title('Subtree Mutation (After)', color='#7ee787', fontsize=12, fontweight='bold')
    
    nodes_after = [(2, 3, '×', '#f97583'), (1, 2, 'z', '#ffa657'), (3, 2, '2', '#58a6ff')]
    edges_after = [(0,1), (0,2)]
    
    for parent, child in edges_after:
        px, py = nodes_after[parent][0], nodes_after[parent][1]
        cx, cy = nodes_after[child][0], nodes_after[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_after:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, 'z × 2', ha='center', va='center', color='#8b949e', fontsize=11)
    
    # === Hoist Mutation ===
    # Original: sin(x + y)
    ax = axes[2, 0]
    ax.set_title('Hoist Mutation (Before)', color='#58a6ff', fontsize=12, fontweight='bold')
    
    nodes = [(2, 3, 'sin', '#d2a8ff'), (2, 2, '+', '#7ee787'), 
             (1.5, 1, 'x', '#7ee787'), (2.5, 1, 'y', '#7ee787')]
    edges = [(0,1), (1,2), (1,3)]
    
    for parent, child in edges:
        px, py = nodes[parent][0], nodes[parent][1]
        cx, cy = nodes[child][0], nodes[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes):
        if i == 1 or i == 2 or i == 3:  # Highlight subtree to hoist
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, 'sin(x + y)', ha='center', va='center', color='#8b949e', fontsize=11)
    
    ax = axes[2, 1]
    ax.text(2, 2, '→', fontsize=40, ha='center', va='center', color='#ffa657', fontweight='bold')
    ax.text(2, 1, 'Hoist\nsubtree up', ha='center', va='center', color='#8b949e', fontsize=10)
    
    # After: x + y
    ax = axes[2, 2]
    ax.set_title('Hoist Mutation (After)', color='#7ee787', fontsize=12, fontweight='bold')
    
    nodes_after = [(2, 3, '+', '#ffa657'), (1.5, 2, 'x', '#ffa657'), (2.5, 2, 'y', '#ffa657')]
    edges_after = [(0,1), (0,2)]
    
    for parent, child in edges_after:
        px, py = nodes_after[parent][0], nodes_after[parent][1]
        cx, cy = nodes_after[child][0], nodes_after[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_after:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, 'x + y', ha='center', va='center', color='#8b949e', fontsize=11)
    
    # === Shrink Mutation ===
    # Original: (x * y) + z
    ax = axes[3, 0]
    ax.set_title('Shrink Mutation (Before)', color='#58a6ff', fontsize=12, fontweight='bold')
    
    nodes = [(2, 3, '+', '#f97583'), (1, 2, '×', '#d2a8ff'), (3, 2, 'z', '#7ee787'),
             (0.5, 1, 'x', '#7ee787'), (1.5, 1, 'y', '#7ee787')]
    edges = [(0,1), (0,2), (1,3), (1,4)]
    
    for parent, child in edges:
        px, py = nodes[parent][0], nodes[parent][1]
        cx, cy = nodes[child][0], nodes[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for i, (x, y, label, color) in enumerate(nodes):
        if i == 1 or i == 3 or i == 4:  # Highlight subtree to shrink
            circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.8)
        else:
            circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                           linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, '(x × y) + z', ha='center', va='center', color='#8b949e', fontsize=11)
    
    ax = axes[3, 1]
    ax.text(2, 2, '→', fontsize=40, ha='center', va='center', color='#ffa657', fontweight='bold')
    ax.text(2, 1, 'Replace with\nterminal', ha='center', va='center', color='#8b949e', fontsize=10)
    
    # After: 5 + z
    ax = axes[3, 2]
    ax.set_title('Shrink Mutation (After)', color='#7ee787', fontsize=12, fontweight='bold')
    
    nodes_after = [(2, 3, '+', '#f97583'), (1, 2, '5', '#ffa657'), (3, 2, 'z', '#7ee787')]
    edges_after = [(0,1), (0,2)]
    
    for parent, child in edges_after:
        px, py = nodes_after[parent][0], nodes_after[parent][1]
        cx, cy = nodes_after[child][0], nodes_after[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, zorder=1)
    
    for x, y, label, color in nodes_after:
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                       linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', zorder=3)
    
    ax.text(2, 0.5, '5 + z', ha='center', va='center', color='#8b949e', fontsize=11)
    
    # Add mutation type labels on the left
    mutation_labels = ['🎯 Point', '🌳 Subtree', '⬆️ Hoist', '⬇️ Shrink']
    for i, label in enumerate(mutation_labels):
        axes[i, 0].text(-0.8, 2, label, ha='center', va='center', fontsize=14, 
                       fontweight='bold', color='#ffa657', rotation=90)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/mutation_types.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_mutation_types()