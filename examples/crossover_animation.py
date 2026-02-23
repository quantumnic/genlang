#!/usr/bin/env python3
"""Animated crossover visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import os

def create_crossover_animation():
    """Animate crossover between two AST trees."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor='#0d1117')
    fig.suptitle('🧬 Crossover Operation — Subtree Exchange', 
                 color='white', fontsize=16, fontweight='bold', y=0.95)
    
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 4)
        ax.axis('off')
    
    titles = ['Parent 1: x + 1', 'Parent 2: x * (x - 2)', 'Offspring: (x - 2) + 1']
    for ax, title in zip(axes, titles):
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    
    # Define tree structures
    # Parent 1: x + 1
    p1_nodes = [(2, 3, '+', '#f97583'), (1, 2, 'x', '#7ee787'), (3, 2, '1', '#58a6ff')]
    p1_edges = [(0, 1), (0, 2)]
    
    # Parent 2: x * (x - 2)  
    p2_nodes = [(2, 3, '*', '#d2a8ff'), (1, 2, 'x', '#7ee787'), (3, 2, '-', '#f97583'),
                (2.5, 1, 'x', '#7ee787'), (3.5, 1, '2', '#58a6ff')]
    p2_edges = [(0, 1), (0, 2), (2, 3), (2, 4)]
    
    # Offspring: (x - 2) + 1
    off_nodes = [(2, 3, '+', '#f97583'), (1, 2, '-', '#f97583'), (3, 2, '1', '#58a6ff'),
                 (0.5, 1, 'x', '#7ee787'), (1.5, 1, '2', '#58a6ff')]
    off_edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    
    def draw_tree(ax, nodes, edges, highlight_nodes=None, alpha=1.0):
        ax.clear()
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 4)
        ax.axis('off')
        
        # Draw edges
        for parent_idx, child_idx in edges:
            px, py = nodes[parent_idx][0], nodes[parent_idx][1]
            cx, cy = nodes[child_idx][0], nodes[child_idx][1]
            color = '#ffa657' if highlight_nodes and (parent_idx in highlight_nodes or child_idx in highlight_nodes) else '#30363d'
            width = 3 if highlight_nodes and (parent_idx in highlight_nodes or child_idx in highlight_nodes) else 2
            ax.plot([px, cx], [py, cy], color=color, linewidth=width, alpha=alpha, zorder=1)
        
        # Draw nodes
        for i, (x, y, label, color) in enumerate(nodes):
            if highlight_nodes and i in highlight_nodes:
                # Highlighted node (bigger, glowing)
                circle = Circle((x, y), 0.35, facecolor='#ffa657', edgecolor='white', 
                               linewidth=3, zorder=2, alpha=alpha)
                ax.add_patch(circle)
                # Glow effect
                for r in [0.45, 0.55]:
                    glow = Circle((x, y), r, facecolor='#ffa657', alpha=0.2*alpha, zorder=1)
                    ax.add_patch(glow)
            else:
                circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                               linewidth=2, zorder=2, alpha=alpha)
                ax.add_patch(circle)
            
            ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                   fontweight='bold', color='white', zorder=3, alpha=alpha)
    
    # Animation frames
    frames = []
    
    # Frame 1-20: Show original trees
    for _ in range(20):
        frames.append(('original', None))
    
    # Frame 21-40: Highlight crossover points
    for _ in range(20):
        frames.append(('highlight', None))
    
    # Frame 41-60: Fade out and swap
    for i in range(20):
        alpha = 1.0 - i / 19.0
        frames.append(('fade', alpha))
    
    # Frame 61-80: Fade in offspring
    for i in range(20):
        alpha = i / 19.0
        frames.append(('offspring', alpha))
    
    # Frame 81-120: Show final result
    for _ in range(40):
        frames.append(('final', None))
    
    def animate(frame_idx):
        frame_type, alpha = frames[frame_idx]
        
        if frame_type == 'original':
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            for ax, title in zip(axes, titles):
                ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
            draw_tree(axes[0], p1_nodes, p1_edges)
            draw_tree(axes[1], p2_nodes, p2_edges)
            
        elif frame_type == 'highlight':
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            for ax, title in zip(axes, titles):
                ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
            draw_tree(axes[0], p1_nodes, p1_edges, highlight_nodes=[2])  # Highlight '1'
            draw_tree(axes[1], p2_nodes, p2_edges, highlight_nodes=[2, 3, 4])  # Highlight subtree 'x - 2'
            
        elif frame_type == 'fade':
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            for ax, title in zip(axes, titles):
                ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
            draw_tree(axes[0], p1_nodes, p1_edges, highlight_nodes=[2], alpha=alpha)
            draw_tree(axes[1], p2_nodes, p2_edges, highlight_nodes=[2, 3, 4], alpha=alpha)
            
        elif frame_type == 'offspring':
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            for ax, title in zip(axes, titles):
                ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
            draw_tree(axes[2], off_nodes, off_edges, alpha=alpha)
            
        elif frame_type == 'final':
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            for ax, title in zip(axes, titles):
                ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
            draw_tree(axes[0], p1_nodes, p1_edges)
            draw_tree(axes[1], p2_nodes, p2_edges)
            draw_tree(axes[2], off_nodes, off_edges)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100, repeat=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/crossover_animation.mp4')
    anim.save(out, writer='ffmpeg', fps=10, dpi=120, savefig_kwargs={'facecolor': '#0d1117'})
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_crossover_animation()