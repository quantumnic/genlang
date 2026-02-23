#!/usr/bin/env python3
"""Animated mutation visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

def create_mutation_animation():
    """Animate mutation of an AST tree."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117')
    fig.suptitle('🎯 Mutation Operation — Random Subtree Replacement', 
                 color='white', fontsize=16, fontweight='bold', y=0.95)
    
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.axis('off')
    
    axes[0].set_title('Original: x + 1', color='white', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_title('Mutated: x + sin(x)', color='white', fontsize=12, fontweight='bold', pad=10)
    
    # Original tree: x + 1
    orig_nodes = [(2, 4, '+', '#f97583'), (1, 3, 'x', '#7ee787'), (3, 3, '1', '#58a6ff')]
    orig_edges = [(0, 1), (0, 2)]
    
    # Mutated tree: x + sin(x)
    mut_nodes = [(2, 4, '+', '#f97583'), (1, 3, 'x', '#7ee787'), (3, 3, 'sin', '#d2a8ff'), (3, 2, 'x', '#7ee787')]
    mut_edges = [(0, 1), (0, 2), (2, 3)]
    
    def draw_tree(ax, nodes, edges, highlight_node=None, highlight_intensity=0, growing_nodes=None, grow_alpha=1.0):
        ax.clear()
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.axis('off')
        
        # Draw edges
        for parent_idx, child_idx in edges:
            px, py = nodes[parent_idx][0], nodes[parent_idx][1]
            cx, cy = nodes[child_idx][0], nodes[child_idx][1]
            
            alpha = grow_alpha if growing_nodes and (parent_idx in growing_nodes or child_idx in growing_nodes) else 1.0
            ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2, alpha=alpha, zorder=1)
        
        # Draw nodes
        for i, (x, y, label, color) in enumerate(nodes):
            alpha = grow_alpha if growing_nodes and i in growing_nodes else 1.0
            
            if highlight_node == i and highlight_intensity > 0:
                # Glowing mutation point
                for r in [0.4, 0.5, 0.6]:
                    glow = Circle((x, y), r, facecolor='#ffa657', 
                                 alpha=0.3 * highlight_intensity, zorder=1)
                    ax.add_patch(glow)
                
                circle = Circle((x, y), 0.25, facecolor='#ffa657', edgecolor='white', 
                               linewidth=3, zorder=2, alpha=alpha)
                ax.add_patch(circle)
            else:
                circle = Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                               linewidth=2, zorder=2, alpha=alpha)
                ax.add_patch(circle)
            
            ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                   fontweight='bold', color='white', zorder=3, alpha=alpha)
    
    # Animation frames
    frames = []
    
    # Phase 1: Show original (30 frames)
    for _ in range(30):
        frames.append(('original', 0, None, 1.0))
    
    # Phase 2: Highlight mutation point with pulsing glow (40 frames)
    for i in range(40):
        intensity = 0.5 + 0.5 * np.sin(i * 0.3)  # Pulsing glow
        frames.append(('highlight', intensity, None, 1.0))
    
    # Phase 3: Flash and fade out old subtree (20 frames)
    for i in range(20):
        intensity = max(0, 1 - i / 10.0) if i < 10 else 0
        frames.append(('flash', intensity, None, 1.0))
    
    # Phase 4: Grow new subtree (30 frames)
    for i in range(30):
        grow_alpha = min(1.0, i / 20.0)
        frames.append(('grow', 0, [2, 3], grow_alpha))
    
    # Phase 5: Show final result (40 frames)
    for _ in range(40):
        frames.append(('final', 0, None, 1.0))
    
    def animate(frame_idx):
        phase, intensity, growing_nodes, grow_alpha = frames[frame_idx]
        
        if phase == 'original':
            draw_tree(axes[0], orig_nodes, orig_edges)
            axes[1].clear()
            axes[1].set_xlim(-1, 5)
            axes[1].set_ylim(-1, 5)
            axes[1].axis('off')
            
        elif phase == 'highlight':
            draw_tree(axes[0], orig_nodes, orig_edges, highlight_node=2, highlight_intensity=intensity)
            axes[1].clear()
            axes[1].set_xlim(-1, 5)
            axes[1].set_ylim(-1, 5)
            axes[1].axis('off')
            
        elif phase == 'flash':
            draw_tree(axes[0], orig_nodes, orig_edges, highlight_node=2, highlight_intensity=intensity)
            axes[1].clear()
            axes[1].set_xlim(-1, 5)
            axes[1].set_ylim(-1, 5)
            axes[1].axis('off')
            
        elif phase == 'grow':
            draw_tree(axes[0], orig_nodes, orig_edges)
            draw_tree(axes[1], mut_nodes, mut_edges, growing_nodes=growing_nodes, grow_alpha=grow_alpha)
            
        elif phase == 'final':
            draw_tree(axes[0], orig_nodes, orig_edges)
            draw_tree(axes[1], mut_nodes, mut_edges)
        
        # Update titles
        axes[0].set_title('Original: x + 1', color='white', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_title('Mutated: x + sin(x)', color='white', fontsize=12, fontweight='bold', pad=10)
    
    # Add explanation text
    fig.text(0.5, 0.08, '1. Select random node → 2. Mutation point glows → 3. Replace with new subtree', 
             ha='center', va='bottom', color='#8b949e', fontsize=11, fontweight='bold')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=150, repeat=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/mutation_animation.mp4')
    anim.save(out, writer='ffmpeg', fps=8, dpi=120, savefig_kwargs={'facecolor': '#0d1117'})
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_mutation_animation()