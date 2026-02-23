#!/usr/bin/env python3
"""Generation timeline showing program evolution."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_generation_timeline():
    """Timeline showing best program at different generations."""
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    # Generation data (getting simpler over time)
    generations = [0, 5, 10, 25, 50]
    expressions = [
        'sin(x * cos(x + 1.2) / (x + 0.8)) + x * 2.1 - sin(x / 0.9)',
        'x * x + sin(x * 1.5) - cos(x) + 2.1',
        'x * x + x * 1.8 + sin(x)',
        'x * x + x + 1.2',
        'x * x + x + 1'
    ]
    fitnesses = [15.7, 3.2, 1.8, 0.4, 0.01]
    complexities = [24, 12, 8, 5, 3]
    
    # Main timeline
    y_pos = 0.5
    ax.plot(generations, [y_pos] * len(generations), color='#58a6ff', linewidth=3, alpha=0.8)
    
    # Generation markers and expressions
    for i, (gen, expr, fitness, complexity) in enumerate(zip(generations, expressions, fitnesses, complexities)):
        # Marker
        ax.scatter(gen, y_pos, s=200, color='#7ee787', edgecolor='white', linewidth=2, zorder=5)
        ax.text(gen, y_pos, f'{gen}', ha='center', va='center', fontweight='bold', 
                color='white', fontsize=10, zorder=6)
        
        # Expression box
        box_y = 0.8 if i % 2 == 0 else 0.2
        
        # Color by fitness (red=bad, green=good)
        color_intensity = max(0.2, min(1.0, 1 - fitness / max(fitnesses)))
        color = plt.cm.RdYlGn(color_intensity)
        
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2, 
                   edgecolor=color, linewidth=2)
        
        ax.text(gen, box_y, f'Gen {gen}\n{expr}\n\nFitness: {fitness}\nNodes: {complexity}',
                ha='center', va='center', fontsize=9, color='white',
                bbox=bbox, fontweight='bold' if gen == 50 else 'normal')
        
        # Arrow from timeline to box
        ax.annotate('', xy=(gen, box_y - 0.08 if box_y > 0.5 else box_y + 0.08), 
                    xytext=(gen, y_pos),
                    arrowprops=dict(arrowstyle='->', color='#8b949e', alpha=0.7))
    
    # Styling
    ax.set_xlim(-2, 52)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Generation', color='white', fontsize=14, fontweight='bold')
    ax.set_title('🧬 Evolution Timeline — Best Program Getting Simpler', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3, color='#30363d')
    
    # Legend
    legend_elements = [
        plt.scatter([], [], s=200, color='#7ee787', edgecolor='white', linewidth=2, label='Generation Marker'),
        plt.Line2D([0], [0], color='#58a6ff', linewidth=3, label='Evolution Timeline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#21262d', 
              edgecolor='#30363d', labelcolor='white')
    
    # Add complexity trend
    ax2 = ax.twinx()
    ax2.plot(generations, complexities, color='#f97583', linewidth=2, marker='o', 
             markersize=6, alpha=0.8, linestyle='--', label='Complexity (nodes)')
    ax2.set_ylabel('Tree Complexity', color='#f97583', fontsize=12)
    ax2.tick_params(colors='#f97583')
    ax2.spines['right'].set_color('#f97583')
    for spine in ['top', 'bottom', 'left']:
        ax2.spines[spine].set_color('#30363d')
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/generation_timeline.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_generation_timeline()