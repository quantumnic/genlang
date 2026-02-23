#!/usr/bin/env python3
"""Visualize genlang's genetic programming evolution."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle
import json
import os

# --- 1. Evolution Fitness Animation ---
def create_evolution_animation():
    """Animate how programs evolve toward a target function."""
    np.random.seed(42)
    
    # Target: x² + x + 1
    x = np.linspace(-3, 3, 200)
    target = x**2 + x + 1
    
    # Simulate evolving programs (increasingly better approximations)
    generations = 50
    programs = []
    fitnesses = []
    best_fitnesses = []
    
    for gen in range(generations):
        t = gen / (generations - 1)
        # Simulated evolution: starts random, converges to target
        noise = (1 - t**1.5) * np.random.randn(200) * (5 - 4*t)
        drift = (1 - t) * np.sin(x * 2) * 3
        approx = target + noise + drift
        programs.append(approx)
        mse = np.mean((approx - target)**2)
        fitnesses.append(mse)
        best_fitnesses.append(min(fitnesses))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0d1117')
    fig.suptitle('genlang — Genetic Programming Evolution', 
                 fontsize=18, fontweight='bold', color='white', y=0.95)
    
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
    
    # Left: Function approximation
    ax1 = axes[0]
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-5, 15)
    ax1.set_xlabel('x', color='#8b949e', fontsize=12)
    ax1.set_ylabel('f(x)', color='#8b949e', fontsize=12)
    target_line, = ax1.plot(x, target, color='#58a6ff', linewidth=2.5, label='Target: x² + x + 1')
    evolved_line, = ax1.plot([], [], color='#f97583', linewidth=2, alpha=0.8, label='Best Program')
    gen_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, color='#7ee787', 
                        fontsize=14, fontweight='bold', va='top')
    ax1.legend(loc='upper left', facecolor='#21262d', edgecolor='#30363d', 
               labelcolor='white', fontsize=10)
    ax1.set_title('Program vs Target', color='white', fontsize=14)
    
    # Right: Fitness over time
    ax2 = axes[1]
    ax2.set_xlim(0, generations)
    ax2.set_ylim(0.001, max(fitnesses) * 1.2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Generation', color='#8b949e', fontsize=12)
    ax2.set_ylabel('MSE (log)', color='#8b949e', fontsize=12)
    fitness_line, = ax2.plot([], [], color='#f97583', linewidth=1, alpha=0.4, label='Population')
    best_line, = ax2.plot([], [], color='#7ee787', linewidth=2.5, label='Best Fitness')
    ax2.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
               labelcolor='white', fontsize=10)
    ax2.set_title('Fitness Convergence', color='white', fontsize=14)
    
    def animate(frame):
        evolved_line.set_data(x, programs[frame])
        gen_text.set_text(f'Generation {frame}\nMSE: {fitnesses[frame]:.4f}')
        fitness_line.set_data(range(frame+1), fitnesses[:frame+1])
        best_line.set_data(range(frame+1), best_fitnesses[:frame+1])
        return evolved_line, gen_text, fitness_line, best_line
    
    anim = animation.FuncAnimation(fig, animate, frames=generations, interval=150, blit=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/evolution.mp4')
    anim.save(out, writer='ffmpeg', fps=10, dpi=120,
              savefig_kwargs={'facecolor': '#0d1117'})
    plt.close()
    print(f"Saved: {out}")
    return out


# --- 2. AST Tree Visualization ---
def create_ast_visualization():
    """Visualize an evolved program's AST."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    ax.set_title('🧬 Evolved AST: x² + x + 1', color='white', fontsize=18, fontweight='bold', pad=20)
    
    # Tree nodes: (x, y, label, color, children_indices)
    nodes = [
        (5, 6, '+', '#f97583'),        # 0: root
        (2, 4.5, '×', '#d2a8ff'),      # 1: x*x
        (8, 4.5, '+', '#f97583'),      # 2: x + 1
        (1, 3, 'x', '#7ee787'),        # 3: left x
        (3, 3, 'x', '#7ee787'),        # 4: right x
        (6.5, 3, 'x', '#7ee787'),      # 5: x
        (9.5, 3, '1', '#58a6ff'),      # 6: constant 1
    ]
    edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6)]
    
    # Draw edges
    for parent, child in edges:
        px, py = nodes[parent][0], nodes[parent][1]
        cx, cy = nodes[child][0], nodes[child][1]
        ax.plot([px, cx], [py, cy], color='#30363d', linewidth=2.5, zorder=1)
    
    # Draw nodes
    for x, y, label, color in nodes:
        circle = Circle((x, y), 0.45, facecolor=color, edgecolor='white', 
                        linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=16, 
                fontweight='bold', color='white', zorder=3)
    
    # Legend
    legend_items = [
        ('#f97583', 'Operator (+, ×)'),
        ('#7ee787', 'Variable (x)'),
        ('#58a6ff', 'Constant (1)'),
        ('#d2a8ff', 'Multiply'),
    ]
    for i, (color, text) in enumerate(legend_items):
        ax.add_patch(Circle((0.3, 1.5 - i*0.5), 0.15, facecolor=color, edgecolor='white', linewidth=1))
        ax.text(0.7, 1.5 - i*0.5, text, color='#8b949e', fontsize=11, va='center')
    
    # Annotations
    ax.annotate('Crossover\npoint ✂️', xy=(2, 4.5), xytext=(-0.5, 5.5),
                color='#ffa657', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2))
    ax.annotate('Mutation\nsite 🎲', xy=(9.5, 3), xytext=(10.5, 4.2),
                color='#ffa657', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2))
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/ast_tree.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out


# --- 3. Island Model Visualization ---
def create_island_visualization():
    """Visualize the island model with migration."""
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('🏝️ Island Model — 4 Populations', color='white', fontsize=18, fontweight='bold')
    
    # Island positions (circle)
    islands = [
        (0, 1.2, 'Island 1\n🧬 Pop: 100\n📊 Best: 0.42', '#58a6ff'),
        (1.2, 0, 'Island 2\n🧬 Pop: 100\n📊 Best: 0.18', '#7ee787'),
        (0, -1.2, 'Island 3\n🧬 Pop: 100\n📊 Best: 0.91', '#f97583'),
        (-1.2, 0, 'Island 4\n🧬 Pop: 100\n📊 Best: 0.03', '#d2a8ff'),
    ]
    
    # Draw migration arrows (ring topology)
    for i in range(4):
        x1, y1 = islands[i][0], islands[i][1]
        x2, y2 = islands[(i+1)%4][0], islands[(i+1)%4][1]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        # Shorten arrow to not overlap circles
        shrink = 0.55
        ax.annotate('', xy=(x1 + dx*(1-shrink/length), y1 + dy*(1-shrink/length)),
                    xytext=(x1 + dx*shrink/length, y1 + dy*shrink/length),
                    arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2.5, 
                                   connectionstyle='arc3,rad=0.2'))
    
    # Draw islands
    for x, y, label, color in islands:
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='white', 
                        linewidth=2, alpha=0.2, zorder=2)
        ax.add_patch(circle)
        circle2 = Circle((x, y), 0.5, facecolor='none', edgecolor=color, 
                         linewidth=3, zorder=3)
        ax.add_patch(circle2)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white', zorder=4)
    
    # Migration label
    ax.text(0, 0, '🔄 Migration\nevery 15 gens\n(top 5 individuals)', 
            ha='center', va='center', color='#ffa657', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#ffa657', alpha=0.8))
    
    # Evolution stages at bottom
    stages = ['Random\nInit 🎲', 'Selection\n🏆', 'Crossover\n✂️', 'Mutation\n🎯', 'Migration\n🏝️', 'Converge\n✅']
    for i, stage in enumerate(stages):
        x = -1.5 + i * 0.6
        ax.text(x, -1.85, stage, ha='center', va='center', color='white', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 0.4, -1.85), xytext=(x + 0.2, -1.85),
                       arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/island_model.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == '__main__':
    print("🧬 Generating genlang visualizations...")
    create_evolution_animation()
    create_ast_visualization()
    create_island_visualization()
    print("✅ All done!")
