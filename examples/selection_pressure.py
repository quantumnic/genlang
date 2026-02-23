#!/usr/bin/env python3
"""Selection pressure effect on convergence."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_selection_pressure():
    """Chart showing weak vs medium vs strong selection pressure."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    # Simulation data
    generations = np.arange(0, 100)
    np.random.seed(42)
    
    # Weak selection (tournament size 2) - slow convergence
    weak_best = []
    weak_avg = []
    best_fitness = 100
    avg_fitness = 100
    
    for gen in generations:
        # Gradual improvement with noise
        improvement = 0.3 + np.random.normal(0, 0.2)
        best_fitness = max(0.01, best_fitness - improvement)
        avg_fitness = max(best_fitness + 5, avg_fitness - improvement * 0.6 + np.random.normal(0, 0.5))
        
        weak_best.append(best_fitness)
        weak_avg.append(avg_fitness)
    
    # Medium selection (tournament size 4) - moderate convergence
    medium_best = []
    medium_avg = []
    best_fitness = 100
    avg_fitness = 100
    
    np.random.seed(42)
    for gen in generations:
        improvement = 0.5 + np.random.normal(0, 0.15)
        best_fitness = max(0.01, best_fitness - improvement)
        avg_fitness = max(best_fitness + 3, avg_fitness - improvement * 0.8 + np.random.normal(0, 0.3))
        
        medium_best.append(best_fitness)
        medium_avg.append(avg_fitness)
    
    # Strong selection (tournament size 8) - fast convergence
    strong_best = []
    strong_avg = []
    best_fitness = 100
    avg_fitness = 100
    
    np.random.seed(42)
    for gen in generations:
        improvement = 0.8 + np.random.normal(0, 0.1)
        best_fitness = max(0.01, best_fitness - improvement)
        avg_fitness = max(best_fitness + 1, avg_fitness - improvement * 1.2 + np.random.normal(0, 0.2))
        
        strong_best.append(best_fitness)
        strong_avg.append(avg_fitness)
    
    # Plot lines
    ax.plot(generations, weak_best, color='#f97583', linewidth=2.5, label='Weak Selection (T=2)', alpha=0.9)
    ax.plot(generations, weak_avg, color='#f97583', linewidth=1.5, linestyle='--', alpha=0.6)
    
    ax.plot(generations, medium_best, color='#ffa657', linewidth=2.5, label='Medium Selection (T=4)', alpha=0.9)
    ax.plot(generations, medium_avg, color='#ffa657', linewidth=1.5, linestyle='--', alpha=0.6)
    
    ax.plot(generations, strong_best, color='#7ee787', linewidth=2.5, label='Strong Selection (T=8)', alpha=0.9)
    ax.plot(generations, strong_avg, color='#7ee787', linewidth=1.5, linestyle='--', alpha=0.6)
    
    # Fill areas between best and avg
    ax.fill_between(generations, weak_best, weak_avg, color='#f97583', alpha=0.1)
    ax.fill_between(generations, medium_best, medium_avg, color='#ffa657', alpha=0.1)
    ax.fill_between(generations, strong_best, strong_avg, color='#7ee787', alpha=0.1)
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness (MSE)', color='white', fontsize=14, fontweight='bold')
    ax.set_title('🏆 Selection Pressure Effect on Convergence Speed', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Legend
    legend = ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
                      labelcolor='white', fontsize=12)
    legend.get_frame().set_alpha(0.9)
    
    # Add annotations
    ax.annotate('Fast convergence\nbut may miss diversity', 
                xy=(70, strong_best[70]), xytext=(50, 5),
                color='#7ee787', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7ee787', alpha=0.8))
    
    ax.annotate('Slow but thorough\nexploration', 
                xy=(80, weak_best[80]), xytext=(60, 20),
                color='#f97583', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#f97583', alpha=0.8))
    
    # Add tournament size explanation
    textstr = 'Tournament Selection:\nT = tournament size\nHigher T = stronger pressure\nSolid = best fitness\nDashed = population average'
    props = dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='#8b949e')
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/selection_pressure.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_selection_pressure()