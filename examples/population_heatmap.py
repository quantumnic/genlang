#!/usr/bin/env python3
"""Population fitness heatmap showing convergence."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_population_heatmap():
    """Heatmap of 100 individuals × 50 generations showing convergence."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Generate realistic GP evolution data
    np.random.seed(42)
    n_individuals = 100
    n_generations = 50
    
    fitness_matrix = np.zeros((n_generations, n_individuals))
    
    # Initialize random population
    fitness_matrix[0] = np.random.exponential(scale=10, size=n_individuals) + 1
    
    # Evolve population - showing convergence
    for gen in range(1, n_generations):
        prev_fitness = fitness_matrix[gen-1]
        
        # Selection bias - better individuals more likely to survive
        selection_prob = 1.0 / (prev_fitness + 0.1)  # Lower fitness = higher prob
        selection_prob /= selection_prob.sum()
        
        # Next generation through selection + variation
        next_gen = []
        for i in range(n_individuals):
            # Select parent based on fitness
            parent_idx = np.random.choice(n_individuals, p=selection_prob)
            parent_fitness = prev_fitness[parent_idx]
            
            # Add small mutation/crossover variation
            variation = np.random.normal(0, max(0.1, parent_fitness * 0.05))
            child_fitness = max(0.01, parent_fitness + variation)
            next_gen.append(child_fitness)
        
        fitness_matrix[gen] = next_gen
        
        # Add occasional "breakthroughs" - sudden improvements
        if gen % 10 == 0 and gen > 0:
            n_breakthroughs = np.random.randint(1, 4)
            for _ in range(n_breakthroughs):
                idx = np.random.randint(n_individuals)
                fitness_matrix[gen][idx] *= 0.3  # Major improvement
    
    # Sort individuals by final fitness for better visualization
    final_order = np.argsort(fitness_matrix[-1])
    fitness_matrix_sorted = fitness_matrix[:, final_order]
    
    # Create heatmap with custom colormap
    im = ax.imshow(fitness_matrix_sorted.T, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='bilinear', alpha=0.9)
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Fitness (MSE)', color='white', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#30363d')
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Individual (sorted by final fitness)', color='white', fontsize=14, fontweight='bold')
    ax.set_title('🔥 Population Fitness Heatmap — Convergence Over Time', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    # Add convergence indicators
    # Best fitness line
    best_fitness = np.min(fitness_matrix_sorted, axis=1)
    best_individual = np.argmin(fitness_matrix_sorted, axis=1)
    
    # Plot best fitness trajectory
    ax2 = ax.twinx()
    ax2.plot(range(n_generations), best_individual, color='#58a6ff', 
             linewidth=3, alpha=0.8, label='Best Individual')
    ax2.set_ylabel('Best Individual ID', color='#58a6ff', fontsize=12)
    ax2.tick_params(colors='#58a6ff')
    ax2.spines['right'].set_color('#58a6ff')
    ax2.set_ylim(0, n_individuals-1)
    
    # Add annotations
    ax.annotate('Diverse\nPopulation', xy=(5, 85), xytext=(15, 85),
                color='white', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f97583', alpha=0.7))
    
    ax.annotate('Convergence\nBegins', xy=(25, 50), xytext=(35, 75),
                color='white', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffa657', alpha=0.7))
    
    ax.annotate('Elite\nSolutions', xy=(45, 15), xytext=(35, 25),
                color='white', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#7ee787', alpha=0.7))
    
    # Statistics text box
    final_best = np.min(fitness_matrix_sorted[-1])
    final_avg = np.mean(fitness_matrix_sorted[-1])
    final_worst = np.max(fitness_matrix_sorted[-1])
    
    stats_text = f'Final Generation Stats:\n• Best: {final_best:.3f}\n• Average: {final_avg:.3f}\n• Worst: {final_worst:.3f}\n• Improvement: {fitness_matrix_sorted[0].mean()/final_avg:.1f}×'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/population_heatmap.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_population_heatmap()