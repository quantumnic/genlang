#!/usr/bin/env python3
"""Visualize population diversity over generations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def create_population_diversity():
    """Show how population diversity changes during evolution."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#0d1117')
    fig.suptitle('🌟 Population Diversity During Evolution', 
                 fontsize=18, fontweight='bold', color='white', y=0.95)
    
    for row in axes:
        for ax in row:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
    
    # Generate simulated evolution data
    n_generations = 100
    population_size = 50
    n_features = 2  # 2D feature space for visualization
    
    # Initialize diverse population
    generations = []
    diversity_scores = []
    mean_fitness = []
    
    for gen in range(n_generations):
        # Population becomes less diverse and fitter over time
        diversity_factor = np.exp(-gen / 40)  # Exponential decay
        fitness_factor = 1 - np.exp(-gen / 30)  # Fitness improvement
        
        if gen == 0:
            # Initial random population (high diversity)
            pop = np.random.multivariate_normal([0, 0], [[4, 0.5], [0.5, 4]], population_size)
            fitness = np.random.exponential(0.2, population_size)
        else:
            # Population converges towards best solutions
            center = np.array([2, 1])  # Target convergence point
            spread = diversity_factor * 3 + 0.2
            pop = np.random.multivariate_normal(center, [[spread, 0.1], [0.1, spread]], population_size)
            
            # Add some outliers (exploration)
            n_outliers = max(1, int(population_size * diversity_factor * 0.3))
            outliers = np.random.multivariate_normal([0, 0], [[6, 0], [0, 6]], n_outliers)
            pop[:n_outliers] = outliers
            
            # Fitness improves and becomes more consistent
            base_fitness = fitness_factor * 0.8
            fitness_var = (1 - fitness_factor) * 0.5 + 0.1
            fitness = np.random.normal(base_fitness, fitness_var, population_size)
            fitness = np.clip(fitness, 0, 1)
        
        generations.append(pop.copy())
        
        # Calculate diversity (average pairwise distance)
        distances = []
        for i in range(population_size):
            for j in range(i+1, population_size):
                dist = np.linalg.norm(pop[i] - pop[j])
                distances.append(dist)
        diversity_scores.append(np.mean(distances))
        mean_fitness.append(np.mean(fitness))
    
    # Plot 1: Early generation (high diversity)
    ax = axes[0, 0]
    early_gen = 5
    pop = generations[early_gen]
    fitness_colors = plt.cm.viridis(np.linspace(0, 1, population_size))
    
    scatter = ax.scatter(pop[:, 0], pop[:, 1], c=range(population_size), 
                        cmap='viridis', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f'Generation {early_gen} — High Diversity', color='#7ee787', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', color='#8b949e')
    ax.set_ylabel('Feature 2', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add diversity circle
    center = np.mean(pop, axis=0)
    radius = np.std(pop) * 1.5
    circle = plt.Circle(center, radius, fill=False, color='#ffa657', linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.text(center[0], center[1]-radius-1, f'Diversity: {diversity_scores[early_gen]:.2f}', 
            ha='center', color='#ffa657', fontweight='bold')
    
    # Plot 2: Late generation (low diversity, converged)
    ax = axes[0, 1]
    late_gen = 80
    pop = generations[late_gen]
    
    scatter = ax.scatter(pop[:, 0], pop[:, 1], c=range(population_size), 
                        cmap='viridis', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.set_title(f'Generation {late_gen} — Low Diversity', color='#58a6ff', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', color='#8b949e')
    ax.set_ylabel('Feature 2', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add diversity circle  
    center = np.mean(pop, axis=0)
    radius = np.std(pop) * 1.5
    circle = plt.Circle(center, radius, fill=False, color='#ffa657', linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.text(center[0], center[1]-radius-1, f'Diversity: {diversity_scores[late_gen]:.2f}', 
            ha='center', color='#ffa657', fontweight='bold')
    
    # Plot 3: Diversity over time
    ax = axes[1, 0]
    ax.plot(range(n_generations), diversity_scores, color='#f97583', linewidth=3, alpha=0.9)
    ax.fill_between(range(n_generations), diversity_scores, alpha=0.3, color='#f97583')
    ax.set_title('Population Diversity Over Time', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', color='#8b949e')
    ax.set_ylabel('Average Pairwise Distance', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add annotations
    ax.annotate('High initial\ndiversity', xy=(5, diversity_scores[5]), xytext=(20, diversity_scores[5]+0.5),
                color='#7ee787', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#7ee787'))
    ax.annotate('Convergence\n(low diversity)', xy=(80, diversity_scores[80]), xytext=(60, diversity_scores[80]+1),
                color='#58a6ff', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#58a6ff'))
    
    # Plot 4: Fitness vs Diversity trade-off
    ax = axes[1, 1]
    
    # Dual y-axis plot
    ax2 = ax.twinx()
    
    # Plot diversity (left axis)
    line1 = ax.plot(range(n_generations), diversity_scores, color='#f97583', 
                    linewidth=3, alpha=0.9, label='Diversity')
    ax.set_ylabel('Diversity', color='#f97583', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#f97583')
    
    # Plot fitness (right axis) 
    line2 = ax2.plot(range(n_generations), mean_fitness, color='#7ee787', 
                     linewidth=3, alpha=0.9, label='Mean Fitness')
    ax2.set_ylabel('Mean Fitness', color='#7ee787', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#7ee787')
    
    ax.set_title('Diversity vs Fitness Trade-off', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', color='#8b949e')
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', facecolor='#21262d', 
             edgecolor='#30363d', labelcolor='white')
    
    # Add exploration vs exploitation phases
    ax.axvspan(0, 30, alpha=0.2, color='#ffa657', label='Exploration')
    ax.axvspan(30, 100, alpha=0.2, color='#58a6ff', label='Exploitation')
    ax.text(15, max(diversity_scores)*0.8, 'Exploration\nPhase', ha='center', 
            color='#ffa657', fontweight='bold', fontsize=11)
    ax.text(65, max(diversity_scores)*0.4, 'Exploitation\nPhase', ha='center', 
            color='#58a6ff', fontweight='bold', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/population_diversity.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_population_diversity()