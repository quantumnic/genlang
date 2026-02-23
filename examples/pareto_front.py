#!/usr/bin/env python3
"""Pareto front for multi-objective GP."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

def create_pareto_front():
    """Multi-objective: fitness vs complexity Pareto front."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    np.random.seed(42)
    
    # Generate population with fitness/complexity trade-off
    n_individuals = 300
    
    # Create realistic GP population data
    individuals = []
    
    for _ in range(n_individuals):
        # Base complexity (tree size)
        complexity = np.random.exponential(15) + 3
        
        # Fitness has inverse relationship with complexity but with noise
        # More complex = potentially better fit but often overfit
        base_fitness = 20 / complexity + np.random.exponential(2)
        noise = np.random.normal(0, base_fitness * 0.3)
        fitness = max(0.1, base_fitness + noise)
        
        individuals.append([complexity, fitness])
    
    individuals = np.array(individuals)
    complexities = individuals[:, 0]
    fitnesses = individuals[:, 1]
    
    # Find Pareto front
    def is_dominated(point, other_points):
        """Check if point is dominated by any other point."""
        for other in other_points:
            # For minimization: other dominates point if other is better in all objectives
            # Complexity: minimize, Fitness: minimize  
            if (other[0] <= point[0] and other[1] <= point[1] and 
                (other[0] < point[0] or other[1] < point[1])):
                return True
        return False
    
    pareto_indices = []
    for i, individual in enumerate(individuals):
        other_individuals = np.delete(individuals, i, axis=0)
        if not is_dominated(individual, other_individuals):
            pareto_indices.append(i)
    
    pareto_points = individuals[pareto_indices]
    dominated_points = np.delete(individuals, pareto_indices, axis=0)
    
    # Sort Pareto points by complexity for drawing
    pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
    
    # Plot dominated solutions
    ax.scatter(dominated_points[:, 0], dominated_points[:, 1], 
              c='#8b949e', alpha=0.4, s=30, label='Dominated Solutions', 
              edgecolors='none')
    
    # Plot Pareto front points
    ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1], 
              c='#7ee787', alpha=0.9, s=80, label='Pareto Front', 
              edgecolors='white', linewidth=1.5, zorder=5)
    
    # Draw Pareto front line
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
           color='#7ee787', linewidth=3, alpha=0.8, zorder=4)
    
    # Shade dominated region
    max_complexity = max(complexities) * 1.1
    max_fitness = max(fitnesses) * 1.1
    
    # Create polygon for dominated region
    polygon_points = []
    for point in pareto_sorted:
        polygon_points.append([point[0], max_fitness])
    polygon_points.append([max_complexity, max_fitness])
    polygon_points.append([max_complexity, max(pareto_sorted[:, 1])])
    
    for point in reversed(pareto_sorted):
        polygon_points.append([max_complexity, point[1]])
    
    dominated_region = Polygon(polygon_points, facecolor='#f97583', 
                              alpha=0.1, edgecolor='none', zorder=1)
    ax.add_patch(dominated_region)
    
    # Highlight some interesting solutions
    # Best fitness (may be complex)
    best_fitness_idx = np.argmin(pareto_sorted[:, 1])
    best_fitness_point = pareto_sorted[best_fitness_idx]
    
    # Simplest solution 
    simplest_idx = np.argmin(pareto_sorted[:, 0])
    simplest_point = pareto_sorted[simplest_idx]
    
    # Balanced solution (knee point)
    # Find point with best fitness/complexity ratio
    ratios = pareto_sorted[:, 1] / pareto_sorted[:, 0]
    balanced_idx = np.argmin(ratios)
    balanced_point = pareto_sorted[balanced_idx]
    
    # Highlight special points
    ax.scatter(*best_fitness_point, c='#58a6ff', s=120, marker='*', 
              edgecolors='white', linewidth=2, zorder=6, label='Best Fitness')
    ax.scatter(*simplest_point, c='#d2a8ff', s=120, marker='s', 
              edgecolors='white', linewidth=2, zorder=6, label='Simplest')  
    ax.scatter(*balanced_point, c='#ffa657', s=120, marker='D', 
              edgecolors='white', linewidth=2, zorder=6, label='Balanced')
    
    # Annotations
    ax.annotate(f'Best Fit\n{best_fitness_point[1]:.2f} MSE\n{best_fitness_point[0]:.0f} nodes', 
                xy=best_fitness_point, xytext=(best_fitness_point[0]+10, best_fitness_point[1]+5),
                color='#58a6ff', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#58a6ff'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor='#58a6ff', alpha=0.9))
    
    ax.annotate(f'Simplest\n{simplest_point[1]:.2f} MSE\n{simplest_point[0]:.0f} nodes', 
                xy=simplest_point, xytext=(simplest_point[0]+15, simplest_point[1]-8),
                color='#d2a8ff', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#d2a8ff'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor='#d2a8ff', alpha=0.9))
    
    ax.annotate(f'Balanced\n{balanced_point[1]:.2f} MSE\n{balanced_point[0]:.0f} nodes', 
                xy=balanced_point, xytext=(balanced_point[0]-20, balanced_point[1]+8),
                color='#ffa657', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffa657'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor='#ffa657', alpha=0.9))
    
    # Labels and title
    ax.set_xlabel('Program Complexity (nodes)', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness (MSE - lower is better)', color='white', fontsize=14, fontweight='bold')
    ax.set_title('📊 Pareto Front — Fitness vs Complexity Trade-off', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    
    # Invert y-axis since lower fitness is better
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Legend
    legend = ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
                      labelcolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Add explanation text
    explanation = ('Pareto Front represents optimal trade-offs:\n'
                  '• No solution dominates these points\n'
                  '• Moving along front trades fitness for simplicity\n'
                  '• Dominated region (shaded) contains suboptimal solutions')
    
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', color='#8b949e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                     edgecolor='#30363d', alpha=0.9))
    
    # Statistics
    n_pareto = len(pareto_indices)
    pareto_percentage = (n_pareto / n_individuals) * 100
    
    stats = f'Population Stats:\n• Total: {n_individuals}\n• Pareto: {n_pareto} ({pareto_percentage:.1f}%)\n• Dominated: {n_individuals - n_pareto}'
    
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', 
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                     edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/pareto_front.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_pareto_front()