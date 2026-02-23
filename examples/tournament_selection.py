#!/usr/bin/env python3
"""Visualize tournament selection mechanism."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import os

def create_tournament_selection():
    """Show how tournament selection works step-by-step."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0d1117')
    fig.suptitle('🏆 Tournament Selection — Survival of the Fittest', 
                 fontsize=18, fontweight='bold', color='white', y=0.95)
    
    for row in axes:
        for ax in row:
            ax.set_facecolor('#0d1117')
            ax.set_xlim(-1, 9)
            ax.set_ylim(-1, 5)
            ax.axis('off')
    
    # Population data (8 individuals)
    population = [
        ('Prog A', 0.92, '#ff6b6b'),  # Low fitness (high error)
        ('Prog B', 0.15, '#4ecdc4'),  # High fitness 
        ('Prog C', 0.78, '#ff9f43'),  # Medium-low fitness
        ('Prog D', 0.05, '#7bed9f'),  # Highest fitness
        ('Prog E', 0.45, '#ffa502'),  # Medium fitness
        ('Prog F', 0.31, '#70a1ff'),  # Medium-high fitness
        ('Prog G', 0.88, '#ff6348'),  # Low fitness
        ('Prog H', 0.22, '#2ed573'),  # High fitness
    ]
    
    positions = [(i % 4 * 2, 3 - i // 4 * 2) for i in range(8)]
    
    # Step 1: Show full population
    ax = axes[0, 0]
    ax.set_title('Step 1: Population (fitness = error)', color='#58a6ff', 
                fontsize=14, fontweight='bold')
    
    for i, ((name, fitness, color), (x, y)) in enumerate(zip(population, positions)):
        # Individual box
        rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle='round,pad=0.05',
                             facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y+0.1, name, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
        ax.text(x, y-0.15, f'{fitness:.2f}', ha='center', va='center', fontsize=9, 
                color='white', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))
    
    # Fitness legend
    ax.text(8, 3, 'Fitness Scale:\n0.00 = Perfect\n1.00 = Worst', 
            ha='center', va='top', color='#8b949e', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))
    
    # Step 2: Select tournament participants  
    ax = axes[0, 1]
    ax.set_title('Step 2: Random Tournament Selection (k=3)', color='#ffa657', 
                fontsize=14, fontweight='bold')
    
    # Randomly select 3 for tournament (fixed for reproducibility)
    tournament_indices = [1, 4, 6]  # Prog B (0.15), Prog E (0.45), Prog G (0.88)
    
    for i, ((name, fitness, color), (x, y)) in enumerate(zip(population, positions)):
        if i in tournament_indices:
            # Highlight selected individuals
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle='round,pad=0.05',
                                 facecolor=color, edgecolor='#ffa657', linewidth=3, alpha=0.9)
            ax.add_patch(rect)
            ax.text(x, y+0.1, name, ha='center', va='center', fontsize=10, 
                    fontweight='bold', color='white')
            ax.text(x, y-0.15, f'{fitness:.2f}', ha='center', va='center', fontsize=9, 
                    color='white', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.8))
            # Selection arrow
            ax.annotate('', xy=(x, y-0.6), xytext=(x, y-0.4),
                       arrowprops=dict(arrowstyle='->', color='#ffa657', lw=2))
        else:
            # Grayed out non-selected
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle='round,pad=0.05',
                                 facecolor='#30363d', edgecolor='#8b949e', linewidth=1, alpha=0.4)
            ax.add_patch(rect)
            ax.text(x, y+0.1, name, ha='center', va='center', fontsize=10, 
                    fontweight='bold', color='#8b949e', alpha=0.6)
            ax.text(x, y-0.15, f'{fitness:.2f}', ha='center', va='center', fontsize=9, 
                    color='#8b949e', alpha=0.6)
    
    ax.text(8, 3, 'Selected for\ntournament', ha='center', va='center', 
            color='#ffa657', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#ffa657'))
    
    # Step 3: Tournament competition
    ax = axes[1, 0]
    ax.set_title('Step 3: Tournament — Best Fitness Wins', color='#7ee787', 
                fontsize=14, fontweight='bold')
    
    # Show tournament participants in arena
    tournament_data = [(population[i][0], population[i][1], population[i][2]) for i in tournament_indices]
    arena_positions = [(2, 3), (4, 3), (6, 3)]
    
    for i, ((name, fitness, color), (x, y)) in enumerate(zip(tournament_data, arena_positions)):
        if fitness == min([t[1] for t in tournament_data]):  # Winner (lowest error)
            # Winner - golden glow
            rect = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8, boxstyle='round,pad=0.05',
                                 facecolor=color, edgecolor='#ffd700', linewidth=4, alpha=0.9)
            ax.add_patch(rect)
            ax.text(x, y+0.2, name, ha='center', va='center', fontsize=11, 
                    fontweight='bold', color='white')
            ax.text(x, y, f'{fitness:.2f}', ha='center', va='center', fontsize=10, 
                    color='white', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.8))
            ax.text(x, y-0.25, '👑 WINNER', ha='center', va='center', fontsize=9, 
                    color='#ffd700', fontweight='bold')
            
            # Victory arrow
            ax.annotate('', xy=(x, 1.5), xytext=(x, y-0.6),
                       arrowprops=dict(arrowstyle='->', color='#7ee787', lw=3))
            ax.text(x, 1.2, 'Selected for\nreproduction', ha='center', va='center',
                   color='#7ee787', fontweight='bold', fontsize=10)
        else:
            # Loser
            rect = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8, boxstyle='round,pad=0.05',
                                 facecolor=color, edgecolor='white', linewidth=2, alpha=0.6)
            ax.add_patch(rect)
            ax.text(x, y+0.2, name, ha='center', va='center', fontsize=11, 
                    fontweight='bold', color='white')
            ax.text(x, y, f'{fitness:.2f}', ha='center', va='center', fontsize=10, 
                    color='white', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))
    
    # Tournament bracket
    ax.text(4, 4.2, '🏟️ TOURNAMENT ARENA', ha='center', va='center', 
            color='white', fontsize=12, fontweight='bold')
    ax.plot([1, 7], [2.2, 2.2], color='#30363d', linewidth=3)
    
    # Step 4: Selection statistics
    ax = axes[1, 1]
    ax.set_title('Step 4: Selection Pressure', color='#d2a8ff', 
                fontsize=14, fontweight='bold')
    
    # Create bar chart showing selection probability
    names = [p[0] for p in population]
    fitnesses = [p[1] for p in population]
    colors = [p[2] for p in population]
    
    # Calculate tournament selection probability (simplified)
    # Higher fitness = lower error = higher selection chance
    selection_probs = []
    for i, fitness in enumerate(fitnesses):
        # Probability of winning tournament of size 3
        better_count = sum(1 for f in fitnesses if f < fitness)  # Lower is better
        worse_count = len(fitnesses) - better_count - 1
        # Simplified probability calculation
        prob = (worse_count + 1) / len(fitnesses) 
        selection_probs.append(prob)
    
    # Normalize
    total_prob = sum(selection_probs)
    selection_probs = [p/total_prob for p in selection_probs]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, selection_probs, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color='white', fontsize=10)
    ax.set_xlabel('Selection Probability', color='#8b949e', fontsize=11)
    ax.tick_params(axis='x', colors='#8b949e')
    ax.set_xlim(0, max(selection_probs) * 1.1)
    
    # Highlight the winner
    winner_idx = tournament_indices[0]  # Prog B
    bars[winner_idx].set_edgecolor('#ffd700')
    bars[winner_idx].set_linewidth(3)
    
    # Add values
    for i, (bar, prob) in enumerate(zip(bars, selection_probs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{prob:.2f}', ha='left', va='center', color='white', fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, color='#30363d', axis='x')
    ax.set_facecolor('#161b22')
    
    # Add explanation
    ax.text(0.5, -1.5, 
           '🎯 Tournament Selection Benefits:\n'
           '• Better individuals more likely to be selected\n'  
           '• Maintains diversity (weaker can still win)\n'
           '• Adjustable selection pressure (tournament size)',
           transform=ax.transAxes, ha='center', va='top',
           color='#8b949e', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#30363d'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/tournament_selection.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_tournament_selection()