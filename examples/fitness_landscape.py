#!/usr/bin/env python3
"""Visualize 3D fitness landscape for genetic programming."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import os

def create_fitness_landscape():
    """Create a rugged 3D fitness landscape showing evolution path."""
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    
    # Create a rugged fitness landscape
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100) 
    X, Y = np.meshgrid(x, y)
    
    # Complex fitness function with multiple peaks and valleys
    Z = (np.sin(X) * np.cos(Y) * 2 + 
         np.exp(-(X**2 + Y**2)/8) * 5 +  # Global optimum
         np.exp(-((X-2)**2 + (Y+1)**2)/3) * 3 +  # Local optimum 1
         np.exp(-((X+1.5)**2 + (Y-2)**2)/2) * 2.5 +  # Local optimum 2
         np.sin(X*Y/3) * 0.5 +
         np.random.normal(0, 0.1, X.shape))  # Add noise
    
    # Create custom colormap (inferno-like)
    colors = ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', 
              '#cf4446', '#ed6925', '#fb9b06', '#f7d03c', '#fcffa4']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('inferno', colors, N=n_bins)
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, edgecolor='none',
                          linewidth=0, antialiased=True, shade=True)
    
    # Evolution path (simulated trajectory climbing to optimum)
    np.random.seed(42)
    n_points = 30
    path_x = np.zeros(n_points)
    path_y = np.zeros(n_points)
    path_z = np.zeros(n_points)
    
    # Start at random position
    path_x[0] = np.random.uniform(-4, 4)
    path_y[0] = np.random.uniform(-4, 4)
    
    # Simulate evolution climbing the landscape
    for i in range(1, n_points):
        # Move towards higher fitness with some randomness
        dx = np.random.normal(0, 0.3)
        dy = np.random.normal(0, 0.3)
        
        # Bias towards global optimum (0,0) as evolution progresses
        bias_factor = i / n_points * 0.8
        dx += -path_x[i-1] * bias_factor
        dy += -path_y[i-1] * bias_factor
        
        path_x[i] = np.clip(path_x[i-1] + dx, -5, 5)
        path_y[i] = np.clip(path_y[i-1] + dy, -5, 5)
    
    # Calculate Z values for path
    for i in range(n_points):
        xi, yi = path_x[i], path_y[i]
        path_z[i] = (np.sin(xi) * np.cos(yi) * 2 + 
                     np.exp(-(xi**2 + yi**2)/8) * 5 +
                     np.exp(-((xi-2)**2 + (yi+1)**2)/3) * 3 +
                     np.exp(-((xi+1.5)**2 + (yi-2)**2)/2) * 2.5 +
                     np.sin(xi*yi/3) * 0.5) + 0.2  # Slight offset above surface
    
    # Plot evolution path
    ax.plot(path_x, path_y, path_z, 'o-', color='#ff6b6b', linewidth=3, 
            markersize=6, alpha=0.9, markeredgecolor='white', markeredgewidth=1,
            label='Evolution Path')
    
    # Mark start and end points
    ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
               c='#ff6b6b', s=120, alpha=1, edgecolors='white', linewidth=2,
               marker='s', label='Start')
    ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
               c='#4ecdc4', s=120, alpha=1, edgecolors='white', linewidth=2,
               marker='*', label='Best Found')
    
    # Mark global optimum
    global_opt_z = np.exp(-0/8) * 5 + 2  # At origin
    ax.scatter([0], [0], [global_opt_z], c='#ffe66d', s=150, alpha=1, 
               edgecolors='white', linewidth=2, marker='D', label='Global Optimum')
    
    # Mark local optima
    local1_z = np.exp(-((2)**2 + (-1)**2)/3) * 3 + 2
    local2_z = np.exp(-((-1.5)**2 + (2)**2)/2) * 2.5 + 2
    ax.scatter([2, -1.5], [-1, 2], [local1_z, local2_z], 
               c='#ff8b94', s=100, alpha=0.8, edgecolors='white', linewidth=1,
               marker='^', label='Local Optima')
    
    # Styling
    ax.set_xlabel('Parameter 1', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel('Parameter 2', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_zlabel('Fitness', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_title('🗻 Fitness Landscape — Evolution Search Path', 
                 fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Set dark background colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.set_edgecolor('#30363d')
    ax.yaxis.pane.set_edgecolor('#30363d')  
    ax.zaxis.pane.set_edgecolor('#30363d')
    
    # Tick colors
    ax.tick_params(axis='x', colors='#8b949e')
    ax.tick_params(axis='y', colors='#8b949e')
    ax.tick_params(axis='z', colors='#8b949e')
    
    # Legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    legend.get_frame().set_facecolor('#21262d')
    legend.get_frame().set_edgecolor('#30363d')
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color('white')
    
    # View angle
    ax.view_init(elev=25, azim=-65)
    
    # Add colorbar
    cb = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cb.set_label('Fitness Value', color='#8b949e', fontsize=11)
    cb.ax.tick_params(colors='#8b949e')
    cb.outline.set_edgecolor('#30363d')
    
    # Add text annotation
    fig.text(0.02, 0.02, 
             '• Evolution navigates rugged fitness landscape\n'
             '• Local optima can trap search (premature convergence)\n'
             '• Crossover and mutation help escape local peaks\n'
             '• Goal: Find global optimum despite landscape complexity',
             fontsize=10, color='#8b949e', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.8))
    
    plt.tight_layout()
    
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/fitness_landscape.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight', 
                edgecolor='none')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_fitness_landscape()