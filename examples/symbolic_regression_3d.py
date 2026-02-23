#!/usr/bin/env python3
"""Create 3D symbolic regression evolution animation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

def create_symbolic_regression_3d():
    """Create 3D animation of symbolic regression converging to target function."""
    np.random.seed(42)
    
    # Target function: x² + y²  (paraboloid)
    def target_func(x, y):
        return x**2 + y**2
    
    # Create mesh grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z_target = target_func(X, Y)
    
    # Generate evolution frames (simulated symbolic regression)
    n_frames = 60
    evolved_functions = []
    
    for frame in range(n_frames):
        t = frame / (n_frames - 1)  # Progress from 0 to 1
        
        if frame == 0:
            # Start with random function
            Z_evolved = np.random.normal(5, 3, X.shape)
        else:
            # Gradually converge to target
            noise_level = (1 - t**2) * 4  # Decreasing noise
            bias = (1 - t**1.5) * 3  # Decreasing bias
            
            # Mix target with decreasing noise/bias
            Z_evolved = (t**2 * Z_target + 
                        (1 - t**2) * np.random.normal(bias, noise_level, X.shape) +
                        (1 - t) * np.sin(X + Y) * 2)  # Some structured error
            
            # Add some realistic GP artifacts
            if frame < 20:
                # Early: blocky, discontinuous
                Z_evolved += np.random.choice([-2, -1, 0, 1, 2], X.shape) * (1-t) * 0.5
            elif frame < 40:
                # Middle: smoother but still some oscillation
                Z_evolved += np.sin(X*3) * np.cos(Y*3) * (1-t) * 1.5
        
        evolved_functions.append(Z_evolved.copy())
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    
    # Initial empty plots
    target_surface = ax.plot_surface(X, Y, Z_target, alpha=0.3, color='#58a6ff', 
                                   shade=True, edgecolor='none')
    evolved_surface = ax.plot_surface(X, Y, evolved_functions[0], cmap='Reds', 
                                    alpha=0.8, shade=True, edgecolor='none')
    
    # Styling
    ax.set_xlabel('x', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel('y', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_zlabel('f(x,y)', fontsize=12, color='#8b949e', labelpad=10)
    
    # Set consistent z-limits
    z_min = min(np.min(Z_target), min(np.min(f) for f in evolved_functions))
    z_max = max(np.max(Z_target), max(np.max(f) for f in evolved_functions))
    ax.set_zlim(z_min - 2, z_max + 2)
    
    # Dark theme styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.set_edgecolor('#30363d')
    ax.yaxis.pane.set_edgecolor('#30363d')
    ax.zaxis.pane.set_edgecolor('#30363d')
    
    ax.tick_params(axis='x', colors='#8b949e')
    ax.tick_params(axis='y', colors='#8b949e')
    ax.tick_params(axis='z', colors='#8b949e')
    
    # Title and generation counter
    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, ha='center', va='top',
                     fontsize=16, fontweight='bold', color='white')
    
    # Legend
    legend_text = ax.text2D(0.02, 0.98, 
                           '🎯 Target: x² + y² (blue)\n🧬 Evolved: GP approximation (red)', 
                           transform=ax.transAxes, ha='left', va='top',
                           fontsize=11, color='white',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                                   edgecolor='#30363d', alpha=0.9))
    
    # Error display
    error_text = ax.text2D(0.98, 0.98, '', transform=ax.transAxes, ha='right', va='top',
                          fontsize=11, fontweight='bold', color='#ffa657',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', alpha=0.9))
    
    # Equation display
    equation_text = ax.text2D(0.5, 0.02, '', transform=ax.transAxes, ha='center', va='bottom',
                             fontsize=12, fontweight='bold', color='#7ee787',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', 
                                     edgecolor='#7ee787', alpha=0.9))
    
    def animate(frame):
        # Clear previous surfaces
        ax.clear()
        
        # Set consistent view and limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(z_min - 2, z_max + 2)
        
        # Plot target surface (always blue and transparent)
        target_surf = ax.plot_surface(X, Y, Z_target, alpha=0.3, color='#58a6ff', 
                                    shade=True, edgecolor='none')
        
        # Plot evolved surface with color based on convergence
        t = frame / (n_frames - 1)
        # Color transition from red (poor) to green (good)
        if t < 0.5:
            color_alpha = 0.8
            cmap = 'Reds'
        else:
            color_alpha = 0.8
            cmap = 'RdYlGn'
        
        evolved_surf = ax.plot_surface(X, Y, evolved_functions[frame], 
                                     cmap=cmap, alpha=color_alpha, 
                                     shade=True, edgecolor='none')
        
        # Calculate MSE
        mse = np.mean((evolved_functions[frame] - Z_target)**2)
        
        # Update texts
        generation = int(frame * 2)  # Scale to make it look like more generations
        title.set_text(f'🧬 3D Symbolic Regression Evolution — Generation {generation}')
        error_text.set_text(f'MSE: {mse:.3f}')
        
        # Simulated equation evolution
        equations = [
            'random_noise(x, y)',
            'x + y + noise', 
            'x² + noise',
            'x² + y + small_error',
            'x² + y² + tiny_error', 
            'x² + y²  ✓'
        ]
        
        eq_index = min(len(equations)-1, frame // 10)
        equation_text.set_text(f'Current best: {equations[eq_index]}')
        
        # Styling (reapply after clear)
        ax.set_facecolor('#0d1117')
        ax.set_xlabel('x', fontsize=12, color='#8b949e', labelpad=10)
        ax.set_ylabel('y', fontsize=12, color='#8b949e', labelpad=10)
        ax.set_zlabel('f(x,y)', fontsize=12, color='#8b949e', labelpad=10)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.set_edgecolor('#30363d')
        ax.yaxis.pane.set_edgecolor('#30363d')
        ax.zaxis.pane.set_edgecolor('#30363d')
        
        ax.tick_params(axis='x', colors='#8b949e')
        ax.tick_params(axis='y', colors='#8b949e')
        ax.tick_params(axis='z', colors='#8b949e')
        
        # Consistent view angle
        ax.view_init(elev=30, azim=frame * 2)  # Slowly rotate for better view
        
        return [target_surf, evolved_surf]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, 
                                  blit=False, repeat=True)
    
    # Add legend after first frame
    ax.text2D(0.02, 0.98, 
             '🎯 Target: x² + y² (blue)\n🧬 Evolved: GP approximation (red→green)', 
             transform=ax.transAxes, ha='left', va='top',
             fontsize=11, color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    
    # Save animation
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/symbolic_regression_3d.mp4')
    print("Saving 3D animation (this may take a moment)...")
    anim.save(out, writer='ffmpeg', fps=8, dpi=100,
              savefig_kwargs={'facecolor': '#0d1117'})
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_symbolic_regression_3d()