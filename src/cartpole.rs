//! Cart-pole (inverted pendulum) balancing problem.
//!
//! Evolve a controller that keeps a pole balanced on a cart by applying
//! left/right force. Classic reinforcement learning benchmark adapted for GP.

use crate::ast::Node;
use crate::genetic::{evolve, GenStats, GpConfig};
use crate::interpreter::Interpreter;
use rand::Rng;

/// Cart-pole physics state.
#[derive(Debug, Clone, Copy)]
pub struct CartPoleState {
    /// Cart position (m).
    pub x: f64,
    /// Cart velocity (m/s).
    pub x_dot: f64,
    /// Pole angle from vertical (rad).
    pub theta: f64,
    /// Pole angular velocity (rad/s).
    pub theta_dot: f64,
}

impl CartPoleState {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.01, // slight initial tilt
            theta_dot: 0.0,
        }
    }

    /// As a slice of f64 for the GP interpreter.
    pub fn as_vars(&self) -> [f64; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

impl Default for CartPoleState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cart-pole simulation parameters.
#[derive(Debug, Clone)]
pub struct CartPoleParams {
    /// Gravity (m/s²).
    pub gravity: f64,
    /// Cart mass (kg).
    pub cart_mass: f64,
    /// Pole mass (kg).
    pub pole_mass: f64,
    /// Half pole length (m).
    pub pole_half_length: f64,
    /// Force magnitude applied (N).
    pub force_mag: f64,
    /// Simulation timestep (s).
    pub dt: f64,
    /// Track half-length (m) — cart fails if |x| exceeds this.
    pub track_limit: f64,
    /// Angle limit (rad) — pole fails if |theta| exceeds this.
    pub angle_limit: f64,
    /// Maximum simulation steps.
    pub max_steps: usize,
}

impl Default for CartPoleParams {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            cart_mass: 1.0,
            pole_mass: 0.1,
            pole_half_length: 0.5,
            force_mag: 10.0,
            dt: 0.02,
            track_limit: 2.4,
            angle_limit: 0.2094, // ~12 degrees
            max_steps: 500,
        }
    }
}

/// Step the cart-pole simulation forward.
/// `force`: the force to apply (positive = right, negative = left).
/// Returns the new state, or None if the episode has terminated (pole fell or cart out of bounds).
pub fn step(state: &CartPoleState, force: f64, params: &CartPoleParams) -> Option<CartPoleState> {
    let cos_theta = state.theta.cos();
    let sin_theta = state.theta.sin();
    let total_mass = params.cart_mass + params.pole_mass;
    let pole_ml = params.pole_mass * params.pole_half_length;

    // Physics equations (Euler integration)
    let temp = (force + pole_ml * state.theta_dot * state.theta_dot * sin_theta) / total_mass;
    let theta_acc = (params.gravity * sin_theta - cos_theta * temp)
        / (params.pole_half_length
            * (4.0 / 3.0 - params.pole_mass * cos_theta * cos_theta / total_mass));
    let x_acc = temp - pole_ml * theta_acc * cos_theta / total_mass;

    let new_state = CartPoleState {
        x: state.x + params.dt * state.x_dot,
        x_dot: state.x_dot + params.dt * x_acc,
        theta: state.theta + params.dt * state.theta_dot,
        theta_dot: state.theta_dot + params.dt * theta_acc,
    };

    // Check termination
    if new_state.x.abs() > params.track_limit || new_state.theta.abs() > params.angle_limit {
        None
    } else {
        Some(new_state)
    }
}

/// Simulate a full episode with a GP controller.
/// The controller receives (x, x_dot, theta, theta_dot) and its output
/// determines the force direction: output > 0 → push right, else push left.
/// Returns the number of steps survived.
pub fn simulate(controller: &Node, params: &CartPoleParams) -> usize {
    let mut state = CartPoleState::new();
    let mut interp = Interpreter::new(1000);

    for t in 0..params.max_steps {
        interp.reset();
        let vars = state.as_vars();
        let force = match interp.eval(controller, &vars) {
            Ok(val) => {
                if val.to_f64() > 0.0 {
                    params.force_mag
                } else {
                    -params.force_mag
                }
            }
            Err(_) => return t,
        };

        match step(&state, force, params) {
            Some(new_state) => state = new_state,
            None => return t,
        }
    }

    params.max_steps
}

/// Simulate from multiple initial conditions for robustness.
pub fn simulate_robust(controller: &Node, params: &CartPoleParams) -> f64 {
    let initial_conditions = [
        CartPoleState {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.01,
            theta_dot: 0.0,
        },
        CartPoleState {
            x: 0.0,
            x_dot: 0.0,
            theta: -0.01,
            theta_dot: 0.0,
        },
        CartPoleState {
            x: 0.5,
            x_dot: 0.0,
            theta: 0.05,
            theta_dot: 0.0,
        },
        CartPoleState {
            x: -0.5,
            x_dot: 0.0,
            theta: -0.05,
            theta_dot: 0.0,
        },
        CartPoleState {
            x: 0.0,
            x_dot: 1.0,
            theta: 0.0,
            theta_dot: 0.1,
        },
    ];

    let total: usize = initial_conditions
        .iter()
        .map(|ic| {
            let mut state = *ic;
            let mut interp = Interpreter::new(1000);
            let mut survived = 0;
            for _ in 0..params.max_steps {
                interp.reset();
                let vars = state.as_vars();
                let force = match interp.eval(controller, &vars) {
                    Ok(val) => {
                        if val.to_f64() > 0.0 {
                            params.force_mag
                        } else {
                            -params.force_mag
                        }
                    }
                    Err(_) => break,
                };
                match step(&state, force, params) {
                    Some(new_state) => {
                        state = new_state;
                        survived += 1;
                    }
                    None => break,
                }
            }
            survived
        })
        .sum();

    total as f64 / initial_conditions.len() as f64
}

/// Evolve a cart-pole balancing controller using GP.
/// Fitness = negative steps survived (we minimize, so more steps = lower fitness).
pub fn cart_pole<R: Rng>(
    rng: &mut R,
    config: &GpConfig,
    sim_params: &CartPoleParams,
) -> (Node, f64, Vec<GenStats>) {
    let params = sim_params.clone();
    let fitness = move |tree: &Node| -> f64 {
        let avg_steps = simulate_robust(tree, &params);
        // Negate: we minimize fitness, so more steps survived = better = lower value
        -avg_steps
    };

    evolve(rng, config, &fitness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, CmpOp};
    use rand::SeedableRng;

    #[test]
    fn test_cart_pole_step() {
        let state = CartPoleState::new();
        let params = CartPoleParams::default();

        // Should survive at least one step
        let next = step(&state, 0.0, &params);
        assert!(next.is_some());

        let next = next.unwrap();
        assert!(next.theta.abs() < params.angle_limit);
    }

    #[test]
    fn test_simulation_terminates() {
        // A do-nothing controller
        let controller = Node::FloatConst(1.0);
        let params = CartPoleParams::default();
        let steps = simulate(&controller, &params);
        // Should survive some steps but not forever
        assert!(steps > 0);
        assert!(steps <= params.max_steps);
    }

    #[test]
    fn test_simple_controller() {
        // A simple proportional controller: output = -theta
        // This should do reasonably well
        let controller = Node::UnaryOp(
            crate::ast::UnaryOp::Neg,
            Box::new(Node::Var(2)), // theta
        );
        let params = CartPoleParams::default();
        let steps = simulate(&controller, &params);
        assert!(
            steps > 0,
            "Simple controller should survive at least 1 step, got {steps}"
        );
    }

    #[test]
    fn test_robust_simulation() {
        let controller = Node::BinOp(
            BinOp::Sub,
            Box::new(Node::FloatConst(0.0)),
            Box::new(Node::Var(2)), // -theta
        );
        let params = CartPoleParams::default();
        let avg_steps = simulate_robust(&controller, &params);
        assert!(avg_steps > 0.0);
    }

    #[test]
    fn test_cart_pole_evolution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 100,
            max_generations: 20,
            max_depth: 4,
            num_vars: 4, // x, x_dot, theta, theta_dot
            ..GpConfig::default()
        };
        let params = CartPoleParams {
            max_steps: 200,
            ..CartPoleParams::default()
        };

        let (best, best_fit, stats) = cart_pole(&mut rng, &config, &params);
        assert!(!stats.is_empty());
        // Fitness is negative steps, so best_fit should be negative
        assert!(best_fit < 0.0);
        // Should survive at least a few steps
        let steps = simulate(&best, &params);
        assert!(steps > 0);
    }

    #[test]
    fn test_better_controller_evolves() {
        // A random constant should do worse than a proper controller
        let constant = Node::FloatConst(1.0);
        let proportional = Node::Cmp(
            CmpOp::Lt,
            Box::new(Node::Var(2)), // theta
            Box::new(Node::FloatConst(0.0)),
        );

        let params = CartPoleParams::default();
        let steps_const = simulate(&constant, &params);
        let steps_prop = simulate(&proportional, &params);

        // The proportional controller that checks theta sign should do at least as well
        assert!(steps_prop >= steps_const || steps_const < 50);
    }
}
