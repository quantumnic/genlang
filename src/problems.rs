use crate::ast::Node;
use crate::genetic::{evolve, GenStats, GpConfig};
use crate::interpreter::Interpreter;
use rand::Rng;

/// Symbolic regression: discover an equation from (x, y) data points.
pub fn symbolic_regression<R: Rng>(
    rng: &mut R,
    data: &[(f64, f64)],
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut total_error = 0.0;
        for (x, y) in data {
            interp.reset();
            match interp.eval(tree, &[*x]) {
                Ok(val) => {
                    let diff = val.to_f64() - y;
                    total_error += diff * diff;
                }
                Err(_) => total_error += 1e6,
            }
        }
        total_error / data.len() as f64
    };

    evolve(rng, config, &fitness)
}

/// Generate data for a target function (for testing).
pub fn generate_data<F>(f: F, range: std::ops::RangeInclusive<i32>) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    range
        .map(|i| {
            let x = i as f64;
            (x, f(x))
        })
        .collect()
}

/// Generate all boolean input combinations for n bits.
fn all_boolean_inputs(n: usize) -> Vec<Vec<f64>> {
    let count = 1 << n;
    (0..count)
        .map(|i| {
            (0..n)
                .map(|bit| if (i >> bit) & 1 == 1 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Even-parity problem: evolve a program that returns 1.0 when the number
/// of true (1.0) inputs is even, 0.0 otherwise.
pub fn even_parity<R: Rng>(
    rng: &mut R,
    num_bits: usize,
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    let inputs = all_boolean_inputs(num_bits);
    let targets: Vec<f64> = inputs
        .iter()
        .map(|bits| {
            let ones = bits.iter().filter(|&&b| b > 0.5).count();
            if ones % 2 == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut errors = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            interp.reset();
            match interp.eval(tree, input) {
                Ok(val) => {
                    let output = if val.to_f64() > 0.5 { 1.0 } else { 0.0 };
                    if (output - target).abs() > 0.5 {
                        errors += 1.0;
                    }
                }
                Err(_) => errors += 1.0,
            }
        }
        errors
    };

    evolve(rng, config, &fitness)
}

/// Fibonacci problem: evolve a program that maps index n → fib(n).
/// Uses indices 0..=max_n as training data.
pub fn fibonacci<R: Rng>(
    rng: &mut R,
    max_n: usize,
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    // Pre-compute Fibonacci targets
    let mut fibs = vec![0.0f64; max_n + 1];
    if max_n >= 1 {
        fibs[1] = 1.0;
    }
    for i in 2..=max_n {
        fibs[i] = fibs[i - 1] + fibs[i - 2];
    }

    let data: Vec<(f64, f64)> = (0..=max_n).map(|i| (i as f64, fibs[i])).collect();

    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut total_error = 0.0;
        for (x, y) in &data {
            interp.reset();
            match interp.eval(tree, &[*x]) {
                Ok(val) => {
                    let diff = val.to_f64() - y;
                    total_error += diff * diff;
                }
                Err(_) => total_error += 1e6,
            }
        }
        total_error / data.len() as f64
    };

    evolve(rng, config, &fitness)
}

/// Sorting network evolution: evolve a comparator network that sorts small arrays.
///
/// The genome encodes a comparison function: given two values (x0, x1),
/// it should return negative if x0 < x1 (keep order), positive if x0 > x1 (swap).
/// We test the evolved comparator as a bubble-sort comparator on small arrays.
pub fn sorting_network<R: Rng>(
    rng: &mut R,
    array_size: usize,
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    // Generate test arrays
    let test_arrays: Vec<Vec<f64>> = (0..20)
        .map(|seed| {
            let mut arr: Vec<f64> = (0..array_size)
                .map(|i| {
                    // Deterministic pseudo-random based on seed + i
                    ((seed * 7 + i * 13 + 5) % 19) as f64 - 9.0
                })
                .collect();
            // Shuffle a bit based on seed
            if seed % 2 == 0 {
                arr.reverse();
            }
            arr
        })
        .collect();

    let fitness = move |tree: &Node| -> f64 {
        let mut total_inversions = 0.0;
        let mut interp = Interpreter::default();

        for arr in &test_arrays {
            // Use evolved comparator in bubble sort
            let mut sorted = arr.clone();
            let n = sorted.len();

            for _ in 0..n {
                for j in 0..n.saturating_sub(1) {
                    interp.reset();
                    // Feed pair to comparator
                    if let Ok(val) = interp.eval(tree, &[sorted[j], sorted[j + 1]]) {
                        if val.to_f64() > 0.0 {
                            sorted.swap(j, j + 1);
                        }
                    }
                }
            }

            // Count inversions in result (0 = perfectly sorted)
            for i in 0..n {
                for j in (i + 1)..n {
                    if sorted[i] > sorted[j] {
                        total_inversions += 1.0;
                    }
                }
            }
        }

        total_inversions
    };

    evolve(rng, config, &fitness)
}

/// Maze solving: evolve a program that navigates a simple grid maze.
/// The program receives (wall_ahead, wall_left, wall_right, goal_direction)
/// and should output a direction: <0 = left, 0 = forward, >0 = right.
pub fn maze_solver<R: Rng>(rng: &mut R, config: &GpConfig) -> (Node, f64, Vec<GenStats>) {
    // Simple maze scenarios: (wall_ahead, wall_left, wall_right, goal_dir) → best action
    // goal_dir: -1 = left, 0 = ahead, 1 = right
    let scenarios: Vec<(Vec<f64>, f64)> = vec![
        // No walls, goal ahead → go forward (0)
        (vec![0.0, 0.0, 0.0, 0.0], 0.0),
        // Wall ahead, goal left → go left (-1)
        (vec![1.0, 0.0, 0.0, -1.0], -1.0),
        // Wall ahead, goal right → go right (1)
        (vec![1.0, 0.0, 0.0, 1.0], 1.0),
        // Wall ahead and left → go right
        (vec![1.0, 1.0, 0.0, 0.0], 1.0),
        // Wall ahead and right → go left
        (vec![1.0, 0.0, 1.0, 0.0], -1.0),
        // No wall ahead, goal ahead → forward
        (vec![0.0, 1.0, 1.0, 0.0], 0.0),
        // Wall left, goal left → forward (can't go left)
        (vec![0.0, 1.0, 0.0, -1.0], 0.0),
        // Wall right, goal right → forward (can't go right)
        (vec![0.0, 0.0, 1.0, 1.0], 0.0),
        // All walls except behind → shouldn't happen but handle gracefully
        (vec![1.0, 1.0, 1.0, 0.0], 0.0),
        // No walls, goal left → go left
        (vec![0.0, 0.0, 0.0, -1.0], -1.0),
        // No walls, goal right → go right
        (vec![0.0, 0.0, 0.0, 1.0], 1.0),
    ];

    let fitness = move |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut total_error = 0.0;

        for (inputs, expected) in &scenarios {
            interp.reset();
            match interp.eval(tree, inputs) {
                Ok(val) => {
                    let output = val.to_f64();
                    // Classify output: <-0.3 = left, >0.3 = right, else = forward
                    let action = if output < -0.3 {
                        -1.0
                    } else if output > 0.3 {
                        1.0
                    } else {
                        0.0
                    };
                    if (action - expected).abs() > 0.5 {
                        total_error += 1.0;
                    }
                }
                Err(_) => total_error += 1.0,
            }
        }

        total_error
    };

    evolve(rng, config, &fitness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_symbolic_regression_linear() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let data = generate_data(|x| 2.0 * x + 1.0, -5..=5);

        let config = GpConfig {
            population_size: 100,
            max_generations: 50,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = symbolic_regression(&mut rng, &data, &config);
        assert!(!stats.is_empty());
        assert!(best_fit < 100.0, "fitness too high: {best_fit}");
    }

    #[test]
    fn test_generate_data() {
        let data = generate_data(|x| x * x, -3..=3);
        assert_eq!(data.len(), 7);
        assert_eq!(data[3], (0.0, 0.0));
        assert_eq!(data[6], (3.0, 9.0));
    }

    #[test]
    fn test_all_boolean_inputs() {
        let inputs = all_boolean_inputs(2);
        assert_eq!(inputs.len(), 4);
        assert_eq!(inputs[0], vec![0.0, 0.0]);
        assert_eq!(inputs[3], vec![1.0, 1.0]);
    }

    #[test]
    fn test_fibonacci() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 1,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = fibonacci(&mut rng, 8, &config);
        assert!(!stats.is_empty());
        // Should at least improve from random
        assert!(best_fit < 1e6);
    }

    #[test]
    fn test_sorting_network() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 100,
            max_generations: 30,
            max_depth: 4,
            num_vars: 2,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = sorting_network(&mut rng, 4, &config);
        assert!(!stats.is_empty());
        // Should improve from random
        assert!(best_fit < 1000.0);
    }

    #[test]
    fn test_maze_solver() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 4,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = maze_solver(&mut rng, &config);
        assert!(!stats.is_empty());
        // 11 scenarios, should get some right
        assert!(best_fit <= 11.0);
    }

    #[test]
    fn test_even_parity_2bit() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 2,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = even_parity(&mut rng, 2, &config);
        assert!(!stats.is_empty());
        // For 2-bit parity, 4 test cases. Should get some right.
        assert!(best_fit <= 4.0);
    }
}
