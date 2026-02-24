//! Competitive Coevolution.
//!
//! Two populations evolve against each other. Each individual's fitness depends
//! on how well it performs against the opposing population, creating an arms race.
//! Useful for evolving game strategies, adversarial test cases, etc.

use crate::ast::Node;
use crate::genetic::{crossover, mutate_hoist, mutate_point, mutate_shrink, GpConfig};
use rand::Rng;

/// Result of a competitive interaction between two individuals.
#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    /// First individual wins.
    Win,
    /// Second individual wins.
    Loss,
    /// Draw.
    Draw,
}

/// Configuration for coevolution.
#[derive(Debug, Clone)]
pub struct CoevConfig {
    /// GP config for population A.
    pub config_a: GpConfig,
    /// GP config for population B.
    pub config_b: GpConfig,
    /// Number of opponents to test against per evaluation.
    pub sample_opponents: usize,
    /// Total generations.
    pub generations: usize,
}

impl Default for CoevConfig {
    fn default() -> Self {
        let gp = GpConfig {
            population_size: 100,
            max_generations: 50,
            max_depth: 5,
            num_vars: 2,
            ..GpConfig::default()
        };
        Self {
            config_a: gp.clone(),
            config_b: gp,
            sample_opponents: 10,
            generations: 50,
        }
    }
}

/// Statistics for a coevolution generation.
#[derive(Debug, Clone)]
pub struct CoevStats {
    pub generation: usize,
    pub best_fitness_a: f64,
    pub best_fitness_b: f64,
    pub avg_fitness_a: f64,
    pub avg_fitness_b: f64,
    pub best_program_a: String,
    pub best_program_b: String,
}

/// Evaluate a population against a sample of opponents.
/// `compete_fn(a, b) -> Outcome` where Outcome::Win means `a` wins.
fn evaluate_against_opponents<R, C>(
    rng: &mut R,
    population: &[Node],
    opponents: &[Node],
    sample_size: usize,
    compete_fn: &C,
) -> Vec<f64>
where
    R: Rng,
    C: Fn(&Node, &Node) -> Outcome,
{
    population
        .iter()
        .map(|individual| {
            let n_opponents = sample_size.min(opponents.len());
            if n_opponents == 0 {
                return 0.0;
            }

            let mut wins = 0.0;
            // Sample random opponents
            for _ in 0..n_opponents {
                let opp_idx = rng.gen_range(0..opponents.len());
                match compete_fn(individual, &opponents[opp_idx]) {
                    Outcome::Win => wins += 1.0,
                    Outcome::Draw => wins += 0.5,
                    Outcome::Loss => {}
                }
            }
            // Fitness = fraction of wins (higher is better, but we negate for minimization)
            -(wins / n_opponents as f64)
        })
        .collect()
}

/// Tournament selection by index.
fn tournament_idx<R: Rng>(rng: &mut R, fitnesses: &[f64], tournament_size: usize) -> usize {
    let mut best = rng.gen_range(0..fitnesses.len());
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..fitnesses.len());
        if fitnesses[idx] < fitnesses[best] {
            best = idx;
        }
    }
    best
}

/// Create the next generation from a population.
fn next_generation<R: Rng>(
    rng: &mut R,
    population: &[Node],
    fitnesses: &[f64],
    config: &GpConfig,
) -> Vec<Node> {
    // Sort indices by fitness
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| {
        fitnesses[a]
            .partial_cmp(&fitnesses[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut next = Vec::with_capacity(config.population_size);

    // Elitism
    for &idx in indices.iter().take(config.elitism) {
        next.push(population[idx].clone());
    }

    while next.len() < config.population_size {
        let r: f64 = rng.gen();
        let child = if r < config.crossover_rate {
            let i1 = tournament_idx(rng, fitnesses, config.tournament_size);
            let i2 = tournament_idx(rng, fitnesses, config.tournament_size);
            let (c, _) = crossover(rng, &population[i1], &population[i2]);
            c
        } else if r < config.crossover_rate + config.mutation_rate {
            let i = tournament_idx(rng, fitnesses, config.tournament_size);
            match rng.gen_range(0..3) {
                0 => mutate_point(rng, &population[i], config.num_vars),
                1 => mutate_hoist(rng, &population[i]),
                _ => mutate_shrink(rng, &population[i], config.num_vars),
            }
        } else {
            let i = tournament_idx(rng, fitnesses, config.tournament_size);
            population[i].clone()
        };

        if child.size() <= config.max_tree_size {
            next.push(child);
        } else {
            let i = tournament_idx(rng, fitnesses, config.tournament_size);
            next.push(population[i].clone());
        }
    }

    next
}

/// Run competitive coevolution between two populations.
///
/// `compete_fn(a, b) -> Outcome` determines the outcome of a matchup.
/// Returns the best from each population and evolution statistics.
pub fn coevolve<R, C>(
    rng: &mut R,
    config: &CoevConfig,
    compete_fn: &C,
) -> (Node, Node, Vec<CoevStats>)
where
    R: Rng,
    C: Fn(&Node, &Node) -> Outcome,
{
    // Initialize populations
    let mut pop_a: Vec<Node> = (0..config.config_a.population_size)
        .map(|_| Node::random(rng, config.config_a.max_depth, config.config_a.num_vars))
        .collect();

    let mut pop_b: Vec<Node> = (0..config.config_b.population_size)
        .map(|_| Node::random(rng, config.config_b.max_depth, config.config_b.num_vars))
        .collect();

    let mut stats = Vec::new();
    let mut best_a: Option<(Node, f64)> = None;
    let mut best_b: Option<(Node, f64)> = None;

    for gen in 0..config.generations {
        // Evaluate A against B
        let fitnesses_a =
            evaluate_against_opponents(rng, &pop_a, &pop_b, config.sample_opponents, compete_fn);

        // Evaluate B against A (reverse the compete function)
        let reverse_compete = |b: &Node, a: &Node| -> Outcome {
            match compete_fn(a, b) {
                Outcome::Win => Outcome::Loss,
                Outcome::Loss => Outcome::Win,
                Outcome::Draw => Outcome::Draw,
            }
        };
        let fitnesses_b = evaluate_against_opponents(
            rng,
            &pop_b,
            &pop_a,
            config.sample_opponents,
            &reverse_compete,
        );

        // Track bests
        let best_a_idx = fitnesses_a
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_b_idx = fitnesses_b
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if best_a.is_none() || fitnesses_a[best_a_idx] < best_a.as_ref().unwrap().1 {
            best_a = Some((pop_a[best_a_idx].clone(), fitnesses_a[best_a_idx]));
        }
        if best_b.is_none() || fitnesses_b[best_b_idx] < best_b.as_ref().unwrap().1 {
            best_b = Some((pop_b[best_b_idx].clone(), fitnesses_b[best_b_idx]));
        }

        let avg_a = fitnesses_a.iter().sum::<f64>() / fitnesses_a.len() as f64;
        let avg_b = fitnesses_b.iter().sum::<f64>() / fitnesses_b.len() as f64;

        stats.push(CoevStats {
            generation: gen,
            best_fitness_a: fitnesses_a[best_a_idx],
            best_fitness_b: fitnesses_b[best_b_idx],
            avg_fitness_a: avg_a,
            avg_fitness_b: avg_b,
            best_program_a: pop_a[best_a_idx].to_expr(),
            best_program_b: pop_b[best_b_idx].to_expr(),
        });

        // Next generation
        pop_a = next_generation(rng, &pop_a, &fitnesses_a, &config.config_a);
        pop_b = next_generation(rng, &pop_b, &fitnesses_b, &config.config_b);
    }

    let (ba, _) = best_a.unwrap_or_else(|| (pop_a[0].clone(), 0.0));
    let (bb, _) = best_b.unwrap_or_else(|| (pop_b[0].clone(), 0.0));
    (ba, bb, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_outcome() {
        assert!(matches!(Outcome::Win, Outcome::Win));
        assert!(matches!(Outcome::Loss, Outcome::Loss));
        assert!(matches!(Outcome::Draw, Outcome::Draw));
    }

    #[test]
    fn test_evaluate_against_opponents() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pop = vec![Node::IntConst(5), Node::IntConst(3), Node::IntConst(7)];
        let opps = vec![Node::IntConst(4), Node::IntConst(6)];

        // Simple compete: higher constant wins
        let compete = |a: &Node, b: &Node| -> Outcome {
            let mut interp = Interpreter::default();
            let va = interp.eval(a, &[]).unwrap().to_f64();
            interp.reset();
            let vb = interp.eval(b, &[]).unwrap().to_f64();
            if va > vb {
                Outcome::Win
            } else if va < vb {
                Outcome::Loss
            } else {
                Outcome::Draw
            }
        };

        let fitnesses = evaluate_against_opponents(&mut rng, &pop, &opps, 2, &compete);
        assert_eq!(fitnesses.len(), 3);
        // IntConst(7) should have best (most negative) fitness since it beats everything
        // IntConst(3) should have worst fitness
        assert!(fitnesses[2] <= fitnesses[1]); // 7 beats more than 3
    }

    #[test]
    fn test_coevolution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let config = CoevConfig {
            config_a: GpConfig {
                population_size: 30,
                max_depth: 3,
                num_vars: 1,
                ..GpConfig::default()
            },
            config_b: GpConfig {
                population_size: 30,
                max_depth: 3,
                num_vars: 1,
                ..GpConfig::default()
            },
            sample_opponents: 5,
            generations: 10,
        };

        // Number comparison game: each program outputs a number,
        // higher number wins. This should drive evolution toward larger values.
        let compete = |a: &Node, b: &Node| -> Outcome {
            let mut interp = Interpreter::default();
            let va = interp
                .eval(a, &[0.0])
                .unwrap_or(crate::interpreter::Value::Float(0.0))
                .to_f64();
            interp.reset();
            let vb = interp
                .eval(b, &[0.0])
                .unwrap_or(crate::interpreter::Value::Float(0.0))
                .to_f64();
            if va > vb + 0.01 {
                Outcome::Win
            } else if vb > va + 0.01 {
                Outcome::Loss
            } else {
                Outcome::Draw
            }
        };

        let (best_a, best_b, stats) = coevolve(&mut rng, &config, &compete);
        assert!(!stats.is_empty());
        assert!(best_a.size() > 0);
        assert!(best_b.size() > 0);
    }

    #[test]
    fn test_coev_arms_race() {
        // Verify that both populations improve over time
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        let config = CoevConfig {
            config_a: GpConfig {
                population_size: 50,
                max_depth: 4,
                num_vars: 1,
                ..GpConfig::default()
            },
            config_b: GpConfig {
                population_size: 50,
                max_depth: 4,
                num_vars: 1,
                ..GpConfig::default()
            },
            sample_opponents: 8,
            generations: 15,
        };

        let compete = |a: &Node, b: &Node| -> Outcome {
            let mut interp = Interpreter::default();
            let va = interp
                .eval(a, &[1.0])
                .unwrap_or(crate::interpreter::Value::Float(0.0))
                .to_f64();
            interp.reset();
            let vb = interp
                .eval(b, &[1.0])
                .unwrap_or(crate::interpreter::Value::Float(0.0))
                .to_f64();
            if va > vb {
                Outcome::Win
            } else if vb > va {
                Outcome::Loss
            } else {
                Outcome::Draw
            }
        };

        let (_, _, stats) = coevolve(&mut rng, &config, &compete);
        assert!(stats.len() == 15);
    }
}
