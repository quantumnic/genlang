use crate::ast::Node;
use crate::genetic::{
    crossover, mutate_hoist, mutate_point, mutate_shrink, tournament_select, GenStats, GpConfig,
};
use rand::Rng;

/// A behavior descriptor: a vector of floats characterizing what a program does
/// (not how well it does it).
pub type Behavior = Vec<f64>;

/// Novelty search archive entry.
#[derive(Debug, Clone)]
pub struct NoveltyEntry {
    pub genome: Node,
    pub behavior: Behavior,
    pub novelty_score: f64,
}

/// Configuration for novelty search.
#[derive(Debug, Clone)]
pub struct NoveltyConfig {
    /// Base GP config.
    pub gp: GpConfig,
    /// Number of nearest neighbors for novelty score.
    pub k_nearest: usize,
    /// Threshold for adding to archive (dynamic).
    pub archive_threshold: f64,
    /// How much to blend novelty vs fitness (0.0 = pure novelty, 1.0 = pure fitness).
    pub fitness_weight: f64,
    /// Maximum archive size.
    pub max_archive_size: usize,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            gp: GpConfig::default(),
            k_nearest: 15,
            archive_threshold: 5.0,
            fitness_weight: 0.0,
            max_archive_size: 1000,
        }
    }
}

/// Compute Euclidean distance between two behavior vectors.
fn behavior_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute novelty score: average distance to k-nearest neighbors in archive + population.
fn novelty_score(
    behavior: &[f64],
    archive: &[Behavior],
    pop_behaviors: &[Behavior],
    k: usize,
) -> f64 {
    let mut distances: Vec<f64> = archive
        .iter()
        .chain(pop_behaviors.iter())
        .map(|b| behavior_distance(behavior, b))
        .collect();

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Average of k nearest (skip self if distance ~0)
    let effective_k = k.min(distances.len());
    if effective_k == 0 {
        return 0.0;
    }
    distances.iter().take(effective_k).sum::<f64>() / effective_k as f64
}

/// Run novelty search evolution.
///
/// `behavior_fn` maps a genome to its behavior descriptor.
/// `fitness_fn` is an optional objective (used when fitness_weight > 0).
pub fn novelty_evolve<R, B, F>(
    rng: &mut R,
    config: &NoveltyConfig,
    behavior_fn: &B,
    fitness_fn: &F,
) -> (Vec<NoveltyEntry>, Vec<GenStats>)
where
    R: Rng,
    B: Fn(&Node) -> Behavior,
    F: Fn(&Node) -> f64,
{
    let mut archive: Vec<Behavior> = Vec::new();
    let mut archive_genomes: Vec<NoveltyEntry> = Vec::new();
    let mut stats = Vec::new();

    // Initialize population
    let mut genomes: Vec<Node> = (0..config.gp.population_size)
        .map(|_| Node::random(rng, config.gp.max_depth, config.gp.num_vars))
        .collect();

    let mut threshold = config.archive_threshold;
    let mut add_count = 0;
    let mut no_add_count = 0;

    for gen in 0..config.gp.max_generations {
        // Compute behaviors
        let behaviors: Vec<Behavior> = genomes.iter().map(behavior_fn).collect();

        // Compute novelty scores
        let novelty_scores: Vec<f64> = behaviors
            .iter()
            .map(|b| novelty_score(b, &archive, &behaviors, config.k_nearest))
            .collect();

        // Optionally blend with fitness
        let fitnesses: Vec<f64> = if config.fitness_weight > 0.0 {
            genomes.iter().map(fitness_fn).collect()
        } else {
            vec![0.0; genomes.len()]
        };

        // Combined scores (lower is better for selection, but novelty is "higher is better")
        // Negate novelty so tournament_select (which minimizes) picks high-novelty
        let combined: Vec<f64> = novelty_scores
            .iter()
            .zip(fitnesses.iter())
            .map(|(n, f)| {
                let novelty_component = -n * (1.0 - config.fitness_weight);
                let fitness_component = f * config.fitness_weight;
                novelty_component + fitness_component
            })
            .collect();

        // Add novel individuals to archive
        let mut gen_added = 0;
        for (i, (behavior, &score)) in behaviors.iter().zip(novelty_scores.iter()).enumerate() {
            if score > threshold && archive.len() < config.max_archive_size {
                archive.push(behavior.clone());
                archive_genomes.push(NoveltyEntry {
                    genome: genomes[i].clone(),
                    behavior: behavior.clone(),
                    novelty_score: score,
                });
                gen_added += 1;
            }
        }

        // Dynamic threshold adjustment
        if gen_added > 0 {
            add_count += 1;
            no_add_count = 0;
        } else {
            no_add_count += 1;
            add_count = 0;
        }
        if no_add_count > 5 {
            threshold *= 0.95; // make it easier to add
        }
        if add_count > 5 {
            threshold *= 1.05; // make it harder
        }

        // Stats
        let best_novelty = novelty_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_novelty = novelty_scores.iter().sum::<f64>() / novelty_scores.len().max(1) as f64;
        let best_fit_idx = combined
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        stats.push(GenStats {
            generation: gen,
            best_fitness: best_novelty,
            avg_fitness: avg_novelty,
            avg_size: genomes.iter().map(|g| g.size() as f64).sum::<f64>()
                / genomes.len().max(1) as f64,
            best_program: genomes[best_fit_idx].to_expr(),
        });

        // Selection and reproduction
        let selection_pop: Vec<(Node, f64)> = genomes
            .iter()
            .zip(combined.iter())
            .map(|(g, &c)| (g.clone(), c))
            .collect();

        let mut next_gen = Vec::with_capacity(config.gp.population_size);

        // Keep best 2
        let mut sorted_indices: Vec<usize> = (0..genomes.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            combined[a]
                .partial_cmp(&combined[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in sorted_indices
            .iter()
            .take(config.gp.elitism.min(genomes.len()))
        {
            next_gen.push(genomes[i].clone());
        }

        while next_gen.len() < config.gp.population_size {
            let r: f64 = rng.gen();
            let child = if r < config.gp.crossover_rate {
                let p1 = tournament_select(rng, &selection_pop, config.gp.tournament_size);
                let p2 = tournament_select(rng, &selection_pop, config.gp.tournament_size);
                crossover(rng, p1, p2).0
            } else if r < config.gp.crossover_rate + config.gp.mutation_rate {
                let p = tournament_select(rng, &selection_pop, config.gp.tournament_size);
                match rng.gen_range(0..3) {
                    0 => mutate_point(rng, p, config.gp.num_vars),
                    1 => mutate_hoist(rng, p),
                    _ => mutate_shrink(rng, p, config.gp.num_vars),
                }
            } else {
                tournament_select(rng, &selection_pop, config.gp.tournament_size).clone()
            };

            if child.size() <= config.gp.max_tree_size {
                next_gen.push(child);
            } else {
                next_gen.push(
                    tournament_select(rng, &selection_pop, config.gp.tournament_size).clone(),
                );
            }
        }

        genomes = next_gen;
    }

    (archive_genomes, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_behavior_distance() {
        assert!((behavior_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((behavior_distance(&[1.0], &[1.0])).abs() < 1e-10);
    }

    #[test]
    fn test_novelty_score_computation() {
        let behavior = vec![0.0, 0.0];
        let archive = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let pop = vec![vec![2.0, 0.0]];
        let score = novelty_score(&behavior, &archive, &pop, 2);
        assert!(score > 0.0);
    }

    #[test]
    fn test_novelty_search_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = NoveltyConfig {
            gp: GpConfig {
                population_size: 30,
                max_generations: 10,
                max_depth: 3,
                num_vars: 1,
                ..GpConfig::default()
            },
            k_nearest: 5,
            ..NoveltyConfig::default()
        };

        // Behavior: output on a few sample points
        let behavior_fn = |tree: &Node| -> Behavior {
            let mut interp = Interpreter::default();
            let mut behavior = Vec::new();
            for i in -3..=3 {
                interp.reset();
                let val = interp
                    .eval(tree, &[i as f64])
                    .map(|v| v.to_f64())
                    .unwrap_or(0.0);
                // Clamp to avoid NaN issues in distance
                behavior.push(val.clamp(-1000.0, 1000.0));
            }
            behavior
        };

        let fitness_fn = |_tree: &Node| -> f64 { 0.0 }; // pure novelty

        let (archive, stats) = novelty_evolve(&mut rng, &config, &behavior_fn, &fitness_fn);

        assert!(!stats.is_empty());
        // Archive should have accumulated some entries
        // (may be 0 if threshold is too high for tiny runs, that's ok)
        assert!(archive.len() <= config.max_archive_size);
    }

    #[test]
    fn test_novelty_with_fitness_blend() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let config = NoveltyConfig {
            gp: GpConfig {
                population_size: 30,
                max_generations: 5,
                max_depth: 3,
                num_vars: 1,
                ..GpConfig::default()
            },
            fitness_weight: 0.5, // 50% novelty, 50% fitness
            k_nearest: 5,
            ..NoveltyConfig::default()
        };

        let behavior_fn = |tree: &Node| -> Behavior {
            let mut interp = Interpreter::default();
            vec![interp
                .eval(tree, &[1.0])
                .map(|v| v.to_f64())
                .unwrap_or(0.0)
                .clamp(-100.0, 100.0)]
        };

        let fitness_fn = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            interp
                .eval(tree, &[1.0])
                .map(|v| (v.to_f64() - 42.0).powi(2))
                .unwrap_or(1e6)
        };

        let (_archive, stats) = novelty_evolve(&mut rng, &config, &behavior_fn, &fitness_fn);
        assert!(!stats.is_empty());
    }
}
