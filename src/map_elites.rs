//! MAP-Elites: Quality-Diversity algorithm.
//!
//! Maintains a grid of behavior descriptors. Each cell holds the best individual
//! found for that behavior niche. Produces a diverse repertoire of high-performing solutions.

use crate::ast::Node;
use crate::genetic::{crossover, mutate_point, mutate_shrink};
use rand::Rng;
use std::collections::HashMap;

/// A cell in the MAP-Elites grid.
#[derive(Debug, Clone)]
pub struct MapCell {
    pub genome: Node,
    pub fitness: f64,
    pub behavior: Vec<usize>,
}

/// MAP-Elites archive: a multi-dimensional grid indexed by discretized behavior descriptors.
#[derive(Debug, Clone)]
pub struct MapElitesArchive {
    /// Map from discretized behavior coordinates to the elite for that cell.
    pub cells: HashMap<Vec<usize>, MapCell>,
    /// Number of bins per behavior dimension.
    pub bins_per_dim: usize,
    /// Number of behavior dimensions.
    pub num_dims: usize,
}

impl MapElitesArchive {
    pub fn new(num_dims: usize, bins_per_dim: usize) -> Self {
        Self {
            cells: HashMap::new(),
            bins_per_dim,
            num_dims,
        }
    }

    /// Discretize a continuous behavior descriptor to grid coordinates.
    pub fn discretize(&self, behavior: &[f64], ranges: &[(f64, f64)]) -> Vec<usize> {
        behavior
            .iter()
            .zip(ranges.iter())
            .map(|(&val, &(lo, hi))| {
                let normalized = (val - lo) / (hi - lo);
                let bin = (normalized * self.bins_per_dim as f64).floor() as usize;
                bin.min(self.bins_per_dim - 1)
            })
            .collect()
    }

    /// Try to place an individual in the archive. Returns true if placed.
    pub fn try_insert(
        &mut self,
        genome: Node,
        fitness: f64,
        behavior: &[f64],
        ranges: &[(f64, f64)],
    ) -> bool {
        let coords = self.discretize(behavior, ranges);

        if let Some(existing) = self.cells.get(&coords) {
            if fitness < existing.fitness {
                self.cells.insert(
                    coords.clone(),
                    MapCell {
                        genome,
                        fitness,
                        behavior: coords,
                    },
                );
                return true;
            }
            false
        } else {
            self.cells.insert(
                coords.clone(),
                MapCell {
                    genome,
                    fitness,
                    behavior: coords,
                },
            );
            true
        }
    }

    /// Get a random elite from the archive.
    pub fn random_elite<R: Rng>(&self, rng: &mut R) -> Option<&MapCell> {
        if self.cells.is_empty() {
            return None;
        }
        let keys: Vec<_> = self.cells.keys().collect();
        let idx = rng.gen_range(0..keys.len());
        self.cells.get(keys[idx])
    }

    /// Coverage: fraction of cells filled.
    pub fn coverage(&self) -> f64 {
        let total_cells = self.bins_per_dim.pow(self.num_dims as u32);
        self.cells.len() as f64 / total_cells as f64
    }

    /// Best fitness across all cells.
    pub fn best_fitness(&self) -> Option<f64> {
        self.cells.values().map(|c| c.fitness).reduce(f64::min)
    }

    /// Get the best individual across all cells.
    pub fn best(&self) -> Option<&MapCell> {
        self.cells.values().min_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of filled cells.
    pub fn filled(&self) -> usize {
        self.cells.len()
    }
}

/// Configuration for MAP-Elites.
#[derive(Debug, Clone)]
pub struct MapElitesConfig {
    /// Total number of iterations (evaluations).
    pub iterations: usize,
    /// Initial random population to seed the archive.
    pub initial_pop: usize,
    /// Max tree depth for random programs.
    pub max_depth: usize,
    /// Max tree size (bloat control).
    pub max_tree_size: usize,
    /// Number of input variables.
    pub num_vars: usize,
    /// Number of bins per behavior dimension.
    pub bins_per_dim: usize,
    /// Behavior descriptor ranges: (min, max) per dimension.
    pub behavior_ranges: Vec<(f64, f64)>,
    /// Mutation rate (vs crossover).
    pub mutation_rate: f64,
}

impl Default for MapElitesConfig {
    fn default() -> Self {
        Self {
            iterations: 5000,
            initial_pop: 200,
            max_depth: 5,
            max_tree_size: 100,
            num_vars: 1,
            bins_per_dim: 10,
            behavior_ranges: vec![(-10.0, 10.0), (-10.0, 10.0)],
            mutation_rate: 0.7,
        }
    }
}

/// Statistics for MAP-Elites iteration batches.
#[derive(Debug, Clone)]
pub struct MapElitesStats {
    pub iteration: usize,
    pub filled_cells: usize,
    pub coverage: f64,
    pub best_fitness: f64,
}

/// Run MAP-Elites algorithm.
///
/// `fitness_fn`: evaluates a program (lower is better).
/// `behavior_fn`: extracts behavior descriptor from a program.
pub fn map_elites_evolve<R, F, B>(
    rng: &mut R,
    config: &MapElitesConfig,
    fitness_fn: &F,
    behavior_fn: &B,
) -> (MapElitesArchive, Vec<MapElitesStats>)
where
    R: Rng,
    F: Fn(&Node) -> f64,
    B: Fn(&Node) -> Vec<f64>,
{
    let num_dims = config.behavior_ranges.len();
    let mut archive = MapElitesArchive::new(num_dims, config.bins_per_dim);
    let mut stats = Vec::new();

    // Phase 1: seed with random individuals
    for _ in 0..config.initial_pop {
        let genome = Node::random(rng, config.max_depth, config.num_vars);
        let fitness = fitness_fn(&genome);
        if fitness.is_finite() {
            let behavior = behavior_fn(&genome);
            archive.try_insert(genome, fitness, &behavior, &config.behavior_ranges);
        }
    }

    // Phase 2: iterate
    let report_interval = (config.iterations / 20).max(1);

    for iter in 0..config.iterations {
        // Pick parent(s) from archive
        let child = if archive.filled() < 2 || rng.gen_bool(config.mutation_rate) {
            // Mutation
            if let Some(parent) = archive.random_elite(rng) {
                if rng.gen_bool(0.5) {
                    mutate_point(rng, &parent.genome, config.num_vars)
                } else {
                    mutate_shrink(rng, &parent.genome, config.num_vars)
                }
            } else {
                Node::random(rng, config.max_depth, config.num_vars)
            }
        } else {
            // Crossover
            let p1 = archive.random_elite(rng).unwrap();
            let p2 = archive.random_elite(rng).unwrap();
            let (c, _) = crossover(rng, &p1.genome, &p2.genome);
            c
        };

        // Bloat control
        if child.size() > config.max_tree_size {
            continue;
        }

        let fitness = fitness_fn(&child);
        if fitness.is_finite() {
            let behavior = behavior_fn(&child);
            archive.try_insert(child, fitness, &behavior, &config.behavior_ranges);
        }

        // Periodic stats
        if iter % report_interval == 0 || iter == config.iterations - 1 {
            stats.push(MapElitesStats {
                iteration: iter,
                filled_cells: archive.filled(),
                coverage: archive.coverage(),
                best_fitness: archive.best_fitness().unwrap_or(f64::INFINITY),
            });
        }
    }

    (archive, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_archive_insert() {
        let mut archive = MapElitesArchive::new(2, 5);
        let ranges = vec![(0.0, 10.0), (0.0, 10.0)];

        let inserted = archive.try_insert(Node::IntConst(1), 5.0, &[2.5, 7.5], &ranges);
        assert!(inserted);
        assert_eq!(archive.filled(), 1);

        // Same cell, worse fitness → rejected
        let inserted = archive.try_insert(Node::IntConst(2), 10.0, &[2.5, 7.5], &ranges);
        assert!(!inserted);

        // Same cell, better fitness → accepted
        let inserted = archive.try_insert(Node::IntConst(3), 1.0, &[2.5, 7.5], &ranges);
        assert!(inserted);
        assert_eq!(archive.filled(), 1);
        assert!((archive.best_fitness().unwrap() - 1.0).abs() < 1e-10);

        // Different cell → accepted
        let inserted = archive.try_insert(Node::IntConst(4), 3.0, &[8.0, 1.0], &ranges);
        assert!(inserted);
        assert_eq!(archive.filled(), 2);
    }

    #[test]
    fn test_discretize() {
        let archive = MapElitesArchive::new(2, 10);
        let ranges = vec![(0.0, 100.0), (-5.0, 5.0)];

        let coords = archive.discretize(&[50.0, 0.0], &ranges);
        assert_eq!(coords, vec![5, 5]);

        let coords = archive.discretize(&[0.0, -5.0], &ranges);
        assert_eq!(coords, vec![0, 0]);

        let coords = archive.discretize(&[100.0, 5.0], &ranges);
        assert_eq!(coords, vec![9, 9]); // clamped to max bin
    }

    #[test]
    fn test_coverage() {
        let mut archive = MapElitesArchive::new(2, 3);
        let ranges = vec![(0.0, 9.0), (0.0, 9.0)];
        assert!((archive.coverage() - 0.0).abs() < 1e-10);

        // Fill 3 out of 9 cells
        archive.try_insert(Node::IntConst(1), 1.0, &[1.0, 1.0], &ranges);
        archive.try_insert(Node::IntConst(2), 1.0, &[4.0, 4.0], &ranges);
        archive.try_insert(Node::IntConst(3), 1.0, &[7.0, 7.0], &ranges);
        assert!((archive.coverage() - 3.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_elites_evolve() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let config = MapElitesConfig {
            iterations: 500,
            initial_pop: 50,
            max_depth: 4,
            max_tree_size: 50,
            num_vars: 1,
            bins_per_dim: 5,
            behavior_ranges: vec![(-10.0, 10.0), (0.0, 50.0)],
            mutation_rate: 0.7,
        };

        // Fitness: how close to x^2
        let fitness_fn = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            let mut error = 0.0;
            for i in -3..=3 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - x * x;
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error / 7.0
        };

        // Behavior: (output at x=0, tree size)
        let behavior_fn = |tree: &Node| -> Vec<f64> {
            let mut interp = Interpreter::default();
            let out_at_0 = interp
                .eval(tree, &[0.0])
                .map(|v| v.to_f64().clamp(-10.0, 10.0))
                .unwrap_or(0.0);
            vec![out_at_0, tree.size() as f64]
        };

        let (archive, stats) = map_elites_evolve(&mut rng, &config, &fitness_fn, &behavior_fn);

        assert!(archive.filled() > 1);
        assert!(!stats.is_empty());
        assert!(archive.best_fitness().unwrap() < 1e6);
    }
}
