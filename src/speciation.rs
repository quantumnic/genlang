use crate::ast::Node;
use crate::genetic::{
    crossover, mutate_hoist, mutate_point, mutate_shrink, tournament_select, GenStats, GpConfig,
};
use rand::Rng;

/// Compute genotypic distance between two programs.
/// Uses a combination of structural and size differences.
pub fn genome_distance(a: &Node, b: &Node) -> f64 {
    let size_diff = (a.size() as f64 - b.size() as f64).abs();
    let depth_diff = (a.depth() as f64 - b.depth() as f64).abs();
    let structural = structural_diff(a, b);
    size_diff * 0.3 + depth_diff * 0.3 + structural * 0.4
}

/// Recursive structural comparison — counts mismatching nodes.
fn structural_diff(a: &Node, b: &Node) -> f64 {
    match (a, b) {
        (Node::IntConst(x), Node::IntConst(y)) => {
            if x == y {
                0.0
            } else {
                0.5
            }
        }
        (Node::FloatConst(x), Node::FloatConst(y)) => ((x - y).abs()).min(1.0),
        (Node::BoolConst(x), Node::BoolConst(y)) => {
            if x == y {
                0.0
            } else {
                1.0
            }
        }
        (Node::Var(x), Node::Var(y)) => {
            if x == y {
                0.0
            } else {
                0.5
            }
        }
        (Node::BinOp(op1, l1, r1), Node::BinOp(op2, l2, r2)) => {
            let op_diff = if op1 == op2 { 0.0 } else { 1.0 };
            op_diff + structural_diff(l1, l2) + structural_diff(r1, r2)
        }
        (Node::UnaryOp(op1, c1), Node::UnaryOp(op2, c2)) => {
            let op_diff = if op1 == op2 { 0.0 } else { 1.0 };
            op_diff + structural_diff(c1, c2)
        }
        (Node::MathFn(f1, c1), Node::MathFn(f2, c2)) => {
            let fn_diff = if f1 == f2 { 0.0 } else { 1.0 };
            fn_diff + structural_diff(c1, c2)
        }
        (Node::Cmp(op1, l1, r1), Node::Cmp(op2, l2, r2)) => {
            let op_diff = if op1 == op2 { 0.0 } else { 1.0 };
            op_diff + structural_diff(l1, l2) + structural_diff(r1, r2)
        }
        (Node::If(a1, b1, c1), Node::If(a2, b2, c2))
        | (Node::Loop(a1, b1, c1), Node::Loop(a2, b2, c2)) => {
            structural_diff(a1, a2) + structural_diff(b1, b2) + structural_diff(c1, c2)
        }
        _ => {
            // Different node types: count max of both sizes as distance
            (a.size().max(b.size())) as f64
        }
    }
}

/// A species: a group of similar individuals.
#[derive(Debug, Clone)]
pub struct Species {
    pub id: usize,
    pub representative: Node,
    pub members: Vec<usize>, // indices into population
    pub best_fitness: f64,
    pub stagnation: usize, // generations without improvement
}

/// Speciated evolution configuration.
#[derive(Debug, Clone)]
pub struct SpeciationConfig {
    /// Base GP config.
    pub gp: GpConfig,
    /// Distance threshold for speciation.
    pub compatibility_threshold: f64,
    /// Maximum stagnation before a species is penalized.
    pub max_stagnation: usize,
    /// Minimum species size.
    pub min_species_size: usize,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            gp: GpConfig::default(),
            compatibility_threshold: 5.0,
            max_stagnation: 15,
            min_species_size: 5,
        }
    }
}

/// Assign individuals to species based on distance to representatives.
fn assign_species(
    population: &[Node],
    species: &mut Vec<Species>,
    threshold: f64,
    next_species_id: &mut usize,
) {
    // Clear old members
    for s in species.iter_mut() {
        s.members.clear();
    }

    for (i, individual) in population.iter().enumerate() {
        let mut assigned = false;
        for s in species.iter_mut() {
            if genome_distance(individual, &s.representative) < threshold {
                s.members.push(i);
                assigned = true;
                break;
            }
        }
        if !assigned {
            // Create new species
            species.push(Species {
                id: *next_species_id,
                representative: individual.clone(),
                members: vec![i],
                best_fitness: f64::INFINITY,
                stagnation: 0,
            });
            *next_species_id += 1;
        }
    }

    // Remove empty species
    species.retain(|s| !s.members.is_empty());
}

/// Run speciated GP evolution (NEAT-inspired).
pub fn speciated_evolve<R, F>(
    rng: &mut R,
    config: &SpeciationConfig,
    fitness_fn: &F,
) -> (Node, f64, Vec<GenStats>)
where
    R: Rng,
    F: Fn(&Node) -> f64,
{
    let pop_size = config.gp.population_size;

    // Initialize population
    let mut population: Vec<Node> = (0..pop_size)
        .map(|_| Node::random(rng, config.gp.max_depth, config.gp.num_vars))
        .collect();
    let mut fitnesses: Vec<f64> = population.iter().map(fitness_fn).collect();

    let mut species_list: Vec<Species> = Vec::new();
    let mut next_species_id = 0;
    let mut stats = Vec::new();

    let mut global_best = population[0].clone();
    let mut global_best_fit = fitnesses[0];

    for gen in 0..config.gp.max_generations {
        // Assign to species
        assign_species(
            &population,
            &mut species_list,
            config.compatibility_threshold,
            &mut next_species_id,
        );

        // Update species fitness and stagnation
        for s in &mut species_list {
            let species_best = s
                .members
                .iter()
                .map(|&i| fitnesses[i])
                .fold(f64::INFINITY, f64::min);

            if species_best < s.best_fitness - 1e-10 {
                s.best_fitness = species_best;
                s.stagnation = 0;
            } else {
                s.stagnation += 1;
            }

            // Update representative to a random member
            if !s.members.is_empty() {
                let rep_idx = s.members[rng.gen_range(0..s.members.len())];
                s.representative = population[rep_idx].clone();
            }
        }

        // Compute adjusted fitness (fitness sharing within species)
        let mut adjusted_fitnesses = fitnesses.clone();
        for s in &species_list {
            let count = s.members.len() as f64;
            let stagnation_penalty = if s.stagnation > config.max_stagnation {
                3.0 // harsh penalty for stagnant species
            } else {
                1.0
            };
            for &idx in &s.members {
                adjusted_fitnesses[idx] = fitnesses[idx] * count * stagnation_penalty;
            }
        }

        // Stats
        let gen_best_idx = fitnesses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if fitnesses[gen_best_idx] < global_best_fit {
            global_best_fit = fitnesses[gen_best_idx];
            global_best = population[gen_best_idx].clone();
        }

        let avg_fit = fitnesses.iter().sum::<f64>() / fitnesses.len().max(1) as f64;
        let avg_size = population.iter().map(|g| g.size() as f64).sum::<f64>()
            / population.len().max(1) as f64;

        stats.push(GenStats {
            generation: gen,
            best_fitness: fitnesses[gen_best_idx],
            avg_fitness: avg_fit,
            avg_size,
            best_program: population[gen_best_idx].to_expr(),
        });

        if fitnesses[gen_best_idx] < 1e-10 {
            break;
        }

        // Allocate offspring per species proportional to inverse adjusted fitness
        let total_inv_fitness: f64 = species_list
            .iter()
            .map(|s| {
                let avg = s
                    .members
                    .iter()
                    .map(|&i| adjusted_fitnesses[i])
                    .sum::<f64>()
                    / s.members.len().max(1) as f64;
                1.0 / (avg + 1e-10)
            })
            .sum();

        let mut offspring_counts: Vec<usize> = species_list
            .iter()
            .map(|s| {
                let avg = s
                    .members
                    .iter()
                    .map(|&i| adjusted_fitnesses[i])
                    .sum::<f64>()
                    / s.members.len().max(1) as f64;
                let share = (1.0 / (avg + 1e-10)) / total_inv_fitness;
                (share * pop_size as f64).round() as usize
            })
            .collect();

        // Ensure at least min_species_size for non-stagnant species
        for (i, s) in species_list.iter().enumerate() {
            if s.stagnation <= config.max_stagnation
                && offspring_counts[i] < config.min_species_size
            {
                offspring_counts[i] = config.min_species_size;
            }
        }

        // Normalize to pop_size
        let total: usize = offspring_counts.iter().sum();
        if total > pop_size {
            // Proportionally reduce
            let scale = pop_size as f64 / total as f64;
            for c in &mut offspring_counts {
                *c = (*c as f64 * scale).ceil() as usize;
            }
        }

        // Produce offspring
        let mut new_population = Vec::with_capacity(pop_size);

        for (species_idx, s) in species_list.iter().enumerate() {
            if s.members.is_empty() {
                continue;
            }

            // Sort species members by fitness
            let mut sorted_members: Vec<usize> = s.members.clone();
            sorted_members.sort_by(|&a, &b| {
                fitnesses[a]
                    .partial_cmp(&fitnesses[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Keep best as elite
            if !sorted_members.is_empty() {
                new_population.push(population[sorted_members[0]].clone());
            }

            let target = offspring_counts[species_idx].saturating_sub(1);
            let species_pop: Vec<(Node, f64)> = sorted_members
                .iter()
                .map(|&i| (population[i].clone(), fitnesses[i]))
                .collect();

            for _ in 0..target {
                if new_population.len() >= pop_size {
                    break;
                }
                let r: f64 = rng.gen();
                let child = if r < config.gp.crossover_rate && species_pop.len() >= 2 {
                    let p1 = tournament_select(
                        rng,
                        &species_pop,
                        config.gp.tournament_size.min(species_pop.len()),
                    );
                    let p2 = tournament_select(
                        rng,
                        &species_pop,
                        config.gp.tournament_size.min(species_pop.len()),
                    );
                    crossover(rng, p1, p2).0
                } else if r < config.gp.crossover_rate + config.gp.mutation_rate {
                    let p = tournament_select(
                        rng,
                        &species_pop,
                        config.gp.tournament_size.min(species_pop.len()),
                    );
                    match rng.gen_range(0..3) {
                        0 => mutate_point(rng, p, config.gp.num_vars),
                        1 => mutate_hoist(rng, p),
                        _ => mutate_shrink(rng, p, config.gp.num_vars),
                    }
                } else {
                    tournament_select(
                        rng,
                        &species_pop,
                        config.gp.tournament_size.min(species_pop.len()),
                    )
                    .clone()
                };

                if child.size() <= config.gp.max_tree_size {
                    new_population.push(child);
                }
            }
        }

        // Fill remaining with random if needed
        while new_population.len() < pop_size {
            new_population.push(Node::random(rng, config.gp.max_depth, config.gp.num_vars));
        }
        new_population.truncate(pop_size);

        population = new_population;
        fitnesses = population.iter().map(fitness_fn).collect();
    }

    (global_best, global_best_fit, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_genome_distance_identical() {
        let a = Node::Var(0);
        let d = genome_distance(&a, &a);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_genome_distance_different() {
        let a = Node::Var(0);
        let b = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        let d = genome_distance(&a, &b);
        assert!(d > 0.0);
    }

    #[test]
    fn test_structural_diff() {
        let a = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        let b = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        assert!(structural_diff(&a, &b).abs() < 1e-10);

        let c = Node::BinOp(
            BinOp::Mul,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        assert!(structural_diff(&a, &c) > 0.0);
    }

    #[test]
    fn test_speciated_evolve() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = SpeciationConfig {
            gp: GpConfig {
                population_size: 60,
                max_generations: 20,
                max_depth: 4,
                num_vars: 1,
                ..GpConfig::default()
            },
            compatibility_threshold: 5.0,
            max_stagnation: 10,
            min_species_size: 3,
        };

        // Target: x * 2
        let fitness = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            let mut error = 0.0;
            for i in -5..=5 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - 2.0 * x;
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error / 11.0
        };

        let (best, best_fit, stats) = speciated_evolve(&mut rng, &config, &fitness);
        assert!(!stats.is_empty());
        assert!(best_fit < 1e6);
        assert!(best.size() > 0);
    }

    #[test]
    fn test_species_assignment() {
        let pop = vec![
            Node::IntConst(1),
            Node::IntConst(2),
            Node::BinOp(
                BinOp::Add,
                Box::new(Node::Var(0)),
                Box::new(Node::BinOp(
                    BinOp::Mul,
                    Box::new(Node::Var(0)),
                    Box::new(Node::IntConst(3)),
                )),
            ),
        ];
        let mut species = Vec::new();
        let mut next_id = 0;
        assign_species(&pop, &mut species, 2.0, &mut next_id);
        // Should have at least 1 species
        assert!(!species.is_empty());
        // All individuals should be assigned
        let total_assigned: usize = species.iter().map(|s| s.members.len()).sum();
        assert_eq!(total_assigned, 3);
    }
}
