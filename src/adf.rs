//! Automatically Defined Functions (ADFs).
//!
//! Programs can define reusable subroutines that are evolved alongside the main body.
//! Each ADF has its own argument list and body. The main program can call ADFs,
//! enabling modular, hierarchical solutions.

use crate::ast::{BinOp, Node};
use crate::interpreter::{ExecError, Interpreter, Value};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// An Automatically Defined Function with its own argument count and body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adf {
    /// Number of arguments this ADF takes.
    pub arity: usize,
    /// The function body (uses Var(0..arity) for its own arguments).
    pub body: Node,
}

/// A program with ADFs: a set of helper functions + a main result-producing branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdfProgram {
    /// Automatically defined functions, called by index.
    pub adfs: Vec<Adf>,
    /// The main result-producing branch.
    pub main_body: Node,
}

/// AST extension for ADF calls within the main body or other ADFs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdfCall {
    /// Index of the ADF to call.
    pub adf_index: usize,
    /// Arguments to pass.
    pub args: Vec<Node>,
}

impl Adf {
    /// Generate a random ADF body.
    pub fn random<R: Rng>(rng: &mut R, arity: usize, max_depth: usize) -> Self {
        Self {
            arity,
            body: Node::random(rng, max_depth, arity),
        }
    }

    /// Total size of the ADF body.
    pub fn size(&self) -> usize {
        self.body.size()
    }
}

impl AdfProgram {
    /// Generate a random ADF program.
    pub fn random<R: Rng>(
        rng: &mut R,
        num_adfs: usize,
        adf_arity: usize,
        max_depth: usize,
        num_vars: usize,
    ) -> Self {
        let adfs: Vec<Adf> = (0..num_adfs)
            .map(|_| Adf::random(rng, adf_arity, max_depth.saturating_sub(1)))
            .collect();

        // Main body can use original vars + has access to ADFs
        // We'll inject ADF calls as BinOp patterns during random gen
        let main_body = Self::random_with_adfs(rng, max_depth, num_vars, &adfs);

        Self { adfs, main_body }
    }

    /// Generate a random node that may contain ADF calls.
    /// ADF calls are encoded as: MathFn(Abs, BinOp(Add, arg1, arg2)) wouldn't work...
    /// Instead we use a simpler approach: the main body is a normal tree,
    /// and we occasionally replace subtrees with "ADF call patterns".
    fn random_with_adfs<R: Rng>(
        rng: &mut R,
        max_depth: usize,
        num_vars: usize,
        adfs: &[Adf],
    ) -> Node {
        // Generate a normal tree, then randomly replace some subtrees with ADF-call-like patterns
        let mut tree = Node::random(rng, max_depth, num_vars);
        if !adfs.is_empty() {
            Self::inject_adf_calls(rng, &mut tree, num_vars, adfs, 0.3);
        }
        tree
    }

    /// Randomly replace some subtrees with ADF call patterns.
    /// An ADF call is encoded as a specific BinOp structure that the evaluator recognizes.
    fn inject_adf_calls<R: Rng>(
        rng: &mut R,
        tree: &mut Node,
        num_vars: usize,
        adfs: &[Adf],
        probability: f64,
    ) {
        let size = tree.size();
        if size < 2 {
            return;
        }
        // Try to replace a few nodes
        let attempts = (size / 3).clamp(1, 3);
        for _ in 0..attempts {
            if rng.gen_bool(probability) {
                let adf_idx = rng.gen_range(0..adfs.len());
                let adf = &adfs[adf_idx];
                // Build ADF call: use a specific pattern BinOp(Div, IntConst(adf_idx), args...)
                let call_node = Self::build_adf_call_node(rng, adf_idx, adf.arity, num_vars);
                let point = rng.gen_range(0..size);
                tree.replace_node(point, call_node);
            }
        }
    }

    /// Build an ADF call encoded as a recognizable AST pattern.
    /// Pattern: If(BoolConst(true), IntConst(adf_idx), BinOp(Add, arg0, arg1))
    /// For single-arg ADFs: If(BoolConst(true), IntConst(adf_idx), arg0)
    fn build_adf_call_node<R: Rng>(
        rng: &mut R,
        adf_idx: usize,
        arity: usize,
        num_vars: usize,
    ) -> Node {
        let args: Vec<Node> = (0..arity).map(|_| Node::random(rng, 1, num_vars)).collect();

        // Encode args into a single tree via nested Add
        let args_tree = if args.is_empty() {
            Node::FloatConst(0.0)
        } else if args.len() == 1 {
            args.into_iter().next().unwrap()
        } else {
            let mut iter = args.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, arg| {
                Node::BinOp(BinOp::Add, Box::new(acc), Box::new(arg))
            })
        };

        // The recognizable pattern: If(BoolConst(true), IntConst(adf_idx), args_tree)
        Node::If(
            Box::new(Node::BoolConst(true)),
            Box::new(Node::IntConst(adf_idx as i64)),
            Box::new(args_tree),
        )
    }

    /// Total size of the entire program (all ADFs + main body).
    pub fn total_size(&self) -> usize {
        self.adfs.iter().map(|a| a.size()).sum::<usize>() + self.main_body.size()
    }

    /// Pretty-print the full ADF program.
    pub fn to_string_pretty(&self) -> String {
        let mut s = String::new();
        for (i, adf) in self.adfs.iter().enumerate() {
            s.push_str(&format!(
                "ADF{i}({}args) = {}\n",
                adf.arity,
                adf.body.to_expr()
            ));
        }
        s.push_str(&format!("MAIN = {}", self.main_body.to_expr()));
        s
    }
}

/// Evaluate an ADF program with given input variables.
/// When encountering an ADF call pattern If(BoolConst(true), IntConst(idx), args),
/// it evaluates the corresponding ADF body with extracted arguments.
pub fn eval_adf_program(
    interp: &mut Interpreter,
    program: &AdfProgram,
    vars: &[f64],
) -> Result<Value, ExecError> {
    eval_with_adfs(interp, &program.main_body, vars, &program.adfs, 0)
}

/// Recursive evaluator that recognizes ADF call patterns.
fn eval_with_adfs(
    interp: &mut Interpreter,
    node: &Node,
    vars: &[f64],
    adfs: &[Adf],
    call_depth: usize,
) -> Result<Value, ExecError> {
    // Prevent infinite ADF recursion
    if call_depth > 10 {
        return Ok(Value::Float(0.0));
    }

    // Check for ADF call pattern: If(BoolConst(true), IntConst(idx), args)
    if let Node::If(cond, then_branch, args_node) = node {
        if let (Node::BoolConst(true), Node::IntConst(idx)) = (cond.as_ref(), then_branch.as_ref())
        {
            let adf_idx = *idx as usize;
            if adf_idx < adfs.len() {
                let adf = &adfs[adf_idx];
                // Extract arguments from args_node
                let mut adf_args = Vec::new();
                extract_adf_args(interp, args_node, vars, adfs, call_depth, &mut adf_args)?;
                // Pad or truncate to match ADF arity
                adf_args.resize(adf.arity, 0.0);
                // Evaluate ADF body with its own arguments
                return eval_with_adfs(interp, &adf.body, &adf_args, adfs, call_depth + 1);
            }
        }
    }

    // Normal evaluation — delegate to interpreter for non-ADF nodes
    interp.eval(node, vars)
}

/// Extract argument values from the ADF call's args subtree.
fn extract_adf_args(
    interp: &mut Interpreter,
    node: &Node,
    vars: &[f64],
    adfs: &[Adf],
    call_depth: usize,
    args: &mut Vec<f64>,
) -> Result<(), ExecError> {
    // If it's a BinOp(Add, left, right), extract from both sides
    if let Node::BinOp(BinOp::Add, left, right) = node {
        extract_adf_args(interp, left, vars, adfs, call_depth, args)?;
        extract_adf_args(interp, right, vars, adfs, call_depth, args)?;
    } else {
        let val = eval_with_adfs(interp, node, vars, adfs, call_depth)?;
        args.push(val.to_f64());
    }
    Ok(())
}

/// Crossover for ADF programs: independently crossover ADFs and main body.
pub fn adf_crossover<R: Rng>(
    rng: &mut R,
    p1: &AdfProgram,
    p2: &AdfProgram,
) -> (AdfProgram, AdfProgram) {
    use crate::genetic::crossover;

    let (main1, main2) = crossover(rng, &p1.main_body, &p2.main_body);

    let adfs1: Vec<Adf> = p1
        .adfs
        .iter()
        .zip(p2.adfs.iter())
        .map(|(a1, a2)| {
            let (body, _) = crossover(rng, &a1.body, &a2.body);
            Adf {
                arity: a1.arity,
                body,
            }
        })
        .collect();

    let adfs2: Vec<Adf> = p2
        .adfs
        .iter()
        .zip(p1.adfs.iter())
        .map(|(a1, a2)| {
            let (body, _) = crossover(rng, &a1.body, &a2.body);
            Adf {
                arity: a1.arity,
                body,
            }
        })
        .collect();

    (
        AdfProgram {
            adfs: adfs1,
            main_body: main1,
        },
        AdfProgram {
            adfs: adfs2,
            main_body: main2,
        },
    )
}

/// Mutate an ADF program: randomly mutate either an ADF body or the main body.
pub fn adf_mutate<R: Rng>(rng: &mut R, program: &AdfProgram, num_vars: usize) -> AdfProgram {
    use crate::genetic::{mutate_point, mutate_shrink};

    let mut result = program.clone();
    let total_parts = result.adfs.len() + 1;
    let which = rng.gen_range(0..total_parts);

    if which < result.adfs.len() {
        // Mutate an ADF
        let adf = &mut result.adfs[which];
        if rng.gen_bool(0.5) {
            adf.body = mutate_point(rng, &adf.body, adf.arity);
        } else {
            adf.body = mutate_shrink(rng, &adf.body, adf.arity);
        }
    } else {
        // Mutate main body
        if rng.gen_bool(0.5) {
            result.main_body = mutate_point(rng, &result.main_body, num_vars);
        } else {
            result.main_body = mutate_shrink(rng, &result.main_body, num_vars);
        }
    }

    result
}

/// Run ADF-enabled GP evolution.
pub fn adf_evolve<R, F>(
    rng: &mut R,
    config: &AdfConfig,
    fitness_fn: &F,
) -> (AdfProgram, f64, Vec<crate::genetic::GenStats>)
where
    R: Rng,
    F: Fn(&AdfProgram) -> f64,
{
    let mut population: Vec<(AdfProgram, f64)> = (0..config.gp.population_size)
        .map(|_| {
            let prog = AdfProgram::random(
                rng,
                config.num_adfs,
                config.adf_arity,
                config.gp.max_depth,
                config.gp.num_vars,
            );
            let fit = fitness_fn(&prog);
            (prog, fit)
        })
        .collect();

    let mut stats = Vec::new();
    let mut best_ever: Option<(AdfProgram, f64)> = None;

    for gen in 0..config.gp.max_generations {
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_fit = population[0].1;
        let avg_fit = population.iter().map(|(_, f)| f).sum::<f64>() / population.len() as f64;
        let avg_size = population
            .iter()
            .map(|(p, _)| p.total_size() as f64)
            .sum::<f64>()
            / population.len() as f64;

        stats.push(crate::genetic::GenStats {
            generation: gen,
            best_fitness: best_fit,
            avg_fitness: avg_fit,
            avg_size,
            best_program: population[0].0.main_body.to_expr(),
        });

        // Track best ever
        if best_ever.is_none() || best_fit < best_ever.as_ref().unwrap().1 {
            best_ever = Some(population[0].clone());
        }

        if best_fit < 1e-10 {
            break;
        }

        // Next generation
        let mut next_gen = Vec::with_capacity(config.gp.population_size);

        // Elitism
        for (prog, _) in population.iter().take(config.gp.elitism) {
            next_gen.push(prog.clone());
        }

        while next_gen.len() < config.gp.population_size {
            let r: f64 = rng.gen();
            let child = if r < config.gp.crossover_rate {
                let i1 = tournament_idx(rng, &population, config.gp.tournament_size);
                let i2 = tournament_idx(rng, &population, config.gp.tournament_size);
                let (c1, _) = adf_crossover(rng, &population[i1].0, &population[i2].0);
                c1
            } else if r < config.gp.crossover_rate + config.gp.mutation_rate {
                let i = tournament_idx(rng, &population, config.gp.tournament_size);
                adf_mutate(rng, &population[i].0, config.gp.num_vars)
            } else {
                let i = tournament_idx(rng, &population, config.gp.tournament_size);
                population[i].0.clone()
            };

            // Bloat control
            if child.total_size() <= config.gp.max_tree_size * (config.num_adfs + 1) {
                next_gen.push(child);
            } else {
                let i = tournament_idx(rng, &population, config.gp.tournament_size);
                next_gen.push(population[i].0.clone());
            }
        }

        population = next_gen
            .into_iter()
            .map(|prog| {
                let fit = fitness_fn(&prog);
                (prog, fit)
            })
            .collect();
    }

    let (best_prog, best_fit) = best_ever.unwrap_or_else(|| population[0].clone());
    (best_prog, best_fit, stats)
}

/// Tournament selection returning index.
fn tournament_idx<R: Rng, T>(rng: &mut R, population: &[(T, f64)], size: usize) -> usize {
    let mut best = rng.gen_range(0..population.len());
    for _ in 1..size {
        let idx = rng.gen_range(0..population.len());
        if population[idx].1 < population[best].1 {
            best = idx;
        }
    }
    best
}

/// Configuration for ADF evolution.
#[derive(Debug, Clone)]
pub struct AdfConfig {
    pub gp: crate::genetic::GpConfig,
    pub num_adfs: usize,
    pub adf_arity: usize,
}

impl Default for AdfConfig {
    fn default() -> Self {
        Self {
            gp: crate::genetic::GpConfig::default(),
            num_adfs: 2,
            adf_arity: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_adf_program_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let prog = AdfProgram::random(&mut rng, 2, 2, 4, 1);
        assert_eq!(prog.adfs.len(), 2);
        assert!(prog.total_size() > 0);
    }

    #[test]
    fn test_adf_program_display() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let prog = AdfProgram::random(&mut rng, 2, 2, 3, 1);
        let s = prog.to_string_pretty();
        assert!(s.contains("ADF0"));
        assert!(s.contains("MAIN"));
    }

    #[test]
    fn test_adf_eval_basic() {
        // ADF0(a, b) = a + b, main = ADF0(x0, 1.0)
        let prog = AdfProgram {
            adfs: vec![Adf {
                arity: 2,
                body: Node::BinOp(BinOp::Add, Box::new(Node::Var(0)), Box::new(Node::Var(1))),
            }],
            main_body: Node::If(
                Box::new(Node::BoolConst(true)),
                Box::new(Node::IntConst(0)), // ADF index 0
                Box::new(Node::BinOp(
                    BinOp::Add,
                    Box::new(Node::Var(0)),          // arg0 = x0
                    Box::new(Node::FloatConst(1.0)), // arg1 = 1.0
                )),
            ),
        };

        let mut interp = Interpreter::default();
        let result = eval_adf_program(&mut interp, &prog, &[5.0]).unwrap();
        assert!((result.to_f64() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_adf_crossover_and_mutate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p1 = AdfProgram::random(&mut rng, 2, 2, 3, 1);
        let p2 = AdfProgram::random(&mut rng, 2, 2, 3, 1);

        let (c1, c2) = adf_crossover(&mut rng, &p1, &p2);
        assert_eq!(c1.adfs.len(), 2);
        assert_eq!(c2.adfs.len(), 2);

        let m = adf_mutate(&mut rng, &p1, 1);
        assert_eq!(m.adfs.len(), 2);
    }

    #[test]
    fn test_adf_evolution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = AdfConfig {
            gp: crate::genetic::GpConfig {
                population_size: 50,
                max_generations: 10,
                max_depth: 4,
                num_vars: 1,
                ..crate::genetic::GpConfig::default()
            },
            num_adfs: 1,
            adf_arity: 2,
        };

        let fitness = |prog: &AdfProgram| -> f64 {
            let mut interp = Interpreter::default();
            let mut error = 0.0;
            for i in -3..=3 {
                let x = i as f64;
                interp.reset();
                match eval_adf_program(&mut interp, prog, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - (x * 2.0);
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error / 7.0
        };

        let (_best, best_fit, stats) = adf_evolve(&mut rng, &config, &fitness);
        assert!(!stats.is_empty());
        assert!(best_fit < 1e6);
    }
}
