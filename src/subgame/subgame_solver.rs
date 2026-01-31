//! Subgame solving for refined Turn/River strategies.
//!
//! A subgame is a portion of the game tree starting at a street transition
//! (Turn or River). Subgames are solved with full precision (no card abstraction)
//! using the boundary data from the blueprint to initialize ranges and constraints.
//!
//! # Safe vs Unsafe Subgame Solving
//!
//! - **Unsafe**: Standard CFR from the boundary. May allow exploitation at boundaries.
//! - **Safe**: Enforces EV floor guarantees from the blueprint using the "gift" mechanism.
//!   This ensures no strategy in the refined subgame can be exploited at the boundary.

use crate::subgame::boundary::BoundaryData;
use crate::Card;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Configuration for subgame solving.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SubgameConfig {
    /// Max iterations for the subgame (safety limit).
    pub iterations: u32,

    /// Delta threshold for convergence (fraction of pot).
    /// Solver stops when |delta| < threshold for `delta_patience` iterations.
    pub delta_threshold: f32,

    /// Number of consecutive iterations where |delta| < threshold before stopping.
    pub delta_patience: u32,

    /// Enable value compression (i16 storage).
    pub enable_compression: bool,

    /// Use safe solving with gift mechanism.
    pub use_safe_solving: bool,

    /// Safety margin for gift mechanism (typically 0.0 to 0.05).
    /// Larger values are more conservative but may slightly reduce EV.
    pub safety_margin: f32,

    /// Whether to print progress during solving.
    pub print_progress: bool,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            iterations: 500,
            delta_threshold: 0.001, // 0.1% of pot
            delta_patience: 3,
            enable_compression: true,
            use_safe_solving: true,
            safety_margin: 0.02,
            print_progress: false,
        }
    }
}

impl SubgameConfig {
    /// Create a fast config for testing.
    pub fn fast() -> Self {
        Self {
            iterations: 100,
            delta_threshold: 0.01, // 1% of pot
            delta_patience: 3,
            enable_compression: false,
            use_safe_solving: false,
            safety_margin: 0.0,
            print_progress: false,
        }
    }

    /// Create a high-precision config.
    pub fn high_precision() -> Self {
        Self {
            iterations: 1000,
            delta_threshold: 0.0005, // 0.05% of pot
            delta_patience: 3,
            enable_compression: true,
            use_safe_solving: true,
            safety_margin: 0.01,
            print_progress: true,
        }
    }
}

/// A subgame for a specific Turn/River runout.
///
/// Contains all the information needed to solve and store a refined
/// strategy for a specific portion of the game tree.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SubgameInfo {
    /// Boundary index this subgame was initialized from.
    pub boundary_idx: usize,

    /// Turn card for this subgame.
    pub turn: Card,

    /// River card (None if Turn-only subgame, e.g., Turn→River).
    pub river: Option<Card>,

    /// Board cards (flop + turn + optional river).
    pub board: Vec<Card>,

    /// Initial pot size.
    pub pot: i32,

    /// Initial stack size.
    pub stack: i32,

    /// Initial ranges [OOP, IP].
    pub ranges: [Vec<f32>; 2],

    /// Whether this subgame has been solved.
    pub is_solved: bool,

    /// Final exploitability achieved.
    pub exploitability: f32,
}

impl SubgameInfo {
    /// Create a subgame info from boundary data.
    pub fn from_boundary(
        boundary: &BoundaryData,
        boundary_idx: usize,
        flop: &[Card; 3],
        turn: Card,
        river: Option<Card>,
    ) -> Self {
        let mut board = Vec::with_capacity(5);
        board.extend_from_slice(flop);
        board.push(turn);
        if let Some(r) = river {
            board.push(r);
        }

        Self {
            boundary_idx,
            turn,
            river,
            board,
            pot: boundary.pot,
            stack: boundary.stack,
            ranges: boundary.ranges.clone(),
            is_solved: false,
            exploitability: f32::MAX,
        }
    }

    /// Check if this is a Turn subgame (includes Turn→River solving).
    #[inline]
    pub fn is_turn_subgame(&self) -> bool {
        self.river.is_none()
    }

    /// Check if this is a River subgame (River only).
    #[inline]
    pub fn is_river_subgame(&self) -> bool {
        self.river.is_some()
    }

    /// Get the board as a slice.
    pub fn board(&self) -> &[Card] {
        &self.board
    }

    /// Get the current street (0 = Flop, 1 = Turn, 2 = River).
    pub fn street(&self) -> u8 {
        if self.river.is_some() {
            2
        } else {
            1
        }
    }

    /// Mark this subgame as solved with the given exploitability.
    pub fn mark_solved(&mut self, exploitability: f32) {
        self.is_solved = true;
        self.exploitability = exploitability;
    }
}

/// Result of a subgame solve operation.
#[derive(Clone, Debug)]
pub struct SubgameSolveResult {
    /// Whether solving succeeded.
    pub success: bool,

    /// Final exploitability achieved.
    pub exploitability: f32,

    /// Number of iterations performed.
    pub iterations_performed: u32,

    /// Whether safety constraints were satisfied.
    pub safety_satisfied: bool,

    /// Any warning or error message.
    pub message: Option<String>,
}

impl SubgameSolveResult {
    /// Create a successful result.
    pub fn success(exploitability: f32, iterations: u32) -> Self {
        Self {
            success: true,
            exploitability,
            iterations_performed: iterations,
            safety_satisfied: true,
            message: None,
        }
    }

    /// Create a failed result.
    pub fn failure(message: &str) -> Self {
        Self {
            success: false,
            exploitability: f32::MAX,
            iterations_performed: 0,
            safety_satisfied: false,
            message: Some(message.to_string()),
        }
    }
}

/// Batch subgame solving statistics.
#[derive(Clone, Debug, Default)]
pub struct BatchSolveStats {
    /// Total subgames attempted.
    pub total: usize,

    /// Successfully solved.
    pub succeeded: usize,

    /// Failed to solve.
    pub failed: usize,

    /// Average exploitability of successful solves.
    pub avg_exploitability: f32,

    /// Maximum exploitability among successful solves.
    pub max_exploitability: f32,

    /// Total solving time in seconds.
    pub total_time_secs: f64,
}

impl BatchSolveStats {
    /// Create a new stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a solve result.
    pub fn record(&mut self, result: &SubgameSolveResult) {
        self.total += 1;
        if result.success {
            self.succeeded += 1;
            let n = self.succeeded as f32;
            self.avg_exploitability =
                (self.avg_exploitability * (n - 1.0) + result.exploitability) / n;
            self.max_exploitability = self.max_exploitability.max(result.exploitability);
        } else {
            self.failed += 1;
        }
    }

    /// Set the total time.
    pub fn set_time(&mut self, secs: f64) {
        self.total_time_secs = secs;
    }
}

impl std::fmt::Display for BatchSolveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchSolve[{}/{} ok, avg={:.4}, max={:.4}, time={:.1}s]",
            self.succeeded,
            self.total,
            self.avg_exploitability,
            self.max_exploitability,
            self.total_time_secs
        )
    }
}

/// Generate all subgame infos for a given flop and boundary.
pub fn generate_subgame_infos(
    boundary: &BoundaryData,
    boundary_idx: usize,
    flop: &[Card; 3],
) -> Vec<SubgameInfo> {
    let mut infos = Vec::new();

    // Generate Turn subgames
    for turn in 0..52u8 {
        if flop.contains(&turn) {
            continue;
        }

        // Turn subgame (includes all river runouts)
        let info = SubgameInfo::from_boundary(boundary, boundary_idx, flop, turn, None);
        infos.push(info);
    }

    infos
}

/// Generate all (turn, river) subgame infos for a given flop and boundary.
pub fn generate_river_subgame_infos(
    boundary: &BoundaryData,
    boundary_idx: usize,
    flop: &[Card; 3],
) -> Vec<SubgameInfo> {
    let mut infos = Vec::new();

    for turn in 0..52u8 {
        if flop.contains(&turn) {
            continue;
        }

        for river in 0..52u8 {
            if flop.contains(&river) || river == turn {
                continue;
            }

            let info =
                SubgameInfo::from_boundary(boundary, boundary_idx, flop, turn, Some(river));
            infos.push(info);
        }
    }

    infos
}

/// Options for batch subgame solving.
#[derive(Clone, Debug)]
pub struct BatchSolveOptions {
    /// Configuration for each subgame solve.
    pub config: SubgameConfig,

    /// Maximum number of parallel threads (0 = use all available).
    pub max_threads: usize,

    /// Whether to print progress.
    pub print_progress: bool,

    /// Progress update interval (number of subgames).
    pub progress_interval: usize,

    /// Stop on first failure.
    pub stop_on_failure: bool,
}

impl Default for BatchSolveOptions {
    fn default() -> Self {
        Self {
            config: SubgameConfig::default(),
            max_threads: 0,
            print_progress: true,
            progress_interval: 10,
            stop_on_failure: false,
        }
    }
}

impl BatchSolveOptions {
    /// Create options for fast batch solving (fewer iterations, no safe solving).
    pub fn fast() -> Self {
        Self {
            config: SubgameConfig::fast(),
            max_threads: 0,
            print_progress: false,
            progress_interval: 50,
            stop_on_failure: false,
        }
    }

    /// Create options for high-precision batch solving.
    pub fn high_precision() -> Self {
        Self {
            config: SubgameConfig::high_precision(),
            max_threads: 0,
            print_progress: true,
            progress_interval: 5,
            stop_on_failure: false,
        }
    }
}

/// Progress callback type for batch solving.
pub type ProgressCallback = Box<dyn Fn(usize, usize, &SubgameSolveResult) + Send + Sync>;

/// Batch solver for parallel subgame solving.
pub struct BatchSolver {
    options: BatchSolveOptions,
    stats: std::sync::Mutex<BatchSolveStats>,
    start_time: std::time::Instant,
}

impl BatchSolver {
    /// Create a new batch solver with default options.
    pub fn new() -> Self {
        Self::with_options(BatchSolveOptions::default())
    }

    /// Create a new batch solver with custom options.
    pub fn with_options(options: BatchSolveOptions) -> Self {
        Self {
            options,
            stats: std::sync::Mutex::new(BatchSolveStats::new()),
            start_time: std::time::Instant::now(),
        }
    }

    /// Get the current statistics.
    pub fn stats(&self) -> BatchSolveStats {
        let mut stats = self.stats.lock().unwrap().clone();
        stats.total_time_secs = self.start_time.elapsed().as_secs_f64();
        stats
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        *self.stats.lock().unwrap() = BatchSolveStats::new();
        self.start_time = std::time::Instant::now();
    }

    /// Record a solve result.
    fn record_result(&self, result: &SubgameSolveResult) {
        self.stats.lock().unwrap().record(result);
    }

    /// Solve a batch of subgames sequentially.
    pub fn solve_sequential<F>(
        &self,
        infos: &mut [SubgameInfo],
        mut solve_fn: F,
    ) -> Vec<SubgameSolveResult>
    where
        F: FnMut(&SubgameInfo, &SubgameConfig) -> SubgameSolveResult,
    {
        let total = infos.len();
        let mut results = Vec::with_capacity(total);

        for (i, info) in infos.iter_mut().enumerate() {
            let result = solve_fn(info, &self.options.config);

            if result.success {
                info.mark_solved(result.exploitability);
            }

            self.record_result(&result);

            if self.options.print_progress && (i + 1) % self.options.progress_interval == 0 {
                let stats = self.stats();
                eprintln!(
                    "Progress: {}/{} ({:.1}%) - {}",
                    i + 1,
                    total,
                    (i + 1) as f64 / total as f64 * 100.0,
                    stats
                );
            }

            if self.options.stop_on_failure && !result.success {
                results.push(result);
                break;
            }

            results.push(result);
        }

        results
    }

    /// Solve a batch of subgames in parallel using rayon.
    #[cfg(feature = "rayon")]
    pub fn solve_parallel<F>(
        &self,
        infos: &mut [SubgameInfo],
        solve_fn: F,
    ) -> Vec<SubgameSolveResult>
    where
        F: Fn(&SubgameInfo, &SubgameConfig) -> SubgameSolveResult + Send + Sync,
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total = infos.len();
        let completed = AtomicUsize::new(0);
        let config = &self.options.config;
        let print_progress = self.options.print_progress;
        let progress_interval = self.options.progress_interval;

        // Configure thread pool if max_threads is set
        let pool = if self.options.max_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.options.max_threads)
                .build()
                .ok()
        } else {
            None
        };

        let solve_one = |info: &mut SubgameInfo| -> SubgameSolveResult {
            let result = solve_fn(info, config);

            if result.success {
                info.mark_solved(result.exploitability);
            }

            self.record_result(&result);

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if print_progress && done % progress_interval == 0 {
                let stats = self.stats();
                eprintln!(
                    "Progress: {}/{} ({:.1}%) - {}",
                    done,
                    total,
                    done as f64 / total as f64 * 100.0,
                    stats
                );
            }

            result
        };

        if let Some(pool) = pool {
            pool.install(|| infos.par_iter_mut().map(solve_one).collect())
        } else {
            infos.par_iter_mut().map(solve_one).collect()
        }
    }
}

impl Default for BatchSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve all subgames for a blueprint in parallel.
///
/// This is the main entry point for batch subgame solving.
///
/// # Arguments
/// * `boundary` - The boundary data from the blueprint
/// * `flop` - The flop cards
/// * `options` - Batch solving options
/// * `solve_fn` - Function to solve a single subgame
///
/// # Returns
/// A tuple of (solved subgame infos, batch statistics)
#[cfg(feature = "rayon")]
pub fn solve_all_subgames<F>(
    boundary: &BoundaryData,
    flop: &[Card; 3],
    options: BatchSolveOptions,
    solve_fn: F,
) -> (Vec<SubgameInfo>, BatchSolveStats)
where
    F: Fn(&SubgameInfo, &SubgameConfig) -> SubgameSolveResult + Send + Sync,
{
    let mut infos = generate_subgame_infos(boundary, 0, flop);
    let solver = BatchSolver::with_options(options);
    let _results = solver.solve_parallel(&mut infos, solve_fn);
    let stats = solver.stats();
    (infos, stats)
}

/// Solve all river subgames for a blueprint in parallel.
#[cfg(feature = "rayon")]
pub fn solve_all_river_subgames<F>(
    boundary: &BoundaryData,
    flop: &[Card; 3],
    options: BatchSolveOptions,
    solve_fn: F,
) -> (Vec<SubgameInfo>, BatchSolveStats)
where
    F: Fn(&SubgameInfo, &SubgameConfig) -> SubgameSolveResult + Send + Sync,
{
    let mut infos = generate_river_subgame_infos(boundary, 0, flop);
    let solver = BatchSolver::with_options(options);
    let _results = solver.solve_parallel(&mut infos, solve_fn);
    let stats = solver.stats();
    (infos, stats)
}

/// Solve subgames for specific turn cards in parallel.
#[cfg(feature = "rayon")]
pub fn solve_turn_subgames<F>(
    boundary: &BoundaryData,
    flop: &[Card; 3],
    turns: &[Card],
    options: BatchSolveOptions,
    solve_fn: F,
) -> (Vec<SubgameInfo>, BatchSolveStats)
where
    F: Fn(&SubgameInfo, &SubgameConfig) -> SubgameSolveResult + Send + Sync,
{
    let mut infos: Vec<SubgameInfo> = turns
        .iter()
        .filter(|&&t| !flop.contains(&t))
        .map(|&turn| SubgameInfo::from_boundary(boundary, 0, flop, turn, None))
        .collect();

    let solver = BatchSolver::with_options(options);
    let _results = solver.solve_parallel(&mut infos, solve_fn);
    let stats = solver.stats();
    (infos, stats)
}

/// A mock solve function for testing (returns success with random exploitability).
pub fn mock_solve(_info: &SubgameInfo, config: &SubgameConfig) -> SubgameSolveResult {
    // Simulate some work
    std::thread::sleep(std::time::Duration::from_micros(100));

    // Return a successful result with exploitability based on delta threshold
    SubgameSolveResult::success(
        config.delta_threshold * 0.8, // Slightly better than threshold
        config.iterations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgame_config_default() {
        let config = SubgameConfig::default();
        assert_eq!(config.iterations, 500);
        assert!(config.use_safe_solving);
    }

    #[test]
    fn test_subgame_config_fast() {
        let config = SubgameConfig::fast();
        assert_eq!(config.iterations, 100);
        assert!(!config.use_safe_solving);
    }

    #[test]
    fn test_subgame_info_from_boundary() {
        let boundary = BoundaryData::new(
            [vec![0.5; 100], vec![0.5; 100]],
            [vec![0.0; 100], vec![0.0; 100]],
            [vec![10.0; 100], vec![10.0; 100]],
            100,
            500,
            vec![0, 1],
            Some(12),
            1,
        );

        let flop = [0, 4, 8];
        let turn = 12u8;
        let river = Some(16u8);

        let info = SubgameInfo::from_boundary(&boundary, 0, &flop, turn, river);

        assert_eq!(info.turn, 12);
        assert_eq!(info.river, Some(16));
        assert_eq!(info.pot, 100);
        assert_eq!(info.stack, 500);
        assert_eq!(info.board, vec![0, 4, 8, 12, 16]);
        assert!(info.is_river_subgame());
        assert!(!info.is_turn_subgame());
    }

    #[test]
    fn test_generate_subgame_infos() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let infos = generate_subgame_infos(&boundary, 0, &flop);

        // Should have 49 turn subgames (52 - 3 flop cards)
        assert_eq!(infos.len(), 49);

        // All should be turn subgames
        assert!(infos.iter().all(|i| i.is_turn_subgame()));
    }

    #[test]
    fn test_generate_river_subgame_infos() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let infos = generate_river_subgame_infos(&boundary, 0, &flop);

        // Should have 49 * 48 = 2352 river subgames
        assert_eq!(infos.len(), 49 * 48);

        // All should be river subgames
        assert!(infos.iter().all(|i| i.is_river_subgame()));
    }

    #[test]
    fn test_subgame_solve_result() {
        let result = SubgameSolveResult::success(0.001, 500);
        assert!(result.success);
        assert!((result.exploitability - 0.001).abs() < 0.0001);

        let result = SubgameSolveResult::failure("Test error");
        assert!(!result.success);
        assert_eq!(result.message, Some("Test error".to_string()));
    }

    #[test]
    fn test_batch_solve_stats() {
        let mut stats = BatchSolveStats::new();

        stats.record(&SubgameSolveResult::success(0.001, 500));
        stats.record(&SubgameSolveResult::success(0.002, 400));
        stats.record(&SubgameSolveResult::failure("error"));

        assert_eq!(stats.total, 3);
        assert_eq!(stats.succeeded, 2);
        assert_eq!(stats.failed, 1);
        assert!((stats.avg_exploitability - 0.0015).abs() < 0.0001);
        assert!((stats.max_exploitability - 0.002).abs() < 0.0001);
    }

    #[test]
    fn test_batch_solve_options() {
        let options = BatchSolveOptions::default();
        assert!(options.print_progress);
        assert_eq!(options.max_threads, 0);

        let fast = BatchSolveOptions::fast();
        assert!(!fast.print_progress);
        assert_eq!(fast.config.iterations, 100);

        let precise = BatchSolveOptions::high_precision();
        assert!(precise.print_progress);
        assert_eq!(precise.config.iterations, 1000);
    }

    #[test]
    fn test_batch_solver_sequential() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let mut infos: Vec<SubgameInfo> = (12..15u8)
            .map(|turn| SubgameInfo::from_boundary(&boundary, 0, &flop, turn, None))
            .collect();

        let options = BatchSolveOptions {
            print_progress: false,
            ..BatchSolveOptions::fast()
        };

        let solver = BatchSolver::with_options(options);
        let results = solver.solve_sequential(&mut infos, |_info, config| {
            SubgameSolveResult::success(config.target_exploitability * 0.9, config.iterations)
        });

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));

        let stats = solver.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.succeeded, 3);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_batch_solver_parallel() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let mut infos: Vec<SubgameInfo> = (12..22u8)
            .filter(|&t| !flop.contains(&t))
            .map(|turn| SubgameInfo::from_boundary(&boundary, 0, &flop, turn, None))
            .collect();

        let options = BatchSolveOptions {
            print_progress: false,
            ..BatchSolveOptions::fast()
        };

        let solver = BatchSolver::with_options(options);
        let results = solver.solve_parallel(&mut infos, |_info, config| {
            SubgameSolveResult::success(config.target_exploitability * 0.9, config.iterations)
        });

        assert_eq!(results.len(), infos.len());
        assert!(results.iter().all(|r| r.success));

        let stats = solver.stats();
        assert_eq!(stats.succeeded, infos.len());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_solve_all_subgames() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let options = BatchSolveOptions {
            print_progress: false,
            ..BatchSolveOptions::fast()
        };

        let (infos, stats) = solve_all_subgames(&boundary, &flop, options, mock_solve);

        assert_eq!(infos.len(), 49); // 52 - 3 flop cards
        assert_eq!(stats.total, 49);
        assert_eq!(stats.succeeded, 49);
        assert!(infos.iter().all(|i| i.is_solved));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_solve_turn_subgames() {
        let boundary = BoundaryData::new(
            [vec![0.5; 10], vec![0.5; 10]],
            [vec![0.0; 10], vec![0.0; 10]],
            [vec![10.0; 10], vec![10.0; 10]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let flop = [0, 4, 8];
        let turns = vec![12, 16, 20, 24, 28];
        let options = BatchSolveOptions {
            print_progress: false,
            ..BatchSolveOptions::fast()
        };

        let (infos, stats) = solve_turn_subgames(&boundary, &flop, &turns, options, mock_solve);

        assert_eq!(infos.len(), 5);
        assert_eq!(stats.total, 5);
        assert_eq!(stats.succeeded, 5);
    }

    #[test]
    fn test_mock_solve() {
        let boundary = BoundaryData::default();
        let info = SubgameInfo::from_boundary(&boundary, 0, &[0, 4, 8], 12, None);
        let config = SubgameConfig::fast();

        let result = mock_solve(&info, &config);

        assert!(result.success);
        assert!(result.exploitability < config.target_exploitability);
    }
}
