//! Boundary data for subgame solving.
//!
//! This module stores boundary information at street transition points
//! (Flop→Turn, Turn→River). This data is used to initialize subgames
//! with correct ranges and expected values from the blueprint solution.
//!
//! # Boundary Data
//!
//! At each street transition, we store:
//! - Reaching probabilities (ranges) for each player
//! - Counterfactual values for safe subgame solving
//! - Expected values for EV floor guarantees
//! - Pot and stack sizes
//! - Action history to reach this point

use crate::Card;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Boundary data at a single street transition point.
///
/// This captures all the information needed to initialize a subgame
/// rooted at this position in the game tree.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct BoundaryData {
    /// Reaching probability for each hand (range).
    /// Index 0 = OOP, Index 1 = IP.
    pub ranges: [Vec<f32>; 2],

    /// Counterfactual values at this node (used for safe solving).
    /// These are the values each hand would achieve if it reached this node.
    pub cfvalues: [Vec<f32>; 2],

    /// Expected values per hand weighted by opponent's range.
    /// Used to set EV floor guarantees in safe subgame solving.
    pub expected_values: [Vec<f32>; 2],

    /// Current pot size (in chips).
    pub pot: i32,

    /// Remaining effective stack.
    pub stack: i32,

    /// Action history to reach this point (sequence of action indices).
    pub action_history: Vec<u16>,

    /// Turn card at this boundary (if Turn→River transition).
    pub turn: Option<Card>,

    /// Board state: 0 = Flop→Turn, 1 = Turn→River.
    pub street: u8,
}

impl Default for BoundaryData {
    fn default() -> Self {
        Self {
            ranges: [Vec::new(), Vec::new()],
            cfvalues: [Vec::new(), Vec::new()],
            expected_values: [Vec::new(), Vec::new()],
            pot: 0,
            stack: 0,
            action_history: Vec::new(),
            turn: None,
            street: 0,
        }
    }
}

impl BoundaryData {
    /// Create a new boundary data with the given parameters.
    pub fn new(
        ranges: [Vec<f32>; 2],
        cfvalues: [Vec<f32>; 2],
        expected_values: [Vec<f32>; 2],
        pot: i32,
        stack: i32,
        action_history: Vec<u16>,
        turn: Option<Card>,
        street: u8,
    ) -> Self {
        Self {
            ranges,
            cfvalues,
            expected_values,
            pot,
            stack,
            action_history,
            turn,
            street,
        }
    }

    /// Check if this boundary has valid data.
    pub fn is_valid(&self) -> bool {
        !self.ranges[0].is_empty() && !self.ranges[1].is_empty()
    }

    /// Get the range for a specific player.
    #[inline]
    pub fn range(&self, player: usize) -> &[f32] {
        &self.ranges[player]
    }

    /// Get the counterfactual values for a specific player.
    #[inline]
    pub fn cfvalue(&self, player: usize) -> &[f32] {
        &self.cfvalues[player]
    }

    /// Get the expected values for a specific player.
    #[inline]
    pub fn expected_value(&self, player: usize) -> &[f32] {
        &self.expected_values[player]
    }

    /// Compute the total reaching probability (sum of range).
    pub fn total_reach(&self, player: usize) -> f32 {
        self.ranges[player].iter().sum()
    }

    /// Compute the average expected value weighted by range.
    pub fn average_ev(&self, player: usize) -> f32 {
        let range = &self.ranges[player];
        let ev = &self.expected_values[player];

        if range.is_empty() {
            return 0.0;
        }

        let total_reach: f32 = range.iter().sum();
        if total_reach <= 0.0 {
            return 0.0;
        }

        let weighted_ev: f32 = range.iter().zip(ev.iter()).map(|(r, e)| r * e).sum();
        weighted_ev / total_reach
    }
}

/// Store for all boundary data extracted from a solved game.
///
/// Boundaries are indexed by:
/// - Flop→Turn: linear index based on action history hash
/// - Turn→River: (parent flop boundary, turn bucket)
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct BoundaryStore {
    /// Boundaries at Flop→Turn transitions.
    /// Each boundary corresponds to a unique action history from root.
    pub flop_to_turn: Vec<BoundaryData>,

    /// Boundaries at Turn→River transitions.
    /// Outer index: flop boundary index
    /// Inner index: turn bucket (from abstraction)
    pub turn_to_river: Vec<Vec<BoundaryData>>,

    /// Number of OOP hands (for validation).
    num_oop_hands: usize,

    /// Number of IP hands (for validation).
    num_ip_hands: usize,
}

impl BoundaryStore {
    /// Create a new empty boundary store.
    pub fn new(num_oop_hands: usize, num_ip_hands: usize) -> Self {
        Self {
            flop_to_turn: Vec::new(),
            turn_to_river: Vec::new(),
            num_oop_hands,
            num_ip_hands,
        }
    }

    /// Add a Flop→Turn boundary.
    pub fn add_flop_boundary(&mut self, boundary: BoundaryData) -> usize {
        let idx = self.flop_to_turn.len();
        self.flop_to_turn.push(boundary);
        // Initialize turn→river storage for this flop boundary
        self.turn_to_river.push(Vec::new());
        idx
    }

    /// Add a Turn→River boundary.
    pub fn add_turn_boundary(&mut self, flop_idx: usize, boundary: BoundaryData) -> usize {
        while self.turn_to_river.len() <= flop_idx {
            self.turn_to_river.push(Vec::new());
        }
        let idx = self.turn_to_river[flop_idx].len();
        self.turn_to_river[flop_idx].push(boundary);
        idx
    }

    /// Get a Flop→Turn boundary by index.
    pub fn get_flop_boundary(&self, idx: usize) -> Option<&BoundaryData> {
        self.flop_to_turn.get(idx)
    }

    /// Get a Turn→River boundary by indices.
    pub fn get_turn_boundary(&self, flop_idx: usize, turn_idx: usize) -> Option<&BoundaryData> {
        self.turn_to_river
            .get(flop_idx)
            .and_then(|v| v.get(turn_idx))
    }

    /// Get all Flop→Turn boundaries.
    pub fn flop_boundaries(&self) -> &[BoundaryData] {
        &self.flop_to_turn
    }

    /// Get Turn→River boundaries for a specific flop boundary.
    pub fn turn_boundaries(&self, flop_idx: usize) -> Option<&[BoundaryData]> {
        self.turn_to_river.get(flop_idx).map(|v| v.as_slice())
    }

    /// Get the total number of boundaries.
    pub fn total_boundaries(&self) -> usize {
        self.flop_to_turn.len()
            + self
                .turn_to_river
                .iter()
                .map(|v| v.len())
                .sum::<usize>()
    }

    /// Get the number of Flop→Turn boundaries.
    pub fn num_flop_boundaries(&self) -> usize {
        self.flop_to_turn.len()
    }

    /// Get the number of Turn→River boundaries for a specific flop boundary.
    pub fn num_turn_boundaries(&self, flop_idx: usize) -> usize {
        self.turn_to_river
            .get(flop_idx)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.flop_to_turn.is_empty()
    }

    /// Clear all boundaries.
    pub fn clear(&mut self) {
        self.flop_to_turn.clear();
        self.turn_to_river.clear();
    }

    /// Validate boundary data consistency.
    pub fn validate(&self) -> Result<(), String> {
        for (i, boundary) in self.flop_to_turn.iter().enumerate() {
            if boundary.ranges[0].len() != self.num_oop_hands {
                return Err(format!(
                    "Flop boundary {} has wrong OOP range size: {} vs {}",
                    i,
                    boundary.ranges[0].len(),
                    self.num_oop_hands
                ));
            }
            if boundary.ranges[1].len() != self.num_ip_hands {
                return Err(format!(
                    "Flop boundary {} has wrong IP range size: {} vs {}",
                    i,
                    boundary.ranges[1].len(),
                    self.num_ip_hands
                ));
            }
        }
        Ok(())
    }
}

/// Safety constraint for safe subgame solving.
///
/// This ensures that the refined subgame solution does not allow
/// exploitation at the boundary - each hand must achieve at least
/// its blueprint expected value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SafetyConstraint {
    /// Minimum EV each hand must achieve (from blueprint).
    pub min_ev_guarantee: [Vec<f32>; 2],

    /// Minimum reach probability to prevent over-folding.
    pub reach_floor: f32,
}

impl Default for SafetyConstraint {
    fn default() -> Self {
        Self {
            min_ev_guarantee: [Vec::new(), Vec::new()],
            reach_floor: 0.0,
        }
    }
}

impl SafetyConstraint {
    /// Create a safety constraint from boundary data.
    pub fn from_boundary(boundary: &BoundaryData, safety_margin: f32) -> Self {
        Self {
            min_ev_guarantee: [
                boundary.expected_values[0].clone(),
                boundary.expected_values[1].clone(),
            ],
            reach_floor: safety_margin,
        }
    }

    /// Check if a hand's EV violates the safety constraint.
    #[inline]
    pub fn is_violated(&self, player: usize, hand_idx: usize, ev: f32) -> bool {
        if hand_idx >= self.min_ev_guarantee[player].len() {
            return false;
        }
        let min_ev = self.min_ev_guarantee[player][hand_idx];
        ev < min_ev - self.reach_floor
    }

    /// Get the deficit (amount below minimum EV).
    #[inline]
    pub fn deficit(&self, player: usize, hand_idx: usize, ev: f32) -> f32 {
        if hand_idx >= self.min_ev_guarantee[player].len() {
            return 0.0;
        }
        let min_ev = self.min_ev_guarantee[player][hand_idx];
        (min_ev - ev).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_data_default() {
        let boundary = BoundaryData::default();
        assert!(!boundary.is_valid());
        assert_eq!(boundary.pot, 0);
        assert_eq!(boundary.stack, 0);
    }

    #[test]
    fn test_boundary_data_new() {
        let ranges = [vec![0.5, 0.5], vec![0.3, 0.7]];
        let cfvalues = [vec![10.0, 20.0], vec![15.0, 25.0]];
        let expected_values = [vec![12.0, 18.0], vec![14.0, 22.0]];

        let boundary = BoundaryData::new(
            ranges.clone(),
            cfvalues.clone(),
            expected_values.clone(),
            100,
            500,
            vec![0, 1, 2],
            Some(12),
            1,
        );

        assert!(boundary.is_valid());
        assert_eq!(boundary.pot, 100);
        assert_eq!(boundary.stack, 500);
        assert_eq!(boundary.action_history, vec![0, 1, 2]);
        assert_eq!(boundary.turn, Some(12));
        assert_eq!(boundary.street, 1);
    }

    #[test]
    fn test_boundary_store_operations() {
        let mut store = BoundaryStore::new(2, 2);
        assert!(store.is_empty());

        // Add a flop boundary
        let flop_boundary = BoundaryData::new(
            [vec![0.5, 0.5], vec![0.5, 0.5]],
            [vec![10.0, 10.0], vec![10.0, 10.0]],
            [vec![10.0, 10.0], vec![10.0, 10.0]],
            100,
            500,
            vec![0],
            None,
            0,
        );
        let flop_idx = store.add_flop_boundary(flop_boundary);
        assert_eq!(flop_idx, 0);
        assert_eq!(store.num_flop_boundaries(), 1);

        // Add a turn boundary
        let turn_boundary = BoundaryData::new(
            [vec![0.4, 0.6], vec![0.6, 0.4]],
            [vec![15.0, 15.0], vec![15.0, 15.0]],
            [vec![15.0, 15.0], vec![15.0, 15.0]],
            150,
            450,
            vec![0, 1],
            Some(12),
            1,
        );
        let turn_idx = store.add_turn_boundary(flop_idx, turn_boundary);
        assert_eq!(turn_idx, 0);
        assert_eq!(store.num_turn_boundaries(flop_idx), 1);

        // Retrieve boundaries
        assert!(store.get_flop_boundary(0).is_some());
        assert!(store.get_turn_boundary(0, 0).is_some());
        assert!(store.get_turn_boundary(0, 1).is_none());
    }

    #[test]
    fn test_safety_constraint() {
        let boundary = BoundaryData::new(
            [vec![0.5, 0.5], vec![0.5, 0.5]],
            [vec![10.0, 20.0], vec![15.0, 25.0]],
            [vec![12.0, 18.0], vec![14.0, 22.0]],
            100,
            500,
            vec![],
            None,
            0,
        );

        let constraint = SafetyConstraint::from_boundary(&boundary, 0.5);

        // Check violation detection
        assert!(constraint.is_violated(0, 0, 11.0)); // 11 < 12 - 0.5
        assert!(!constraint.is_violated(0, 0, 12.0)); // 12 >= 12 - 0.5

        // Check deficit calculation
        assert!((constraint.deficit(0, 0, 10.0) - 2.0).abs() < 0.001);
        assert!(constraint.deficit(0, 0, 15.0).abs() < 0.001);
    }

    #[test]
    fn test_average_ev() {
        let boundary = BoundaryData::new(
            [vec![0.25, 0.75], vec![0.5, 0.5]],
            [vec![0.0, 0.0], vec![0.0, 0.0]],
            [vec![10.0, 20.0], vec![15.0, 25.0]],
            100,
            500,
            vec![],
            None,
            0,
        );

        // OOP: (0.25 * 10 + 0.75 * 20) / 1.0 = (2.5 + 15) / 1.0 = 17.5
        let avg_ev_oop = boundary.average_ev(0);
        assert!((avg_ev_oop - 17.5).abs() < 0.001);

        // IP: (0.5 * 15 + 0.5 * 25) / 1.0 = (7.5 + 12.5) / 1.0 = 20.0
        let avg_ev_ip = boundary.average_ev(1);
        assert!((avg_ev_ip - 20.0).abs() < 0.001);
    }
}
