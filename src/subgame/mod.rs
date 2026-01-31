//! Subgame solving framework for decomposed poker solving.
//!
//! This module implements a subgame solving approach that:
//! 1. Generates a coarse "blueprint" strategy using card abstraction (EHS2)
//! 2. Refines Turn/River subgames with high precision on-demand
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Desktop Client (UNCHANGED)                        │
//! │  game_load_file() → game.play() → game.strategy()                   │
//! └──────────────────────────────┬──────────────────────────────────────┘
//!                                │ Same API
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                 PostFlopGame (Enhanced Internally)                   │
//! │  ┌─────────────────┐    ┌────────────────────────────────────────┐ │
//! │  │  Archive Reader │───▶│  Transparent Subgame Loading           │ │
//! │  │  (.pfs file)    │    │  - Flop: immediate from blueprint      │ │
//! │  └─────────────────┘    │  - Turn/River: load chunk on-demand    │ │
//! │                         └────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - `subgame`: Enable subgame solving framework (requires `bincode`)
//! - `subgame-safe`: Enable safe solving with gift mechanism

mod abstraction;
mod archive;
mod blueprint;
mod boundary;
mod integration;
mod stitching;
mod subgame_solver;

pub use abstraction::*;
pub use archive::*;
pub use blueprint::*;
pub use boundary::*;
pub use integration::*;
pub use stitching::*;
pub use subgame_solver::*;

#[cfg(feature = "rayon")]
#[allow(unused_imports)]
use rayon::prelude::*;

use crate::Card;

/// Number of valid turn cards given a flop (52 - 3 = 49)
pub const NUM_TURN_CARDS: usize = 49;

/// Number of valid river cards given a turn (52 - 4 = 48)
pub const NUM_RIVER_CARDS: usize = 48;

/// Total possible turn-river runouts: 49 * 48 = 2352
pub const NUM_RUNOUTS: usize = NUM_TURN_CARDS * NUM_RIVER_CARDS;

/// Get all valid turn cards for a given flop
#[inline]
pub fn valid_turn_cards(flop: &[Card; 3]) -> impl Iterator<Item = Card> + '_ {
    (0..52u8).filter(|c| !flop.contains(c))
}

/// Get all valid river cards for a given flop and turn
#[inline]
pub fn valid_river_cards(flop: &[Card; 3], turn: Card) -> impl Iterator<Item = Card> + '_ {
    (0..52u8).filter(move |c| !flop.contains(c) && *c != turn)
}

/// Get all (turn, river) runouts for a given flop
pub fn all_runouts(flop: &[Card; 3]) -> Vec<(Card, Card)> {
    let mut runouts = Vec::with_capacity(NUM_RUNOUTS);
    for turn in valid_turn_cards(flop) {
        for river in valid_river_cards(flop, turn) {
            runouts.push((turn, river));
        }
    }
    runouts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_turn_cards() {
        let flop = [0, 4, 8]; // 2c, 3c, 4c
        let turns: Vec<_> = valid_turn_cards(&flop).collect();
        assert_eq!(turns.len(), 49);
        assert!(!turns.contains(&0));
        assert!(!turns.contains(&4));
        assert!(!turns.contains(&8));
    }

    #[test]
    fn test_valid_river_cards() {
        let flop = [0, 4, 8];
        let turn = 12; // 5c
        let rivers: Vec<_> = valid_river_cards(&flop, turn).collect();
        assert_eq!(rivers.len(), 48);
        assert!(!rivers.contains(&0));
        assert!(!rivers.contains(&4));
        assert!(!rivers.contains(&8));
        assert!(!rivers.contains(&12));
    }

    #[test]
    fn test_all_runouts() {
        let flop = [0, 4, 8];
        let runouts = all_runouts(&flop);
        assert_eq!(runouts.len(), 49 * 48);

        // Check uniqueness
        let mut seen = std::collections::HashSet::new();
        for (t, r) in &runouts {
            assert!(seen.insert((*t, *r)), "Duplicate runout: ({}, {})", t, r);
            assert!(!flop.contains(t));
            assert!(!flop.contains(r));
            assert_ne!(t, r);
        }
    }
}
