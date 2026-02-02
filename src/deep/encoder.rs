//! Information set encoder for converting game states to neural network inputs.

// Card utilities from parent crate used for encoding

/// Dimension of encoded information set tensor.
/// - Board cards: 5 * 52 = 260 (one-hot, padded for flop/turn)
/// - Hole cards: 2 * 52 = 104 (one-hot)
/// - Pot ratio: 1 (normalized)
/// - Stack ratio: 1 (normalized)
/// - Street indicator: 3 (one-hot: flop/turn/river)
/// - Actions taken: varies based on max_actions setting
pub const BASE_ENCODING_DIM: usize = 260 + 104 + 1 + 1 + 3;

/// Maximum number of actions to encode in history
pub const MAX_ACTION_HISTORY: usize = 10;

/// Dimension per action in history (action type + size)
pub const ACTION_ENCODING_DIM: usize = 5; // fold, check, call, bet/raise, size

/// Total encoding dimension
pub const ENCODING_DIM: usize = BASE_ENCODING_DIM + MAX_ACTION_HISTORY * ACTION_ENCODING_DIM;

/// Encodes game state into a fixed-size tensor for neural network input.
#[derive(Clone, Debug)]
pub struct InfoSetEncoder {
    /// Starting pot size for normalization
    starting_pot: f32,
    /// Effective stack for normalization
    effective_stack: f32,
}

impl InfoSetEncoder {
    /// Create a new encoder with game parameters for normalization.
    pub fn new(starting_pot: i32, effective_stack: i32) -> Self {
        Self {
            starting_pot: starting_pot as f32,
            effective_stack: effective_stack as f32,
        }
    }

    /// Get the output dimension of the encoder.
    pub fn dim(&self) -> usize {
        ENCODING_DIM
    }

    /// Encode a card as one-hot vector (52 dimensions).
    /// Card ID: 4 * rank + suit (rank: 0-12, suit: 0-3)
    fn encode_card(&self, card: u8, output: &mut [f32]) {
        debug_assert!(output.len() >= 52);
        if card < 52 {
            output[card as usize] = 1.0;
        }
        // If card is NOT_DEALT (255), leave all zeros
    }

    /// Encode board cards (flop + turn + river) as concatenated one-hot vectors.
    pub fn encode_board(&self, flop: [u8; 3], turn: u8, river: u8, output: &mut [f32]) {
        debug_assert!(output.len() >= 260);

        // Flop cards (always present)
        self.encode_card(flop[0], &mut output[0..52]);
        self.encode_card(flop[1], &mut output[52..104]);
        self.encode_card(flop[2], &mut output[104..156]);

        // Turn card (may be NOT_DEALT = 255)
        self.encode_card(turn, &mut output[156..208]);

        // River card (may be NOT_DEALT = 255)
        self.encode_card(river, &mut output[208..260]);
    }

    /// Encode hole cards as concatenated one-hot vectors.
    pub fn encode_hole_cards(&self, card1: u8, card2: u8, output: &mut [f32]) {
        debug_assert!(output.len() >= 104);

        // Canonicalize order (lower card first)
        let (c1, c2) = if card1 <= card2 {
            (card1, card2)
        } else {
            (card2, card1)
        };

        self.encode_card(c1, &mut output[0..52]);
        self.encode_card(c2, &mut output[52..104]);
    }

    /// Encode pot and stack sizes (normalized).
    pub fn encode_pot_stack(&self, current_pot: i32, remaining_stack: i32, output: &mut [f32]) {
        debug_assert!(output.len() >= 2);

        // Normalize to [0, ~1] range
        output[0] = (current_pot as f32) / (self.starting_pot + 2.0 * self.effective_stack);
        output[1] = (remaining_stack as f32) / self.effective_stack;
    }

    /// Encode street as one-hot (flop=0, turn=1, river=2).
    pub fn encode_street(&self, street: usize, output: &mut [f32]) {
        debug_assert!(output.len() >= 3);
        debug_assert!(street < 3);
        output[street] = 1.0;
    }

    /// Encode action history.
    /// Each action: [fold, check, call, bet/raise, normalized_size]
    pub fn encode_action_history(&self, actions: &[(ActionType, i32)], output: &mut [f32]) {
        debug_assert!(output.len() >= MAX_ACTION_HISTORY * ACTION_ENCODING_DIM);

        for (i, (action_type, amount)) in actions.iter().take(MAX_ACTION_HISTORY).enumerate() {
            let offset = i * ACTION_ENCODING_DIM;
            match action_type {
                ActionType::Fold => output[offset] = 1.0,
                ActionType::Check => output[offset + 1] = 1.0,
                ActionType::Call => output[offset + 2] = 1.0,
                ActionType::Bet | ActionType::Raise => {
                    output[offset + 3] = 1.0;
                    output[offset + 4] =
                        (*amount as f32) / (self.starting_pot + 2.0 * self.effective_stack);
                }
            }
        }
    }

    /// Encode a complete information set.
    ///
    /// # Arguments
    /// * `flop` - Flop cards (3 cards)
    /// * `turn` - Turn card (or NOT_DEALT)
    /// * `river` - River card (or NOT_DEALT)
    /// * `hole_cards` - Player's hole cards
    /// * `current_pot` - Current pot size
    /// * `remaining_stack` - Player's remaining stack
    /// * `street` - Current street (0=flop, 1=turn, 2=river)
    /// * `actions` - Action history
    ///
    /// # Returns
    /// Fixed-size vector of floats for neural network input
    pub fn encode(
        &self,
        flop: [u8; 3],
        turn: u8,
        river: u8,
        hole_cards: (u8, u8),
        current_pot: i32,
        remaining_stack: i32,
        street: usize,
        actions: &[(ActionType, i32)],
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; ENCODING_DIM];

        let mut offset = 0;

        // Board cards (260 dims)
        self.encode_board(flop, turn, river, &mut output[offset..offset + 260]);
        offset += 260;

        // Hole cards (104 dims)
        self.encode_hole_cards(hole_cards.0, hole_cards.1, &mut output[offset..offset + 104]);
        offset += 104;

        // Pot and stack (2 dims)
        self.encode_pot_stack(current_pot, remaining_stack, &mut output[offset..offset + 2]);
        offset += 2;

        // Street (3 dims)
        self.encode_street(street, &mut output[offset..offset + 3]);
        offset += 3;

        // Action history
        self.encode_action_history(actions, &mut output[offset..]);

        output
    }
}

/// Action types for encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionType {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
}

impl ActionType {
    /// Convert from the game's Action enum.
    pub fn from_action(action: crate::Action) -> Option<Self> {
        match action {
            crate::Action::Fold => Some(ActionType::Fold),
            crate::Action::Check => Some(ActionType::Check),
            crate::Action::Call => Some(ActionType::Call),
            crate::Action::Bet(_) => Some(ActionType::Bet),
            crate::Action::Raise(_) => Some(ActionType::Raise),
            crate::Action::AllIn(_) => Some(ActionType::Raise), // Treat all-in as raise
            crate::Action::Chance(_) | crate::Action::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimensions() {
        let encoder = InfoSetEncoder::new(100, 500);
        assert_eq!(encoder.dim(), ENCODING_DIM);
    }

    #[test]
    fn test_card_encoding() {
        let encoder = InfoSetEncoder::new(100, 500);
        let mut output = vec![0.0f32; 52];

        // Ace of spades: rank=12, suit=0 => id=48
        encoder.encode_card(48, &mut output);
        assert_eq!(output[48], 1.0);
        assert_eq!(output.iter().filter(|&&x| x == 1.0).count(), 1);
    }

    #[test]
    fn test_full_encoding() {
        let encoder = InfoSetEncoder::new(100, 500);

        // Flop: Td9d6h (need actual card IDs)
        // T=8, 9=7, 6=4; d=1, h=2
        let flop = [4 * 8 + 1, 4 * 7 + 1, 4 * 4 + 2]; // Td, 9d, 6h
        let turn = 255; // NOT_DEALT
        let river = 255; // NOT_DEALT
        let hole_cards = (4 * 12 + 0, 4 * 11 + 0); // As, Ks

        let encoded = encoder.encode(
            flop,
            turn,
            river,
            hole_cards,
            150, // pot
            450, // stack
            0,   // flop
            &[],
        );

        assert_eq!(encoded.len(), ENCODING_DIM);

        // Check street encoding (should be flop)
        let street_offset = 260 + 104 + 2;
        assert_eq!(encoded[street_offset], 1.0); // flop
        assert_eq!(encoded[street_offset + 1], 0.0); // not turn
        assert_eq!(encoded[street_offset + 2], 0.0); // not river
    }
}
