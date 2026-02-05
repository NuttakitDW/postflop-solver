use super::*;
use crate::action_tree::*;
use crate::bet_size::*;
use crate::card::*;
use crate::interface::{Game, GameNode};
use serde_json::{json, Value};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Serialize an Action to JSON
fn serialize_action(action: &Action) -> Value {
    match action {
        Action::None => json!("None"),
        Action::Fold => json!("Fold"),
        Action::Check => json!("Check"),
        Action::Call => json!("Call"),
        Action::Bet(amount) => json!({"Bet": amount}),
        Action::Raise(amount) => json!({"Raise": amount}),
        Action::AllIn(amount) => json!({"AllIn": amount}),
        Action::Chance(card) => json!({"Chance": *card}),
    }
}

/// Serialize CardConfig to JSON
fn serialize_card_config(config: &CardConfig) -> Value {
    let flop: Vec<u8> = config.flop.to_vec();

    let turn = if config.turn == NOT_DEALT {
        Value::Null
    } else {
        json!(config.turn)
    };

    let river = if config.river == NOT_DEALT {
        Value::Null
    } else {
        json!(config.river)
    };

    // Convert ranges to arrays (already in Vec<f32> format from raw_data())
    let oop_range = config.range[0].raw_data();
    let ip_range = config.range[1].raw_data();

    json!({
        "flop": flop,
        "turn": turn,
        "river": river,
        "oop_range": oop_range,
        "ip_range": ip_range,
    })
}

/// Serialize BetSize to JSON
fn serialize_bet_size(bet_size: &BetSize) -> Value {
    match bet_size {
        BetSize::PotRelative(ratio) => json!({"PotRelative": ratio}),
        BetSize::PrevBetRelative(ratio) => json!({"PrevBetRelative": ratio}),
        BetSize::Additive(pot, prev) => json!({"Additive": {"pot": pot, "prev": prev}}),
        BetSize::Geometric(amount, ratio) => json!({"Geometric": {"amount": amount, "ratio": ratio}}),
        BetSize::AllIn => json!("AllIn"),
    }
}

/// Serialize BetSizeOptions to JSON
fn serialize_bet_size_options(options: &BetSizeOptions) -> Value {
    json!({
        "bet": options.bet.iter().map(serialize_bet_size).collect::<Vec<_>>(),
        "raise": options.raise.iter().map(serialize_bet_size).collect::<Vec<_>>(),
    })
}

/// Serialize DonkSizeOptions to JSON
fn serialize_donk_size_options(options: &Option<DonkSizeOptions>) -> Value {
    match options {
        Some(donk) => json!({
            "donk": donk.donk.iter().map(serialize_bet_size).collect::<Vec<_>>(),
        }),
        None => Value::Null,
    }
}

/// Serialize TreeConfig to JSON
fn serialize_tree_config(config: &TreeConfig) -> Value {
    let initial_state = match config.initial_state {
        BoardState::Flop => "flop",
        BoardState::Turn => "turn",
        BoardState::River => "river",
    };

    json!({
        "initial_state": initial_state,
        "starting_pot": config.starting_pot,
        "effective_stack": config.effective_stack,
        "rake_rate": config.rake_rate,
        "rake_cap": config.rake_cap,
        "flop_bet_sizes": [
            serialize_bet_size_options(&config.flop_bet_sizes[0]),
            serialize_bet_size_options(&config.flop_bet_sizes[1]),
        ],
        "turn_bet_sizes": [
            serialize_bet_size_options(&config.turn_bet_sizes[0]),
            serialize_bet_size_options(&config.turn_bet_sizes[1]),
        ],
        "river_bet_sizes": [
            serialize_bet_size_options(&config.river_bet_sizes[0]),
            serialize_bet_size_options(&config.river_bet_sizes[1]),
        ],
        "turn_donk_sizes": serialize_donk_size_options(&config.turn_donk_sizes),
        "river_donk_sizes": serialize_donk_size_options(&config.river_donk_sizes),
        "add_allin_threshold": config.add_allin_threshold,
        "force_allin_threshold": config.force_allin_threshold,
        "merging_threshold": config.merging_threshold,
        "max_raises_per_street": config.max_raises_per_street,
    })
}

/// Decompress strategy data from a node
fn decompress_strategy(
    node: &PostFlopNode,
    num_hands: usize,
    is_compression_enabled: bool,
) -> Vec<f32> {
    if is_compression_enabled {
        let compressed = node.strategy_compressed();
        let scale = node.strategy_scale();
        let scale_factor = scale / (u16::MAX as f32);

        #[cfg(feature = "rayon")]
        {
            compressed
                .par_iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            compressed
                .iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }
    } else {
        node.strategy().to_vec()
    }
}

/// Decompress cfvalues data from a node
fn decompress_cfvalues(
    node: &PostFlopNode,
    num_hands: usize,
    is_compression_enabled: bool,
) -> Vec<f32> {
    if is_compression_enabled {
        let compressed = node.cfvalues_compressed();
        let scale = node.cfvalue_scale();
        let scale_factor = scale / (i16::MAX as f32);

        #[cfg(feature = "rayon")]
        {
            compressed
                .par_iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            compressed
                .iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }
    } else {
        node.cfvalues().to_vec()
    }
}

/// Decompress IP cfvalues data from a node
fn decompress_cfvalues_ip(
    node: &PostFlopNode,
    num_hands: usize,
    is_compression_enabled: bool,
) -> Option<Vec<f32>> {
    if node.num_elements_ip == 0 {
        return None;
    }

    if is_compression_enabled {
        let compressed = node.cfvalues_ip_compressed();
        let scale = node.cfvalue_ip_scale();
        let scale_factor = scale / (i16::MAX as f32);

        #[cfg(feature = "rayon")]
        {
            Some(
                compressed
                    .par_iter()
                    .map(|&val| val as f32 * scale_factor)
                    .collect()
            )
        }

        #[cfg(not(feature = "rayon"))]
        {
            Some(
                compressed
                    .iter()
                    .map(|&val| val as f32 * scale_factor)
                    .collect()
            )
        }
    } else {
        Some(node.cfvalues_ip().to_vec())
    }
}

/// Decompress chance cfvalues data from a node
fn decompress_cfvalues_chance(
    node: &PostFlopNode,
    num_hands: usize,
    is_compression_enabled: bool,
) -> Vec<f32> {
    if is_compression_enabled {
        let compressed = node.cfvalues_chance_compressed();
        let scale = node.cfvalue_chance_scale();
        let scale_factor = scale / (i16::MAX as f32);

        #[cfg(feature = "rayon")]
        {
            compressed
                .par_iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            compressed
                .iter()
                .map(|&val| val as f32 * scale_factor)
                .collect()
        }
    } else {
        node.cfvalues_chance().to_vec()
    }
}

/// Serialize a single node to JSON
fn serialize_node(
    node: &PostFlopNode,
    index: usize,
    num_oop_hands: usize,
    num_ip_hands: usize,
    is_solved: bool,
    is_compression_enabled: bool,
    locking_strategy: &BTreeMap<usize, Vec<f32>>,
) -> Value {
    let is_terminal = node.player & PLAYER_TERMINAL_FLAG != 0;
    let is_chance = node.player & PLAYER_CHANCE_FLAG != 0;
    let player = (node.player & PLAYER_MASK) as usize;

    // Build children indices array
    let mut children = Vec::new();
    let children_nodes = node.children();
    for i in 0..children_nodes.len() {
        children.push((node.children_offset as usize) + i);
    }

    let mut node_json = json!({
        "index": index,
        "prev_action": serialize_action(&node.prev_action),
        "player": player,
        "turn": if node.turn == NOT_DEALT { Value::Null } else { json!(node.turn) },
        "river": if node.river == NOT_DEALT { Value::Null } else { json!(node.river) },
        "is_locked": node.is_locked,
        "amount": node.amount,
        "children": children,
        "is_terminal": is_terminal,
        "is_chance": is_chance,
    });

    // Add strategy if it's a player node (not terminal, not chance)
    if !is_terminal && !is_chance && node.num_children > 0 {
        let num_hands = if player == 0 { num_oop_hands } else { num_ip_hands };

        // Check if there's a locked strategy for this node
        let strategy = if let Some(locked) = locking_strategy.get(&index) {
            locked.clone()
        } else {
            decompress_strategy(node, num_hands, is_compression_enabled)
        };

        node_json["strategy"] = json!(strategy);

        // Add cfvalues if solved
        if is_solved && node.num_elements > 0 {
            let cfvalues = decompress_cfvalues(node, num_hands, is_compression_enabled);
            node_json["cfvalues"] = json!(cfvalues);

            if let Some(cfvalues_ip) = decompress_cfvalues_ip(node, num_hands, is_compression_enabled) {
                node_json["cfvalues_ip"] = json!(cfvalues_ip);
            }
        }
    }

    // Add chance node cfvalues if solved and it's a chance node
    if is_chance && is_solved && node.num_elements > 0 {
        let num_hands = if player == 0 { num_oop_hands } else { num_ip_hands };
        let cfvalues_chance = decompress_cfvalues_chance(node, num_hands, is_compression_enabled);
        node_json["cfvalues_chance"] = json!(cfvalues_chance);
    }

    node_json
}

impl PostFlopGame {
    /// Export the complete game data to a JSON Value including all strategies and EVs
    pub fn to_json_value(&self) -> Result<Value, String> {
        to_json_value_internal(self)
    }

    /// Estimate the size of the JSON export in bytes
    pub fn estimate_json_size(&self) -> usize {
        estimate_json_size_internal(self)
    }
}

/// Export the complete PostFlopGame to a JSON Value (internal function)
fn to_json_value_internal(game: &PostFlopGame) -> Result<Value, String> {
    // Get basic metadata
    let is_solved = game.is_solved();
    let storage_mode = match game.storage_mode() {
        BoardState::Flop => "flop",
        BoardState::Turn => "turn",
        BoardState::River => "river",
    };
    let is_compression_enabled = game.is_compression_enabled();

    // Get configuration
    let card_config = game.card_config();
    let tree_config = game.tree_config();

    // Get hand data
    let oop_cards = game.private_cards(0);
    let ip_cards = game.private_cards(1);
    let num_oop_hands = oop_cards.len();
    let num_ip_hands = ip_cards.len();

    #[cfg(feature = "rayon")]
    let (oop_private_cards, ip_private_cards) = rayon::join(
        || oop_cards.par_iter().map(|&(c1, c2)| vec![c1, c2]).collect::<Vec<Vec<u8>>>(),
        || ip_cards.par_iter().map(|&(c1, c2)| vec![c1, c2]).collect::<Vec<Vec<u8>>>(),
    );

    #[cfg(not(feature = "rayon"))]
    let (oop_private_cards, ip_private_cards) = (
        oop_cards.iter().map(|&(c1, c2)| vec![c1, c2]).collect::<Vec<Vec<u8>>>(),
        ip_cards.iter().map(|&(c1, c2)| vec![c1, c2]).collect::<Vec<Vec<u8>>>(),
    );

    let oop_initial_weights = game.initial_weights(0).to_vec();
    let ip_initial_weights = game.initial_weights(1).to_vec();

    // Get equity and expected values if solved and normalized weights are cached
    let (oop_equity, ip_equity, oop_ev, ip_ev, oop_eqr, ip_eqr) = if is_solved && game.is_normalized_weight_cached {
        let oop_equity_vec = game.equity(0);
        let ip_equity_vec = game.equity(1);
        let oop_ev_vec = game.expected_values(0);
        let ip_ev_vec = game.expected_values(1);

        // Calculate EQR (Equity Realization) = EV / Equity
        // Handle division by zero
        #[cfg(feature = "rayon")]
        let (oop_eqr_vec, ip_eqr_vec) = rayon::join(
            || oop_ev_vec.par_iter().zip(oop_equity_vec.par_iter())
                .map(|(&ev, &eq)| if eq > 0.01 { ev / eq } else { 0.0 })
                .collect::<Vec<f32>>(),
            || ip_ev_vec.par_iter().zip(ip_equity_vec.par_iter())
                .map(|(&ev, &eq)| if eq > 0.01 { ev / eq } else { 0.0 })
                .collect::<Vec<f32>>(),
        );

        #[cfg(not(feature = "rayon"))]
        let (oop_eqr_vec, ip_eqr_vec) = (
            oop_ev_vec.iter().zip(oop_equity_vec.iter())
                .map(|(&ev, &eq)| if eq > 0.01 { ev / eq } else { 0.0 })
                .collect::<Vec<f32>>(),
            ip_ev_vec.iter().zip(ip_equity_vec.iter())
                .map(|(&ev, &eq)| if eq > 0.01 { ev / eq } else { 0.0 })
                .collect::<Vec<f32>>(),
        );

        (Some(oop_equity_vec), Some(ip_equity_vec), Some(oop_ev_vec), Some(ip_ev_vec), Some(oop_eqr_vec), Some(ip_eqr_vec))
    } else {
        (None, None, None, None, None, None)
    };

    // Serialize added and removed lines (parallel processing)
    #[cfg(feature = "rayon")]
    let (added_lines, removed_lines) = rayon::join(
        || game.added_lines().par_iter().map(|line| line.iter().map(serialize_action).collect()).collect::<Vec<Vec<Value>>>(),
        || game.removed_lines().par_iter().map(|line| line.iter().map(serialize_action).collect()).collect::<Vec<Vec<Value>>>(),
    );

    #[cfg(not(feature = "rayon"))]
    let (added_lines, removed_lines) = (
        game.added_lines().iter().map(|line| line.iter().map(serialize_action).collect()).collect::<Vec<Vec<Value>>>(),
        game.removed_lines().iter().map(|line| line.iter().map(serialize_action).collect()).collect::<Vec<Vec<Value>>>(),
    );

    // Get locking strategy map
    let locking_strategy = &game.locking_strategy;

    // Serialize all nodes (parallel processing for speed)
    let total_nodes: usize = game.num_nodes.iter().map(|&x| x as usize).sum();

    #[cfg(feature = "rayon")]
    let nodes: Vec<Value> = (0..total_nodes)
        .into_par_iter()
        .map(|i| {
            let node = game.node_arena[i].lock();
            serialize_node(
                &node,
                i,
                num_oop_hands,
                num_ip_hands,
                is_solved,
                is_compression_enabled,
                locking_strategy,
            )
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let nodes: Vec<Value> = {
        let mut nodes = Vec::with_capacity(total_nodes);
        for i in 0..total_nodes {
            let node = game.node_arena[i].lock();
            let node_json = serialize_node(
                &node,
                i,
                num_oop_hands,
                num_ip_hands,
                is_solved,
                is_compression_enabled,
                locking_strategy,
            );
            nodes.push(node_json);
        }
        nodes
    };

    // Build the complete JSON structure
    // Use a simple timestamp format (ISO 8601-like)
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let exported_at = format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        1970 + (timestamp / 31_536_000),
        ((timestamp % 31_536_000) / 2_592_000) + 1,
        ((timestamp % 2_592_000) / 86_400) + 1,
        (timestamp % 86_400) / 3600,
        (timestamp % 3600) / 60,
        timestamp % 60
    );

    Ok(json!({
        "version": 1,
        "exported_at": exported_at,
        "is_solved": is_solved,
        "storage_mode": storage_mode,
        "compression_enabled": is_compression_enabled,
        "configuration": {
            "card_config": serialize_card_config(card_config),
            "tree_config": serialize_tree_config(tree_config),
            "added_lines": added_lines,
            "removed_lines": removed_lines,
        },
        "hand_data": {
            "oop_private_cards": oop_private_cards,
            "ip_private_cards": ip_private_cards,
            "oop_initial_weights": oop_initial_weights,
            "ip_initial_weights": ip_initial_weights,
            "oop_equity": oop_equity,
            "ip_equity": ip_equity,
            "oop_ev": oop_ev,
            "ip_ev": ip_ev,
            "oop_eqr": oop_eqr,
            "ip_eqr": ip_eqr,
        },
        "nodes": nodes,
    }))
}

/// Estimate the JSON size in bytes (internal function)
fn estimate_json_size_internal(game: &PostFlopGame) -> usize {
    let total_nodes: usize = game.num_nodes.iter().map(|&x| x as usize).sum();
    let num_oop_hands = game.private_cards(0).len();
    let num_ip_hands = game.private_cards(1).len();

    // Rough estimation:
    // - Base node structure: ~200 bytes
    // - Strategy: num_actions * num_hands * 8 bytes (assuming average 3 actions)
    // - CFValues: num_actions * num_hands * 8 bytes
    // - CFValues IP: num_hands * 8 bytes
    let avg_num_actions = 3;
    let avg_hands = (num_oop_hands + num_ip_hands) / 2;

    let bytes_per_node = 200 + (avg_num_actions * avg_hands * 8) + (avg_num_actions * avg_hands * 8) + (avg_hands * 8);

    // Configuration overhead (rough estimate)
    let config_overhead = 50_000;

    // Hand data overhead
    let hand_data_overhead = (num_oop_hands + num_ip_hands) * 100;

    config_overhead + hand_data_overhead + (total_nodes * bytes_per_node)
}
