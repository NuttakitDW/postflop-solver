// [File format]
// The file consists of a header and a body. The header is as follows:
//  - Magic number (4 bytes): 90 57 f1 09
//  - Version number (1 byte): 1
//  - Compression type (1 byte): 0 (none), 1 (zstd)
//  - Data type (1 byte): 0 (game), 1 (bunching)
//  - Estimated memory usage (`VarIntEncoding`)
//  - Memo string
//
// `VarIntEncoding`: https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md#varintencoding

use crate::bunching::*;
use crate::game::*;
use crate::interface::*;
use bincode::{Decode, Encode};
use std::cell::Cell;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[cfg(feature = "logging")]
use std::time::Instant;

const MAGIC: u32 = 0x09f15790;
const VERSION: u8 = 2;

/// Output mode for saving game data.
///
/// Controls what data is included when saving a game to file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Full output with all data (strategies + regrets).
    /// Required for resuming CFR iterations.
    #[default]
    Full = 0,
    /// Display-only output (strategies only, no regrets).
    /// Smaller file size (~50% reduction), suitable for UI display.
    /// Cannot be used to resume CFR iterations.
    Display = 1,
}

impl OutputMode {
    /// Returns true if this is display mode (regrets are omitted).
    #[inline]
    pub fn is_display(&self) -> bool {
        matches!(self, OutputMode::Display)
    }
}

// Thread-local storage for output mode during encoding
thread_local! {
    pub(crate) static ENCODE_OUTPUT_MODE: Cell<OutputMode> = Cell::new(OutputMode::Full);
}

// Thread-local storage for output mode during decoding (set by load functions)
thread_local! {
    pub(crate) static DECODE_OUTPUT_MODE: Cell<OutputMode> = Cell::new(OutputMode::Full);
}

/// Macro for conditional logging
#[cfg(feature = "logging")]
macro_rules! log_info {
    ($($arg:tt)*) => {
        log::info!($($arg)*);
    };
}

#[cfg(not(feature = "logging"))]
macro_rules! log_info {
    ($($arg:tt)*) => {};
}

/// File metadata without full load
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub version: u8,
    pub compression_type: u8,
    pub data_type: u8,
    pub output_mode: OutputMode,
    pub estimated_memory_usage: u64,
    pub memo: String,
    pub file_size: u64,
}

/// Timing breakdown for file loading
#[derive(Debug, Clone, Default)]
pub struct LoadTimings {
    pub header_ms: u64,
    pub decompression_ms: u64,
    pub deserialization_ms: u64,
    pub init_ms: u64,
    pub total_ms: u64,
}

#[doc(hidden)]
pub enum DataType {
    Game = 0,
    Bunching = 1,
}

/// A trait for data that can be saved into a file.
pub trait FileData: Decode<()> + Encode {
    #[doc(hidden)]
    fn data_type() -> DataType;
    #[doc(hidden)]
    fn is_ready_to_save(&self) -> bool;
    #[doc(hidden)]
    fn estimated_memory_usage(&self) -> u64;
}

fn encode_into_std_write<E: Encode, W: Write>(
    val: E,
    writer: &mut W,
    err_msg: &str,
) -> Result<usize, String> {
    bincode::encode_into_std_write(val, writer, bincode::config::standard())
        .map_err(|e| format!("{}: {}", err_msg, e))
}

/// Saves data into a standard writer.
///
/// This function serializes the `data` into the `writer`.
/// This is useful if you want to save the data into a custom writer like `Vec<u8>`, but if you want
/// to save the data into a file, use [`save_data_to_file`] instead.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `writer`: The writer to write the data into.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
pub fn save_data_into_std_write<T: FileData, W: Write>(
    data: &T,
    memo: &str,
    writer: &mut W,
    compression_level: Option<i32>,
) -> Result<(), String> {
    save_data_into_std_write_with_mode(data, memo, writer, compression_level, OutputMode::Full)
}

/// Saves data into a standard writer with a specified output mode.
///
/// This function serializes the `data` into the `writer` using the specified output mode.
/// This is useful if you want to save the data into a custom writer like `Vec<u8>`, but if you want
/// to save the data into a file, use [`save_data_to_file_with_mode`] instead.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `writer`: The writer to write the data into.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
/// - `output_mode`: The output mode to use. `Full` includes all data (strategies + regrets),
///   `Display` includes only strategies (smaller file, ~50% size reduction).
pub fn save_data_into_std_write_with_mode<T: FileData, W: Write>(
    data: &T,
    memo: &str,
    writer: &mut W,
    compression_level: Option<i32>,
    output_mode: OutputMode,
) -> Result<(), String> {
    if !data.is_ready_to_save() {
        return Err("Data is not ready to save".to_string());
    }

    #[cfg(not(feature = "zstd"))]
    if compression_level.is_some() {
        return Err("Compression is not supported".to_string());
    }

    // Set the output mode for encoding
    ENCODE_OUTPUT_MODE.with(|c| c.set(output_mode));

    encode_into_std_write(MAGIC, writer, "Failed to write magic number")?;
    encode_into_std_write(VERSION, writer, "Failed to write version number")?;

    let compression_type = compression_level.is_some() as u8;
    encode_into_std_write(compression_type, writer, "Failed to write compression type")?;

    encode_into_std_write(T::data_type() as u8, writer, "Failed to write data type")?;
    encode_into_std_write(output_mode as u8, writer, "Failed to write output mode")?;
    encode_into_std_write(
        data.estimated_memory_usage(),
        writer,
        "Failed to write memory usage",
    )?;

    encode_into_std_write(memo, writer, "Failed to write memo")?;

    if compression_level.is_none() {
        encode_into_std_write(data, writer, "Failed to write data")?;
        writer
            .flush()
            .map_err(|e| format!("Failed to flush writer: {}", e))?;
    }

    #[cfg(feature = "zstd")]
    if let Some(compression_level) = compression_level {
        let mut zstd_encoder = zstd::stream::Encoder::new(writer, compression_level)
            .map_err(|e| format!("Failed to create zstd encoder: {}", e))?;

        #[cfg(feature = "rayon")]
        zstd_encoder
            .multithread(rayon::current_num_threads() as u32)
            .map_err(|e| format!("Failed to enable multithreaded zstd encoder: {}", e))?;

        encode_into_std_write(data, &mut zstd_encoder, "Failed to write data")?;
        zstd_encoder
            .finish()
            .map_err(|e| format!("Failed to finish zstd encoder: {}", e))?
            .flush()
            .map_err(|e| format!("Failed to flush writer: {}", e))?;
    }

    Ok(())
}

/// Saves data into a file.
///
/// This function serializes the `data` into a file specified by `path`.
/// If the file already exists, it will be overwritten.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `path`: The path to the file to save.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
pub fn save_data_to_file<T: FileData, P: AsRef<Path>>(
    data: &T,
    memo: &str,
    path: P,
    compression_level: Option<i32>,
) -> Result<(), String> {
    save_data_to_file_with_mode(data, memo, path, compression_level, OutputMode::Full)
}

/// Saves data into a file with a specified output mode.
///
/// This function serializes the `data` into a file specified by `path` using the specified output mode.
/// If the file already exists, it will be overwritten.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `path`: The path to the file to save.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
/// - `output_mode`: The output mode to use. `Full` includes all data (strategies + regrets),
///   `Display` includes only strategies (smaller file, ~50% size reduction).
pub fn save_data_to_file_with_mode<T: FileData, P: AsRef<Path>>(
    data: &T,
    memo: &str,
    path: P,
    compression_level: Option<i32>,
    output_mode: OutputMode,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);
    save_data_into_std_write_with_mode(data, memo, &mut writer, compression_level, output_mode)
}

fn decode_from_std_read<D: Decode<()>, R: Read>(reader: &mut R, err_msg: &str) -> Result<D, String> {
    bincode::decode_from_std_read(reader, bincode::config::standard())
        .map_err(|e| format!("{}: {}", err_msg, e))
}

/// Loads data from a standard reader.
///
/// This function deserializes the data from the `reader`.
/// This is useful if you want to load the data from a custom reader like `Vec<u8>`, but if you want
/// to load the data from a file, use [`load_data_from_file`] instead.
///
/// # Arguments
///
/// - `reader`: The reader to read the data from.
/// - `max_memory_usage`: The maximum memory usage allowed for the data (in bytes). If `None`, no
///   limit is set. If the estimated memory usage exceeds this value, `Err` is returned.
///
/// # Returns
///
/// A tuple of the deserialized data (either a [`PostFlopGame`] or a [`BunchingData`]) and the memo
/// string.
pub fn load_data_from_std_read<T: FileData, R: Read>(
    reader: &mut R,
    max_memory_usage: Option<u64>,
) -> Result<(T, String), String> {
    #[cfg(feature = "logging")]
    let start = Instant::now();

    let magic: u32 = decode_from_std_read(reader, "Failed to read magic number")?;
    if magic != MAGIC {
        return Err("Magic number is invalid".to_string());
    }
    log_info!("[LOAD] Magic number validated");

    let version: u8 = decode_from_std_read(reader, "Failed to read version number")?;
    // Support both version 1 (legacy) and version 2 (with output mode)
    if version != 1 && version != VERSION {
        return Err(format!("Version number {} is not supported (expected 1 or {})", version, VERSION));
    }
    log_info!("[LOAD] Version: {}", version);

    let compression_type: u8 = decode_from_std_read(reader, "Failed to read compression type")?;
    if compression_type > 1 {
        return Err("Compression type is invalid".to_string());
    }
    log_info!("[LOAD] Compression: {}", if compression_type == 0 { "none" } else { "zstd" });

    #[cfg(not(feature = "zstd"))]
    if compression_type == 1 {
        return Err("Compression is not supported".to_string());
    }

    let data_type: u8 = decode_from_std_read(reader, "Failed to read data type")?;
    if data_type != T::data_type() as u8 {
        return Err("Data type is invalid".to_string());
    }
    log_info!("[LOAD] Data type: {}", data_type);

    // Read output mode (version 2+) or default to Full (version 1)
    let output_mode = if version >= 2 {
        let mode_byte: u8 = decode_from_std_read(reader, "Failed to read output mode")?;
        match mode_byte {
            0 => OutputMode::Full,
            1 => OutputMode::Display,
            _ => return Err(format!("Invalid output mode: {}", mode_byte)),
        }
    } else {
        OutputMode::Full
    };
    log_info!("[LOAD] Output mode: {:?}", output_mode);

    // Set the output mode for decoding
    DECODE_OUTPUT_MODE.with(|c| c.set(output_mode));

    let estimated_memory_usage: u64 = decode_from_std_read(reader, "Failed to read memory usage")?;
    log_info!("[LOAD] Estimated memory: {} MB", estimated_memory_usage / 1_048_576);

    if let Some(max_memory_usage) = max_memory_usage {
        if estimated_memory_usage > max_memory_usage {
            return Err(format!(
                "Estimated memory usage ({} MB) exceeds limit ({} MB)",
                estimated_memory_usage / 1_048_576,
                max_memory_usage / 1_048_576
            ));
        }
    }

    let memo: String = decode_from_std_read(reader, "Failed to read memo")?;
    log_info!("[LOAD] Memo length: {} chars", memo.len());

    log_info!(
        "[LOAD] Starting {}...",
        if compression_type == 0 { "deserialization" } else { "decompression + deserialization" }
    );

    #[cfg(feature = "logging")]
    let decompress_start = Instant::now();

    #[cfg(not(feature = "zstd"))]
    let data: T = decode_from_std_read(reader, "Failed to read data")?;
    #[cfg(feature = "zstd")]
    let data: T = if compression_type == 0 {
        decode_from_std_read(reader, "Failed to read data")?
    } else {
        let mut zstd_decoder = zstd::stream::Decoder::new(reader)
            .map_err(|e| format!("Failed to create zstd decoder: {}", e))?;
        decode_from_std_read(&mut zstd_decoder, "Failed to read data")?
    };

    #[cfg(feature = "logging")]
    {
        let decompress_elapsed = decompress_start.elapsed();
        let total_elapsed = start.elapsed();
        log_info!(
            "[LOAD] Complete! Deserialization took {:?}, total {:?}",
            decompress_elapsed,
            total_elapsed
        );
    }

    Ok((data, memo))
}

/// Read file header without loading the full game tree.
///
/// This is useful for pre-validation before committing to a full load,
/// such as checking memory requirements or file metadata.
pub fn read_file_info<P: AsRef<Path>>(path: P) -> Result<FileInfo, String> {
    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open file: {}", e))?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = BufReader::new(file);

    let magic: u32 = decode_from_std_read(&mut reader, "Failed to read magic number")?;
    if magic != MAGIC {
        return Err("Magic number is invalid".to_string());
    }

    let version: u8 = decode_from_std_read(&mut reader, "Failed to read version number")?;
    // Support both version 1 (legacy) and version 2 (with output mode)
    if version != 1 && version != VERSION {
        return Err(format!("Version number {} is not supported (expected 1 or {})", version, VERSION));
    }

    let compression_type: u8 = decode_from_std_read(&mut reader, "Failed to read compression type")?;
    if compression_type > 1 {
        return Err("Compression type is invalid".to_string());
    }

    let data_type: u8 = decode_from_std_read(&mut reader, "Failed to read data type")?;

    // Read output mode (version 2+) or default to Full (version 1)
    let output_mode = if version >= 2 {
        let mode_byte: u8 = decode_from_std_read(&mut reader, "Failed to read output mode")?;
        match mode_byte {
            0 => OutputMode::Full,
            1 => OutputMode::Display,
            _ => return Err(format!("Invalid output mode: {}", mode_byte)),
        }
    } else {
        OutputMode::Full
    };

    let estimated_memory_usage: u64 = decode_from_std_read(&mut reader, "Failed to read memory usage")?;

    let memo: String = decode_from_std_read(&mut reader, "Failed to read memo")?;

    Ok(FileInfo {
        version,
        compression_type,
        data_type,
        output_mode,
        estimated_memory_usage,
        memo,
        file_size,
    })
}

/// Loads data from a file.
///
/// This function deserializes the data from a file specified by `path`.
///
/// # Arguments
///
/// - `path`: The path to the file to load.
/// - `max_memory_usage`: The maximum memory usage allowed for the data (in bytes). If `None`, no
///   limit is set. If the estimated memory usage exceeds this value, `Err` is returned.
///
/// # Returns
///
/// A tuple of the deserialized data (either a [`PostFlopGame`] or a [`BunchingData`]) and the memo
/// string.
pub fn load_data_from_file<T: FileData, P: AsRef<Path>>(
    path: P,
    max_memory_usage: Option<u64>,
) -> Result<(T, String), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);
    load_data_from_std_read(&mut reader, max_memory_usage)
}

impl FileData for PostFlopGame {
    fn data_type() -> DataType {
        DataType::Game
    }

    fn is_ready_to_save(&self) -> bool {
        self.is_solved()
    }

    fn estimated_memory_usage(&self) -> u64 {
        self.target_memory_usage()
    }
}

impl FileData for BunchingData {
    fn data_type() -> DataType {
        DataType::Bunching
    }

    fn is_ready_to_save(&self) -> bool {
        self.is_ready()
    }

    fn estimated_memory_usage(&self) -> u64 {
        self.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::*;
    use crate::card::*;
    use crate::range::*;
    use crate::utility::*;

    #[test]
    #[cfg(feature = "zstd")]
    fn save_and_load_file_compressed() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            flop_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            turn_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        // save
        save_data_to_file(&game, "", "tmpfile-zstd.flop", Some(3)).unwrap();

        // load
        let mut game: PostFlopGame = load_data_from_file("tmpfile-zstd.flop", None).unwrap().0;

        // remove tmpfile
        std::fs::remove_file("tmpfile-zstd.flop").unwrap();

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 45.0).abs() < 1e-4);
        assert!((root_ev_ip - 15.0).abs() < 1e-4);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn save_and_load_display_mode() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            flop_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            turn_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        // Set target storage mode to Flop so that storage2 is actually written in Full mode
        // (River mode already omits storage2 to save space)
        game.set_target_storage_mode(BoardState::Flop).unwrap();

        // save in full mode
        save_data_to_file_with_mode(&game, "", "tmpfile-full.flop", Some(3), OutputMode::Full).unwrap();
        let full_size = std::fs::metadata("tmpfile-full.flop").unwrap().len();

        // save in display mode
        save_data_to_file_with_mode(&game, "", "tmpfile-display.flop", Some(3), OutputMode::Display).unwrap();
        let display_size = std::fs::metadata("tmpfile-display.flop").unwrap().len();

        // display mode should be smaller (no storage2/regrets)
        assert!(display_size < full_size, "display_size={} should be < full_size={}", display_size, full_size);

        // load display mode file and verify it's usable for display
        let file_info = read_file_info("tmpfile-display.flop").unwrap();
        assert_eq!(file_info.output_mode, OutputMode::Display);

        let mut game_display: PostFlopGame = load_data_from_file("tmpfile-display.flop", None).unwrap().0;
        game_display.cache_normalized_weights();

        // equities should still be correct
        let weights_oop = game_display.normalized_weights(0);
        let root_equity_oop = compute_average(&game_display.equity(0), weights_oop);
        assert!((root_equity_oop - 0.5).abs() < 1e-5);

        // cleanup
        std::fs::remove_file("tmpfile-full.flop").unwrap();
        std::fs::remove_file("tmpfile-display.flop").unwrap();
    }

}
