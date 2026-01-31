//! Archive format for subgame solving (.pfs files).
//!
//! The .pfs (Postflop Solver) archive format is a ZIP-based bundle containing:
//! - manifest.json: Version, config, metadata
//! - blueprint.bin: Flop strategies + bucketed Turn/River
//! - boundaries.bin: Ranges/EVs at street transitions
//! - abstraction.bin: Card-to-bucket mapping tables
//! - subgames/index.bin: Subgame offset table for random access
//! - subgames/t*.bin: Individual subgame data files
//!
//! This module requires the `subgame-archive` feature.

#[cfg(feature = "subgame-archive")]
use std::collections::BTreeMap;
#[cfg(feature = "subgame-archive")]
use std::fs::File;
#[cfg(feature = "subgame-archive")]
use std::io::{BufReader, Read, Write};
#[cfg(feature = "subgame-archive")]
use std::path::Path;

#[cfg(feature = "subgame-archive")]
use lru::LruCache;
#[cfg(feature = "subgame-archive")]
use zip::{ZipArchive, ZipWriter};

use crate::subgame::blueprint::Blueprint;
use crate::Card;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Magic number for .pfs files.
pub const PFS_MAGIC: u32 = 0x50465300; // "PFS\0"

/// Current archive format version.
pub const PFS_VERSION: u32 = 1;

/// Manifest for a .pfs archive.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct PfsManifest {
    /// Archive format version.
    pub version: u32,

    /// Format identifier.
    pub format: String,

    /// Creation timestamp (ISO 8601).
    pub created: String,

    /// Configuration used to generate this archive.
    pub config: PfsConfig,

    /// Game configuration.
    pub game: PfsGameInfo,

    /// Statistics about the archive contents.
    pub stats: PfsStats,
}

/// Configuration stored in manifest.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct PfsConfig {
    /// Number of turn buckets.
    pub turn_buckets: u8,

    /// Number of river buckets per turn.
    pub river_buckets: u8,

    /// Clustering method (e.g., "ehs2").
    pub clustering_method: String,

    /// Whether safe solving was used.
    pub safe_solving: bool,

    /// Number of blueprint iterations.
    pub blueprint_iterations: u32,

    /// Number of subgame iterations.
    pub subgame_iterations: u32,
}

/// Game information stored in manifest.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct PfsGameInfo {
    /// Flop board cards.
    pub board: [Card; 3],

    /// Starting pot size.
    pub starting_pot: i32,

    /// Effective stack.
    pub effective_stack: i32,

    /// OOP range string.
    pub oop_range: String,

    /// IP range string.
    pub ip_range: String,
}

/// Statistics stored in manifest.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct PfsStats {
    /// Total number of subgames.
    pub total_subgames: u32,

    /// Number of solved subgames.
    pub solved_subgames: u32,

    /// Blueprint exploitability (fraction of pot).
    pub blueprint_exploitability: f32,

    /// Average subgame exploitability.
    pub avg_subgame_exploitability: f32,
}

impl Default for PfsManifest {
    fn default() -> Self {
        Self {
            version: PFS_VERSION,
            format: "pfs-subgame".to_string(),
            created: String::new(),
            config: PfsConfig::default(),
            game: PfsGameInfo::default(),
            stats: PfsStats::default(),
        }
    }
}

impl PfsManifest {
    /// Create a new manifest with current timestamp.
    pub fn new(config: PfsConfig, game: PfsGameInfo, stats: PfsStats) -> Self {
        Self {
            version: PFS_VERSION,
            format: "pfs-subgame".to_string(),
            created: chrono_now(),
            config,
            game,
            stats,
        }
    }
}

/// Subgame index entry for random access.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SubgameIndexEntry {
    /// Turn card.
    pub turn: Card,

    /// River card (255 if turn-only subgame).
    pub river: Card,

    /// File name in archive.
    pub filename: String,

    /// Offset within the file (for multi-subgame files).
    pub offset: u64,

    /// Compressed size.
    pub compressed_size: u64,

    /// Uncompressed size.
    pub uncompressed_size: u64,
}

/// Index of all subgames in an archive.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SubgameIndex {
    /// Entries indexed by (turn * 52 + river).
    entries: Vec<SubgameIndexEntry>,

    /// Turn card to file index mapping.
    turn_to_file: Vec<usize>,
}

impl SubgameIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: SubgameIndexEntry) {
        self.entries.push(entry);
    }

    /// Get entry for a specific turn/river combination.
    pub fn get(&self, turn: Card, river: Option<Card>) -> Option<&SubgameIndexEntry> {
        let river = river.unwrap_or(255);
        self.entries
            .iter()
            .find(|e| e.turn == turn && e.river == river)
    }

    /// Get all entries.
    pub fn entries(&self) -> &[SubgameIndexEntry] {
        &self.entries
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Get current timestamp as ISO 8601 string.
fn chrono_now() -> String {
    // Simple timestamp without external dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Archive builder for creating .pfs files.
#[derive(Default)]
pub struct PfsArchiveBuilder {
    manifest: PfsManifest,
    blueprint: Option<Blueprint>,
    subgame_data: Vec<(Card, Option<Card>, Vec<u8>)>,
}

impl PfsArchiveBuilder {
    /// Create a new archive builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the manifest.
    pub fn manifest(mut self, manifest: PfsManifest) -> Self {
        self.manifest = manifest;
        self
    }

    /// Set the blueprint.
    pub fn blueprint(mut self, blueprint: Blueprint) -> Self {
        self.blueprint = Some(blueprint);
        self
    }

    /// Add subgame data.
    pub fn add_subgame(mut self, turn: Card, river: Option<Card>, data: Vec<u8>) -> Self {
        self.subgame_data.push((turn, river, data));
        self
    }

    /// Build to bytes (for in-memory archives).
    #[cfg(feature = "bincode")]
    pub fn build_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        // Write magic number
        buffer.extend_from_slice(&PFS_MAGIC.to_le_bytes());

        // Write version
        buffer.extend_from_slice(&PFS_VERSION.to_le_bytes());

        // Encode manifest
        let manifest_bytes = bincode::encode_to_vec(&self.manifest, bincode::config::standard())
            .map_err(|e| format!("Failed to encode manifest: {}", e))?;
        buffer.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&manifest_bytes);

        // Encode blueprint if present
        if let Some(ref blueprint) = self.blueprint {
            let blueprint_bytes =
                bincode::encode_to_vec(blueprint, bincode::config::standard())
                    .map_err(|e| format!("Failed to encode blueprint: {}", e))?;
            buffer.extend_from_slice(&(blueprint_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&blueprint_bytes);
        } else {
            buffer.extend_from_slice(&0u32.to_le_bytes());
        }

        // Build subgame index
        let mut index = SubgameIndex::new();
        let mut offset = 0u64;
        for (turn, river, data) in &self.subgame_data {
            let entry = SubgameIndexEntry {
                turn: *turn,
                river: river.unwrap_or(255),
                filename: format!("t{:02}_r{:02}.bin", turn, river.unwrap_or(255)),
                offset,
                compressed_size: data.len() as u64,
                uncompressed_size: data.len() as u64,
            };
            index.add(entry);
            offset += data.len() as u64;
        }

        // Encode index
        let index_bytes = bincode::encode_to_vec(&index, bincode::config::standard())
            .map_err(|e| format!("Failed to encode index: {}", e))?;
        buffer.extend_from_slice(&(index_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&index_bytes);

        // Write subgame data
        for (_, _, data) in &self.subgame_data {
            buffer.extend_from_slice(data);
        }

        Ok(buffer)
    }
}

/// Archive reader for .pfs files.
pub struct PfsArchiveReader {
    /// Manifest.
    pub manifest: PfsManifest,

    /// Blueprint (loaded on demand or immediately).
    pub blueprint: Option<Blueprint>,

    /// Subgame index.
    pub index: SubgameIndex,

    /// Raw data for subgames (for in-memory reading).
    subgame_data: Vec<u8>,

    /// Offset where subgame data starts (reserved for future random access).
    #[allow(dead_code)]
    subgame_data_offset: usize,
}

impl PfsArchiveReader {
    /// Read from bytes.
    #[cfg(feature = "bincode")]
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut offset = 0;

        // Read and verify magic
        if data.len() < 8 {
            return Err("Data too short".to_string());
        }
        let magic = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if magic != PFS_MAGIC {
            return Err(format!("Invalid magic number: {:08x}", magic));
        }
        offset += 4;

        // Read version
        let version = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if version != PFS_VERSION {
            return Err(format!("Unsupported version: {}", version));
        }
        offset += 4;

        // Read manifest
        let manifest_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let (manifest, _): (PfsManifest, _) =
            bincode::decode_from_slice(&data[offset..offset + manifest_len], bincode::config::standard())
                .map_err(|e| format!("Failed to decode manifest: {}", e))?;
        offset += manifest_len;

        // Read blueprint
        let blueprint_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let blueprint = if blueprint_len > 0 {
            let (bp, _): (Blueprint, _) =
                bincode::decode_from_slice(&data[offset..offset + blueprint_len], bincode::config::standard())
                    .map_err(|e| format!("Failed to decode blueprint: {}", e))?;
            offset += blueprint_len;
            Some(bp)
        } else {
            None
        };

        // Read index
        let index_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let (index, _): (SubgameIndex, _) =
            bincode::decode_from_slice(&data[offset..offset + index_len], bincode::config::standard())
                .map_err(|e| format!("Failed to decode index: {}", e))?;
        offset += index_len;

        // Store subgame data reference
        let subgame_data_offset = offset;
        let subgame_data = data[offset..].to_vec();

        Ok(Self {
            manifest,
            blueprint,
            index,
            subgame_data,
            subgame_data_offset,
        })
    }

    /// Get subgame data for a specific turn/river.
    pub fn get_subgame_data(&self, turn: Card, river: Option<Card>) -> Option<&[u8]> {
        let entry = self.index.get(turn, river)?;
        let start = entry.offset as usize;
        let end = start + entry.compressed_size as usize;
        if end <= self.subgame_data.len() {
            Some(&self.subgame_data[start..end])
        } else {
            None
        }
    }

    /// Check if a subgame exists.
    pub fn has_subgame(&self, turn: Card, river: Option<Card>) -> bool {
        self.index.get(turn, river).is_some()
    }

    /// Get the number of subgames.
    pub fn num_subgames(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfs_manifest_default() {
        let manifest = PfsManifest::default();
        assert_eq!(manifest.version, PFS_VERSION);
        assert_eq!(manifest.format, "pfs-subgame");
    }

    #[test]
    fn test_subgame_index() {
        let mut index = SubgameIndex::new();
        assert!(index.is_empty());

        index.add(SubgameIndexEntry {
            turn: 12,
            river: 16,
            filename: "t12_r16.bin".to_string(),
            offset: 0,
            compressed_size: 1000,
            uncompressed_size: 2000,
        });

        assert_eq!(index.len(), 1);
        assert!(index.get(12, Some(16)).is_some());
        assert!(index.get(12, Some(20)).is_none());
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn test_archive_roundtrip() {
        use crate::subgame::abstraction::AbstractionMapping;
        use crate::subgame::boundary::BoundaryStore;

        let board = [0, 4, 8];
        let config = crate::subgame::blueprint::BlueprintConfig::fast();
        let abstraction =
            AbstractionMapping::compute(&board, &config.abstraction);
        let boundaries = BoundaryStore::new(10, 10);

        let blueprint = Blueprint::new(
            board,
            abstraction,
            boundaries,
            config,
            0.01,
            10,
            10,
        );

        let manifest = PfsManifest::new(
            PfsConfig {
                turn_buckets: 5,
                river_buckets: 5,
                clustering_method: "ehs2".to_string(),
                safe_solving: true,
                blueprint_iterations: 100,
                subgame_iterations: 50,
            },
            PfsGameInfo {
                board,
                starting_pot: 100,
                effective_stack: 500,
                oop_range: "AA".to_string(),
                ip_range: "KK".to_string(),
            },
            PfsStats {
                total_subgames: 1,
                solved_subgames: 1,
                blueprint_exploitability: 0.01,
                avg_subgame_exploitability: 0.001,
            },
        );

        // Build archive
        let builder = PfsArchiveBuilder::new()
            .manifest(manifest)
            .blueprint(blueprint)
            .add_subgame(12, Some(16), vec![1, 2, 3, 4]);

        let bytes = builder.build_bytes().unwrap();

        // Read archive
        let reader = PfsArchiveReader::from_bytes(&bytes).unwrap();

        assert_eq!(reader.manifest.version, PFS_VERSION);
        assert!(reader.blueprint.is_some());
        assert_eq!(reader.num_subgames(), 1);
        assert!(reader.has_subgame(12, Some(16)));
        assert!(!reader.has_subgame(12, Some(20)));

        let data = reader.get_subgame_data(12, Some(16)).unwrap();
        assert_eq!(data, &[1, 2, 3, 4]);
    }
}
