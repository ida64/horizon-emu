//! Machine snapshots: everything a running machine holds, saved so that a
//! machine built the same way continues from it.
//!
//! A machine is built from its boot parameters, images and keys the same
//! way every time: the same devices at the same addresses, wired the same
//! way. A restore therefore builds the machine as for a fresh boot and then
//! puts back the state of each part. Each part with state of its own
//! implements [`Snapshot`]: it saves what it holds into bytes and takes the
//! same bytes back. Nothing is derived again on restore that the part
//! could not derive at any other moment.
//!
//! A part that holds state and cannot save it makes the snapshot fail
//! with [`SnapshotError::Unsupported`], naming it; a restore never leaves
//! a part as built when the saved machine had moved it on.
//!
//! What a part saves follows these rules:
//! * Times are guest time ([`crate::clock::GuestClock`]), never host
//!   instants: a deadline is saved as the guest time it falls at.
//! * Work a thread of the part has scheduled (a timer, a completion due
//!   later) is state like any other: saved as what is due and when, and
//!   scheduled again on restore.
//! * Host resources (sockets, files, GPU objects) are not saved. A cache of
//!   something in guest memory or in the part's own state is dropped and
//!   rebuilt; a connection to the host's network is lost, as when the link
//!   drops, and says so.
//! * Wiring (interrupt lines, callbacks, children's places) is not saved:
//!   the restoring machine has it already.

use serde::{Serialize, de::DeserializeOwned};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// A part cannot save or restore its state in the state it is in, or
    /// at all yet; the string names it and why.
    Unsupported(String),
    /// Saved bytes that do not decode as the part's state, or do not fit
    /// the machine being restored.
    Corrupt(String),
    /// Reading or writing the snapshot's files.
    Io(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "cannot snapshot {what}"),
            Self::Corrupt(what) => write!(f, "snapshot does not fit: {what}"),
            Self::Io(what) => write!(f, "snapshot files: {what}"),
        }
    }
}

/// A part of the machine with state of its own.
///
/// Saving and restoring happen with the machine paused: the cores parked
/// and guest time ([`crate::clock::GuestClock`]) standing still.
pub trait Snapshot: Send + Sync {
    /// Brings work the part does on its own, not on a timer, to a point
    /// where its state is complete in what `save` sees (the GPU finishes
    /// the channel it is running and writes its render targets back to
    /// memory). Called for every part before any part saves.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        Ok(())
    }
    /// Its state, encoded.
    fn save(&self) -> Result<Vec<u8>, SnapshotError>;
    /// Takes back what `save` of the same part, in a machine built the same
    /// way, returned, including work the part had scheduled.
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError>;
}

impl std::error::Error for SnapshotError {}

/// Encodes a part's state (bincode).
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, SnapshotError> {
    bincode::serialize(value).map_err(|error| SnapshotError::Corrupt(error.to_string()))
}

/// Decodes a part's state; `what` names the part for the error.
pub fn decode<T: DeserializeOwned>(what: &str, bytes: &[u8]) -> Result<T, SnapshotError> {
    bincode::deserialize(bytes).map_err(|error| SnapshotError::Corrupt(format!("{what}: {error}")))
}

/// serde for fixed-size arrays of any length (serde derives them only up to
/// 32 elements): `#[serde(with = "crate::snapshot::array")]`.
pub mod array {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        array: &[T; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(array.iter())
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[T; N], D::Error> {
        let items = Vec::<T>::deserialize(deserializer)?;
        let len = items.len();
        items
            .try_into()
            .map_err(|_| D::Error::custom(format!("{len} elements where {N} were saved")))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Wide {
        #[serde(with = "array")]
        words: [u32; 100],
    }

    #[test]
    fn long_arrays_round_trip_and_a_wrong_length_is_refused() {
        let wide = Wide {
            words: std::array::from_fn(|index| index as u32 * 3),
        };
        let bytes = encode(&wide).unwrap();
        assert_eq!(decode::<Wide>("wide", &bytes).unwrap(), wide);
        let short = encode(&vec![1u32; 99]).unwrap();
        assert!(matches!(
            decode::<Wide>("wide", &short),
            Err(SnapshotError::Corrupt(_))
        ));
    }
}
