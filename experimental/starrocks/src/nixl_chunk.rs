//! Sirius envelope on unpatched `PInternalService.transmit_chunk`.
//!
//! NIXL exchange control rides the FE-advertised brpc port and the existing `transmit_chunk`
//! method. The protobuf carries routing (`finst_id`, `node_id`, `sender_id`, `sequence`, `eos`);
//! the attachment is `SRNX`, a kind byte, then a little-endian body. GPU bytes never ride this RPC.
//! Native `ChunkPB`, pass-through, and pipeline-level shuffle are rejected so a stock BE cannot be
//! misread as Sirius.
//!
//! | Kind | Request body | Reply attachment |
//! |---|---|---|
//! | `Md` | sender agent metadata | receiver agent metadata |
//! | `Alloc` | batch layout | `u64 token, i32 device, u32 n, n x (u64 addr, u64 len)` |
//! | `Packed` | `u64 token, u64 rows, u32 n, n x (u32 len, utf-8 name)` | empty |
//! | `Release` | `u64 token` | empty |

use std::net::SocketAddr;
use std::sync::Arc;

use crate::fragment_executor::{FragmentExecutor, OutputDrain, SenderSlot};
use crate::proto::starrocks::PTransmitChunkParams;

const MAGIC: [u8; 4] = *b"SRNX";
const MD: u8 = 1;
const ALLOC: u8 = 2;
const PACKED: u8 = 3;
const RELEASE: u8 = 4;

/// Request attachment after `SRNX`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NixlEnvelope {
    /// The sender's agent metadata. The receiver replies with its own.
    Md(Vec<u8>),
    /// Allocate receive buffers for a batch with this layout.
    Alloc(Vec<u8>),
    /// The buffers under `token` were written: one batch of `rows`. Token 0 carries no batch
    /// (an EOS frame); real tokens start at 1.
    Packed {
        token: u64,
        rows: u64,
        names: Vec<String>,
    },
    /// Free a receive token that will never be announced.
    Release(u64),
}

impl NixlEnvelope {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MAGIC);
        match self {
            Self::Md(blob) => {
                out.push(MD);
                out.extend_from_slice(blob);
            }
            Self::Alloc(layout) => {
                out.push(ALLOC);
                out.extend_from_slice(layout);
            }
            Self::Packed { token, rows, names } => {
                out.push(PACKED);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&rows.to_le_bytes());
                out.extend_from_slice(&(names.len() as u32).to_le_bytes());
                for name in names {
                    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
                    out.extend_from_slice(name.as_bytes());
                }
            }
            Self::Release(token) => {
                out.push(RELEASE);
                out.extend_from_slice(&token.to_le_bytes());
            }
        }
        out
    }

    /// Decodes a request attachment, rejecting truncated and trailing bytes.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader(bytes);
        if reader.take(4)? != MAGIC {
            return Err(
                "transmit_chunk attachment is not an SRNX envelope; native ChunkPB must not \
                 ride this path"
                    .to_string(),
            );
        }
        let envelope = match reader.take(1)?[0] {
            MD => Self::Md(reader.rest()),
            ALLOC => Self::Alloc(reader.rest()),
            PACKED => {
                let token = reader.u64()?;
                let rows = reader.u64()?;
                let count = reader.u32()?;
                let names = (0..count)
                    .map(|_| {
                        let len = reader.u32()? as usize;
                        String::from_utf8(reader.take(len)?.to_vec())
                            .map_err(|err| format!("SRNX Packed name is not utf-8: {err}"))
                    })
                    .collect::<Result<_, _>>()?;
                Self::Packed { token, rows, names }
            }
            RELEASE => Self::Release(reader.u64()?),
            kind => return Err(format!("unknown SRNX kind {kind}")),
        };
        reader.finish()?;
        Ok(envelope)
    }
}

/// Receive buffers a peer allocated for one batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocReply {
    /// Names the buffers in the peer's direct exchange.
    pub token: u64,
    /// The peer's CUDA device, which its buffers are registered under.
    pub device: i32,
    /// `(address, length)` of each buffer, pairing with the sender's.
    pub buffers: Vec<(u64, u64)>,
}

impl AllocReply {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + 16 * self.buffers.len());
        out.extend_from_slice(&self.token.to_le_bytes());
        out.extend_from_slice(&self.device.to_le_bytes());
        out.extend_from_slice(&(self.buffers.len() as u32).to_le_bytes());
        for (addr, len) in &self.buffers {
            out.extend_from_slice(&addr.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }
        out
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader(bytes);
        let token = reader.u64()?;
        let device = reader.u32()? as i32;
        let count = reader.u32()? as usize;
        if reader.0.len() != count.saturating_mul(16) {
            return Err(format!(
                "Alloc reply announces {count} buffers but carries {} bytes for them",
                reader.0.len()
            ));
        }
        let buffers = (0..count)
            .map(|_| Ok((reader.u64()?, reader.u64()?)))
            .collect::<Result<_, String>>()?;
        Ok(Self {
            token,
            device,
            buffers,
        })
    }
}

/// Little-endian cursor over an SRNX body.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.0.len() < len {
            return Err("SRNX envelope is truncated".to_string());
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn rest(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0).to_vec()
    }

    fn finish(&self) -> Result<(), String> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!("SRNX envelope has {} trailing bytes", self.0.len()))
        }
    }
}

/// This CN's side of the NIXL exchange: what `transmit_chunk` serves to peers, and the hop that
/// ships a parked sender output to one.
pub trait NixlEndpoint: Send + Sync + std::fmt::Debug {
    /// This agent's metadata, cached at bring-up. Serving it never waits on the transport thread,
    /// which would deadlock two CNs that open hops to each other at once.
    fn local_md(&self) -> Vec<u8>;

    /// Receive buffers for a batch with `layout`, without waiting for memory.
    fn allocate(&self, layout: &[u8]) -> Result<AllocReply, String>;

    /// Frees a receive token that will not be pushed. Unknown and consumed tokens are ignored.
    fn release(&self, token: u64);

    /// Receive and export buffers this CN's direct exchange still holds.
    fn outstanding(&self) -> usize;

    /// Lets a fully received batch spill to host while it waits for its receiver. The token is
    /// still pushed and released as before.
    fn seal(&self, token: u64) -> Result<(), String>;

    /// Writes every batch parked under `slot` into `peer`'s pool, announcing each, then EOS.
    fn send(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        names: Vec<String>,
        executor: Arc<dyn FragmentExecutor>,
    ) -> Result<(), String>;

    /// Writes every batch of each hop's drain into its peer's pool as the fragment produces it,
    /// serving all hops at once, then sends each EOS. Returns once every drain ended, or on the
    /// first error, after which no hop sends its EOS.
    fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String>;
}

/// One remote output streamed while its fragment runs.
#[derive(Debug)]
pub struct StreamHop {
    pub peer: SocketAddr,
    pub slot: SenderSlot,
    pub names: Vec<String>,
    pub drain: Box<dyn OutputDrain>,
}

/// Rejects StarRocks native shuffle so a stock BE `ChunkPB` cannot be treated as Sirius.
pub(crate) fn reject_native_chunk(params: &PTransmitChunkParams) -> Result<(), String> {
    if params.use_pass_through.unwrap_or(false) {
        return Err("pass-through transmit_chunk is not supported".to_string());
    }
    if params.is_pipeline_level_shuffle.unwrap_or(false) {
        return Err("pipeline-level shuffle transmit_chunk is not supported".to_string());
    }
    if !params.chunks.is_empty() {
        return Err(
            "transmit_chunk ChunkPB payloads are not supported; Sirius NIXL control must be in \
             the SRNX attachment"
                .to_string(),
        );
    }
    Ok(())
}

/// Params for Md, Alloc, and Release. Routing is unset so these cannot be ingested as shuffle.
pub(crate) fn control_params() -> PTransmitChunkParams {
    packed_params(None, 0, false)
}

/// Params for a Packed frame from `slot`'s sender to its receiver exchange.
pub(crate) fn packed_params(slot: Option<SenderSlot>, seq: i64, eos: bool) -> PTransmitChunkParams {
    PTransmitChunkParams {
        finst_id: slot.map(|slot| slot.fragment_instance_id.to_proto()),
        node_id: slot.map(|slot| slot.node_id),
        sender_id: slot.map(|slot| slot.sender_id),
        be_number: Some(0),
        eos: Some(eos),
        sequence: Some(seq),
        chunks: Vec::new(),
        query_statistics: None,
        use_pass_through: Some(false),
        is_pipeline_level_shuffle: Some(false),
        driver_sequences: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::starrocks::ChunkPb;
    use crate::result_store::FragmentInstanceId;

    #[test]
    fn envelopes_round_trip() {
        for envelope in [
            NixlEnvelope::Md(b"agent-md".to_vec()),
            NixlEnvelope::Alloc(b"SXD1-layout".to_vec()),
            NixlEnvelope::Packed {
                token: 9,
                rows: 7,
                names: vec!["region".into(), "sum".into()],
            },
            NixlEnvelope::Packed {
                token: 0,
                rows: 0,
                names: vec!["eos".into()],
            },
            NixlEnvelope::Release(9),
        ] {
            let encoded = envelope.encode();
            assert_eq!(&encoded[..4], b"SRNX");
            assert_eq!(NixlEnvelope::decode(&encoded).unwrap(), envelope);
        }
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        let mut trailing = NixlEnvelope::Release(9).encode();
        trailing.push(0);
        let packed = NixlEnvelope::Packed {
            token: 1,
            rows: 1,
            names: vec!["a".into()],
        }
        .encode();
        for bytes in [
            b"PRPC\x04".as_slice(),
            b"SRNX".as_slice(),
            b"SRNX\x09".as_slice(),
            b"SRNX\x04\x01\x00".as_slice(),
            trailing.as_slice(),
            &packed[..packed.len() - 1],
        ] {
            assert!(NixlEnvelope::decode(bytes).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn alloc_reply_round_trips_and_rejects_a_length_mismatch() {
        let reply = AllocReply {
            token: 3,
            device: 1,
            buffers: vec![(0xB000, 64), (0xC000, 8)],
        };
        let bytes = reply.encode();
        assert_eq!(AllocReply::decode(&bytes).unwrap(), reply);
        let mut trailing = bytes.clone();
        trailing.push(0);
        for bad in [&bytes[..bytes.len() - 1], trailing.as_slice(), &bytes[..12]] {
            assert!(AllocReply::decode(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn native_chunks_are_rejected() {
        assert!(reject_native_chunk(&control_params()).is_ok());
        let mut pass = control_params();
        pass.use_pass_through = Some(true);
        let mut pipeline = control_params();
        pipeline.is_pipeline_level_shuffle = Some(true);
        let mut chunks = control_params();
        chunks.chunks.push(ChunkPb::default());
        for params in [pass, pipeline, chunks] {
            assert!(reject_native_chunk(&params).is_err());
        }
    }

    #[test]
    fn packed_params_carry_the_slot_routing() {
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(11, 22),
            node_id: 2,
            sender_id: 1,
        };
        let params = packed_params(Some(slot), 3, true);
        assert!(reject_native_chunk(&params).is_ok());
        assert_eq!(
            FragmentInstanceId::from(params.finst_id.as_ref().unwrap()),
            slot.fragment_instance_id
        );
        assert_eq!(
            (
                params.node_id,
                params.sender_id,
                params.sequence,
                params.eos
            ),
            (Some(2), Some(1), Some(3), Some(true))
        );
        assert_eq!(control_params().finst_id, None);
    }
}
