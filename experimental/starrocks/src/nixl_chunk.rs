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
//! | `Packed` | `u64 token, u64 rows, u32 n, n x (u32 len, utf-8 name)` | empty |
//! | `Topology` | `i64 query hi, i64 query lo, i32 filter id` | see [`encode_topology`] |
//! | `ShareAbandoned` | `i64 instance hi, i64 instance lo, i32 filter id` | empty |
//! | `RingOpen` | see [`crate::ring_wire::LaneOpen`] | see [`crate::ring_wire::LaneGrant`] |
//! | `RingClose` | `u64 lane` | empty |

use std::net::SocketAddr;
use std::sync::Arc;

use starrocks_thrift::types::TNetworkAddress;

use crate::fragment_executor::{FragmentExecutor, OutputDrain, SenderSlot};
use crate::proto::starrocks::PTransmitChunkParams;
use crate::result_store::FragmentInstanceId;
use crate::ring_wire::{Reader, put_bytes, put_strings};
use crate::runtime_filters::FilterTopology;

const MAGIC: [u8; 4] = *b"SRNX";
const MD: u8 = 1;
const PACKED: u8 = 3;
const TOPOLOGY: u8 = 5;
const SHARE_ABANDONED: u8 = 6;
const RING_OPEN: u8 = 8;
const RING_CLOSE: u8 = 9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NixlEnvelope {
    /// The sender's agent metadata. The receiver replies with its own.
    Md(Vec<u8>),
    /// One batch of `rows` landed under `token` on the receiver. Token 0 carries no batch
    /// (an EOS frame); real tokens start at 1.
    Packed {
        token: u64,
        rows: u64,
        names: Vec<String>,
    },
    /// Who probes a partitioned join's runtime filter, asked of its merge node.
    Topology {
        query: FragmentInstanceId,
        filter_id: i32,
    },
    /// A holder of runtime filter `filter_id` won't send its share to scan `instance`, which
    /// stops waiting and runs unfiltered.
    ShareAbandoned {
        instance: FragmentInstanceId,
        filter_id: i32,
    },
    /// Open a ring-exchange lane for the sender routed by the params: the body is a
    /// [`crate::ring_wire::LaneOpen`], the reply a [`crate::ring_wire::LaneGrant`].
    RingOpen(Vec<u8>),
    /// The sender is done with ring lane `lane`; its frames go back to the receiver.
    RingClose(u64),
}

impl NixlEnvelope {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MAGIC);
        match self {
            Self::Md(blob) => {
                out.push(MD);
                out.extend_from_slice(blob);
            }
            Self::Packed { token, rows, names } => {
                out.push(PACKED);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&rows.to_le_bytes());
                put_strings(&mut out, names);
            }
            Self::RingOpen(body) => {
                out.push(RING_OPEN);
                out.extend_from_slice(body);
            }
            Self::RingClose(lane) => {
                out.push(RING_CLOSE);
                out.extend_from_slice(&lane.to_le_bytes());
            }
            Self::Topology {
                query: id,
                filter_id,
            }
            | Self::ShareAbandoned {
                instance: id,
                filter_id,
            } => {
                out.push(if matches!(self, Self::Topology { .. }) {
                    TOPOLOGY
                } else {
                    SHARE_ABANDONED
                });
                let id = id.to_proto();
                out.extend_from_slice(&id.hi.to_le_bytes());
                out.extend_from_slice(&id.lo.to_le_bytes());
                out.extend_from_slice(&filter_id.to_le_bytes());
            }
        }
        out
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes, "SRNX envelope");
        if reader.take(4)? != MAGIC {
            return Err(
                "transmit_chunk attachment is not an SRNX envelope; native ChunkPB must not \
                 ride this path"
                    .to_string(),
            );
        }
        let envelope = match reader.u8()? {
            MD => Self::Md(reader.rest()),
            PACKED => Self::Packed {
                token: reader.u64()?,
                rows: reader.u64()?,
                names: reader.strings("SRNX Packed name")?,
            },
            TOPOLOGY => Self::Topology {
                query: FragmentInstanceId::from_halves(reader.u64()? as i64, reader.u64()? as i64),
                filter_id: reader.u32()? as i32,
            },
            SHARE_ABANDONED => Self::ShareAbandoned {
                instance: FragmentInstanceId::from_halves(
                    reader.u64()? as i64,
                    reader.u64()? as i64,
                ),
                filter_id: reader.u32()? as i32,
            },
            RING_OPEN => Self::RingOpen(reader.rest()),
            RING_CLOSE => Self::RingClose(reader.u64()?),
            kind => return Err(format!("unknown SRNX kind {kind}")),
        };
        reader.finish()?;
        Ok(envelope)
    }
}

/// The `Topology` reply: `u8 known`; when known, `u32 shares, u32 n`, then per prober
/// `i64 instance hi, i64 instance lo, i32 port, u32 len, utf-8 host`.
pub(crate) fn encode_topology(topology: Option<&FilterTopology>) -> Vec<u8> {
    let Some(topology) = topology else {
        return vec![0];
    };
    let mut out = vec![1];
    out.extend_from_slice(&(topology.shares as u32).to_le_bytes());
    out.extend_from_slice(&(topology.probers.len() as u32).to_le_bytes());
    for (instance, address) in &topology.probers {
        let instance = instance.to_proto();
        out.extend_from_slice(&instance.hi.to_le_bytes());
        out.extend_from_slice(&instance.lo.to_le_bytes());
        out.extend_from_slice(&address.port.to_le_bytes());
        put_bytes(&mut out, address.hostname.as_bytes());
    }
    out
}

pub(crate) fn decode_topology(bytes: &[u8]) -> Result<Option<FilterTopology>, String> {
    let mut reader = Reader::new(bytes, "SRNX envelope");
    if reader.take(1)?[0] == 0 {
        reader.finish()?;
        return Ok(None);
    }
    let shares = reader.u32()? as usize;
    let count = reader.u32()?;
    let probers = (0..count)
        .map(|_| {
            let instance =
                FragmentInstanceId::from_halves(reader.u64()? as i64, reader.u64()? as i64);
            let port = reader.u32()? as i32;
            let host = reader.string("SRNX Topology host")?;
            Ok((instance, TNetworkAddress::new(host, port)))
        })
        .collect::<Result<_, String>>()?;
    reader.finish()?;
    Ok(Some(FilterTopology { shares, probers }))
}

/// This CN's side of the NIXL exchange: what `transmit_chunk` serves to peers, and the hop that
/// ships a parked sender output to one.
pub trait NixlEndpoint: Send + Sync + std::fmt::Debug {
    /// This agent's metadata, cached at bring-up. Serving it never waits on the transport thread,
    /// which would deadlock two CNs that open hops to each other at once.
    fn local_md(&self) -> Vec<u8>;

    /// Frees a receive token that will not be pushed. Unknown and consumed tokens are ignored.
    fn release(&self, token: u64);

    /// Received batches this CN's exchange inbox still holds.
    fn outstanding(&self) -> usize;

    /// Ships every batch parked under `slot` to `peer`, then EOS. Blocks until the hop ends. On
    /// error the peer gets no EOS.
    fn send(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        names: Vec<String>,
        executor: Arc<dyn FragmentExecutor>,
    ) -> Result<(), String>;

    /// Ships every batch of each hop's drain to its peer as the fragment produces it, serving all
    /// hops at once, then sends each EOS. Returns once every drain ended, or on the
    /// first error, after which no hop sends its EOS.
    fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String>;

    /// Opens a ring-exchange lane for the sender `request` routes, from a `RingOpen` body;
    /// returns the encoded grant. Only the ring transport has lanes.
    fn ring_open(&self, _request: &PTransmitChunkParams, _body: &[u8]) -> Result<Vec<u8>, String> {
        Err("this CN's exchange transport has no ring lanes".to_string())
    }

    /// Takes back ring lane `lane`'s frames once its sender is done with it.
    fn ring_close(&self, _lane: u64) -> Result<(), String> {
        Err("this CN's exchange transport has no ring lanes".to_string())
    }

    /// Where the ring transport hands the batches it lands. Other transports land nothing.
    fn set_delivery(&self, _delivery: std::sync::Arc<dyn RingDelivery>) {}

    /// Stops the transport and drops its delivery, which holds the service. Idempotent.
    fn shut_down(&self) {}

    /// Sends `peer` a control message (`Topology`, `ShareAbandoned`) and returns its reply.
    fn control(&self, peer: SocketAddr, envelope: &NixlEnvelope) -> Result<Vec<u8>, String>;
}

#[derive(Debug)]
pub struct StreamHop {
    pub peer: SocketAddr,
    pub slot: SenderSlot,
    pub names: Vec<String>,
    pub drain: Box<dyn OutputDrain>,
}

/// Where the ring transport hands a batch it landed whole: `token` holds it in this CN's direct
/// exchange, and `request` routes it like a Packed frame with that sequence number.
pub trait RingDelivery: Send + Sync {
    /// Pushes the batch to its receiver; on an error the caller releases the token.
    fn deliver(
        &self,
        request: &PTransmitChunkParams,
        token: u64,
        rows: u64,
        names: &[String],
    ) -> Result<(), String>;
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

/// Params for Md and RingClose. Routing is unset so these cannot be ingested as shuffle.
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
            NixlEnvelope::Topology {
                query: FragmentInstanceId::from_halves(-3, 1 << 40),
                filter_id: 7,
            },
            NixlEnvelope::ShareAbandoned {
                instance: FragmentInstanceId::from_halves(4, -2),
                filter_id: 3,
            },
            NixlEnvelope::RingOpen(b"lane-open".to_vec()),
            NixlEnvelope::RingClose(9),
        ] {
            let encoded = envelope.encode();
            assert_eq!(&encoded[..4], b"SRNX");
            assert_eq!(NixlEnvelope::decode(&encoded).unwrap(), envelope);
        }
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        let mut trailing = NixlEnvelope::RingClose(9).encode();
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
    fn a_topology_reply_round_trips_and_rejects_trailing_bytes() {
        let topology = FilterTopology {
            shares: 4,
            probers: vec![
                (
                    FragmentInstanceId::from_halves(5, -1),
                    TNetworkAddress::new("127.0.0.1".to_string(), 9102),
                ),
                (
                    FragmentInstanceId::from_halves(5, 2),
                    TNetworkAddress::new("cn-3".to_string(), 9132),
                ),
            ],
        };
        let bytes = encode_topology(Some(&topology));
        assert_eq!(decode_topology(&bytes).unwrap(), Some(topology));
        assert_eq!(decode_topology(&encode_topology(None)).unwrap(), None);
        let mut trailing = bytes.clone();
        trailing.push(0);
        for bad in [&bytes[..bytes.len() - 1], trailing.as_slice(), &[][..]] {
            assert!(decode_topology(bad).is_err(), "{bad:?}");
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
