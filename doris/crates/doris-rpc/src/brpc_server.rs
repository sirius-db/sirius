//! bRPC "baidu_std" protocol handler for inter-BE exchange.
//!
//! Regular Doris BEs use bRPC (not gRPC) for `transmit_block` calls.
//! This module parses the baidu_std wire format and dispatches to our
//! existing ExchangeBuffer.
//!
//! Wire format (12-byte header per frame):
//!   bytes 0-3:  "PRPC" magic
//!   bytes 4-7:  body_size (u32 big-endian)
//!   bytes 8-11: meta_size (u32 big-endian)
//! Body = [meta (RpcMeta pb)] [payload (request/response pb)] [attachment]

use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{info, instrument, warn};

use crate::exchange_buffer::{ExchangeBuffer, ExchangeKey};
use doris_proto::brpc::policy::{RpcMeta, RpcResponseMeta};
use doris_proto::doris::{PStatus, PTransmitDataParams, PTransmitDataResult};

const BRPC_MAGIC: &[u8; 4] = b"PRPC";
const HEADER_SIZE: usize = 12;

/// Handle a single bRPC connection (multiple sequential frames).
#[instrument(skip_all, fields(peer = %stream.peer_addr().map_or("unknown".to_string(), |a| a.to_string())))]
pub async fn handle_brpc_connection(mut stream: TcpStream, exchange_buffer: ExchangeBuffer) {
    info!("bRPC connection accepted");
    loop {
        match handle_one_frame(&mut stream, &exchange_buffer).await {
            Ok(true) => continue,
            Ok(false) => break, // clean EOF
            Err(e) => {
                warn!(error = %e, "bRPC frame error");
                break;
            }
        }
    }
}

/// Process one bRPC frame. Returns Ok(true) to continue, Ok(false) on EOF.
async fn handle_one_frame(
    stream: &mut TcpStream,
    exchange_buffer: &ExchangeBuffer,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    // Read 12-byte header.
    let mut header = [0u8; HEADER_SIZE];
    match stream.read_exact(&mut header).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e.into()),
    }

    // Validate magic.
    if &header[0..4] != BRPC_MAGIC {
        return Err(format!("bad bRPC magic: {:?}", &header[0..4]).into());
    }

    let body_size = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
    let meta_size = u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize;

    if body_size == 0 {
        return Ok(true); // keepalive/empty frame
    }

    // Read entire body.
    let mut body = vec![0u8; body_size];
    stream.read_exact(&mut body).await?;

    // Parse RpcMeta from first meta_size bytes.
    let meta = RpcMeta::decode(&body[..meta_size])?;
    let correlation_id = meta.correlation_id.unwrap_or(0);

    let req_meta = meta.request.as_ref();
    let service = req_meta.and_then(|m| m.service_name.as_deref()).unwrap_or("");
    let method = req_meta.and_then(|m| m.method_name.as_deref()).unwrap_or("");

    // Payload is after meta, before any attachment.
    let attachment_size = meta.attachment_size.unwrap_or(0) as usize;
    let payload_end = body_size - attachment_size;
    let payload = &body[meta_size..payload_end];

    // Dispatch.
    let (resp_bytes, error_code, error_text) = match (service, method) {
        (_, "transmit_block") => {
            match handle_transmit_block(payload, exchange_buffer) {
                Ok(resp) => (resp.encode_to_vec(), 0, String::new()),
                Err(e) => {
                    let err_msg = e.to_string();
                    let resp = PTransmitDataResult {
                        status: Some(PStatus {
                            status_code: 1, // INTERNAL_ERROR
                            error_msgs: vec![err_msg.clone()],
                        }),
                        ..Default::default()
                    };
                    (resp.encode_to_vec(), 1, err_msg)
                }
            }
        }
        _ => {
            // Unknown method — return error response.
            let err_msg = format!("unsupported bRPC method: {}/{}", service, method);
            warn!(%service, %method, "unknown bRPC method");
            (Vec::new(), -1, err_msg)
        }
    };

    // Build response.
    let resp_meta = RpcMeta {
        response: Some(RpcResponseMeta {
            error_code: Some(error_code),
            error_text: if error_text.is_empty() {
                None
            } else {
                Some(error_text)
            },
        }),
        correlation_id: Some(correlation_id),
        ..Default::default()
    };
    let resp_meta_bytes = resp_meta.encode_to_vec();
    let resp_body_size = resp_meta_bytes.len() + resp_bytes.len();

    // Write 12-byte response header.
    let mut resp_header = [0u8; HEADER_SIZE];
    resp_header[0..4].copy_from_slice(BRPC_MAGIC);
    resp_header[4..8].copy_from_slice(&(resp_body_size as u32).to_be_bytes());
    resp_header[8..12].copy_from_slice(&(resp_meta_bytes.len() as u32).to_be_bytes());

    stream.write_all(&resp_header).await?;
    stream.write_all(&resp_meta_bytes).await?;
    stream.write_all(&resp_bytes).await?;
    stream.flush().await?;

    Ok(true)
}

/// Handle a transmit_block request — same logic as the gRPC handler.
fn handle_transmit_block(
    payload: &[u8],
    exchange_buffer: &ExchangeBuffer,
) -> Result<PTransmitDataResult, Box<dyn std::error::Error + Send + Sync>> {
    let req = PTransmitDataParams::decode(payload)?;

    let query_id = req
        .query_id
        .as_ref()
        .map(|id| (id.hi, id.lo))
        .unwrap_or((0, 0));
    let key = ExchangeKey {
        query_id,
        node_id: req.node_id,
    };

    let mut block_count = 0u32;

    // Buffer single block (old style).
    if let Some(block) = req.block {
        block_count += 1;
        exchange_buffer.add_block(&key, req.sender_id, Some(block), false);
    }

    // Buffer multiple blocks (new style).
    for block in req.blocks {
        block_count += 1;
        exchange_buffer.add_block(&key, req.sender_id, Some(block), false);
    }

    // Signal EOS for this sender.
    let all_done = if req.eos {
        exchange_buffer.add_block(&key, req.sender_id, None, true)
    } else {
        false
    };

    info!(
        ?query_id,
        node_id = req.node_id,
        sender_id = req.sender_id,
        block_count,
        eos = req.eos,
        all_done,
        "bRPC transmit_block"
    );

    Ok(PTransmitDataResult {
        status: Some(PStatus {
            status_code: 0, // OK
            error_msgs: vec![],
        }),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use doris_proto::doris::{PBlock, PUniqueId};
    use prost::Message;

    fn empty_block() -> PBlock {
        PBlock {
            column_metas: vec![],
            column_values: None,
            compressed: None,
            uncompressed_size: None,
            compression_type: None,
            be_exec_version: None,
        }
    }

    #[test]
    fn test_handle_transmit_block_single_block() {
        let buf = ExchangeBuffer::new();
        let k = ExchangeKey {
            query_id: (100, 200),
            node_id: 5,
        };
        buf.register(k.clone(), 1);

        let req = PTransmitDataParams {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            node_id: 5,
            sender_id: 0,
            be_number: 0,
            eos: false,
            row_batch: None,
            packet_seq: 0,
            query_statistics: None,
            block: Some(empty_block()),
            transfer_by_attachment: None,
            query_id: Some(PUniqueId { hi: 100, lo: 200 }),
            exec_status: None,
            blocks: vec![],
        };

        let payload = req.encode_to_vec();
        let result = handle_transmit_block(&payload, &buf).unwrap();
        assert_eq!(result.status.unwrap().status_code, 0);

        // Block should be buffered but not done yet (no EOS).
        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_handle_transmit_block_eos() {
        let buf = ExchangeBuffer::new();
        let k = ExchangeKey {
            query_id: (10, 20),
            node_id: 3,
        };
        buf.register(k.clone(), 1);

        // Send data block.
        let data_req = PTransmitDataParams {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            node_id: 3,
            sender_id: 0,
            be_number: 0,
            eos: false,
            row_batch: None,
            packet_seq: 0,
            query_statistics: None,
            block: Some(empty_block()),
            transfer_by_attachment: None,
            query_id: Some(PUniqueId { hi: 10, lo: 20 }),
            exec_status: None,
            blocks: vec![],
        };
        handle_transmit_block(&data_req.encode_to_vec(), &buf).unwrap();

        // Send EOS.
        let eos_req = PTransmitDataParams {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            node_id: 3,
            sender_id: 0,
            be_number: 0,
            eos: true,
            row_batch: None,
            packet_seq: 1,
            query_statistics: None,
            block: None,
            transfer_by_attachment: None,
            query_id: Some(PUniqueId { hi: 10, lo: 20 }),
            exec_status: None,
            blocks: vec![],
        };
        handle_transmit_block(&eos_req.encode_to_vec(), &buf).unwrap();

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_handle_transmit_block_multiple_blocks_new_style() {
        let buf = ExchangeBuffer::new();
        let k = ExchangeKey {
            query_id: (1, 2),
            node_id: 1,
        };
        buf.register(k.clone(), 1);

        // Send multiple blocks in `blocks` field (new style).
        let req = PTransmitDataParams {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            node_id: 1,
            sender_id: 0,
            be_number: 0,
            eos: true,
            row_batch: None,
            packet_seq: 0,
            query_statistics: None,
            block: None,
            transfer_by_attachment: None,
            query_id: Some(PUniqueId { hi: 1, lo: 2 }),
            exec_status: None,
            blocks: vec![empty_block(), empty_block(), empty_block()],
        };

        handle_transmit_block(&req.encode_to_vec(), &buf).unwrap();

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_handle_transmit_block_no_query_id() {
        // When query_id is None, should default to (0, 0).
        let buf = ExchangeBuffer::new();
        let k = ExchangeKey {
            query_id: (0, 0),
            node_id: 1,
        };
        buf.register(k.clone(), 1);

        let req = PTransmitDataParams {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            node_id: 1,
            sender_id: 0,
            be_number: 0,
            eos: true,
            row_batch: None,
            packet_seq: 0,
            query_statistics: None,
            block: Some(empty_block()),
            transfer_by_attachment: None,
            query_id: None, // no query_id
            exec_status: None,
            blocks: vec![],
        };

        handle_transmit_block(&req.encode_to_vec(), &buf).unwrap();

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
    }
}
