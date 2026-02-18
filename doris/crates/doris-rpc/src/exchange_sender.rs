//! bRPC exchange sender: send PBlock data to remote BEs via `transmit_block`.
//!
//! When this BE executes a scan/leaf fragment whose output feeds an EXCHANGE_NODE
//! on a remote BE, we serialize the result into PBlocks and send them using
//! bRPC baidu_std wire format (same protocol as regular Doris BEs).
//!
//! Wire format (12-byte header per frame):
//!   bytes 0-3:  "PRPC" magic
//!   bytes 4-7:  body_size (u32 big-endian)
//!   bytes 8-11: meta_size (u32 big-endian)
//! Body = [meta (RpcMeta pb)] [payload (PTransmitDataParams pb)]

use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::info;

use doris_proto::brpc::policy::{RpcMeta, RpcRequestMeta};
use doris_proto::doris::{PBlock, PTransmitDataParams, PTransmitDataResult, PUniqueId};

const BRPC_MAGIC: &[u8; 4] = b"PRPC";
const HEADER_SIZE: usize = 12;

/// Destination for exchange data.
#[derive(Debug, Clone)]
pub struct ExchangeDest {
    /// Remote BE bRPC address (host:port).
    pub brpc_addr: String,
    /// Fragment instance ID on the remote BE.
    pub finst_id: (i64, i64),
}

/// Send a PBlock (or EOS) to a remote BE via bRPC transmit_block.
///
/// Opens a TCP connection, sends one bRPC frame, reads the response, and closes.
/// For efficiency in production, connection pooling could be added later.
pub async fn send_transmit_block(
    dest: &ExchangeDest,
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
    block: Option<PBlock>,
    eos: bool,
    packet_seq: i64,
) -> Result<(), String> {
    let finst_id = PUniqueId {
        hi: dest.finst_id.0,
        lo: dest.finst_id.1,
    };
    let q_id = PUniqueId {
        hi: query_id.0,
        lo: query_id.1,
    };

    let params = PTransmitDataParams {
        finst_id,
        node_id,
        sender_id,
        be_number: 0,
        eos,
        row_batch: None,
        packet_seq,
        query_statistics: None,
        block,
        transfer_by_attachment: None,
        query_id: Some(q_id),
        exec_status: None,
        blocks: vec![],
    };

    let payload = params.encode_to_vec();

    // Build RpcMeta for transmit_block
    let meta = RpcMeta {
        request: Some(RpcRequestMeta {
            service_name: Some("doris.PBackendService".to_string()),
            method_name: Some("transmit_block".to_string()),
            ..Default::default()
        }),
        response: None,
        correlation_id: Some(packet_seq),
        compress_type: None,
        attachment_size: None,
    };
    let meta_bytes = meta.encode_to_vec();

    let body_size = meta_bytes.len() + payload.len();

    // Build 12-byte header
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(BRPC_MAGIC);
    header[4..8].copy_from_slice(&(body_size as u32).to_be_bytes());
    header[8..12].copy_from_slice(&(meta_bytes.len() as u32).to_be_bytes());

    // Connect and send
    let mut stream = TcpStream::connect(&dest.brpc_addr)
        .await
        .map_err(|e| format!("connect to {}: {e}", dest.brpc_addr))?;

    stream
        .write_all(&header)
        .await
        .map_err(|e| format!("write header: {e}"))?;
    stream
        .write_all(&meta_bytes)
        .await
        .map_err(|e| format!("write meta: {e}"))?;
    stream
        .write_all(&payload)
        .await
        .map_err(|e| format!("write payload: {e}"))?;
    stream
        .flush()
        .await
        .map_err(|e| format!("flush: {e}"))?;

    // Read response
    let mut resp_header = [0u8; HEADER_SIZE];
    stream
        .read_exact(&mut resp_header)
        .await
        .map_err(|e| format!("read response header: {e}"))?;

    if &resp_header[0..4] != BRPC_MAGIC {
        return Err(format!(
            "bad response magic: {:?}",
            &resp_header[0..4]
        ));
    }

    let resp_body_size = u32::from_be_bytes(resp_header[4..8].try_into().unwrap()) as usize;
    let resp_meta_size = u32::from_be_bytes(resp_header[8..12].try_into().unwrap()) as usize;

    let mut resp_body = vec![0u8; resp_body_size];
    stream
        .read_exact(&mut resp_body)
        .await
        .map_err(|e| format!("read response body: {e}"))?;

    // Check RpcMeta for errors
    let resp_meta = RpcMeta::decode(&resp_body[..resp_meta_size])
        .map_err(|e| format!("decode response meta: {e}"))?;

    if let Some(resp) = &resp_meta.response {
        if resp.error_code.unwrap_or(0) != 0 {
            return Err(format!(
                "bRPC error {}: {}",
                resp.error_code.unwrap_or(-1),
                resp.error_text.as_deref().unwrap_or("unknown")
            ));
        }
    }

    // Parse PTransmitDataResult for status
    let payload_start = resp_meta_size;
    let payload_end = resp_body_size;
    if payload_end > payload_start {
        let result = PTransmitDataResult::decode(&resp_body[payload_start..payload_end])
            .map_err(|e| format!("decode response: {e}"))?;
        if let Some(status) = result.status {
            if status.status_code != 0 {
                return Err(format!(
                    "transmit_block error: {}",
                    status.error_msgs.join("; ")
                ));
            }
        }
    }

    Ok(())
}

/// Send Arrow IPC result to all exchange destinations as PBlocks.
///
/// Converts the IPC bytes to a PBlock, then sends to each destination
/// with data blocks followed by an EOS signal.
pub async fn send_exchange_result(
    ipc_bytes: &[u8],
    destinations: &[ExchangeDest],
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
) -> Result<(), String> {
    use super::arrow_to_pblock::arrow_ipc_to_pblock;

    let (pblock, num_rows) = arrow_ipc_to_pblock(ipc_bytes)?;

    info!(
        num_rows,
        num_dests = destinations.len(),
        node_id,
        "sending exchange result via bRPC"
    );

    // Send data block to all destinations
    for dest in destinations {
        if num_rows > 0 {
            send_transmit_block(
                dest,
                query_id,
                node_id,
                sender_id,
                Some(pblock.clone()),
                false,
                0,
            )
            .await
            .map_err(|e| format!("send data to {}: {e}", dest.brpc_addr))?;
        }

        // Send EOS
        send_transmit_block(dest, query_id, node_id, sender_id, None, true, 1)
            .await
            .map_err(|e| format!("send EOS to {}: {e}", dest.brpc_addr))?;

        info!(
            dest = %dest.brpc_addr,
            node_id,
            "bRPC exchange: transmit_block sent (rows={}, eos=true)",
            num_rows,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_dest_debug() {
        let dest = ExchangeDest {
            brpc_addr: "127.0.0.1:8060".to_string(),
            finst_id: (1, 2),
        };
        assert!(format!("{:?}", dest).contains("127.0.0.1:8060"));
    }
}
