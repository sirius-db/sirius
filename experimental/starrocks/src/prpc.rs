use std::io::{self, Read, Write};

use anyhow::{Context, Result, anyhow};
use prost::Message;

const PRPC_MAGIC: &[u8; 4] = b"PRPC";
const PRPC_HEAD_SIZE: usize = 12;
const MAX_PRPC_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
const PRPC_SUCCESS: i32 = 0;
const PRPC_SERVICE_NOT_FOUND: i32 = 1001;
const PRPC_ERROR: i32 = 2001;

#[derive(Clone, PartialEq, Message)]
struct RpcMeta {
    #[prost(message, optional, tag = "1")]
    request: Option<RpcRequestMeta>,
    #[prost(message, optional, tag = "2")]
    response: Option<RpcResponseMeta>,
    #[prost(int32, optional, tag = "3")]
    compress_type: Option<i32>,
    #[prost(int64, optional, tag = "4")]
    correlation_id: Option<i64>,
    #[prost(int32, optional, tag = "5")]
    attachment_size: Option<i32>,
    #[prost(message, optional, tag = "6")]
    chunk_info: Option<ChunkInfo>,
    #[prost(bytes = "vec", optional, tag = "7")]
    authentication_data: Option<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct RpcRequestMeta {
    #[prost(string, required, tag = "1")]
    service_name: String,
    #[prost(string, required, tag = "2")]
    method_name: String,
    #[prost(int64, optional, tag = "3")]
    log_id: Option<i64>,
    #[prost(int64, optional, tag = "4")]
    trace_id: Option<i64>,
    #[prost(int64, optional, tag = "5")]
    span_id: Option<i64>,
    #[prost(int64, optional, tag = "6")]
    parent_span_id: Option<i64>,
    #[prost(message, repeated, tag = "7")]
    ext_fields: Vec<RpcRequestMetaExtField>,
    #[prost(bytes = "vec", optional, tag = "110")]
    extra_param: Option<Vec<u8>>,
    #[prost(string, optional, tag = "111")]
    trace_key: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct RpcRequestMetaExtField {
    #[prost(string, required, tag = "1")]
    key: String,
    #[prost(string, required, tag = "2")]
    value: String,
}

#[derive(Clone, PartialEq, Message)]
struct RpcResponseMeta {
    #[prost(int32, optional, tag = "1")]
    error_code: Option<i32>,
    #[prost(string, optional, tag = "2")]
    error_text: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct ChunkInfo {
    #[prost(int64, required, tag = "1")]
    stream_id: i64,
    #[prost(int64, required, tag = "2")]
    chunk_id: i64,
}

#[derive(Clone, Debug)]
pub(crate) struct Request {
    pub(crate) service_name: String,
    pub(crate) method_name: String,
    pub(crate) body: Vec<u8>,
    pub(crate) attachment: Vec<u8>,
}

impl Request {
    pub(crate) fn new(
        service_name: impl Into<String>,
        method_name: impl Into<String>,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> Self {
        Self {
            service_name: service_name.into(),
            method_name: method_name.into(),
            body,
            attachment,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Response {
    pub(crate) body: Vec<u8>,
    pub(crate) attachment: Vec<u8>,
}

impl Response {
    pub(crate) fn new(body: Vec<u8>) -> Self {
        Self {
            body,
            attachment: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Error {
    code: i32,
    text: String,
}

impl Error {
    pub(crate) fn service_not_found(text: impl Into<String>) -> Self {
        Self {
            code: PRPC_SERVICE_NOT_FOUND,
            text: text.into(),
        }
    }

    pub(crate) fn method_not_found(text: impl Into<String>) -> Self {
        Self::service_not_found(text)
    }

    pub(crate) fn method_not_implemented(method_name: impl std::fmt::Display) -> Self {
        Self::method_not_found(format!(
            "method '{method_name}' is not implemented by the Rust CN"
        ))
    }

    pub(crate) fn invalid_request(
        method_name: impl std::fmt::Display,
        err: impl std::fmt::Display,
    ) -> Self {
        Self::server(format!("invalid {method_name} request: {err}"))
    }

    fn server(text: impl Into<String>) -> Self {
        Self {
            code: PRPC_ERROR,
            text: text.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub(crate) struct Frame {
    meta: RpcMeta,
    body: Vec<u8>,
    attachment: Vec<u8>,
}

impl Frame {
    pub(crate) fn request(&self) -> std::result::Result<Request, Error> {
        let Some(request) = self.meta.request.as_ref() else {
            return Err(Error::server("missing PRPC request metadata"));
        };

        Ok(Request::new(
            request.service_name.clone(),
            request.method_name.clone(),
            self.body.clone(),
            self.attachment.clone(),
        ))
    }

    pub(crate) fn into_response_frame(
        self,
        response: std::result::Result<Response, Error>,
    ) -> Self {
        match response {
            Ok(response) => self.response(PRPC_SUCCESS, None, response.body, response.attachment),
            Err(error) => self.response(error.code, Some(error.text), Vec::new(), Vec::new()),
        }
    }

    fn response(
        &self,
        error_code: i32,
        error_text: Option<String>,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> Self {
        Self {
            meta: RpcMeta {
                request: None,
                response: Some(RpcResponseMeta {
                    error_code: Some(error_code),
                    error_text,
                }),
                compress_type: Some(0),
                correlation_id: self.meta.correlation_id,
                attachment_size: Some(attachment.len().min(i32::MAX as usize) as i32),
                chunk_info: None,
                authentication_data: None,
            },
            body,
            attachment,
        }
    }

    #[cfg(test)]
    fn for_request(
        service_name: impl Into<String>,
        method_name: impl Into<String>,
        body: Vec<u8>,
        attachment: Vec<u8>,
        correlation_id: Option<i64>,
    ) -> Self {
        Self {
            meta: RpcMeta {
                request: Some(RpcRequestMeta {
                    service_name: service_name.into(),
                    method_name: method_name.into(),
                    log_id: None,
                    trace_id: None,
                    span_id: None,
                    parent_span_id: None,
                    ext_fields: Vec::new(),
                    extra_param: None,
                    trace_key: None,
                }),
                response: None,
                compress_type: Some(0),
                correlation_id,
                attachment_size: None,
                chunk_info: None,
                authentication_data: None,
            },
            body,
            attachment,
        }
    }
}

pub(crate) fn read_frame(stream: &mut impl Read) -> Result<Option<Frame>> {
    let mut header = [0u8; PRPC_HEAD_SIZE];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::UnexpectedEof
                    | io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
            ) =>
        {
            return Ok(None);
        }
        Err(err) => return Err(err).context("failed to read PRPC header"),
    }

    if &header[..4] != PRPC_MAGIC {
        return Err(anyhow!("invalid PRPC magic code"));
    }

    let message_size = i32::from_be_bytes(header[4..8].try_into().unwrap());
    let meta_size = i32::from_be_bytes(header[8..12].try_into().unwrap());
    if message_size < 0 || meta_size < 0 {
        return Err(anyhow!("negative PRPC message size"));
    }
    let message_size = message_size as usize;
    let meta_size = meta_size as usize;
    if message_size > MAX_PRPC_MESSAGE_SIZE {
        return Err(anyhow!("PRPC message exceeds maximum size"));
    }
    if meta_size > message_size {
        return Err(anyhow!("PRPC meta size exceeds message size"));
    }

    let mut payload = vec![0u8; message_size];
    stream
        .read_exact(&mut payload)
        .context("failed to read PRPC body")?;
    let meta = RpcMeta::decode(&payload[..meta_size]).context("failed to decode PRPC metadata")?;

    if meta.compress_type.unwrap_or(0) != 0 {
        return Err(anyhow!(
            "compressed PRPC payloads are not supported by the Rust CN"
        ));
    }
    if meta.chunk_info.is_some() {
        return Err(anyhow!(
            "chunked PRPC payloads are not supported by the Rust CN"
        ));
    }

    let attachment_size = meta.attachment_size.unwrap_or(0);
    if attachment_size < 0 {
        return Err(anyhow!("negative PRPC attachment size"));
    }
    let attachment_size = attachment_size as usize;
    let body_size = message_size - meta_size;
    if attachment_size > body_size {
        return Err(anyhow!("PRPC attachment size exceeds payload size"));
    }

    let body_start = meta_size;
    let body_end = message_size - attachment_size;
    let body = payload[body_start..body_end].to_vec();
    let attachment = payload[body_end..].to_vec();

    Ok(Some(Frame {
        meta,
        body,
        attachment,
    }))
}

pub(crate) fn write_frame(stream: &mut impl Write, frame: &Frame) -> Result<()> {
    let bytes = encode_frame(frame);
    stream
        .write_all(&bytes)
        .context("failed to write PRPC response")?;
    stream.flush().context("failed to flush PRPC response")?;
    Ok(())
}

pub(crate) fn encode_frame(frame: &Frame) -> Vec<u8> {
    let mut meta = frame.meta.clone();
    meta.attachment_size = Some(frame.attachment.len().min(i32::MAX as usize) as i32);
    let meta_bytes = meta.encode_to_vec();
    let message_size = meta_bytes.len() + frame.body.len() + frame.attachment.len();

    let mut bytes = Vec::with_capacity(PRPC_HEAD_SIZE + message_size);
    bytes.extend_from_slice(PRPC_MAGIC);
    bytes.extend_from_slice(&(message_size as i32).to_be_bytes());
    bytes.extend_from_slice(&(meta_bytes.len() as i32).to_be_bytes());
    bytes.extend_from_slice(&meta_bytes);
    bytes.extend_from_slice(&frame.body);
    bytes.extend_from_slice(&frame.attachment);
    bytes
}

#[cfg(test)]
mod tests {
    use std::{
        net::{TcpListener, TcpStream},
        thread,
    };

    use super::*;

    #[test]
    fn frame_round_trip_preserves_attachment_and_correlation() {
        let request = Frame::for_request(
            "PInternalService",
            "exec_plan_fragment",
            b"request body".to_vec(),
            b"thrift attachment".to_vec(),
            Some(42),
        );

        let bytes = encode_frame(&request);
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            read_frame(&mut stream).unwrap().unwrap()
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(&bytes).unwrap();
        let decoded = join.join().unwrap();
        let decoded_request = decoded.request().unwrap();

        assert_eq!(decoded.meta.correlation_id, Some(42));
        assert_eq!(decoded_request.attachment, b"thrift attachment");
        assert_eq!(decoded_request.body, b"request body");
    }
}
