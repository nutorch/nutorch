//! Wire types for the PoC protocol: newline-delimited JSON over a Unix
//! socket. Deliberately throwaway (issue 0002) — debuggability over merit.
//!
//! Since issue 0005, table ops travel as a generic form:
//! `{"op":"<name>","tensors":["h1",…],"params":{…}}` →
//! `{"ok":true,"handles":["h1",…]}`. The bespoke ops (tensor, value, and the
//! lifecycle family) keep their dedicated shapes. Errors carry a
//! machine-readable `code` alongside the message.

use serde::{Deserialize, Serialize};

/// The bespoke (non-table) requests, deserialized by tag.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Bespoke {
    Tensor {
        data: serde_json::Value,
        dtype: Option<String>,
    },
    Value {
        handle: String,
    },
    /// Free tensors by handle, or all of them. Exactly one of the two
    /// fields must be meaningfully present (validated in dispatch;
    /// `all: false` counts as "not requested").
    Free {
        handles: Option<Vec<String>>,
        all: Option<bool>,
    },
    Status,
    #[serde(rename = "set_ttl")]
    SetTtl {
        ttl: String,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum Request {
    Bespoke(Bespoke),
    /// A table op (issue 0005): name resolved against nutorch_ops::OPS.
    Table {
        name: String,
        tensors: Vec<String>,
        params: serde_json::Map<String, serde_json::Value>,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Response {
    Handle {
        ok: bool,
        handle: String,
    },
    Handles {
        ok: bool,
        handles: Vec<String>,
    },
    Value {
        ok: bool,
        value: serde_json::Value,
    },
    Error {
        ok: bool,
        code: &'static str,
        error: String,
    },
}

impl Response {
    pub fn handle(handle: String) -> Self {
        Response::Handle { ok: true, handle }
    }

    pub fn handles(handles: Vec<String>) -> Self {
        Response::Handles { ok: true, handles }
    }

    pub fn value(value: serde_json::Value) -> Self {
        Response::Value { ok: true, value }
    }

    pub fn error(code: &'static str, message: impl Into<String>) -> Self {
        Response::Error {
            ok: false,
            code,
            error: message.into(),
        }
    }
}
