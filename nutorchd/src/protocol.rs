//! Wire types for the PoC protocol: newline-delimited JSON over a Unix
//! socket. Deliberately throwaway (issue 0002) — debuggability over merit.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Request {
    Tensor {
        data: serde_json::Value,
        device: Option<String>,
        dtype: Option<String>,
    },
    Value {
        handle: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Response {
    Handle { ok: bool, handle: String },
    Value { ok: bool, value: serde_json::Value },
    Error { ok: bool, error: String },
}

impl Response {
    pub fn handle(handle: String) -> Self {
        Response::Handle { ok: true, handle }
    }

    pub fn value(value: serde_json::Value) -> Self {
        Response::Value { ok: true, value }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Response::Error {
            ok: false,
            error: message.into(),
        }
    }
}
