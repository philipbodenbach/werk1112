//! Transport-neutral Werk runtime control protocol.
//!
//! Axum is intentionally absent from this module. Transports adapt the typed
//! contract exposed here instead of defining runtime semantics themselves.

mod client;
mod service;
mod types;

pub use client::{ClientError, WerkProtocolClient};
pub use service::{
    BoxControlFuture, ControlContext, ProtocolError, ProtocolErrorBody, ProtocolErrorCode,
    ProtocolResult, WerkControl,
};
pub use types::*;
