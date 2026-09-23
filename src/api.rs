//! HTTP API façade.
//!
//! State, routing, chat, media handling, responses, and tests live in focused
//! submodules while the historical public API remains available here.

mod anthropic;
mod automatic1111;
mod chat;
mod cors;
mod generation;
mod media;
mod response;
mod router;
mod state;
mod werk;

#[cfg(test)]
mod tests;

pub use cors::CorsOrigin;
pub use router::{router, serve, serve_with_listener};
pub use state::{ApiState, PromptOptionsResolver};
