//! HTTP API façade.
//!
//! State, routing, chat, media handling, responses, and tests live in focused
//! submodules while the historical public API remains available here.

mod anthropic;
mod automatic1111;
mod chat;
mod cors;
mod documents;
mod extended;
mod files;
mod generation;
mod media;
mod observability;
mod response;
mod router;
mod state;
mod tools;
mod werk;

#[cfg(test)]
mod tests;

pub use cors::CorsOrigin;
pub use router::{router, serve, serve_with_listener};
pub use state::{ApiState, PromptOptionsResolver};
