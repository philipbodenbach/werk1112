//! HTTP API façade.
//!
//! State, routing, chat, media handling, responses, and tests live in focused
//! submodules while the historical public API remains available here.

mod automatic1111;
mod chat;
mod cors;
mod media;
mod response;
mod router;
mod state;
mod werk;

#[cfg(test)]
mod tests;

pub use cors::CorsOrigin;
pub use router::{router, serve};
pub use state::{ApiState, PromptOptionsResolver};
