//! Optional vendor transports. No credentials or network needed by local models.
pub mod action;
pub mod cli;
pub(crate) mod http;
pub mod media;
#[cfg(test)]
pub(crate) mod tests;
