#[cfg(all(
    any(feature = "release-linux", feature = "release-windows"),
    not(feature = "candle-cuda")
))]
compile_error!("Linux and Windows release artifacts must compile Candle CUDA support.");

#[cfg(all(feature = "release-macos-apple-silicon", not(feature = "metal")))]
compile_error!("macOS Apple Silicon release artifacts must compile Candle Metal support.");

pub mod api;
pub mod api_keys;
pub mod backend;
pub mod banner;
pub mod cli;
pub mod model_store;
pub mod openai;
pub mod runtime_planner;
