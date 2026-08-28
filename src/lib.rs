#[cfg(all(
    any(
        feature = "release-linux",
        feature = "release-linux-aarch64",
        feature = "release-windows"
    ),
    not(feature = "candle-cuda")
))]
compile_error!("Linux and Windows release artifacts must compile Candle CUDA support.");

#[cfg(all(
    feature = "release-linux",
    not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))
))]
compile_error!("release-linux is only valid for the x86_64-unknown-linux-gnu target");

#[cfg(all(
    feature = "release-linux-aarch64",
    not(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu"))
))]
compile_error!("release-linux-aarch64 is only valid for the aarch64-unknown-linux-gnu target");

#[cfg(all(
    feature = "release-linux-strix-halo",
    not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))
))]
compile_error!("release-linux-strix-halo is only valid for the x86_64-unknown-linux-gnu target");

#[cfg(all(feature = "release-linux", feature = "release-linux-strix-halo"))]
compile_error!(
    "release-linux and release-linux-strix-halo are mutually exclusive platform profiles"
);

#[cfg(all(feature = "release-macos-apple-silicon", not(feature = "metal")))]
compile_error!("macOS Apple Silicon release artifacts must compile Candle Metal support.");

pub mod api;
pub mod api_keys;
pub mod backend;
pub mod banner;
pub mod capabilities;
pub mod cli;
pub mod inference;
pub mod inference_service;
pub mod media_cli;
pub mod media_companion;
pub mod model_store;
pub mod openai;
pub mod runtime_planner;
