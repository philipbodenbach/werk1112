fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        // Building the full Clap command tree in debug mode exceeds the Windows
        // main thread's default 1 MiB stack. Reserve 8 MiB for the CLI executable
        // in every profile; leave the committed size at the linker's default.
        println!("cargo:rustc-link-arg-bin=werk=/STACK:8388608");
    }
}
