#[tokio::main]
async fn main() -> std::process::ExitCode {
    match werk1112::cli::run_from_env().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            werk1112::cli::print_error(&error);
            std::process::ExitCode::FAILURE
        }
    }
}
