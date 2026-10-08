use crate::terminal::{self, Stream};
use std::{
    fmt,
    io::{self, Write},
};

/// Keep terminal styling out of backend diagnostics and API responses.
#[derive(Debug)]
pub(super) struct RuntimeSetupRequired {
    pub detail: String,
    pub backend: String,
}

impl fmt::Display for RuntimeSetupRequired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for RuntimeSetupRequired {}

impl RuntimeSetupRequired {
    fn body(&self) -> String {
        let (dependencies, install, alternative) = if self.backend == "vllm" {
            ("vLLM", "werk backend install vllm", "WERK_VLLM_PYTHON")
        } else {
            (
                "Text-analysis",
                "werk backend install text-analysis",
                "WERK_TEXT_PYTHON",
            )
        };
        format!(
            "{dependencies} dependencies are missing or incompatible.\n\nINSTALL\n  {install}\n\nNEXT\nRestart your werk serve command after installation.\n\nALTERNATIVE\nSet {alternative} to a compatible Python environment."
        )
    }
}

pub(super) fn print_error(error: &anyhow::Error) {
    if !terminal::interactive(Stream::Err) {
        let _ = writeln!(io::stderr().lock(), "Error: {error:?}");
    } else if let Some(setup) = error.downcast_ref::<RuntimeSetupRequired>() {
        terminal::panel(Stream::Err, "Backend setup required", &setup.body());
    } else {
        terminal::panel(
            Stream::Err,
            "Werk could not complete this command",
            &format!("{error:?}"),
        );
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn console_modes_keep_all_structured_outputs_unadorned() {
        for args in [
            vec!["werk", "list", "--json"],
            vec!["werk", "gpus"],
            vec!["werk", "inspect", "test/model"],
            vec!["werk", "runtime", "info"],
            vec!["werk", "temp", "path"],
            vec!["werk", "parameters", "--example"],
            vec!["werk", "parameters", "--sources"],
            vec!["werk", "top", "--once", "--json"],
            vec!["werk", "run", "test/model", "hello", "--json"],
            vec!["werk", "cache", "list", "--json"],
        ] {
            let matches = super::super::Cli::command()
                .try_get_matches_from(&args)
                .unwrap();
            assert!(super::super::console_command(&matches).1, "{args:?}");
        }
        for args in [
            vec!["werk"],
            vec!["werk", "serve"],
            vec!["werk", "list"],
            vec!["werk", "backend", "list"],
            vec!["werk", "chat", "test/model"],
        ] {
            let matches = super::super::Cli::command()
                .try_get_matches_from(&args)
                .unwrap();
            assert!(!super::super::console_command(&matches).1, "{args:?}");
        }
    }
}
