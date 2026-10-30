//! Persist installer diagnostics independently of the UI's bounded live output.
use super::*;
use std::io::{BufReader, Write};

pub(super) struct RunLog {
    file: Option<fs::File>,
    pub(super) path: PathBuf,
}

impl RunLog {
    pub(super) fn create(store: &ModelStore) -> Result<Self> {
        let root = store.home().join("logs/components");
        fs::create_dir_all(&root)?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let (file, path) = tempfile::Builder::new()
            .prefix(&format!("run-{stamp}-"))
            .suffix(".log")
            .tempfile_in(root)?
            .keep()?;
        Ok(Self {
            file: Some(file),
            path,
        })
    }

    pub(super) fn record(&mut self, line: &str) -> Result<()> {
        if let Some(file) = &mut self.file {
            let result = writeln!(file, "{line}").and_then(|_| file.flush());
            if let Err(error) = result {
                self.file = None;
                return Err(error).with_context(|| format!("Cannot write {}", self.path.display()));
            }
        }
        Ok(())
    }
}

/// Drain both pipes concurrently, but deliver and persist lines on one thread.
pub(super) fn run(command: &mut Command, mut emit: impl FnMut(String)) -> Result<bool> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let out = child.stdout.take().context("Missing installer stdout")?;
    let err = child.stderr.take().context("Missing installer stderr")?;
    let (tx, rx) = mpsc::channel();
    let stream = |reader: Box<dyn Read + Send>, tx: mpsc::Sender<String>| {
        thread::spawn(move || {
            let mut line = Vec::new();
            for byte in BufReader::new(reader).bytes() {
                let byte = byte?;
                if byte == b'\n' || byte == b'\r' || line.len() >= 8192 {
                    if !line.is_empty() {
                        let _ = tx.send(crate::terminal::clean(&String::from_utf8_lossy(&line)));
                        line.clear();
                    }
                    if byte == b'\n' || byte == b'\r' {
                        continue;
                    }
                }
                line.push(byte);
            }
            if !line.is_empty() {
                let _ = tx.send(crate::terminal::clean(&String::from_utf8_lossy(&line)));
            }
            Ok::<_, io::Error>(())
        })
    };
    let a = stream(Box::new(out), tx.clone());
    let b = stream(Box::new(err), tx);
    for line in rx {
        emit(line);
    }
    let status = child.wait()?;
    a.join()
        .map_err(|_| anyhow::anyhow!("Installer stdout reader failed"))??;
    b.join()
        .map_err(|_| anyhow::anyhow!("Installer stderr reader failed"))??;
    emit(format!("Installer exited with {status}"));
    Ok(status.success())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn failed_installer_keeps_stdout_stderr_and_exit_status_after_reopen() {
        let temp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(temp.path().to_path_buf())).unwrap();
        let mut log = RunLog::create(&store).unwrap();
        let path = log.path.clone();
        let success = run(
            Command::new("sh").args([
                "-c",
                "printf 'progress\\r'; printf 'dependency conflict' >&2; exit 7",
            ]),
            |line| log.record(&line).unwrap(),
        )
        .unwrap();
        assert!(!success);
        drop(log);
        let saved = fs::read_to_string(path).unwrap();
        assert!(saved.contains("progress\n"));
        assert!(saved.contains("dependency conflict\n"));
        assert!(saved.contains("exit status: 7"));
    }
}
