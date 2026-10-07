//! Shared console presentation. Enabled by the CLI only; library users, protocol
//! payloads and redirected streams keep their original bytes.
use std::{
    env, fmt,
    io::{self, BufRead, IsTerminal, Write},
    process::{Command, ExitStatus, Stdio},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

pub const CYAN: (u8, u8, u8) = (0, 220, 255);
pub const BLUE: (u8, u8, u8) = (59, 130, 246);
pub const INDIGO: (u8, u8, u8) = (99, 102, 241);
pub const VIOLET: (u8, u8, u8) = (139, 92, 246);
pub const PINK: (u8, u8, u8) = (255, 79, 195);
static ENABLED: AtomicBool = AtomicBool::new(false);
static HUMAN_STDOUT: AtomicBool = AtomicBool::new(false);
static OUTPUT: Mutex<()> = Mutex::new(());
static TITLE: OnceLock<String> = OnceLock::new();
static PROGRESS: OnceLock<indicatif::MultiProgress> = OnceLock::new();

pub fn progress(bar: indicatif::ProgressBar) -> indicatif::ProgressBar {
    PROGRESS.get_or_init(indicatif::MultiProgress::new).add(bar)
}

#[derive(Clone, Copy)]
pub enum Stream {
    Out,
    Err,
}

pub fn init(human_stdout: bool, title: String) {
    ENABLED.store(true, Ordering::Relaxed);
    HUMAN_STDOUT.store(human_stdout, Ordering::Relaxed);
    let _ = TITLE.set(title);
}

pub fn session_heading() {
    if let Some(title) = TITLE.get() {
        heading(
            Stream::Out,
            &format!("WERK · {title} · v{}", env!("CARGO_PKG_VERSION")),
        );
    }
}

pub fn interactive(stream: Stream) -> bool {
    ENABLED.load(Ordering::Relaxed)
        && !env::var("TERM").is_ok_and(|value| value.eq_ignore_ascii_case("dumb"))
        && match stream {
            Stream::Out => HUMAN_STDOUT.load(Ordering::Relaxed) && io::stdout().is_terminal(),
            Stream::Err => io::stderr().is_terminal(),
        }
}

pub fn color(stream: Stream) -> bool {
    interactive(stream) && env::var_os("NO_COLOR").is_none()
}

pub fn paint(text: &str, rgb: (u8, u8, u8), bold: bool, enabled: bool) -> String {
    if !enabled {
        return text.to_string();
    }
    let (r, g, b) = rgb;
    format!(
        "\x1b[{}38;2;{r};{g};{b}m{text}\x1b[0m",
        if bold { "1;" } else { "" }
    )
}

pub fn clean(text: &str) -> String {
    console::strip_ansi_codes(text)
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\n' | '\t'))
        .collect()
}

fn highlight(text: &str, colors: bool) -> String {
    let lower = text.trim_start().to_ascii_lowercase();
    if lower.starts_with("error") || lower.starts_with("failed") || lower.starts_with("warning") {
        return paint(text, PINK, false, colors);
    }
    // Preserve padding, so tables stay aligned after styling.
    let mut result = String::new();
    for word in text.split_inclusive(char::is_whitespace) {
        let trimmed = word.trim_end();
        let tint = match trimmed
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase()
            .as_str()
        {
            "ok" | "ready" | "available" | "installed" | "enabled" | "success" | "running"
            | "200" | "201" | "202" | "204" => Some(CYAN),
            "failed" | "error" | "missing" | "unavailable" | "warn" | "warning" | "rejected"
            | "disabled" | "400" | "401" | "403" | "404" | "422" | "429" | "500" | "503" => {
                Some(PINK)
            }
            "optional" | "pending" | "loading" | "fallback" => Some(VIOLET),
            _ if trimmed.starts_with("http://") || trimmed.starts_with("https://") => Some(CYAN),
            _ => None,
        };
        if let Some(tint) = tint {
            result.push_str(&paint(trimmed, tint, false, colors));
            result.push_str(&word[trimmed.len()..]);
        } else {
            result.push_str(word);
        }
    }
    // Commands and inline code are the actionable part of a diagnostic.
    if let Some(start) = text.find("werk ")
        && (start == 0 || text[..start].ends_with(|ch: char| ch.is_whitespace() || ch == '`'))
    {
        let end = text[start..]
            .find(['`', ';', '\n'])
            .map_or(text.len(), |n| start + n);
        return format!(
            "{}{}{}",
            &text[..start],
            paint(&text[start..end], CYAN, true, colors),
            &text[end..]
        );
    }
    result
}

#[cfg(test)]
pub(crate) fn render(text: &str, colors: bool) -> String {
    render_width(text, colors, usize::MAX)
}

fn wrapped_lines(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for line in text.lines() {
        // Keep commands and complete unbroken paths copyable.
        if console::measure_text_width(line) <= width || line.trim_start().starts_with("werk ") {
            lines.push(line.to_string());
            continue;
        }
        let mut current = String::new();
        for part in line.split_inclusive(char::is_whitespace) {
            if !current.trim().is_empty()
                && console::measure_text_width(&current)
                    + console::measure_text_width(part.trim_end())
                    > width
            {
                lines.push(current.trim_end().to_string());
                current = "  ".into();
            }
            current.push_str(part);
        }
        lines.push(current.trim_end().to_string());
    }
    lines
}

fn render_width(text: &str, colors: bool, width: usize) -> String {
    let text = clean(text);
    let mut result = String::new();
    for line in wrapped_lines(&text, width.saturating_sub(3).max(12)) {
        let line = line.as_str();
        if line.is_empty() {
            result.push_str(&format!("{}\n", paint("│", VIOLET, false, colors)));
            continue;
        }
        let upper = line.chars().any(char::is_alphabetic) && !line.chars().any(char::is_lowercase);
        let styled = if line.starts_with("[werk") && line.contains("] ") {
            let (label, message) = line.split_once("] ").unwrap();
            format!(
                "{} {}",
                paint(&format!("{label}]"), VIOLET, false, colors),
                highlight(message, colors)
            )
        } else if upper {
            paint(line, BLUE, true, colors)
        } else if !line.contains(char::is_whitespace) {
            paint(line, CYAN, true, colors)
        } else if line.ends_with(':') {
            paint(line, VIOLET, true, colors)
        } else if let Some((label, value)) = line.split_once(": ") {
            if label.len() < 42 && !label.contains("http") {
                format!(
                    "{} {}",
                    paint(&format!("{label}:"), BLUE, false, colors),
                    highlight(value, colors)
                )
            } else {
                highlight(line, colors)
            }
        } else {
            highlight(line, colors)
        };
        result.push_str(&format!(
            "{}  {styled}\n",
            paint("│", VIOLET, false, colors)
        ));
    }
    result
}

pub fn line(stream: Stream, args: fmt::Arguments<'_>) {
    let text = args.to_string();
    let output = if interactive(stream) {
        let width = match stream {
            Stream::Out => console::Term::stdout().size().1,
            Stream::Err => console::Term::stderr().size().1,
        };
        render_width(&text, color(stream), usize::from(width))
    } else {
        format!("{text}\n")
    };
    write(stream, &output);
}

pub fn write(stream: Stream, text: &str) {
    let _guard = OUTPUT.lock().unwrap_or_else(|e| e.into_inner());
    let emit = || match stream {
        Stream::Out => {
            let _ = io::stdout().lock().write_all(text.as_bytes());
        }
        Stream::Err => {
            let _ = io::stderr().lock().write_all(text.as_bytes());
        }
    };
    if let Some(progress) = PROGRESS.get() {
        progress.suspend(emit);
    } else {
        emit();
    }
}

pub fn heading(stream: Stream, title: &str) {
    if interactive(stream) {
        let colors = color(stream);
        write(
            stream,
            &format!(
                "\n{} {}\n",
                paint("╭─", VIOLET, false, colors),
                paint(&clean(title), PINK, true, colors)
            ),
        );
    }
}

pub fn finish() {
    if interactive(Stream::Out) {
        write(
            Stream::Out,
            &format!("{}\n", paint("╰─ Done", CYAN, false, color(Stream::Out))),
        );
    }
}

pub fn panel(stream: Stream, title: &str, text: &str) {
    if interactive(stream) {
        heading(stream, title);
        line(stream, format_args!("{text}"));
        write(
            stream,
            &format!("{}\n", paint("╰─", VIOLET, false, color(stream))),
        );
    } else {
        line(stream, format_args!("{title}\n{text}"));
    }
}

pub fn prompt(role: &str) -> String {
    if !interactive(Stream::Out) {
        return format!("{role}> ");
    }
    let (label, tint) = if role == "you" {
        ("you", CYAN)
    } else {
        ("werk", PINK)
    };
    format!(
        "{} {} ",
        paint(label, tint, true, color(Stream::Out)),
        paint("›", VIOLET, true, color(Stream::Out))
    )
}

/// A line-buffered adapter for the existing statistics/report writers.
/// Only presentation uses this; JSON and inference content never do.
pub struct ReportWriter {
    stream: Stream,
    pending: Vec<u8>,
}
impl ReportWriter {
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            pending: Vec::new(),
        }
    }
}
impl Write for ReportWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(data);
        while let Some(end) = self.pending.iter().position(|&b| b == b'\n') {
            let bytes: Vec<_> = self.pending.drain(..=end).collect();
            line(
                self.stream,
                format_args!("{}", String::from_utf8_lossy(&bytes[..bytes.len() - 1])),
            );
        }
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if !self.pending.is_empty() {
            let bytes = std::mem::take(&mut self.pending);
            write(self.stream, &String::from_utf8_lossy(&bytes));
        }
        Ok(())
    }
}
impl Drop for ReportWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Tee installer tools through the same console stream without buffering their
/// entire output or changing noninteractive command execution.
pub fn command_status(command: &mut Command) -> io::Result<ExitStatus> {
    if !interactive(Stream::Err) {
        return command.status();
    }
    let label = command.get_program().to_string_lossy().into_owned();
    heading(
        Stream::Err,
        &format!(
            "Installing · {}",
            std::path::Path::new(&label)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ),
    );
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    std::thread::scope(|scope| {
        let read = |reader: Box<dyn io::Read + Send>| {
            let mut reader = io::BufReader::new(reader);
            loop {
                let mut buffer = Vec::new();
                // Bound individual tool lines (some progress printers omit LF).
                match io::Read::take(&mut reader, 16 * 1024).read_until(b'\n', &mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => line(
                        Stream::Err,
                        format_args!("{}", String::from_utf8_lossy(&buffer).trim_end()),
                    ),
                }
            }
        };
        scope.spawn(move || read(Box::new(stdout)));
        scope.spawn(move || read(Box::new(stderr)));
        child.wait()
    })
}

pub fn clap_styles() -> clap::builder::Styles {
    use clap::builder::styling::{AnsiColor, Color, RgbColor, Style};
    let accent = |(r, g, b)| {
        Style::new()
            .fg_color(Some(Color::Rgb(RgbColor(r, g, b))))
            .bold()
    };
    clap::builder::Styles::styled()
        .header(accent(VIOLET))
        .usage(accent(PINK))
        .literal(accent(CYAN))
        .placeholder(accent(BLUE))
        .error(accent(PINK))
        .valid(accent(CYAN))
        .invalid(Style::new().fg_color(Some(AnsiColor::Magenta.into())))
}

#[macro_export]
macro_rules! ui_println {
    () => { std::println!() };
    ($($args:tt)*) => {{
        if $crate::terminal::interactive($crate::terminal::Stream::Out) {
            $crate::terminal::line($crate::terminal::Stream::Out, format_args!($($args)*));
        } else { std::println!($($args)*); }
    }};
}
#[macro_export]
macro_rules! ui_eprintln {
    () => { std::eprintln!() };
    ($($args:tt)*) => {{
        if $crate::terminal::interactive($crate::terminal::Stream::Err) {
            $crate::terminal::line($crate::terminal::Stream::Err, format_args!($($args)*));
        } else { std::eprintln!($($args)*); }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn styling_keeps_table_padding_and_commands_and_removes_control_injection() {
        let text =
            "RUNTIME     STATUS\nCUDA        ready\nInstall: werk backend install text-analysis";
        assert_eq!(
            console::strip_ansi_codes(&render(text, true)),
            render(text, false)
        );
        assert!(
            render(text, true).contains("\x1b[1;38;2;0;220;255mwerk backend install text-analysis")
        );
        assert!(render("bad\x1b[2Jmodel\r\x07", true).contains("badmodel"));
        assert!(!render(text, false).contains('\x1b'));
    }
}
