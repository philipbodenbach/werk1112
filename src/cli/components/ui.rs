use super::*;
use crossterm::{
    event::{self, Event as TerminalEvent, KeyCode, KeyEventKind, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, Paragraph, Row, Table, TableState, Wrap},
};
use std::{
    collections::{HashSet, VecDeque},
    time::Instant,
};

const CYAN: Color = Color::Rgb(0, 220, 255);
const BG: Color = Color::Rgb(9, 13, 23);
const MUTED: Color = Color::Rgb(145, 157, 181);

struct Screen;
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            event::DisableMouseCapture,
            LeaveAlternateScreen
        );
    }
}

#[derive(Default)]
struct App {
    inventory: Option<Inventory>,
    table: TableState,
    selected: HashSet<String>,
    filter: usize,
    logs: VecDeque<String>,
    scanning: bool,
    running: bool,
    confirm: Option<Vec<Action>>,
    queue: Vec<(String, String)>,
    current: String,
    done: usize,
    failed: usize,
    started: Option<Instant>,
    tick: usize,
    exiting: bool,
    queue_view: bool,
    confirm_offset: usize,
}

impl App {
    fn rows(&self) -> Vec<&Component> {
        self.inventory
            .as_ref()
            .map(|i| {
                i.components
                    .iter()
                    .filter(|c| match self.filter {
                        1 => c.status == "update available",
                        2 => c.status.contains("missing"),
                        3 => c.installed.is_some(),
                        _ => true,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    fn log(&mut self, line: String) {
        self.logs.push_back(line);
        while self.logs.len() > 500 {
            self.logs.pop_front();
        }
    }
    fn navigate(&mut self, delta: isize) {
        let len = if self.queue_view {
            self.queue.len()
        } else {
            self.rows().len()
        };
        if len > 0 {
            self.table.select(Some(
                self.table
                    .selected()
                    .unwrap_or(0)
                    .saturating_add_signed(delta)
                    .min(len - 1),
            ));
        }
    }
    fn toggle(&mut self) {
        let id = self
            .rows()
            .get(self.table.selected().unwrap_or(0))
            .filter(|r| r.action.is_some())
            .map(|c| c.id.clone());
        if let Some(id) = id {
            if !self.selected.remove(&id) {
                self.selected.insert(id);
            }
        }
    }
    fn plan(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        if let Some(inventory) = &self.inventory {
            for row in &inventory.components {
                if self.selected.contains(&row.id) {
                    if let Some(action) = &row.action {
                        if !actions.contains(action) {
                            actions.push(action.clone());
                        }
                    }
                }
            }
        }
        actions
    }
    fn mark(&mut self, missing: bool) {
        if let Some(inventory) = &self.inventory {
            for row in &inventory.components {
                let eligible = if missing {
                    matches!(row.action, Some(Action::SetupModel(_)))
                        || row.status == "missing / required"
                } else {
                    matches!(
                        row.action,
                        Some(
                            Action::UpdateBackend(_)
                                | Action::UpdateModel(_)
                                | Action::UpdateExternalBackend { .. }
                                | Action::UpdateWerk(_)
                                | Action::UpdateProjector { .. }
                        )
                    )
                };
                if eligible && row.action.is_some() {
                    self.selected.insert(row.id.clone());
                }
            }
        }
    }
}

pub(super) fn run(store: ModelStore, offline: bool) -> Result<()> {
    anyhow::ensure!(
        io::stdin().is_terminal() && io::stdout().is_terminal(),
        "The component UI needs a terminal; use werk components --list --json or werk update --all --yes"
    );
    enable_raw_mode()?;
    let _screen = Screen;
    execute!(
        io::stdout(),
        EnterAlternateScreen,
        event::EnableMouseCapture
    )?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let scan_cancel = Arc::new(AtomicBool::new(false));
    let mut app = App {
        scanning: true,
        ..Default::default()
    };
    refresh(store.clone(), offline, tx.clone(), scan_cancel.clone());
    loop {
        for event in rx.try_iter() {
            match event {
                Event::Inventory(result) => {
                    app.scanning = false;
                    match result {
                        Ok(inventory) => {
                            app.inventory = Some(inventory);
                            app.table.select(Some(0));
                            app.log("Inventory refreshed. Select components to review an installation/update plan.".into());
                        }
                        Err(e) => app.log(format!("Inventory failed: {e}")),
                    }
                }
                Event::Started(i, label) => {
                    app.current = label;
                    app.queue[i].1 = "running".into();
                }
                Event::Log(line) => app.log(line),
                Event::Done(i, success) => {
                    app.done += 1;
                    if !success {
                        app.failed += 1;
                    }
                    app.queue[i].1 = if success { "completed" } else { "FAILED" }.into();
                    app.log(format!("{}: {}", app.queue[i].0, app.queue[i].1));
                }
                Event::Finished => {
                    app.running = false;
                    for item in &mut app.queue {
                        if item.1 == "queued" {
                            item.1 = "skipped".into();
                        }
                    }
                    app.current = format!(
                        "Finished: {} completed, {} failed. Restart running services to load updated components.",
                        app.done.saturating_sub(app.failed),
                        app.failed
                    );
                    app.log(app.current.clone());
                    app.selected.clear();
                    if !app.exiting {
                        app.scanning = true;
                        scan_cancel.store(false, Ordering::Relaxed);
                        refresh(store.clone(), offline, tx.clone(), scan_cancel.clone());
                    }
                }
            }
        }
        if app.exiting && !app.scanning && !app.running {
            break;
        }
        app.tick = app.tick.wrapping_add(1);
        terminal.draw(|frame| draw(frame, &mut app))?;
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            TerminalEvent::Key(key) if key.kind == KeyEventKind::Press => {
                if key.code == KeyCode::Char('c')
                    && key.modifiers.contains(event::KeyModifiers::CONTROL)
                {
                    if !app.running && !app.scanning {
                        break;
                    }
                    app.exiting = true;
                    cancel.store(true, Ordering::Relaxed);
                    scan_cancel.store(true, Ordering::Relaxed);
                    continue;
                }
                if app.confirm.is_some() {
                    match key.code {
                        KeyCode::Down | KeyCode::PageDown => {
                            app.confirm_offset = app
                                .confirm_offset
                                .saturating_add(1)
                                .min(app.confirm.as_ref().unwrap().len().saturating_sub(1));
                        }
                        KeyCode::Up | KeyCode::PageUp => {
                            app.confirm_offset = app.confirm_offset.saturating_sub(1);
                        }
                        KeyCode::Char('y') => {
                            let actions = app.confirm.take().unwrap();
                            app.queue = actions
                                .iter()
                                .map(|a| (a.label(), "queued".into()))
                                .collect();
                            app.done = 0;
                            app.failed = 0;
                            app.running = true;
                            app.queue_view = true;
                            app.table.select(Some(0));
                            app.started = Some(Instant::now());
                            cancel.store(false, Ordering::Relaxed);
                            execute_queue(store.clone(), actions, cancel.clone(), tx.clone());
                        }
                        KeyCode::Esc | KeyCode::Char('n' | 'q') => {
                            app.confirm = None;
                        }
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Down | KeyCode::Char('j') => app.navigate(1),
                    KeyCode::Up | KeyCode::Char('k') => app.navigate(-1),
                    KeyCode::PageDown => app.navigate(10),
                    KeyCode::PageUp => app.navigate(-10),
                    KeyCode::Home => app.table.select(Some(0)),
                    KeyCode::End => app.navigate(isize::MAX),
                    KeyCode::Tab | KeyCode::Char('f') => {
                        app.filter = (app.filter + 1) % 4;
                        app.table.select(Some(0));
                    }
                    KeyCode::Char('q') | KeyCode::Esc if !app.running => {
                        if !app.scanning {
                            break;
                        }
                        app.exiting = true;
                        scan_cancel.store(true, Ordering::Relaxed);
                    }
                    KeyCode::Char('v') => {
                        app.queue_view = !app.queue_view;
                        app.table.select(Some(0));
                    }
                    KeyCode::Char('q' | 'c') if app.running => {
                        if key.code == KeyCode::Char('q') {
                            app.exiting = true;
                        }
                        cancel.store(true, Ordering::Relaxed);
                        app.log("Pending operations cancelled. The active installer will finish before exit.".into());
                    }
                    KeyCode::Char(' ') if !app.running && !app.scanning && !app.queue_view => {
                        app.toggle()
                    }
                    KeyCode::Char('a') if !app.running && !app.scanning => app.mark(false),
                    KeyCode::Char('m') if !app.running && !app.scanning => app.mark(true),
                    KeyCode::Char('x') if !app.running => app.selected.clear(),
                    KeyCode::Char('r') if !app.running && !app.scanning => {
                        app.scanning = true;
                        scan_cancel.store(false, Ordering::Relaxed);
                        refresh(store.clone(), offline, tx.clone(), scan_cancel.clone());
                    }
                    KeyCode::Enter if !app.running && !app.scanning && !app.queue_view => {
                        if app.selected.is_empty() {
                            app.toggle();
                        }
                        let plan = app.plan();
                        if !plan.is_empty() {
                            app.confirm_offset = 0;
                            app.confirm = Some(plan);
                        }
                    }
                    _ => {}
                }
            }
            TerminalEvent::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollDown => app.navigate(3),
                MouseEventKind::ScrollUp => app.navigate(-3),
                _ => {}
            },
            _ => {}
        }
    }
    Ok(())
}

fn panel(title: impl Into<Line<'static>>) -> Block<'static> {
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(69, 84, 117)))
}

fn short(value: &Option<String>) -> String {
    value.as_deref().unwrap_or("—").chars().take(100).collect()
}

fn draw(frame: &mut Frame, app: &mut App) {
    frame.render_widget(
        Block::default().style(Style::default().bg(BG).fg(Color::White)),
        frame.area(),
    );
    if frame.area().height < 18 || frame.area().width < 60 {
        frame.render_widget(
            Paragraph::new("Enlarge terminal to at least 60 × 18. q exits when idle.")
                .wrap(Wrap { trim: true }),
            frame.area(),
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(5),
        Constraint::Length(3),
        Constraint::Length(6),
        Constraint::Length(2),
    ])
    .split(frame.area());
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][app.tick % 10];
    let host = app
        .inventory
        .as_ref()
        .map(|i| i.host.as_str())
        .unwrap_or("Detecting host and installed components…");
    frame.render_widget(
        Paragraph::new(format!(
            "{host}{}",
            if app.inventory.as_ref().is_some_and(|i| i.offline) {
                " · OFFLINE"
            } else {
                ""
            }
        ))
        .block(panel(Line::from(vec![
            Span::styled(
                " WERK / Components ",
                Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
            ),
            Span::raw(if app.scanning {
                format!(" {spinner} scanning")
            } else {
                String::new()
            }),
        ]))),
        areas[0],
    );
    let rows = app.rows();
    let selected = app.table.selected().unwrap_or(0);
    let mut detail = rows
        .get(selected)
        .map(|c| {
            format!(
                "{}\n{}{}",
                c.id,
                c.detail,
                if c.required_by.is_empty() {
                    String::new()
                } else {
                    format!("\nRequired by: {}", c.required_by.join(", "))
                }
            )
        })
        .unwrap_or_else(|| "No components in this filter.".into());
    let mut table_rows: Vec<_> = rows
        .iter()
        .map(|c| {
            let color = if c.status.contains("error") || c.status.contains("blocked") {
                Color::LightRed
            } else if c.status.contains("update") {
                CYAN
            } else if c.status.contains("missing") {
                Color::Yellow
            } else {
                MUTED
            };
            Row::new(vec![
                if app.selected.contains(&c.id) {
                    "[x]"
                } else if c.action.is_some() {
                    "[ ]"
                } else {
                    " · "
                }
                .into(),
                c.name.clone(),
                c.status.clone(),
                short(&c.installed),
                short(&c.available),
            ])
            .style(Style::default().fg(color))
        })
        .collect();
    if app.queue_view {
        table_rows = app
            .queue
            .iter()
            .map(|(name, status)| {
                Row::new(vec![
                    String::new(),
                    name.clone(),
                    status.clone(),
                    String::new(),
                    String::new(),
                ])
                .style(Style::default().fg(if status == "FAILED" {
                    Color::LightRed
                } else {
                    CYAN
                }))
            })
            .collect();
        detail = app
            .queue
            .get(selected)
            .map(|(name, status)| format!("{name}\nStatus: {status}"))
            .unwrap_or_else(|| "No operations have run. Press v for the inventory.".into());
    }
    let filter = if app.queue_view {
        "Operation queue / Results"
    } else {
        ["All compatible", "Updates", "Missing", "Installed"][app.filter]
    };
    let table = Table::new(
        table_rows,
        [
            Constraint::Length(3),
            Constraint::Percentage(25),
            Constraint::Length(21),
            Constraint::Percentage(30),
            Constraint::Percentage(25),
        ],
    )
    .header(
        Row::new(["", "Component", "Status", "Installed", "Upstream"])
            .style(Style::default().fg(CYAN))
            .bottom_margin(1),
    )
    .block(panel(format!(
        " {filter} · {} selected ",
        app.selected.len()
    )))
    .row_highlight_style(
        Style::default()
            .bg(Color::Rgb(30, 48, 72))
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("› ");
    frame.render_stateful_widget(table, areas[1], &mut app.table);
    frame.render_widget(
        Paragraph::new(crate::terminal::clean(&detail))
            .wrap(Wrap { trim: true })
            .block(panel(" Details / Dependencies ")),
        areas[2],
    );
    let ratio = if app.queue.is_empty() {
        0.0
    } else {
        app.done as f64 / app.queue.len() as f64
    };
    let label = if app.running {
        format!(
            "{spinner} {}/{} · {} · {}s",
            app.done,
            app.queue.len(),
            app.current,
            app.started.map_or(0, |t| t.elapsed().as_secs())
        )
    } else if app.scanning {
        format!("{spinner} Reading component metadata…")
    } else if app.queue.is_empty() {
        "Select components, then Enter to review the plan".into()
    } else {
        app.current.clone()
    };
    frame.render_widget(
        Gauge::default()
            .block(panel(" Progress "))
            .gauge_style(Style::default().fg(CYAN))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(label),
        areas[3],
    );
    let log = app
        .logs
        .iter()
        .rev()
        .take(areas[4].height.saturating_sub(2) as usize)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    frame.render_widget(
        Paragraph::new(log)
            .style(Style::default().fg(MUTED))
            .block(panel(" Live installer output / Results ")),
        areas[4],
    );
    frame.render_widget(Paragraph::new("↑↓/PgUp/PgDn scroll · Space select · a all updates · m missing · Enter review\nf filter · v queue · r rescan · x clear · c cancel pending · q quit").style(Style::default().fg(CYAN)), areas[5]);
    if let Some(actions) = &app.confirm {
        let width = frame.area().width.saturating_sub(6).min(100);
        let height = (actions.len() as u16 + 7).min(frame.area().height.saturating_sub(2));
        let area = ratatui::layout::Rect::new(
            (frame.area().width - width) / 2,
            (frame.area().height - height) / 2,
            width,
            height,
        );
        let mut lines = actions
            .iter()
            .enumerate()
            .skip(app.confirm_offset)
            .take(height.saturating_sub(6) as usize)
            .map(|(i, a)| format!("{}. {}", i + 1, a.label()))
            .collect::<Vec<_>>();
        if actions.len() > lines.len() + app.confirm_offset {
            lines.push(format!(
                "… {} more selected operations",
                actions.len() - lines.len() - app.confirm_offset
            ));
        }
        lines.push(format!("\n{} operations. Dependencies included; supported pins remain.\n[↑↓] Scroll plan   [y] Execute   [n/Esc] Back", actions.len()));
        frame.render_widget(Clear, area);
        frame.render_widget(
            Paragraph::new(lines.join("\n"))
                .wrap(Wrap { trim: true })
                .block(panel(" Review installation / update plan "))
                .style(Style::default().bg(BG).fg(Color::White)),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_renders_small_and_large_inventory_without_mutations() {
        for (width, height) in [(50, 12), (80, 25), (140, 45)] {
            let mut terminal =
                Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            let mut app = App::default();
            terminal.draw(|f| draw(f, &mut app)).unwrap();
            app.confirm = Some(vec![Action::UpdateBackend("llama-cpu".into())]);
            terminal.draw(|f| draw(f, &mut app)).unwrap();
        }
    }
}
