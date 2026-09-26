use super::{App, clean};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Gauge, Paragraph, Row, Table, TableState, Wrap},
};
use std::collections::VecDeque;

const BG: Color = Color::Rgb(9, 13, 23);
const PANEL: Color = Color::Rgb(15, 21, 35);
const TEXT: Color = Color::Rgb(224, 230, 248);
const MUTED: Color = Color::Rgb(129, 143, 174);
const CYAN: Color = Color::Rgb(0, 220, 255);
const BLUE: Color = Color::Rgb(59, 130, 246);
const INDIGO: Color = Color::Rgb(99, 102, 241);
const VIOLET: Color = Color::Rgb(139, 92, 246);
const PINK: Color = Color::Rgb(255, 79, 195);
const BRAND: [Color; 5] = [CYAN, BLUE, INDIGO, VIOLET, PINK];

fn panel(title: &str, color: Color) -> Block<'static> {
    Block::default()
        .title(Line::styled(
            format!(" {title} "),
            Style::default().fg(color).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .style(Style::default().bg(PANEL).fg(TEXT))
}
fn live_panel(title: &str, color: Color, app: &App) -> Block<'static> {
    let block = panel(title, color);
    if !llama(app) || !app.animation || app.paused || app.error.is_some() || app.snapshot.is_none()
    {
        return block;
    }
    let marker = ["▁", "▂", "▄", "▆", "█", "▆", "▄", "▂"][app.tick as usize / 2 % 8];
    block.title(
        Line::styled(format!(" {marker} LIVE "), Style::default().fg(color))
            .alignment(Alignment::Right),
    )
}

fn number(value: Option<f64>, unit: &str) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.1} {unit}"))
        .unwrap_or_else(|| "n/a".into())
}
fn tokens(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "—".into())
}
fn native(app: &App, key: &str) -> Option<f64> {
    app.snapshot
        .as_ref()?
        .backends
        .get(app.worker)
        .filter(|b| b.available)?
        .gauges
        .get(key)
        .copied()
}
fn llama(app: &App) -> bool {
    app.snapshot
        .as_ref()
        .and_then(|s| s.backends.get(app.worker))
        .is_some_and(|b| b.backend.starts_with("llama.cpp"))
}

fn duration(seconds: f64) -> String {
    let s = seconds.max(0.) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

pub(super) fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BG).fg(TEXT)),
        area,
    );
    if area.width < 32 || area.height < 10 {
        frame.render_widget(
            Paragraph::new("WERK TOP\nResize to at least 32 × 10\nq to quit")
                .style(Style::default().fg(CYAN)),
            area,
        );
        return;
    }
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(area);
    header(frame, sections[0], app);
    let body = sections[1];
    if body.height < 14 {
        let parts = Layout::vertical([Constraint::Length(4), Constraint::Min(2)]).split(body);
        let text = if llama(app) {
            format!(
                "Decode ≈ {}    Active {}\nCPU experts RAM {} / {}",
                number(app.rates.decode_estimate, "tok/s"),
                app.snapshot.as_ref().map_or(0, |s| s.totals.active),
                number(
                    native(app, "cpu_expert_resident_bytes").map(|n| n / 1073741824.),
                    "GiB"
                ),
                number(
                    native(app, "cpu_expert_weight_bytes").map(|n| n / 1073741824.),
                    "GiB"
                )
            )
        } else {
            format!(
                "Decode ≈ {}    Expert hits {}\nActive {}    Offload {}",
                number(app.rates.decode_estimate, "tok/s"),
                number(app.rates.expert_hit_ratio.map(|v| v * 100.), "%"),
                app.snapshot
                    .as_ref()
                    .map_or_else(|| "—".into(), |s| s.totals.active.to_string()),
                number(
                    app.rates.read_bytes_per_second.map(|v| v / 1048576.),
                    "MiB/s"
                )
            )
        };
        frame.render_widget(Paragraph::new(text).block(panel("LIVE", CYAN)), parts[0]);
        requests(frame, parts[1], app);
    } else if body.width >= 96 && body.height >= 20 {
        let card_height = if llama(app) {
            (body.height / 3).clamp(8, 12)
        } else {
            8
        };
        let rows = Layout::vertical([
            Constraint::Length(card_height),
            Constraint::Length(card_height),
            Constraint::Min(4),
        ])
        .split(body);
        let upper = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        let lower = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);
        inference(frame, upper[0], app);
        memory(frame, upper[1], app);
        cache(frame, lower[0], app);
        offload(frame, lower[1], app);
        requests(frame, rows[2], app);
    } else {
        let rows = Layout::vertical([
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Min(3),
        ])
        .split(body);
        inference(frame, rows[0], app);
        let cards = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);
        memory(frame, cards[0], app);
        cache(frame, cards[1], app);
        requests(frame, rows[2], app);
    }
    let status = if let Some(error) = &app.error {
        format!("DISCONNECTED · {}", clean(error))
    } else if app.paused {
        "DISPLAY PAUSED · inference continues".into()
    } else if app.demo {
        "SIMULATED DATA · no server connection".into()
    } else {
        format!(
            "{} · sample age {}",
            clean(&app.target),
            number(app.last_received.map(|t| t.elapsed().as_secs_f64()), "s")
        )
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                status,
                Style::default().fg(if app.error.is_some() { PINK } else { MUTED }),
            ),
            Line::from(vec![
                Span::styled(" q ", Style::default().fg(BG).bg(CYAN)),
                Span::raw(" quit  "),
                Span::styled("Space", Style::default().fg(PINK)),
                Span::raw(" pause  ↑↓ select  Tab details  b worker  a animation"),
            ]),
        ]),
        sections[2],
    );
    if app.details {
        details(frame, app);
    }
}
fn header(frame: &mut Frame, area: Rect, app: &App) {
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let status = if app.demo {
        "DEMO"
    } else if app.paused {
        "PAUSED"
    } else if app.error.is_some() {
        "OFFLINE"
    } else if app.snapshot.is_some() {
        "LIVE"
    } else {
        "CONNECTING"
    };
    let mut spans = vec![Span::raw(" ")];
    for (i, s) in ["W", "E", "R", "K", "1112"].iter().enumerate() {
        spans.push(Span::styled(*s, Style::default().fg(BRAND[i]).bold()));
    }
    spans.push(Span::styled("  TOP", Style::default().fg(TEXT).bold()));
    spans.push(Span::styled(
        format!(
            "   {} {status}",
            if app.animation {
                spinner[app.tick as usize % spinner.len()]
            } else {
                "●"
            }
        ),
        Style::default().fg(if app.error.is_some() { PINK } else { CYAN }),
    ));
    if let Some(s) = &app.snapshot {
        spans.push(Span::styled(
            format!("   ↑ {}", duration(s.uptime_seconds)),
            Style::default().fg(MUTED),
        ));
    }
    let model = app
        .snapshot
        .as_ref()
        .and_then(|s| {
            s.backends
                .get(app.worker)
                .map(|b| format!("{}  /  {}", clean(&b.model), clean(&b.backend)))
                .or_else(|| s.requests.first().map(|r| clean(&r.model)))
        })
        .unwrap_or_else(|| "Inference Router · live observability".into());
    let rail = (0..area.width)
        .map(|i| {
            Span::styled(
                "━",
                Style::default().fg(BRAND[((i as u64 * 5 / u64::from(area.width.max(1)))
                    + (if app.animation { app.tick / 15 } else { 0 }))
                    as usize
                    % 5]),
            )
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(spans),
            Line::styled(format!(" {model}"), Style::default().fg(MUTED)),
            Line::from(rail),
        ]),
        area,
    );
}
fn inference(frame: &mut Frame, area: Rect, app: &App) {
    let block = live_panel("INFERENCE", CYAN, app);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let active = app.snapshot.as_ref().map_or(0, |s| s.totals.active);
    let phase = if app.rates.decode_estimate.is_some() && active > 0 {
        "DECODE"
    } else if active > 0 {
        "WORKING"
    } else {
        "IDLE"
    };
    let top = format!("{phase}   ≈ {}", number(app.rates.decode_estimate, "tok/s"));
    frame.render_widget(
        Paragraph::new(top).style(Style::default().fg(CYAN).bold()),
        Rect { height: 1, ..inner },
    );
    if inner.height > 2 {
        graph(
            frame,
            Rect {
                x: inner.x,
                y: inner.y + 1,
                width: inner.width,
                height: inner.height - 2,
            },
            &app.history,
            CYAN,
        );
    }
    if inner.height > 1 {
        frame.render_widget(
            Paragraph::new(if llama(app) {
                format!(
                    "Active {active}   Output {} tokens   ~ interval estimate",
                    native(app, "active_output_tokens")
                        .map(|n| format!("{n:.0}"))
                        .unwrap_or_else(|| "n/a".into())
                )
            } else {
                format!(
                    "Active {active}   Queued {}   ~ interval estimate",
                    native(app, "requests_waiting")
                        .map(|n| format!("{n:.0}"))
                        .unwrap_or_else(|| "n/a".into())
                )
            })
            .style(Style::default().fg(MUTED)),
            Rect {
                x: inner.x,
                y: inner.bottom() - 1,
                width: inner.width,
                height: 1,
            },
        );
    }
}
fn memory(frame: &mut Frame, area: Rect, app: &App) {
    let block = live_panel("MEMORY", VIOLET, app);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let host = app.snapshot.as_ref().and_then(|s| s.memory.as_ref());
    let (resident, budget, label_name) = if llama(app) {
        let capacity = host.and_then(|m| m.host.capacity_bytes).map(|n| n as f64);
        (
            app.snapshot.as_ref().and_then(super::host_used_bytes),
            capacity,
            "Host incl. cache",
        )
    } else {
        (
            native(app, "expert_cache_resident_bytes"),
            native(app, "expert_cache_budget_bytes"),
            "Experts",
        )
    };
    let ratio = resident
        .zip(budget)
        .and_then(|(a, b)| (b > 0.).then_some((a / b).clamp(0., 1.)))
        .unwrap_or(0.);
    let label = format!(
        "{label_name} {} / {}",
        number(resident.map(|n| n / 1073741824.), "GiB"),
        number(budget.map(|n| n / 1073741824.), "GiB")
    );
    frame.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label(label)
            .gauge_style(Style::default().fg(VIOLET).bg(BG))
            .use_unicode(true),
        Rect { height: 1, ..inner },
    );
    let lines = vec![
        Line::raw(if llama(app) {
            format!(
                "Worker RSS {}",
                number(
                    native(app, "process_resident_bytes").map(|n| n / 1073741824.),
                    "GiB"
                )
            )
        } else {
            format!(
                "Effective expert budget {}",
                number(
                    native(app, "expert_cache_effective_budget_bytes").map(|n| n / 1073741824.),
                    "GiB"
                )
            )
        }),
        Line::raw(format!(
            "Host available {}",
            number(
                host.and_then(|m| m.host.available_bytes)
                    .map(|n| n as f64 / 1073741824.),
                "GiB"
            )
        )),
        Line::raw(format!(
            "{} {}",
            if llama(app) {
                "Host pressure"
            } else {
                "Pressure"
            },
            host.map(|m| format!(
                "{:?}",
                if llama(app) {
                    m.host.pressure
                } else {
                    m.overall_pressure
                }
            )
            .to_uppercase())
                .unwrap_or_else(|| "N/A".into())
        )),
        Line::styled(
            format!(
                "Swap {}",
                number(
                    app.snapshot
                        .as_ref()
                        .and_then(|s| s.host_swap_used_bytes)
                        .map(|v| v as f64 / 1073741824.),
                    "GiB"
                )
            ),
            Style::default().fg(MUTED),
        ),
    ];
    let plot_height = if llama(app) {
        inner.height.saturating_sub(5)
    } else {
        0
    };
    graph(
        frame,
        Rect {
            y: inner.y + 1,
            height: plot_height,
            ..inner
        },
        &app.memory_history,
        VIOLET,
    );
    if inner.height > 1 {
        frame.render_widget(
            Paragraph::new(lines),
            Rect {
                y: inner.y + 1 + plot_height,
                height: inner.height - 1 - plot_height,
                ..inner
            },
        );
    }
}
fn cache(frame: &mut Frame, area: Rect, app: &App) {
    if llama(app) {
        let block = live_panel("PROMPT CACHE", BLUE, app);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let hit = native(app, "prompt_cache_hit_ratio");
        frame.render_widget(
            Gauge::default()
                .ratio(hit.unwrap_or(0.).clamp(0., 1.))
                .label(format!(
                    "Prompt cache hit rate {}",
                    number(hit.map(|n| n * 100.), "%")
                ))
                .gauge_style(Style::default().fg(BLUE).bg(BG)),
            Rect { height: 1, ..inner },
        );
        let plot_height = inner.height.saturating_sub(4);
        graph(
            frame,
            Rect {
                y: inner.y + 1,
                height: plot_height,
                ..inner
            },
            &app.prompt_hit_history,
            BLUE,
        );
        if inner.height > 1 {
            frame.render_widget(Paragraph::new(format!(
                "Cached prompt {} / {} tokens\nContext {} / {} tokens\nKV prefix reuse · slots {}",
                tokens(native(app, "active_cached_tokens").map(|n| n as u64)),
                tokens(native(app, "active_prompt_tokens").map(|n| n as u64)),
                tokens(native(app, "context_used_tokens").map(|n| n as u64)),
                tokens(native(app, "context_capacity_tokens").map(|n| n as u64)),
                tokens(native(app, "slots_total").map(|n| n as u64)))),
                Rect { y: inner.y + 1 + plot_height, height: inner.height - 1 - plot_height, ..inner });
        }
        return;
    }
    let block = live_panel("EXPERT CACHE", BLUE, app);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(format!(
            "Interval hit rate {}",
            number(app.rates.expert_hit_ratio.map(|n| n * 100.), "%")
        ))
        .style(Style::default().fg(CYAN).bold()),
        Rect { height: 1, ..inner },
    );
    if inner.height > 2 {
        graph(
            frame,
            Rect {
                x: inner.x,
                y: inner.y + 1,
                width: inner.width,
                height: inner.height - 2,
            },
            &app.hit_history,
            BLUE,
        );
    }
    let total = app
        .snapshot
        .as_ref()
        .and_then(|s| s.backends.get(app.worker))
        .and_then(|b| {
            let h = *b.counters.get("expert_cache_hits_total")?;
            let m = *b.counters.get("expert_cache_misses_total")?;
            (h + m > 0).then(|| h as f64 / (h + m) as f64 * 100.)
        });
    if inner.height > 1 {
        frame.render_widget(
            Paragraph::new(format!(
                "Lifetime {} · gaps mean no sample",
                number(total, "%")
            ))
            .style(Style::default().fg(MUTED)),
            Rect {
                x: inner.x,
                y: inner.bottom() - 1,
                width: inner.width,
                height: 1,
            },
        );
    }
}
fn offload(frame: &mut Frame, area: Rect, app: &App) {
    if llama(app) {
        let cpu = if native(app, "cpu_moe_all") == Some(1.) {
            "all".into()
        } else {
            native(app, "cpu_moe_layers")
                .map(|n| format!("{n:.0}"))
                .unwrap_or_else(|| "default".into())
        };
        let gpu = native(app, "gpu_layers_requested")
            .map(|n| format!("{n:.0}"))
            .unwrap_or_else(|| "default".into());
        let accelerator = app
            .snapshot
            .as_ref()
            .and_then(|s| s.memory.as_ref())
            .map(|m| &m.accelerator);
        let capacity = accelerator.and_then(|m| m.capacity_bytes);
        let used = capacity
            .zip(accelerator.and_then(|m| m.available_bytes))
            .map(|(c, a)| c.saturating_sub(a));
        let block = live_panel("CPU / RAM OFFLOAD", PINK, app);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let resident = native(app, "cpu_expert_resident_bytes");
        let weights = native(app, "cpu_expert_weight_bytes");
        let ratio = resident
            .zip(weights)
            .filter(|(_, w)| *w > 0.)
            .map_or(0., |(r, w)| (r / w).clamp(0., 1.));
        frame.render_widget(
            Gauge::default()
                .ratio(ratio)
                .label(format!(
                    "CPU experts in RAM {} / {}",
                    number(resident.map(|n| n / 1073741824.), "GiB"),
                    number(weights.map(|n| n / 1073741824.), "GiB")
                ))
                .gauge_style(Style::default().fg(PINK).bg(BG)),
            Rect { height: 1, ..inner },
        );
        let plot_height = inner.height.saturating_sub(5);
        graph(
            frame,
            Rect {
                y: inner.y + 1,
                height: plot_height,
                ..inner
            },
            &app.expert_ram_history,
            PINK,
        );
        let pressure = accelerator
            .map(|m| format!("{:?}", m.pressure).to_uppercase())
            .unwrap_or_else(|| "N/A".into());
        if inner.height > 1 {
            frame.render_widget(Paragraph::new(format!(
                "CPU expert layers {cpu} · mmap RAM\nGPU memory {} / {}\nGPU pressure {pressure}\nGPU layers requested {gpu}",
                number(used.map(|n| n as f64 / 1073741824.), "GiB"),
                number(capacity.map(|n| n as f64 / 1073741824.), "GiB"))),
                Rect { y: inner.y + 1 + plot_height, height: inner.height - 1 - plot_height, ..inner });
        }
        return;
    }
    let block = live_panel("OFFLOAD", PINK, app);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(format!(
            "Expert reads {}",
            number(
                app.rates.read_bytes_per_second.map(|v| v / 1048576.),
                "MiB/s"
            )
        ))
        .style(Style::default().fg(PINK).bold()),
        Rect { height: 1, ..inner },
    );
    if inner.height > 2 {
        graph(
            frame,
            Rect {
                x: inner.x,
                y: inner.y + 1,
                width: inner.width,
                height: inner.height - 2,
            },
            &app.read_history,
            PINK,
        );
    }
    if inner.height > 1 {
        frame.render_widget(
            Paragraph::new("Logical reads include the OS file cache")
                .style(Style::default().fg(MUTED)),
            Rect {
                x: inner.x,
                y: inner.bottom() - 1,
                width: inner.width,
                height: 1,
            },
        );
    }
}
fn graph(frame: &mut Frame, area: Rect, history: &VecDeque<Option<f64>>, color: Color) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let samples = history
        .iter()
        .rev()
        .take(area.width as usize)
        .rev()
        .collect::<Vec<_>>();
    let max = samples.iter().filter_map(|v| **v).fold(1., f64::max) * 1.12;
    let fill = match color {
        Color::Rgb(r, g, b) => Color::Rgb(r / 4 + 12, g / 4 + 16, b / 4 + 26),
        _ => MUTED,
    };
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut lines = vec![];
    for y in 0..area.height {
        let mut spans = vec![Span::raw(" ".repeat(area.width as usize - samples.len()))];
        for sample in &samples {
            let (ch, tint) = match sample {
                None => (if y + 1 == area.height { '·' } else { ' ' }, MUTED),
                Some(v) => {
                    let level = (*v / max * f64::from(area.height) * 8.
                        - f64::from(area.height - 1 - y) * 8.)
                        .clamp(0., 8.);
                    (
                        glyphs[level.round() as usize],
                        if level >= 8. { fill } else { color },
                    )
                }
            };
            spans.push(Span::styled(ch.to_string(), Style::default().fg(tint)));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), area);
}
fn requests(frame: &mut Frame, area: Rect, app: &App) {
    let wide = area.width >= 80;
    let entries = app
        .snapshot
        .as_ref()
        .map(|s| s.requests.as_slice())
        .unwrap_or(&[]);
    let rows = entries.iter().map(|r| {
        let mut cells = vec![
            r.id.to_string(),
            clean(&r.state),
            tokens(r.prompt_tokens),
            tokens(r.output_tokens),
            duration(r.elapsed_seconds),
        ];
        if wide {
            cells.insert(3, tokens(r.cached_tokens));
            cells.push(number(r.decode_tokens_per_second, "tok/s"));
        }
        Row::new(cells).style(Style::default().fg(if r.state == "error" {
            PINK
        } else if r.state == "streaming" {
            CYAN
        } else {
            TEXT
        }))
    });
    let headers = if wide {
        vec![
            "#", "State", "Prompt", "Cached", "Output", "Elapsed", "Decode",
        ]
    } else {
        vec!["#", "State", "Prompt", "Output", "Elapsed"]
    };
    let widths = if wide {
        vec![
            Constraint::Length(5),
            Constraint::Min(16),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(12),
        ]
    } else {
        vec![
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(9),
        ]
    };
    let table = Table::new(rows, widths)
        .header(
            Row::new(headers)
                .style(Style::default().fg(MUTED))
                .bottom_margin(1),
        )
        .block(panel("REQUESTS · backend-reported usage", INDIGO))
        .row_highlight_style(
            Style::default()
                .bg(Color::Rgb(32, 35, 62))
                .add_modifier(Modifier::BOLD),
        );
    let mut state = TableState::default()
        .with_selected((!entries.is_empty()).then(|| app.selected.min(entries.len() - 1)));
    frame.render_stateful_widget(table, area, &mut state);
}
fn details(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let width = area.width.saturating_sub(4).min(76);
    let height = area.height.saturating_sub(4).min(15);
    let popup = Rect::new(
        (area.width - width) / 2,
        (area.height - height) / 2,
        width,
        height,
    );
    let selected = app
        .snapshot
        .as_ref()
        .and_then(|s| s.requests.get(app.selected));
    let text=selected.map(|r|format!("Model     {}\nState     {}\nPrompt    {} tokens\nCached    {} tokens\nOutput    {} tokens\nElapsed   {}\nFirst output {}\nDecode    {}\nPrefill   {}\n\nTab / Enter closes details",clean(&r.model),r.state,tokens(r.prompt_tokens),tokens(r.cached_tokens),tokens(r.output_tokens),duration(r.elapsed_seconds),number(r.first_output_seconds,"s"),number(r.decode_tokens_per_second,"tok/s"),number(r.prefill_tokens_per_second,"tok/s"))).unwrap_or_else(||"No request selected\n\nTab closes details".into());
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: true })
            .block(panel("REQUEST DETAIL", PINK)),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    #[test]
    fn observability_llama_panels_show_context_memory_and_placement() {
        let args = super::super::TopArgs {
            url: "http://localhost:11434".into(),
            api_key: None,
            interval_ms: 2000,
            once: false,
            json: false,
            no_animation: true,
            demo: false,
        };
        let mut app = App::new(&args);
        let mut snapshot = super::super::demo(3.);
        snapshot.backends[0].backend = "llama.cpp / CUDA".into();
        use crate::werk_protocol::{MemoryStatusResponse, MemoryTierStatus, PressureLevel};
        let host = MemoryTierStatus {
            capacity_bytes: Some(100 * 1073741824),
            available_bytes: Some(90 * 1073741824),
            managed_bytes: 0,
            reserved_bytes: 0,
            pressure: PressureLevel::Normal,
        };
        let accelerator = MemoryTierStatus {
            pressure: PressureLevel::Emergency,
            ..host.clone()
        };
        snapshot.memory = Some(MemoryStatusResponse {
            observed_at_unix_ms: 0,
            overall_pressure: PressureLevel::Emergency,
            topology: "discrete".into(),
            host,
            accelerator,
            last_action_unix_ms: None,
            counters: Default::default(),
        });
        snapshot.backends[0].gauges.extend([
            ("decode_tokens_per_second_estimate".into(), 12.),
            ("context_used_tokens".into(), 1200.),
            ("context_capacity_tokens".into(), 32768.),
            ("active_cached_tokens".into(), 1000.),
            ("active_output_tokens".into(), 200.),
            ("process_resident_bytes".into(), 1073741824.),
            ("cpu_moe_layers".into(), 38.),
            ("cpu_expert_weight_bytes".into(), 60. * 1073741824.),
            ("cpu_expert_resident_bytes".into(), 59. * 1073741824.),
            ("prompt_cache_hit_ratio".into(), 0.9),
        ]);
        app.update(snapshot);
        let mut terminal = Terminal::new(TestBackend::new(140, 36)).unwrap();
        terminal.draw(|f| draw(f, &app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        for value in [
            "12.0 tok/s",
            "PROMPT CACHE",
            "Prompt cache hit rate 90.0 %",
            "CPU experts in RAM 59.0 GiB / 60.0 GiB",
            "1200 / 32768",
            "Worker RSS 1.0 GiB",
            "Host incl. cache 98.0 GiB / 100.0 GiB",
            "Host pressure NORMAL",
            "GPU pressure EMERGENCY",
            "CPU expert layers 38",
        ] {
            assert!(text.contains(value), "missing {value}");
        }
        assert!(!text.contains("EXPERT CACHE"));
        assert!(!text.contains("disk-read"));
        assert_eq!(app.expert_ram_history.back(), Some(&Some(59.)));
        assert_eq!(app.prompt_hit_history.back(), Some(&Some(90.)));
        // Freeze sample age; only the animation tick changes between draws.
        app.last_received = None;
        // All four panels indicate live refresh, including stable memory data.
        app.animation = true;
        app.tick = 0;
        terminal.draw(|f| draw(f, &app)).unwrap();
        let first = terminal.backend().buffer().clone();
        let text = first.content.iter().map(|c| c.symbol()).collect::<String>();
        assert_eq!(text.matches("▁ LIVE").count(), 4);
        app.tick = 8;
        terminal.draw(|f| draw(f, &app)).unwrap();
        assert_ne!(&first, terminal.backend().buffer());
        app.animation = false;
        terminal.draw(|f| draw(f, &app)).unwrap();
        let still = terminal.backend().buffer().clone();
        app.tick = 16;
        terminal.draw(|f| draw(f, &app)).unwrap();
        assert_eq!(&still, terminal.backend().buffer());
    }

    #[test]
    fn observability_omlx_keeps_existing_mac_panels_and_layout() {
        let args = super::super::TopArgs {
            url: "http://localhost:11434".into(),
            api_key: None,
            interval_ms: 2000,
            once: false,
            json: false,
            no_animation: false,
            demo: true,
        };
        let mut app = App::new(&args);
        app.update(super::super::demo(1.));
        app.update(super::super::demo(3.));
        let mut terminal = Terminal::new(TestBackend::new(160, 48)).unwrap();
        terminal.draw(|f| draw(f, &app)).unwrap();
        let buffer = terminal.backend().buffer();
        let row = |y: usize| {
            buffer.content[y * 160..(y + 1) * 160]
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
        };
        assert!(row(4).contains("MEMORY"));
        assert!(row(12).contains("EXPERT CACHE"));
        assert!(row(12).contains("OFFLOAD"));
        assert!(row(20).contains("REQUESTS"));
        let text = buffer
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        for label in [
            "Experts",
            "Effective expert budget",
            "Interval hit rate",
            "Expert reads",
            "Logical reads include the OS file cache",
        ] {
            assert!(
                text.contains(label),
                "missing original oMLX display: {label}"
            );
        }
        for label in [
            "PROMPT CACHE",
            "CPU / RAM OFFLOAD",
            "Host pressure",
            "▁ LIVE",
        ] {
            assert!(
                !text.contains(label),
                "llama.cpp-only display leaked into oMLX: {label}"
            );
        }
    }

    #[test]
    fn observability_layouts_render_at_small_medium_and_large_sizes() {
        let args = super::super::TopArgs {
            url: "http://localhost:11434".into(),
            api_key: None,
            interval_ms: 2000,
            once: false,
            json: false,
            no_animation: false,
            demo: true,
        };
        let mut app = App::new(&args);
        app.update(super::super::demo(1.));
        app.update(super::super::demo(3.));
        for index in 2..90 {
            app.update(super::super::demo(index as f64 * 2.));
        }
        for (w, h) in [(20, 5), (32, 10), (60, 22), (100, 30), (160, 48)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &app)).unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>();
            assert!(text.contains("WERK"));
            if w >= 100 {
                assert!(text.contains("OFFLOAD"));
                assert!(text.contains("SIMULATED"));
            }
            if let Some(path) = std::env::var_os("WERK_TOP_CAPTURE_DIR") {
                let directory = std::path::PathBuf::from(path);
                std::fs::create_dir_all(&directory).unwrap();
                let cells:Vec<_>=terminal.backend().buffer().content.iter().map(|c|serde_json::json!({"text":c.symbol(),"fg":format!("{:?}",c.fg),"bg":format!("{:?}",c.bg)})).collect();
                std::fs::write(
                    directory.join(format!("top-{w}x{h}.json")),
                    serde_json::to_vec(&serde_json::json!({"width":w,"height":h,"cells":cells}))
                        .unwrap(),
                )
                .unwrap();
            }
            app.details = true;
            terminal.draw(|f| draw(f, &app)).unwrap();
            app.details = false;
        }
    }
}
