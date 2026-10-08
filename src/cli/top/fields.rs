//! Session-local field selection; never changes the server or persisted settings.
use super::{App, clean};
use crossterm::event::KeyCode;

pub(super) const LABELS: &[&str] = &[
    "Task",
    "State",
    "Request duration",
    "Input tokens",
    "Output tokens",
    "Cached prompt tokens",
    "Time to first output",
    "Decode rate",
    "Prefill rate",
    "Runtime",
    "Device",
    "Precision",
    "Model weight cache",
    "Load duration",
    "Inference duration",
    "Worker duration",
    "Results",
    "Runtime attempts",
    "Active requests",
    "Host free",
    "Host swap",
    "Accelerator available",
];

pub(super) struct Selection {
    pub applied: Option<Vec<bool>>,
    pub draft: Option<Vec<bool>>,
    pub cursor: usize,
    pub scroll: u16,
    pub order: Vec<usize>,
    pub columns: usize,
    pub column_major: bool,
    pub split: usize, // 0 bottom, 1 right, 2 hidden
    pub placement: bool,
    pub selected: usize,
    pub viewport_width: u16,
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            applied: None,
            draft: None,
            cursor: 0,
            scroll: 0,
            order: (0..LABELS.len()).collect(),
            columns: 2,
            column_major: false,
            split: 0,
            placement: false,
            selected: 0,
            viewport_width: 120,
        }
    }
}
impl Selection {
    fn defaults(analysis: bool) -> Vec<bool> {
        (0..LABELS.len())
            .map(|i| {
                if analysis {
                    matches!(i, 0..=3 | 9..=20)
                } else {
                    matches!(i, 0..=8 | 18..=21)
                }
            })
            .collect()
    }
    pub fn visible(&self) -> Vec<usize> {
        self.order
            .iter()
            .copied()
            .filter(|i| self.applied.as_ref().is_some_and(|a| a[*i]))
            .collect()
    }
    pub fn effective_columns(&self, width: u16) -> usize {
        self.columns.min((width / 24).max(1) as usize)
    }
    pub fn coordinates(&self, index: usize, count: usize, columns: usize) -> (usize, usize) {
        if self.column_major {
            let rows = count.div_ceil(columns).max(1);
            (index / rows, index % rows)
        } else {
            (index % columns, index / columns)
        }
    }
    fn move_selected(&mut self, key: KeyCode) {
        let visible = self.visible();
        if visible.is_empty() {
            return;
        }
        self.selected = self.selected.min(visible.len() - 1);
        let columns = self.effective_columns(self.viewport_width);
        let (x, y) = self.coordinates(self.selected, visible.len(), columns);
        let destination = match key {
            KeyCode::Left => x.checked_sub(1).map(|x| (x, y)),
            KeyCode::Right => Some((x + 1, y)),
            KeyCode::Up => y.checked_sub(1).map(|y| (x, y)),
            KeyCode::Down => Some((x, y + 1)),
            _ => None,
        };
        if let Some(target) = (0..visible.len())
            .find(|i| Some(self.coordinates(*i, visible.len(), columns)) == destination)
        {
            let source = self
                .order
                .iter()
                .position(|i| *i == visible[self.selected])
                .unwrap();
            let destination = self
                .order
                .iter()
                .position(|i| *i == visible[target])
                .unwrap();
            self.order.swap(source, destination);
            self.selected = target;
        }
    }
    pub fn handle(&mut self, key: KeyCode, analysis: bool) -> bool {
        if let Some(draft) = &mut self.draft {
            match key {
                KeyCode::Up => self.cursor = self.cursor.saturating_sub(1),
                KeyCode::Down => self.cursor = (self.cursor + 1).min(LABELS.len() - 1),
                KeyCode::Char(' ') => draft[self.cursor] = !draft[self.cursor],
                KeyCode::Enter => {
                    self.applied = self.draft.take();
                    self.scroll = 0;
                    self.selected = 0;
                }
                KeyCode::Esc | KeyCode::Char('v') | KeyCode::Char('q') => self.draft = None,
                KeyCode::Char('R') => *self = Self::default(),
                _ => {}
            }
            return true;
        }
        if self.placement {
            match key {
                KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right => {
                    self.move_selected(key);
                    return true;
                }
                KeyCode::Tab => {
                    self.selected = (self.selected + 1) % self.visible().len().max(1);
                    return true;
                }
                KeyCode::BackTab => {
                    self.selected = self
                        .selected
                        .checked_sub(1)
                        .unwrap_or(self.visible().len().saturating_sub(1));
                    return true;
                }
                KeyCode::Esc | KeyCode::Enter => {
                    self.placement = false;
                    return true;
                }
                _ => {}
            }
        }
        match key {
            KeyCode::Char('v') => {
                self.draft = Some(
                    self.applied
                        .clone()
                        .unwrap_or_else(|| Self::defaults(analysis)),
                );
                self.cursor = 0;
                true
            }
            KeyCode::Char('R') => {
                *self = Self::default();
                true
            }
            KeyCode::Char('e' | 'c' | 'r' | 's' | 'o') => {
                self.applied.get_or_insert_with(|| Self::defaults(analysis));
                match key {
                    KeyCode::Char('e') => self.placement = !self.placement,
                    KeyCode::Char('c') => self.columns = self.columns % 3 + 1,
                    KeyCode::Char('r') => self.column_major = !self.column_major,
                    KeyCode::Char('s') => self.split = (self.split + 1) % 3,
                    KeyCode::Char('o') => {
                        self.order.reverse();
                        self.selected = self
                            .visible()
                            .len()
                            .saturating_sub(1)
                            .saturating_sub(self.selected);
                    }
                    _ => {}
                }
                self.scroll = 0;
                true
            }
            KeyCode::PageUp if self.applied.is_some() => {
                self.scroll = self.scroll.saturating_sub(2);
                true
            }
            KeyCode::PageDown if self.applied.is_some() => {
                self.scroll = (self.scroll + 2).min(self.visible().len().saturating_sub(1) as u16);
                true
            }
            _ => false,
        }
    }
}

pub(super) fn value(app: &App, index: usize) -> String {
    let snapshot = app.snapshot.as_ref();
    let request = snapshot.and_then(|s| s.requests.get(app.selected));
    let analysis = request.and_then(|r| r.analysis.as_ref());
    let number = |n: Option<f64>, unit: &str| {
        n.map(|n| {
            if unit == "s" {
                format!("{n:.2} {unit}")
            } else {
                format!("{n:.3} {unit}")
            }
        })
        .unwrap_or_else(|| "n/a".into())
    };
    let count = |n: Option<u64>| n.map(|n| n.to_string()).unwrap_or_else(|| "n/a".into());
    let text = |s: Option<&str>| clean(s.unwrap_or("n/a"));
    match index {
        0 => text(request.map(|r| {
            r.analysis
                .as_ref()
                .map_or("text-generation", |a| a.task.as_str())
        })),
        1 => text(request.map(|r| r.state.as_str())),
        2 => number(request.map(|r| r.elapsed_seconds), "s"),
        3 => count(request.and_then(|r| r.prompt_tokens)),
        4 => count(request.and_then(|r| r.output_tokens)),
        5 => count(request.and_then(|r| r.cached_tokens)),
        6 => number(request.and_then(|r| r.first_output_seconds), "s"),
        7 => number(request.and_then(|r| r.decode_tokens_per_second), "tok/s"),
        8 => number(request.and_then(|r| r.prefill_tokens_per_second), "tok/s"),
        9 => text(analysis.and_then(|a| a.runtime.as_deref())),
        10 => text(analysis.and_then(|a| a.device.as_deref())),
        11 => text(analysis.and_then(|a| a.dtype.as_deref())),
        12 => analysis
            .and_then(|a| a.model_cache_hit)
            .map(|hit| if hit { "hit" } else { "miss" })
            .unwrap_or("n/a")
            .into(),
        13 => number(analysis.and_then(|a| a.load_seconds), "s"),
        14 => number(analysis.and_then(|a| a.inference_seconds), "s"),
        15 => number(analysis.and_then(|a| a.worker_seconds), "s"),
        16 => count(analysis.and_then(|a| a.results)),
        17 => count(analysis.map(|a| u64::from(a.attempts))),
        18 => count(snapshot.map(|s| s.totals.active)),
        19 => number(
            snapshot
                .and_then(|s| s.host_memory_free_bytes)
                .map(|n| n as f64 / 1073741824.),
            "GiB",
        ),
        20 => number(
            snapshot
                .and_then(|s| s.host_swap_used_bytes)
                .map(|n| n as f64 / 1073741824.),
            "GiB",
        ),
        21 => number(
            snapshot
                .and_then(|s| s.memory.as_ref())
                .and_then(|m| m.accelerator.available_bytes)
                .map(|n| n as f64 / 1073741824.),
            "GiB",
        ),
        _ => "n/a".into(),
    }
}

#[cfg(test)]
mod observability_tests {
    use super::*;
    #[test]
    fn placement_swaps_neighbours_and_respects_rows_columns_and_boundaries() {
        let mut fields = Selection::default();
        fields.handle(KeyCode::Char('e'), true);
        let before = fields.visible();
        fields.handle(KeyCode::Right, true);
        assert_eq!(fields.visible()[1], before[0]);
        assert_eq!(fields.selected, 1);
        fields.handle(KeyCode::Down, true);
        assert_eq!(fields.visible()[3], before[0]);
        fields.handle(KeyCode::Tab, true);
        assert_eq!(fields.selected, 4);
        fields.handle(KeyCode::Char('r'), true);
        assert!(fields.column_major);
        fields.selected = 0;
        let before = fields.visible();
        fields.handle(KeyCode::Down, true);
        assert_eq!(fields.visible()[1], before[0]);
        fields.handle(KeyCode::Char('c'), true);
        assert_eq!(fields.columns, 3);
        fields.handle(KeyCode::Char('s'), true);
        assert_eq!(fields.split, 1);
        fields.viewport_width = 23;
        let before = fields.visible();
        fields.handle(KeyCode::Right, true);
        assert_eq!(fields.visible(), before); // Narrow terminal has only one column.
        fields.handle(KeyCode::Char('o'), true);
        assert_eq!(
            fields.visible(),
            before.into_iter().rev().collect::<Vec<_>>()
        );
        fields.handle(KeyCode::Esc, true);
        assert!(!fields.placement);
        assert!(!fields.handle(KeyCode::Down, true)); // Normal arrows select requests again.
        fields.handle(KeyCode::Char('R'), true);
        assert!(fields.applied.is_none());
        assert_eq!(fields.columns, 2);
    }
    #[test]
    fn selection_is_optional_cancellable_and_session_local() {
        let mut fields = Selection::default();
        assert!(fields.handle(KeyCode::Char('v'), true));
        assert!(!fields.draft.as_ref().unwrap()[7]); // Analysis has no decode rate.
        fields.handle(KeyCode::Char(' '), true);
        fields.handle(KeyCode::Esc, true);
        assert!(fields.applied.is_none());
        fields.handle(KeyCode::Char('v'), true);
        fields.handle(KeyCode::Char(' '), true);
        fields.handle(KeyCode::Enter, true);
        assert!(!fields.applied.as_ref().unwrap()[0]);
        fields.handle(KeyCode::Char('v'), false);
        assert!(!fields.draft.as_ref().unwrap()[0]); // Custom fields survive task changes.
        fields.handle(KeyCode::Char('R'), false);
        assert!(fields.applied.is_none());
        assert!(fields.draft.is_none());
        fields.handle(KeyCode::Char('v'), false);
        assert!(fields.draft.as_ref().unwrap()[7]); // Generation defaults restored.
        assert!(Selection::default().applied.is_none());
    }
}
