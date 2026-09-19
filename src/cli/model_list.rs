use anyhow::Result;
use console::{Alignment, Term, measure_text_width, pad_str};
use std::{io::Write, path};

use crate::model_store::{ModelManifest, ModelStorage, ModelStore};

const HEADERS: [&str; 7] = [
    "MODEL",
    "LAYOUT",
    "FAMILY",
    "ARCHITECTURE",
    "TASKS",
    "STORAGE",
    "PATH",
];

pub(super) fn print(store: &ModelStore, manifests: &[ModelManifest]) -> Result<()> {
    let width = usize::from(Term::stdout().size().1);
    let output = format(store, manifests, width)?;
    std::io::stdout().lock().write_all(output.as_bytes())?;
    Ok(())
}

fn format(
    store: &ModelStore,
    manifests: &[ModelManifest],
    terminal_width: usize,
) -> Result<String> {
    if manifests.is_empty() {
        return Ok(format!(
            "No matching models installed in {}\n",
            store.home().display()
        ));
    }

    let rows = manifests
        .iter()
        .map(|manifest| {
            Ok([
                manifest.id.clone(),
                manifest.metadata.repository_layout.to_string(),
                manifest
                    .metadata
                    .family
                    .as_deref()
                    .unwrap_or("-")
                    .to_string(),
                manifest.architecture.as_deref().unwrap_or("-").to_string(),
                super::join_display(&manifest.metadata.tasks),
                match &manifest.storage {
                    ModelStorage::Managed => "managed",
                    ModelStorage::External { .. } => "external",
                }
                .to_string(),
                path::absolute(store.model_location(manifest))?
                    .display()
                    .to_string(),
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let mut widths = HEADERS.map(measure_text_width);
    for row in &rows {
        for (width, value) in widths.iter_mut().zip(row) {
            *width = (*width).max(measure_text_width(value));
        }
    }

    let mut output = String::new();
    let table_width = widths.iter().sum::<usize>() + 2 * (HEADERS.len() - 1);
    if table_width <= terminal_width {
        append_row(&mut output, HEADERS, &widths);
        for row in &rows {
            append_row(&mut output, row.each_ref().map(String::as_str), &widths);
        }
    } else {
        // Keep complete names and copyable paths when a table would wrap.
        for (index, row) in rows.iter().enumerate() {
            if index > 0 {
                output.push('\n');
            }
            output.push_str(&row[0]);
            output.push('\n');
            for (label, value) in [
                "Layout",
                "Family",
                "Architecture",
                "Tasks",
                "Storage",
                "Path",
            ]
            .iter()
            .zip(&row[1..])
            {
                output.push_str(&format!("  {label:<12}  {value}\n"));
            }
        }
    }
    Ok(output)
}

fn append_row(output: &mut String, values: [&str; 7], widths: &[usize; 7]) {
    for (index, value) in values.into_iter().enumerate() {
        if index > 0 {
            output.push_str("  ");
        }
        if index + 1 == values.len() {
            output.push_str(value);
        } else {
            output.push_str(&pad_str(value, widths[index], Alignment::Left, None));
        }
    }
    output.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelFormat, ModelSource};

    #[test]
    fn mixed_storage_lists_actual_locations_at_both_terminal_widths() {
        let root = std::env::temp_dir().join("werk-list-storage");
        let store = ModelStore::resolve(Some(root.join("local"))).unwrap();
        let managed = ModelManifest {
            id: "small-local".to_string(),
            source: ModelSource::LocalPath {
                path: "source".to_string(),
            },
            storage: ModelStorage::Managed,
            format: ModelFormat::Gguf,
            architecture: None,
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/small.gguf".to_string()),
            backend: "llama-server".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        let mut external = managed.clone();
        external.id = "large-external".to_string();
        external.storage = ModelStorage::External {
            path: root.join("raid").join("large"),
        };
        let managed_path = path::absolute(store.model_dir(&managed.id)).unwrap();
        let external_path = path::absolute(root.join("raid").join("large")).unwrap();
        let registration_path = path::absolute(store.model_dir(&external.id)).unwrap();
        for width in [40, 4096] {
            let output = format(&store, &[managed.clone(), external.clone()], width).unwrap();
            assert!(output.contains("small-local"));
            assert!(output.contains("large-external"));
            assert!(output.contains("managed"));
            assert!(output.contains("external"));
            assert!(output.contains(&managed_path.display().to_string()));
            assert!(output.contains(&external_path.display().to_string()));
            assert!(!output.contains(&registration_path.display().to_string()));
        }
    }
}
