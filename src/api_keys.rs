use anyhow::{Context, Result, bail};
use std::{
    env, fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

const KEY_BYTES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKeyEntry {
    pub name: String,
    pub key: String,
}

pub fn default_api_keys_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("api-keys.toml"))
}

pub fn load_api_keys_file(path: &Path) -> Result<Vec<ApiKeyEntry>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read API key file {}", path.display()))?;
    let entries = parse_api_keys_toml(&text)
        .with_context(|| format!("failed to parse API key file {}", path.display()))?;
    if entries.is_empty() {
        bail!("API key file {} does not contain any keys", path.display());
    }
    Ok(entries)
}

pub fn generate_api_key() -> Result<String> {
    let mut bytes = [0u8; KEY_BYTES];
    getrandom::getrandom(&mut bytes)
        .map_err(|err| anyhow::anyhow!("failed to generate random API key: {err}"))?;
    Ok(format!("sk-werk-{}", hex(&bytes)))
}

pub fn write_api_keys_file(path: &Path, name: &str, force: bool) -> Result<ApiKeyEntry> {
    let name = normalized_name(name);
    let entry = ApiKeyEntry {
        name,
        key: generate_api_key()?,
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let text = api_keys_file_text(&entry);
    let mut options = OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    #[cfg(unix)]
    {
        options.mode(0o600);
    }

    let mut file = options
        .open(path)
        .with_context(|| format!("failed to create API key file {}", path.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("failed to write API key file {}", path.display()))?;

    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to restrict permissions on {}", path.display()))?;
    }

    Ok(entry)
}

pub fn parse_api_keys_toml(text: &str) -> Result<Vec<ApiKeyEntry>> {
    let mut entries = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_key: Option<String> = None;
    let mut in_keys_table = false;

    for (index, raw_line) in text.lines().enumerate() {
        let line_no = index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[keys]]" {
            finish_entry(&mut entries, &mut current_name, &mut current_key, line_no)?;
            in_keys_table = true;
            continue;
        }

        let Some((field, value)) = line.split_once('=') else {
            bail!("invalid API key TOML at line {line_no}: expected key = \"value\"");
        };
        let field = field.trim();
        let value = parse_toml_string(value.trim(), line_no)?;
        match field {
            "name" => {
                in_keys_table = true;
                current_name = Some(normalized_name(&value));
            }
            "key" => {
                in_keys_table = true;
                current_key = Some(validate_key(value, line_no)?);
            }
            _ if in_keys_table => {}
            _ => {}
        }
    }

    finish_entry(
        &mut entries,
        &mut current_name,
        &mut current_key,
        text.lines().count() + 1,
    )?;
    Ok(entries)
}

fn finish_entry(
    entries: &mut Vec<ApiKeyEntry>,
    name: &mut Option<String>,
    key: &mut Option<String>,
    line_no: usize,
) -> Result<()> {
    match (name.take(), key.take()) {
        (None, None) => Ok(()),
        (name, Some(key)) => {
            entries.push(ApiKeyEntry {
                name: name.unwrap_or_else(|| "default".to_string()),
                key,
            });
            Ok(())
        }
        (_, None) => bail!("invalid API key TOML before line {line_no}: key is missing"),
    }
}

fn validate_key(value: String, line_no: usize) -> Result<String> {
    if value.trim().is_empty() {
        bail!("invalid API key TOML at line {line_no}: key must not be empty");
    }
    if value.chars().any(char::is_control) {
        bail!("invalid API key TOML at line {line_no}: key contains control characters");
    }
    Ok(value)
}

fn normalized_name(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        "default".to_string()
    } else {
        value.to_string()
    }
}

fn api_keys_file_text(entry: &ApiKeyEntry) -> String {
    format!(
        "# Werk1112 API keys. Keep this file private and do not commit it.\n\
         # Clients must send: Authorization: Bearer <key>\n\n\
         [[keys]]\n\
         name = \"{}\"\n\
         key = \"{}\"\n",
        escape_toml_string(&entry.name),
        escape_toml_string(&entry.key)
    )
}

fn parse_toml_string(value: &str, line_no: usize) -> Result<String> {
    let Some(rest) = value.strip_prefix('"') else {
        bail!("invalid API key TOML at line {line_no}: value must be a quoted string");
    };
    let mut out = String::new();
    let mut escaped = false;
    for (offset, ch) in rest.char_indices() {
        if escaped {
            let decoded = match ch {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            };
            out.push(decoded);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => {
                let trailing = rest[offset + ch.len_utf8()..].trim();
                if !trailing.is_empty() {
                    bail!(
                        "invalid API key TOML at line {line_no}: unexpected text after quoted string"
                    );
                }
                return Ok(out);
            }
            other => out.push(other),
        }
    }
    bail!("invalid API key TOML at line {line_no}: unterminated quoted string")
}

fn strip_comment(line: &str) -> &str {
    let mut escaped = false;
    let mut in_string = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..index],
            _ => {}
        }
    }
    line
}

fn escape_toml_string(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            other => vec![other],
        })
        .collect()
}

fn config_dir() -> Result<PathBuf> {
    if let Some(path) = nonempty_env_path("XDG_CONFIG_HOME") {
        return Ok(path.join("werk1112"));
    }
    if cfg!(windows)
        && let Some(path) = nonempty_env_path("APPDATA")
    {
        return Ok(path.join("Werk1112"));
    }
    if let Some(home) = nonempty_env_path("HOME").or_else(|| nonempty_env_path("USERPROFILE")) {
        return Ok(home.join(".config").join("werk1112"));
    }
    bail!("could not determine config directory; pass --path")
}

fn nonempty_env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_api_keys_toml() {
        let entries = parse_api_keys_toml(
            r#"
            [[keys]]
            name = "open-webui"
            key = "sk-werk-one"

            [[keys]]
            name = "lm-studio"
            key = "sk-werk-two" # comment
            "#,
        )
        .unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "open-webui");
        assert_eq!(entries[0].key, "sk-werk-one");
        assert_eq!(entries[1].name, "lm-studio");
        assert_eq!(entries[1].key, "sk-werk-two");
    }

    #[test]
    fn generated_api_key_has_stable_prefix_and_entropy_length() {
        let key = generate_api_key().unwrap();
        assert!(key.starts_with("sk-werk-"));
        assert_eq!(key.len(), "sk-werk-".len() + KEY_BYTES * 2);
    }
}
