use serde_json::Value;

pub(super) fn is_meaningful_name(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "none" | "automatic" | "use same checkpoint"
    )
}

pub(super) fn harmless_default(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(value) => value.as_f64() == Some(0.0),
        Value::String(value) => !is_meaningful_name(value),
        Value::Array(values) => values.is_empty() || values.iter().all(harmless_default),
        Value::Object(values) => values.is_empty() || values.values().all(harmless_default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn harmless_defaults_are_narrow_and_recursive() {
        assert!(harmless_default(&Value::Null));
        assert!(harmless_default(&json!({"nested": false})));
        assert!(harmless_default(&json!("Automatic")));
        assert!(!harmless_default(&json!(true)));
        assert!(!harmless_default(&json!({"weight": 1})));
    }
}
