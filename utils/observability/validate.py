"""Validate shipped monitoring assets without external Python dependencies."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
dashboard = json.loads((root / "grafana-dashboard.json").read_text())
assert dashboard["uid"] == "werk-observability"
assert dashboard["__inputs"][0]["name"] == "DS_PROMETHEUS"
ids = set()
for panel in dashboard["panels"]:
    assert panel["id"] not in ids
    ids.add(panel["id"])
    assert panel["datasource"]["uid"] == "${DS_PROMETHEUS}"
    for target in panel["targets"]:
        assert "werk_" in target["expr"] or 'job="werk"' in target["expr"]
assert len(ids) >= 10
expressions = {target['expr'] for panel in dashboard['panels'] for target in panel['targets']}
for metric in ['decode_tokens_per_second', 'prefill_tokens_per_second', 'first_output_seconds',
               'load_seconds', 'inference_seconds', 'worker_seconds', 'model_cache_hit', 'results']:
    assert f'werk_last_request_{metric}' + '{job="werk"}' in expressions
config = (root / "prometheus.yml").read_text()
assert "metrics_path: /metrics" in config
assert "credentials_file:" in config
assert "sk-werk-" not in config
print(f"Validated {len(ids)} Grafana panels and Prometheus configuration")
