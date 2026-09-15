# CLI/Serve-Parität und automatische Expertenbudgets

Stand: 2026-09-12, Mac mit 48 GiB Unified Memory, oMLX 0.6.4,
`mlx-community/DeepSeek-V4-Flash-2bit-DQ`.

`werk chat` und `werk serve` verwenden denselben oMLX-Decoder. Der zuvor
verglichene CLI-Verlauf hatte 8 GiB Experten-Cache und zehn gespeicherte
Nachrichten; Serve lief mit 24 GiB und anderer Historie. Daraus ließ sich kein
langsamerer CLI-Decoder ableiten. Eine kontrollierte Messung bestätigt nun
vergleichbare Geschwindigkeit, identische Texte und bei identischem Budget
identische Expertenzugriffe.

## Umsetzung

- Ohne `WERK_OMLX_EXPERT_CACHE_MB` oder mit dem Wert `auto` wählen beide
  Befehle für den unterstützten DeepSeek-V4-Offload denselben Budgetalgorithmus.
  Die Auswahl erfolgt beim Workerstart aus aktuellem verfügbarem RAM,
  effektivem Metal-Limit und den tatsächlichen Basis-/Expert-Gewichtsgrößen.
  Reserviert werden Systemspielraum, Basisgewichte und temporärer Workspace.
  Die vorhandene native Speicherprüfung begrenzt das Arbeitsset zusätzlich
  vor Prompt-Verarbeitung und an Prefill-Chunk-Grenzen für KV/Attention.
- Es gibt keine feste 24-GiB- oder 1-TiB-Grenze für Auto. Obergrenze ist die
  kleinere Menge aus nutzbarem Speicher und sämtlichen Experten des Modells.
  Die Berechnung für TiB-Systeme wurde mit simulierten Speicherwerten geprüft,
  nicht auf einer solchen Maschine gemessen.
- Positive MiB-Werte bleiben explizite Obergrenzen. Insbesondere 8 GiB und
  noch kleinere Caches bleiben unterstützt; nicht residente Experten werden
  weiterhin aus dem Checkpoint nachgeladen. `0` wählt den normalen nativen
  Loader. Auto gilt derzeit für unterstützte affine DeepSeek-V4-Checkpoints
  auf oMLX 0.6.4; andere Architekturen/Laufzeitversionen behalten ihren nativen
  Ladepfad. Kernel-Limits werden nicht verändert.
- CLI-Verbose trennt jetzt gesamte, gecachte und neu evaluierte Prompt-Tokens.
  Bei 778 Gesamt-Tokens, 757 gecachten Tokens und 9,25 Sekunden Prefill sind
  das 21 neu evaluierte Tokens und 2,27 Tokens/s, nicht 84,11 Tokens/s.
  Die Cacheanzahl wird strukturiert aus der Laufzeit übernommen, nicht aus
  Diagnose-Strings herausgelesen. Unbekannte Cachewerte bleiben unbekannt.
- Serve protokolliert zusätzlich dieselben nativen Prefill-/Decode-Dauern,
  die Zeit zum ersten Token und die neu evaluierten Prompt-Tokens.
- Ein HTTP-Regressionstest vergleicht die tatsächlich gesendeten Mehrturn-
  Requests des persistenten CLI-Pfads mit dem Serve-Backendpfad, einschließlich
  Nachrichten, Seed, Sampling, Thinking und gleichem 8-GiB-Budget.

## Reale Vergleichsmessung

Zwei identische deutsche Rust-Fragen, frische leere CLI-Sitzung und frischer
Serve-Worker, Temperatur 0, Top-p 0,95, Seed 42, Thinking aus, max. 64 Tokens.
Die zweite HTTP-Anfrage enthält exakt die Antwort der ersten CLI-Anfrage.
Es läuft immer nur ein großer Modellworker. Serve nutzt einen eigenen Testport
18434; alle Testprozesse wurden anschließend beendet. Der OS-Dateicache wurde
nicht geleert. Die Messwerte sind einzelne kontrollierte Durchläufe, keine
Mediane oder universellen Leistungszusagen.

| Budget / Pfad | Turn 1 gesamt | Turn 1 Decode | Turn 2 gesamt | Turn 2 erstes Token¹ | Turn 2 Decode | Turn 2 gecacht |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 GiB / CLI | 36,618 s | 2,46 tok/s | 22,138 s | 8,96 s | 2,35 tok/s | 86 / 108 |
| 8 GiB / Serve | 37,346 s | 2,41 tok/s | 22,187 s | 8,98 s | 2,35 tok/s | 86 / 108 |
| Auto / CLI | 26,751 s | 3,96 tok/s | 9,538 s | 4,35 s | 5,98 tok/s | 86 / 108 |
| Auto / Serve | 27,211 s | 3,91 tok/s | 8,838 s | 3,95 s | 6,43 tok/s | 86 / 108 |

¹ Native oMLX-Zeit, bei Serve getrennt von der im JSON-Bericht ebenfalls
enthaltenen clientseitigen Zeit bis zum ersten sichtbaren SSE-Text.

Bei 8 GiB liegen die Folgeturns rund 0,05 Sekunden auseinander. Experten-Hits,
Misses und gelesene Bytes stimmen sogar exakt überein: im zweiten Turn
5.902 Hits, 4.559 Misses und 38.189.137.920 gelesene Bytes. Das belegt denselben
Inferenzaufwand; diese Bytes enthalten OS-Dateicache-Lesezugriffe.

Auto wählte beim CLI 27.662 MiB Obergrenze und bei Serve 29.303 MiB; der
verfügbare Speicher beim Start war unterschiedlich. Nach nativer Anpassung
waren im Folgeturn ungefähr 27,01 bzw. 28,12 GiB Experten resident. Der
CLI-Folgeturn ist damit gegenüber 8 GiB rund 2,32× schneller insgesamt und
2,54× schneller beim Decode. Serve erreicht etwa 2,51× bzw. 2,74×. Auto fordert
keine identischen Bytebudgets bei unterschiedlichem freien Systemspeicher.

Alle verglichenen Antworten sind zwischen CLI, Serve, 8 GiB und Auto gleich.
Der erste Turn erreicht absichtlich das Testlimit von 64 Tokens, der zweite
endet nach 31 Tokens mit `stop`. Dies ist ein Paritäts- und Geschwindigkeitstest,
kein Nachweis eines allgemeinen Qualitätsfixes oder einer robusten Begrenzung
auf genau einen Satz. Alte auffällige Gesprächshistorien wurden nicht gelöscht.

Im früheren CLI-Log ist das 680-Token-Endpräfix der ersten Antwort kein exaktes
Präfix der erneut tokenisierten nächsten Anfrage. Der kürzere 428-Token-Prefix
passt. Tokenizer-/Template-Nachprüfung bestätigt diesen Unterschied; der Cache
muss dann sicher auf das kürzere passende Präfix zurückfallen. Die genaue
Ursache der abweichenden Roh-Tokenfolge und der damaligen Textdegeneration ist
damit nicht bewiesen. Ein nicht passender KV-Zustand wird nicht erzwungen.

Rohdaten und reproduzierbares Testskript:
[8-GiB-Bericht](benchmarks/2026-09-12-omlx-parity/8192-report.json),
[Auto-Bericht](benchmarks/2026-09-12-omlx-parity/auto-report.json),
[Vergleichsskript](benchmarks/2026-09-12-omlx-parity/compare.py).
Die jeweiligen CLI-/Serve-Logs liegen im selben Verzeichnis. Ein anfänglicher
Testskriptfehler hatte den CLI-Tokenlimit-Hinweis als Antworttext eingelesen;
der betroffene HTTP-Lauf wurde verworfen und mit korrekt identischer Historie
wiederholt. Die Tabelle enthält ausschließlich den korrigierten Vergleich.

## Starten

~~~bash
WERK_OMLX_THINKING=0 WERK_OMLX_EXPERT_CACHE_MB=auto \
werk --backend omlx chat mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose --persistence

WERK_OMLX_THINKING=0 WERK_OMLX_EXPERT_CACHE_MB=auto \
werk --backend omlx serve --model mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose --persistence
~~~

Für einen kontrollierten Vergleich außerdem dieselbe Historie, Temperatur,
Top-p, Seed und Tokenobergrenze verwenden. CLI `--persistence` stellt die
bisherige Konversation wieder her; `--session NAME` wählt eine andere Sitzung.
Explizite Client- oder Umgebungsbudgets wie `8192` überschreiben Auto weiterhin.

## Prüfstatus

- 48 oMLX-Rust-Tests einschließlich neuer CLI-/Serve-Request-Parität und
  modellgebundener Auto-Aktivierung bestanden.
- 69 native Expert-/Persistenz-/Qualitätstests auf Metal bestanden, darunter
  numerische Prüfungen mit kleinem Cache und simulierte Auto-Budgetgrenzen.
- 81 Python-Expert-/Probe-Tests: 65 bestanden, 16 hardwaregebundene Tests im
  Standard-Python-Lauf übersprungen; der separate native Lauf prüft diese.
- Vollständige Rust-Suite: 895/900 bestanden. Fünf Fehler außerhalb der
  geänderten Inferenzpfade bleiben bestehen:
  `cli::tests::fallback_diagnostic_retains_the_executed_route_without_verbose_or_reprobe`,
  `cli::tests::repeated_route_notes_are_deduplicated_and_recovery_resets_them`,
  `runtime_planner::tests::missing_or_incompatible_preferred_runtime_reports_the_actual_fallback`,
  `runtime_planner::tests::safetensors_auto_prefers_an_available_rocm_vllm_runtime`,
  `werk_protocol::client::tests::client_parses_envelope_and_sends_bearer_without_leaking_it`.
  Der erste genannte Runtime-Planner-Test und der Protokollclient-Test schlagen
  auch isoliert fehl (fehlende Fallback-Notiz bzw. Transport-Fehler 22).
- `cargo fmt --check` und `git diff --check` bestanden. Release installiert;
  Benchmark-, Release- und installiertes Binary anhand SHA-256 abgeglichen.
