# CUDA-Offload: Prüfstand 2026-09-18

WSL/Linux, RTX 3090 mit 24 GiB VRAM, CUDA Toolkit 13.0, etwa 62 GiB Host-RAM.
Der native Build verwendet den in `src/backend/llama_server/cuda_offload.rs`
festgelegten Fork-Commit mit deaktiviertem vollständigem mmap-Prefetch bei
aktivem Expertencache. Bestehender Werk-CUDA-Adapter, normale Chat-Sessions und
die vorhandenen privaten Slot-Snapshot-Funktionen.

## Echter Chat mit KV-Wiederaufnahme

Qwen3-4B Q4_K_M, vorhandene GGUF-Datei, CUDA, Kontext 2048, Batch 256,
Microbatch 64, Temperatur 0, maximal 16 Ausgabetokens. Erster Prozess:
Zahl 37 merken und erneut abfragen. Zweiter Prozess: dieselbe persistente
Session öffnen und die Zahl erneut abfragen. Antworten: `OK`, `37`, `37`.

| Turn | Prompt-Token | Davon wiederverwendet | Neu ausgewertet | Nachweis |
| --- | ---: | ---: | ---: | --- |
| Kalt | 23 | 0 | 23 | [Log](qwen3-cold-warm.log) |
| Im selben Prozess | 49 | 19 | 30 | [Log](qwen3-cold-warm.log) |
| Nach Prozessneustart | 76 | 45 | 31 | [Log](qwen3-restart.log) |

Beim Neustart wurde ein Snapshot mit 51 Token eingelesen. Die 45 tatsächlich
wiederverwendeten Prompt-Token stammen aus der nativen Antwort-Usage, nicht
aus der Snapshot-Größe oder der Anzahl gespeicherter Nachrichten. Der neue
Snapshot enthält 78 Token und belegt 11.503.724 Bytes.

Das ist ein Funktionsnachweis. Die Modelldatei lag für diesen kleinen Test
unter `/mnt/f`; Modellladen dauerte etwa 35 Sekunden und war nicht Gegenstand
eines SSD-Leistungsvergleichs. Snapshot-I/O trägt zur gesamten Turn-Latenz bei.
Qwen3-4B ist dicht und besitzt keine PLE-Tabelle: Dieser Lauf beweist weder
MoE-Offload noch N-Gramm-Offload an einem Großmodell.

## Native Expertentests und Grenzen

Der native CUDA-Testlauf bestand unter anderem Cache-Matmul, gemappten Prefill
mit Overflow, pageable H2D-Staging und mehrere Tests begrenzter Host-Budgets.
Der gesamte Lauf ist **nicht grün**: Ein Test zusätzlicher Host-Registrierungen
bricht mit `auxiliary_alias_not_identity` ab. Unter diesem WSL-CUDA-Treiber
entspricht die GPU-Adresse dort nicht der CPU-Adresse. Der Fork lehnt die
Konfiguration ab. Der gezielte gruppierte Lauf erreicht denselben Fehler.
Für den Testbuild war außerdem ein fehlendes `<stdexcept>`-Include in
`tests/test-moe-cache-registry.cpp` nötig; das verändert den Server-Build nicht.

DeepSeek V4 meldet 256 Experten, sechs aktive Experten und einen gemeinsamen
Experten. Der hier sichtbare Download enthält nur den 5.250.016-Byte-Metadaten-
Shard der alten Q2_K-Variante; der vollständige Q2_K_S-Checkpoint stand für
die Prüfung nicht zur Verfügung. Der echte DeepSeek-Offload-Test bleibt offen.

Der optionale CUDA-Build hat einen Cache in Slots pro Expertentensor und
einen separaten Grenzwert für gepinnten Host-Speicher. Unterstützte PLE-Tabellen
werden über native Lazy-Reads und den OS-Dateicache bedient. Das entspricht
nicht oMLX-Auto-Budgets oder dessen eigenständigem begrenztem N-Gramm-LRU.

## Vorhandene Mac-Testfolge mit CUDA wiederverwenden

Die bestehende [Flash-Chat-Testfolge](../2026-09-12-flash-offload/reproduce_chat.py)
akzeptiert jetzt `WERK_FLASH_BACKEND=cuda` und verwendet standardmäßig das
installierte `werk`. Dieselben elf Fragen und derselbe Neustartmodus bleiben
erhalten. Nach `werk backend install llama-cuda-offload` und mit vollständigem
lokalem Modell:

```bash
export WERK_FLASH_BACKEND=cuda
export WERK_FLASH_SESSION=cuda-auto-test

python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_chat.py \
  ggml-org/DeepSeek-V4-Flash-GGUF deepseek
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_chat.py \
  ggml-org/DeepSeek-V4-Flash-GGUF deepseek restart
```

`WERK_FLASH_MODEL_HOME` wählt optional den Store, `WERK_FLASH_REPORT_DIR` das
Berichtsverzeichnis und `WERK_FLASH_WERK` eine andere Werk-Installation.
`WERK_LLAMA_ARGS` kann das dokumentierte CUDA-Testprofil überschreiben.
JSON-Berichte enthalten die nativen Cache-Treffer pro Turn und für CUDA den
Nachweis einer Snapshot-Wiederaufnahme. Ein erfolgreicher Prozess-Exit allein
ist kein Nachweis für korrektes Expertencaching oder Antwortqualität.

Der breite Rust-Testlauf ergab 929 bestandene und 27 fehlgeschlagene Tests.
Die Fehler liegen im bestehenden oMLX-Testbereich (`Argument list too long`
beim Starten des Python-Workers und ein darauf bezogener CLI-Test). Der
Worker-Startfehler wurde separat am unveränderten `HEAD` reproduziert.
Der abschließende gezielte llama.cpp-Lauf bestand alle 54 Tests, einschließlich
Prozesswechsel, beschädigter Snapshots, abgebrochener Streams und Cache-Zähler.
