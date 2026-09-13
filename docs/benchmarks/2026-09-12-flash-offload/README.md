# Flash-Offload: Messstand 2026-09-13

Apple Silicon, 48 GB Unified Memory, oMLX 0.6.4; Werk aus diesem Arbeitsstand.
Qwen-Checkpoint vollständig lokal geprüft. Die Manifeste im Modellstore binden
die Dateien über Prüfsummen; eine ursprüngliche Git-Revision ist für den bereits
importierten Qwen-Download nicht mehr vorhanden.

## Reale Qwen-Läufe

Alle Läufe: Thinking aus, Temperatur 0, maximal 16 Ausgabetokens, Persistenz an,
1 GiB separater N-Gramm-Cache. Elf kurze Deutsch-/Englisch-Aufgaben mit
Rechnen und Erinnerung an die Zahl 37. Alle elf Antworten stimmen jeweils.

| Lauf | Expertenbudget | Warmer Median je Turn | Rohdaten |
| --- | --- | ---: | --- |
| CLI, seriell | 8 GiB | 18,217 s | [JSON](qwen-chat.json) |
| CLI, gruppiert | 8 GiB | 13,000 s | [JSON](qwen-grouped-chat.json) |
| Serve, gruppiert | 8 GiB | 12,654 s | [JSON](qwen-grouped-serve.json) |
| CLI, Auto/gruppiert | effektiv 22.399.793.128 Bytes | 4,173 s | [JSON](qwen-auto-chat.json) |

Median der Turns 2–11, eine Sitzung je Variante. Vollständige CLI-Ausgaben liegen
jeweils in der gleichnamigen `.log`-Datei. Serve bestätigt für jeden Turn den
erwarteten Text, Finish-Grund und vollständiges SSE einschließlich `[DONE]`.
Die HTTP-Usage enthielt keine Cached-Token-Felder; diese bleiben dort unbekannt.
CLI/Serve unterscheiden sich beim 8-GiB-Gesamtzeitmedian um etwa 2,7 %.

Das sind kurze Diagnosen mit meist 1–2 Ausgabetokens, kein allgemeiner
Qualitätsbenchmark und keine belastbare Messung längerer Decode-Leistung.
OS-Dateicache und paralleler GLM-Download waren nicht kontrolliert.
Auto verändert das Budget; dessen Beschleunigung ist deshalb kein isolierter
Nachweis für die Gruppierungsoptimierung.

Beim [zweiten Neustart](qwen-restart2-chat.json) mit unverändertem Build wurden
356 von 389 Prompttokens wiederverwendet, die Antwort blieb `37`.
Der [erste Neustart](qwen-restart-chat.json) nach Änderung der Modellmetadaten
war ein korrekter Cachemiss und zählt nicht als Nachweis nativer Wiederaufnahme.
Ältere Gruppierungsläufe enthalten noch Nullwerte für einzelne I/O-/Eval-Zähler;
der Auto-Lauf verwendet die korrigierte Instrumentierung. Lesebytes sind
logische Dateizugriffe, keine Messung physischer SSD-Transfers.

## Prüfung und verbleibende Grenzen

- 53 gezielte native Offload-/Textworker-/Probe-Tests bestanden, einschließlich
  Prefill, zehn Decode-Schritten, unabhängigen Budgets, Verdrängung und exakt
  gleicher Fortsetzung nach SSD-Neuöffnung auf kleinen nativen Modellen.
- 69 bestehende DeepSeek-Experten-/Persistenz-/Qualitätstests bestanden.
- Rust: 58 oMLX-, 60 Modellstore- und 6 Chatoptions-Tests bestanden.
  Ein vollständiger Bibliothekslauf ist auf diesem Mac nicht vollständig grün:
  vier bestehende CUDA-/ROCm-Fallbacktests setzen eine andere Hostplattform
  voraus. Der separat wiederholte lokale Worker-Test bestand.
- ComfyUI: 238 Tests. n8n: 99 Tests, Build, Lint und realer Loader mit acht
  Custom Nodes sowie importierten Image→Vision-/Text→Text-Workflows bestanden.
  Das ersetzt keinen manuellen Test jeder Benutzeroberfläche mit dem Großmodell.
- GLM war bei der ersten Abnahme nur auf kleinen nativen Modellen und anhand
  der 22 Tensorheader geprüft; der damalige Download wurde abgebrochen. Der
  später vom Benutzer regulär installierte Checkpoint wird im Abschnitt
  „GLM: realer lokaler Checkpoint“ separat bewertet.
- Die geprüften DeepSeek-/GLM-Checkpoints enthalten keine N-Gramm-Tabellen.
  Dort ist N-Gramm-Offload nicht anwendbar, während MoE unabhängig bleibt.

## Qwen mit Tool-Katalog und Tool-Rückgabe

Die Qwen-Probe prüft zusätzlich den nativen `qwen3_coder`-Parser, dessen
Typkonvertierung und die tatsächliche Tokenizer-/Template-Anbindung.
Geänderte lokale Templates invalidieren die zwischengespeicherte Probe.
Unverifizierte Parser und die bisher ungeprüfte GLM-Tool-Anbindung bleiben gesperrt.

[Fünf echte API-Anfragen](qwen-tools.json) bestanden mit Auto-Expertenbudget,
1 GiB N-Gramm-Cache, Thinking aus und Persistenz:

1. Rust-Erklärung trotz angehängtem Tool-Katalog: vollständiger Text-Stream.
2. JSON-Antwort mit `add`, Ganzzahlargumenten `17` und `25`, Tool-ID und
   Finish-Grund `tool_calls`.
3. Tool-Ergebnis `42` wird in der normalen Folgeantwort korrekt verwendet.
4. Derselbe Tool-Aufruf als SSE mit vollständigen Argumenten und `[DONE]`.
5. Tool-Rückgabe als SSE: korrekte Antwort und vollständiger Abschluss.

Das ist ein direkter Werk-HTTP-Test des von Open WebUI verwendeten Vertrags,
kein manueller Browsernachweis. Der Testserver wurde danach beendet.
Wiederholen: `python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_tools.py`.
Ergebnisse landen unter `/tmp/werk-qwen-tools.json` oder `WERK_FLASH_TOOL_REPORT`.

## Automatischer N-Gramm-Cache und großer Tool-Katalog

`WERK_OMLX_NGRAM_CACHE_MB=auto` startet auf diesem Qwen mit 64 MiB und
wächst nach tatsächlicher Zeilenverdrängung innerhalb des gemeinsamen
Speicherbudgets. Die Geräteobergrenze ist keine Vorabreservierung.
API (`werk.omlx.ngram_cache_mb: "auto"`), ComfyUI und n8n können Auto
auch gegenüber einem festen Serverbudget auswählen; manuelle Werte bleiben erhalten.

Der 34-Tool-Test verwendet 28.425 Bytes Tool-Schema, `tool_choice=auto`,
Thinking aus, gruppierte Experten und diskbasierte Persistenz. Beide Fragen
sind normale Rust-Fragen ohne Anweisung zum Abschalten von Tools. Der
synthetische Katalog ist größer als der gemeldete Open-WebUI-Katalog mit
21.003 Bytes; die Zeiten sind deshalb kein kontrollierter Vorher/Nachher-Vergleich.

[Abgeschlossener Lauf](qwen-large-tools-auto.json), beide Antworten korrekt,
`finish_reason=stop` und vollständiger SSE-Abschluss mit `[DONE]`:

| Anfrage | Prompttokens | Wiederverwendet | Erstes Textstück | Gesamt |
| --- | ---: | ---: | ---: | ---: |
| Kalter Rust-Satz | 6.684 | 0 | 327,55 s | 339,54 s |
| Neue Frage, gleicher Tool-Katalog | 6.682 | 6.144 | 46,73 s | 61,62 s |

Der N-Gramm-Cache blieb bei 64 MiB Kapazität; zuletzt waren etwa 6 MiB
belegt, ohne Zeilenverdrängung. Die Korrektur ermöglicht diesen großen
Tool-Katalog, beseitigt aber nicht den teuren kalten SSD-Prefill. Die zweite
Anfrage profitiert von Prefix-Wiederverwendung; 538 Prompttokens mussten
weiterhin berechnet werden. Ein festes 8-GiB-N-Gramm-Budget ist für diesen
Lauf durch die tatsächlich genutzten Zeilen nicht begründet.

Die Diagnoseversuche bleiben nachvollziehbar:

- [Vor zusätzlicher Prefill-Reserve](qwen-large-tools-before-reserve.json):
  nativer Speicherabbruch bei 36,0 GiB und einer Hard-Watermark von 35,6 GiB.
- [Vor nativer Chunk-Anpassung](qwen-large-tools-before-native-fallback.json):
  die lokale Mindestcache-Prüfung verhinderte den nativen Fallback.
- [Vor Korrektur verzögerter Freigaben](qwen-large-tools-before-deferred-reclaim.status.json):
  native Prognosen über 30 GiB begrenzten den Cache auf einen Experten.
  Dieser Lauf wurde nach der Diagnose kontrolliert beendet.

Die Textadapter lassen zusätzliche Transientreserve außerhalb der Gewichtscaches.
Ist ein angeforderter Chunk zu groß, bleiben die minimalen Gewichte verfügbar,
bevor die native Chunkwahl weiter verkleinert oder ablehnt. Verzögerte physische
Freigaben zuvor verdrängter Gewichte werden nicht erneut als Arbeitsreserve
angelernt; gleichzeitig wachsender aktiver MLX-Speicher bleibt sichtbar.
Die nativen Speichergrenzen bleiben wirksam.

Gezielte Validierung dieser Erweiterung: 118 native Tests und der zusätzliche
Regressionstest gegen oMLXs echten Reclaim-Tracker bestanden; 50 oMLX- und
6 Chatoptions-Rusttests, ComfyUI 239 Tests, n8n 99 Tests sowie Build/Lint bestanden.

Wiederholen: `python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_large_tools.py`.
Ausgabe: `/tmp/werk-qwen-large-tools-auto.json` oder `WERK_FLASH_LARGE_TOOL_REPORT`,
mit `.log` und `.status.json` daneben. Der Test verwendet einen eigenen Server
mit zufälligem API-Schlüssel; zur Speichermessung darf kein zweiter großer
Modellworker gleichzeitig laufen. Dies ist ein direkter HTTP-Test, kein manueller
Open-WebUI-Browsernachweis.

## Leseoptimierung für Qwen

Dateideskriptoren bleiben in einem begrenzten LRU geöffnet. Kleine Tensoren
verwenden den direkt gelesenen, unabhängig besessenen Bytepuffer. Bei gruppierter
Ausführung werden unmittelbar benachbarte, tatsächlich benötigte Expertenbereiche
zusammen gelesen. Die zurückgegebenen Zeilen besitzen eigene Puffer, damit ein
verbleibender Experte nach Verdrängung seiner Nachbarn keine ganze Gruppe im
Speicher hält. Read-ahead ist auf 128 MiB begrenzt; zusammen mit Zeilenkopien
passt es in die bestehende Arbeitsreserve. Kleine Budgets und verstreute Bereiche
verwenden weiterhin einzelne Lesezugriffe. Dateiidentitätsprüfungen bleiben aktiv.

Derselbe 34-Tool-Test, gleiche Fragen und identische Antworten, jeweils vollständiger
SSE-Abschluss; Angaben in Sekunden. Jede Variante wurde einmal ausgeführt:

| Variante | Kalt gesamt | Kalt erstes Textstück | Warm gesamt | Warm erstes Textstück | Warm Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Vorher](qwen-large-tools-auto.json) | 339,54 | 327,55 | 61,62 | 46,73 | 2,76 |
| [Deskriptoren und weniger Kopien](qwen-io-optimized.json) | 309,38 | 298,38 | 56,48 | 42,97 | 3,03 |
| [Zusätzlich zusammengefasste Lesezugriffe](qwen-coalesced.json) | 217,94 | 206,63 | 46,65 | 31,48 | 2,70 |

Im letzten Lauf sinkt die Gesamtzeit gegenüber dem Ausgangslauf kalt um 35,8 %,
warm um 24,3 %. Die Ausgabe selbst erreicht noch nicht die gewünschten etwa
6 Tokens/s. Die Verbesserung betrifft überwiegend die Promptverarbeitung.
Auto und die native Speicherregelung bleiben aktiv: Das obere Expertenbudget
betrug beim mittleren Lauf rund 25,8 GiB, beim letzten rund 22,5 GiB.
OS-Dateicache und verfügbarer Systemspeicher sind nicht kontrolliert; diese
Einzelläufe sind keine statistisch abgesicherte Geschwindigkeitsgarantie.
N-Gramm-Zeilen blieben bei etwa 6 MiB ohne Verdrängung.

Ein [isolierter Leser-Test](qwen-reader-microbenchmark.json) mit 675 MiB in
2.304 Lesezugriffen pro Durchlauf ergab im warmen OS-Dateicache 143,1 ms gegenüber
50,8 ms Median nach Deskriptor-/Kopieroptimierung. Das ist keine Messung
physischer SSD-Bandbreite und kein Faktor für die gesamte Inferenz.
Ein [gather_qmm-Prototyp](qwen-gather-microbenchmark.json) wurde verworfen:
bei einem Decode-Token war die gebündelte Matrixoperation langsamer
(1,20 statt 1,02 ms), trotz kleiner Vorteile bei größeren Batches.

[DeepSeek mit demselben Tool-Katalog](deepseek-same-tools.json) benötigte kalt
190,97 s und warm 53,54 s. Es erhielt in diesem Auto-Lauf allerdings nur rund
14 GiB Expertenbudget; auch die Tokenisierung unterscheidet sich (warm 244 statt
538 neue Prompttokens). Dieser Lauf beweist deshalb keine erreichte Gleichheit
bei der Decode-Geschwindigkeit und ersetzt keinen Vergleich bei festem Budget.

### Kurze Antworten mit gleichem Expertenbudget

Zusätzliche Diagnose mit denselben beiden Rust-Fragen, 24 GiB Expertenbudget,
Thinking aus und Persistenz. Der Tool-Katalog wurde ausschließlich für diesen
Kurzprompt-Vergleich weggelassen; die Prüfung mit 34 Tools oben bleibt maßgeblich
für den Open-WebUI-Anwendungsfall. Kein gemeinsamer Prefix wurde wiederverwendet.

| Modell | Kalt gesamt | Warm gesamt | Warm erstes Textstück | Warm Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| [Qwen](qwen-short-24g.json) | 23,45 s | 9,81 s | 2,99 s | 4,55 |
| [DeepSeek](deepseek-short-24g.json) | 20,89 s | 14,41 s | 4,75 s | 3,01 |

Qwen ist in dieser warmen Kurzdiagnose schneller als DeepSeek. Das ersetzt
keinen längeren Decode-Benchmark: Die Antworten umfassen nur 31 beziehungsweise
29 Tokens und die Modelle erzeugen unterschiedliche Texte. Die früher gemeldeten
etwa 6 Tokens/s von DeepSeek wurden hier nicht reproduziert. Für Open WebUI mit
großem Tool-Katalog bleibt insbesondere der kalte Prefill deutlich langsamer.

Der [erste DeepSeek-Start](deepseek-short-24g-before-memory-release.log) direkt
nach Qwen scheiterte korrekt an HTTP 507: Die dynamische Speichergrenze lag noch
bei 21 GiB gegenüber 29 GiB Ladebedarf. Nach Prüfung, dass keine Modellworker
mehr liefen und Systemspeicher wieder frei war, bestand derselbe Lauf unverändert.
Zwischen den Modellen muss die tatsächliche Speicherfreigabe abgewartet werden;
ein Ende des äußeren Serve-Prozesses allein reicht für den Vergleich nicht.

Wiederholen mit `WERK_FLASH_TOOL_COUNT=0 WERK_FLASH_EXPERT_CACHE_MB=24576`,
`WERK_FLASH_MODEL` und getrennten `WERK_FLASH_LARGE_TOOL_REPORT`-Dateien über
`reproduce_large_tools.py`. Der Standard bleibt der 34-Tool-Test.

Validierung der Leseoptimierung: 140 native Tests bestanden, einschließlich
Dateiersetzung, kurzer Lesezugriffe, begrenzter Deskriptoren, unabhängiger Puffer,
Sparse-/Budget-Fallback und Aufräumen nach Lesefehlern. Kleine native Qwen-/GLM-
Modelle prüfen weiterhin Prefill, Decode und exakte SSD-Fortsetzung. Release-Build
und `git diff --check` bestanden. ComfyUI/n8n benötigen für diese interne
Optimierung keine weiteren Parameter.

Der Reproducer erlaubt nun `WERK_FLASH_MODEL` und
`WERK_FLASH_EXPERT_CACHE_MB` (voreingestellt `auto`). Archivierte `.log`-Dateien
enthalten die tatsächlichen Budgets und Phasenmessungen.

## GLM: realer lokaler Checkpoint

`Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`, Architektur `glm5_next`, liegt inzwischen
regulär im Werk-Modellstore. Das geprüfte Tensorinventar enthält 95.126.814.720
Bytes Experten, 9.567.393.404 Bytes Text-Basis, keine N-Gramm-Tabellen sowie
Vision-/MTP-Gewichte, die der Textadapter ausschließt.

Die reale Ladeprüfung erforderte zwei Korrekturen:

- [Vision-Attribut](glm-local-before-vision-fix.log): `Module`-Kinder über
  `setattr(..., None)` entfernen, damit der native GLM-Sanitizer weiterhin
  auf `vision_model` zugreifen kann.
- [Affine Forget-Gate-Projektionen](glm-local-before-quantization-fix.log):
  Gewichte, Scales/Biases und Q8-Overrides gemeinsam in den nativen
  `forget_gate`-Namensraum überführen. Die strikte Gewichtsprüfung bleibt aktiv.

140 native Tests bestehen mit beiden Korrekturen. Der erweiterte Regressionstest
enthält eine echte kleine GLM-Vision-Konfiguration und quantisierte Forget-Gate-
Triplets im flachen Namensraum dieses Checkpoints; Referenz-Prefill, Decode und
SSD-Fortsetzung werden weiter geprüft.

[Elf echte CLI-Turns](glm-local-auto-chat.json) liefern elf korrekte Antworten
mit regulärem `stop`: Zahlenrechnung, Deutsch/Englisch und wiederholte Erinnerung
an `37`. Experten: Auto/gruppiert, oberes Budget in diesem Lauf 18.211 MiB,
Persistenz an, Temperatur 0, maximal 256 generierte Tokens. Erster Turn 67,41 s,
warmer Median 31,27 s. Die angezeigten Decode-Raten liegen ungefähr bei
1,4–1,6 Tokens/s und zählen auch Reasoning; sie sind keine Rate sichtbarer Wörter.

**Auto ist für diesen GLM noch nicht zuverlässig freigegeben.** Mit mehr freiem
RAM wurden beim [Neustart](glm-local-restart-chat-auto-memory-abort.log) und
[Serve](glm-local-serve-auto-memory-abort.json) obere Budgets von 23.937 bzw.
24.292 MiB gewählt. Beide Anfragen wurden von der nativen Speicherwache
abgebrochen (36,4–36,5 GB Nutzung bei 35,6 GB Hard-Watermark). Die Speicherwache
bleibt aktiv; die erfolgreiche kleinere Auto-Auswahl beweist keine sichere
höhere Residenz. Diese Grenze wird durch die beiden Loader-Korrekturen nicht behoben.

Mit [festen 16 GiB](glm-local-16g-restart-chat.json) wird dieselbe gespeicherte
Sitzung mit 22 Nachrichten korrekt wiederaufgenommen; Antwort `37`, 43,44 s,
regulärer Abschluss. Bei diesem Wechsel des Expertenbudgets wurden null native
Präfixtokens wiederverwendet; er belegt die Gesprächswiederherstellung, nicht
allein die native SSD-Präfixwiederverwendung.

Beim [zweiten Neustart mit unveränderten 16 GiB](glm-local-16g-reopen-restart-chat.json)
wurden 24 gespeicherte Nachrichten geladen und 238 von 265 Prompttokens aus
dem nativen persistenten Präfix wiederverwendet. Korrekte Antwort `37`, 28,59 s,
regulärer Abschluss. Damit ist die native Wiederaufnahme auch am großen GLM
bei dieser Konfiguration nachgewiesen.

[Serve mit 16 GiB](glm-local-16g-serve.json): drei korrekte Textantworten
(`OK`, `37`, `5`) in 68,44 / 69,98 / 26,13 s, jeweils `stop` und vollständiges
SSE einschließlich `[DONE]`. Das erste sichtbare Textstück kam erst nach
67,76 / 69,34 / 25,51 s. Native First-Token-Phasen zählen dagegen bereits
den Beginn der vorgelagerten Generierung; sie sind hier keine sichtbare Textlatenz.

Die anschließende echte Tool-Anfrage wurde mit HTTP 400 und
`unsupported_tool_calling` abgewiesen. Der aktuelle GLM-Textadapter unterstützt
daher gewöhnliche Textanfragen, aber nicht den von Open WebUI mitgeschickten
Tool-Vertrag. Die erfolgreiche Qwen-Tool-Abnahme lässt sich nicht auf GLM übertragen.

Das mitgelieferte Template öffnet immer einen Reasoning-Block und liest
`enable_thinking` nicht. `WERK_OMLX_THINKING=0` wurde angefordert, schaltet bei
diesem Checkpoint das Reasoning aber nicht aus. Deshalb 256 statt der in den
Qwen-Kurztests verwendeten 16 Ausgabetokens. GLM-Tool-Calling, Vision und MTP sind
mit diesem Textadapter nicht freigegeben. N-Gramm-Offload ist mangels Tabellen
nicht anwendbar.

Die Reproducer unterstützen `WERK_FLASH_MAX_TOKENS`, für CLI zusätzlich
`WERK_FLASH_EXPERT_CACHE_MB`, für Serve `WERK_FLASH_PROMPT_LIMIT` und optional
`WERK_FLASH_CHECK_TOOLS=1`. Der CLI-Reproducer prüft nun auch abgeschlossene
Turns und Generierungsfehler: Ein interaktiver CLI-Prozess kann trotz fehlgeschlagener
Anfrage regulär mit Exitcode 0 enden, wie der archivierte Auto-Neustart zeigt.

## Wiederholen

Die Skripte verwenden `target/release/werk` und schreiben nach
`/tmp/werk-flash-reproduction`, alternativ `WERK_FLASH_REPORT_DIR`.
Sie verändern keine archivierten Messdaten. CLI startet standardmäßig eine
frische Sitzung; für einen kontrollierten Neustart `WERK_FLASH_SESSION` bei
beiden Aufrufen identisch setzen. Ein erneuter Lauf ersetzt seine Ergebnisdatei.
Serve verwendet einen eigenen lokalen Server mit zufälligem API-Schlüssel.

```bash
cargo build --release
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_chat.py \
  pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit qwen compare-grouped
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_chat.py \
  pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit qwen compare-auto
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_serve.py \
  pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit qwen-grouped grouped 8192
```

`compare-serial` prüft den konservativen CLI-Modus. `restart` sendet nur die
Erinnerungsfrage. Bei GLM kann `WERK_FLASH_MODEL_HOME` dessen isolierten
Modellstore auswählen; voreingestellt ist `/private/tmp/werk-glm-offload-validation`.
