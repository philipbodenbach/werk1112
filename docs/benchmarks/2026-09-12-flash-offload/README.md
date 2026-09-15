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
Unverifizierte Parser bleiben gesperrt. Der spätere GLM-Test steht im
Abschnitt „GLM mit Tool-Katalog und Tool-Rückgabe“.

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

**Befund vor der Speicherkorrektur vom 14. September:** Auto war für diesen GLM noch nicht zuverlässig. Mit mehr freiem
RAM wurden beim [Neustart](glm-local-restart-chat-auto-memory-abort.log) und
[Serve](glm-local-serve-auto-memory-abort.json) obere Budgets von 23.937 bzw.
24.292 MiB gewählt. Beide Anfragen wurden von der nativen Speicherwache
abgebrochen (36,4–36,5 GB Nutzung bei 35,6 GB Hard-Watermark). Die Speicherwache
bleibt aktiv; die erfolgreiche kleinere Auto-Auswahl beweist keine sichere
höhere Residenz. Die beiden damaligen Loader-Korrekturen beheben diese Grenze nicht.
Die zusätzliche Attention-Speicherkorrektur und neue Messungen stehen weiter unten.

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

Vor der zusätzlichen Parseranbindung wurde die echte Tool-Anfrage mit HTTP 400 und
`unsupported_tool_calling` abgewiesen. Der damalige GLM-Textadapter unterstützte
daher gewöhnliche Textanfragen, aber nicht den von Open WebUI mitgeschickten
Tool-Vertrag. Die erfolgreiche Qwen-Tool-Abnahme lässt sich nicht auf GLM übertragen.

Das mitgelieferte Template öffnet immer einen Reasoning-Block und liest
`enable_thinking` nicht. `WERK_OMLX_THINKING=0` wurde angefordert, schaltet bei
diesem Checkpoint das Reasoning aber nicht aus. Deshalb 256 statt der in den
Qwen-Kurztests verwendeten 16 Ausgabetokens. Die damalige Abnahme umfasste
kein GLM-Tool-Calling; Vision und MTP bleiben außerhalb dieses Textadapters. N-Gramm-Offload ist mangels Tabellen
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

## GLM Auto und Reasoning-Aufwand, 2026-09-14

Die native GLM-Linear-Attention hält beim ersten Forward zusammengefasste
Q/K/V-/Gate-Gewichte zusätzlich zu den ursprünglichen Projektionen. Beim lokalen
Vontra-Checkpoint sind das **3.4308 GiB (3.683.811.328 Bytes)**. Diese persistenten
Kopien fehlten bisher sowohl im Auto-Basisbudget als auch in der gemessenen
Speicherbelegung vor dem ersten Prompt. Der Textloader berücksichtigt sie jetzt
im Inventar und wertet die nativen Kopien bereits beim Laden aus. Die erste
Prefill-Zulassung sieht damit rund 13,88 statt 10,11 GB Basisbelegung. Gemischte
Projektionsquantisierung behält den nativen ungefusionierten Pfad. Kein nativer
Speicherschutz wird abgeschaltet; explizite Expertenbudgets bleiben nutzbar.

Zusätzlich wird der vom GLM-Template ausgewertete `reasoning_effort` über
`WERK_OMLX_REASONING_EFFORT=low|high|max` und API
`werk.omlx.reasoning_effort` weitergegeben. ComfyUI Text Config und n8n Text
Chat Options bieten dafür eine unabhängige Auswahl. `inherit`/Weglassen erhält
den bisherigen Standard; eine Request-Änderung lädt die Gewichte nicht neu.
`low` ist eine Aufwandseinstellung, kein garantiertes Abschalten von Reasoning.

Private Serve-Läufe, dieselbe erste Frage „Merke dir die Zahl 37. Antworte nur
mit OK.“, Temperatur 0, maximal 256 Ausgabetokens, Auto/gruppierte Experten,
Persistenz an, `thinking=0`, keine Tools:

| Lauf | Erste Antwort | Zeit bis sichtbarer Text | Gesamtzeit | Generierte Tokens |
| --- | --- | ---: | ---: | ---: |
| [Vorher, Modellstandard](glm-opt-baseline.json) | OK | 62,76 s | 63,38 s | 82 |
| [Speicherkorrektur, Modellstandard](glm-opt-after-memory-baseline.json) | OK | 66,33 s | 67,01 s | 82 |
| [Speicherkorrektur, low](glm-opt-after-low.json) | OK | 14,87 s | 15,40 s | 2 |

Die beiden folgenden Serve-Turns mit `low` liefern korrekt `37` in **10,19 s**
und `5` in **14,24 s**, mit 3 bzw. 13 generierten Tokens. Alle drei Streams enden
mit `stop` und `[DONE]`. Die kürzere erste Antwort dauert rund 77 % weniger als
mit korrigierter Speicherplanung und Modellstandard. Die Speicheränderung allein
beschleunigt den Test nicht; der große Unterschied kommt vom geringeren
Reasoning-Umfang. Aus diesen kurzen Ausgaben folgt keine entsprechende Steigerung
der reinen Decode-Rate oder eine allgemeine Qualitätsaussage.

Die effektiven Expertenbudgets waren nicht identisch: vorher 20,37 GiB,
Speicherkorrektur/Standard 18,13 GiB, mit `low` anfangs 19,76 GiB und zuletzt
18,58 GiB. Auto richtet sich nach dem verfügbaren RAM. Betriebssystem-Dateicaches
sind nicht kontrolliert; aufgezeichnete Lesebytes sind logische Reads und können
aus dem Dateicache kommen. Die Messung isoliert daher nicht jeden Performancefaktor.

Reproduktion der drei Serve-Turns (installierter lokaler Checkpoint erforderlich):

```sh
cargo build --release --locked
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_auto.py custom-low low
```

Ein Label mit `baseline` misst nur den ersten Prompt; ohne zweiten Parameter
bleibt Reasoning auf dem Modellstandard. Der Reproducer beendet seinen privaten
Server auch bei Fehlern. Er schreibt neue JSON-/Logdateien unter dem gewählten
Label; für einen neuen Vergleich ein neues Label verwenden.

[Elf CLI-Turns mit Auto/low](glm-opt-low-chat.json) sind ebenfalls vollständig
und korrekt: **11/11**, erste Antwort **14,75 s**, warmer Median **9,77 s**.
Das identische kurze Aufgabenpaket hatte zuvor mit Modellstandard einen warmen
Median von 31,27 s. Der langsamste neue Turn benötigt weiterhin **38,79 s**;
`low` garantiert keine gleichbleibend kurze Antwort. Im letzten Turn werden
191 von 213 Prompttokens wiederverwendet. Das obere Expertenbudget beträgt
20,29 GiB, die wirksame Residenz wird während der Anfragen reduziert.

Prüfungen der Änderung: 77 oMLX-bezogene Rust-Tests, zusätzlich sechs API-Tests
mit explizitem Reasoning-Feld einschließlich Typ-/Wertefehlern und Streaming;
acht native Textadapter-Tests mit GLM/Qwen-Mathematik und Cache-Neuöffnung;
240 ComfyUI-Tests; 100 n8n-Tests sowie Build, Lint und tatsächlicher Host-Loader.

[Prozessneustart mit derselben Auto/low-Sitzung](glm-opt-low-restart-chat.json):
22 gespeicherte Nachrichten wiederhergestellt, korrekte Antwort **37** in
**13,33 s**, **216 von 239 Prompttokens** aus dem nativen persistenten Präfix.
Auch dieser Lauf endet regulär; der Testprozess und sein privater Worker werden
beendet. Auto/low besteht damit hier Kaltstart, elf Chat-Turns, drei Serve-Turns
und native Wiederaufnahme. Lange Kontexte und allgemeine Modellqualität bleiben
außerhalb dieses kurzen Abnahmepakets.

## GLM mit Tool-Katalog und Tool-Rückgabe

Die bisherige Ablehnung von Open-WebUI-Anfragen mit Tools lag an der fehlenden
GLM-Freigabe in Werks Kompatibilitätsprobe. Das installierte MLX-LM enthält bereits
den zum Vontra-Template passenden `glm47`-Parser für
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`.
Die Probe verifiziert nun auch für `glm5_next` die tatsächliche Template-Auswahl,
die Tokenizer-Verkabelung, den vollständigen nativen Parser samt Hilfsfunktionen
und seine Begrenzungsmarken. Unpassende oder geänderte Parser bleiben abgewiesen.
Tools werden weiterhin an den nativen Server weitergegeben.

Native Tests prüfen GLM und Qwen einschließlich Zahl-/String-/Boolean-/Objekt-
Argumenten, parameterlosen Funktionen, Template-Vorrang, explizitem Abschalten
und geänderten Parserfunktionen. Beide Tests, alle 34 Probe-Tests und 77
Rust-oMLX-Tests bestehen. Die Korrektur ist im Release-Binary installiert.
Ein erster Rust-Lauf in der Sandbox konnte keine HTTP-Testserver/Prozessgruppen
anlegen; derselbe Lauf mit den erforderlichen Berechtigungen besteht vollständig.

Die großen API-Abnahmen verwenden einen freien Modell-Arbeitsbereich, ohne
gleichzeitig laufenden alten GLM-Worker:

```sh
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_tools.py
WERK_FLASH_MODEL=Vontra/GLM-5.3-Flash-MLX-oQ2-MTP \
WERK_OMLX_REASONING_EFFORT=low \
WERK_FLASH_LARGE_TOOL_REPORT=/tmp/werk-glm-large-tools.json \
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_large_tools.py
```

Der erste Reproducer prüft eine normale Textantwort mit Tool-Katalog sowie einen
`add(17,25)`-Aufruf und dessen Tool-Rückantwort, jeweils als JSON und SSE. Der
zweite prüft zwei gewöhnliche Textfragen mit 34 angebotenen Tools.

Die [fünf echten Anfragen](glm-tools.json) bestehen nach der Optimierung des
Attention-Speichers: normale Rust-Antwort mit angebotenem Tool,
`add(17,25)` als strukturierter Tool-Call und anschließende Antwort `42`,
jeweils als JSON und SSE. Streaming liefert `tool_calls`/`stop` und `[DONE]`.
Der erste Tool-Aufruf dauert 42,68 s, die später wiederholte Anfrage 8,76 s;
die zugehörigen Rückantworten dauern 6,32 bzw. 4,36 s. Das ist eine reale
Tool-Abnahme, keine allgemeine Modellqualitätsaussage.

Auch [zwei echte Textstreams mit 34 Tools](glm-large-tools.json) bestehen:
synthetischer Open-WebUI-ähnlicher Katalog mit 28.425 JSON-Bytes, Auto-Experten,
Reasoning low und Disk-Persistenz. Erster Rust-Satz **231,33 s**, danach eine
Ownership-Frage **31,30 s**. Beide Antworten sind vollständig und enden mit
`stop`/`[DONE]`; es erfolgt kein unbeabsichtigter Tool-Aufruf. Der zweite Aufruf
verwendet **6.144 von 6.169 Prompttokens** wieder. Die erste Promptverarbeitung
dauert 210,24 s; der große Katalog wird beim zweiten Aufruf weitgehend aus dem
Cache bedient. Das ist ein Cachevergleich innerhalb derselben Version, kein
Nachweis eines siebenfachen Decode-Gewinns durch die Speicheroptimierung.
Der Browser selbst wurde hier nicht automatisiert; geprüft wird der HTTP-/SSE-
Vertrag mit angebotenem Tool-Katalog. Die Testworker wurden beendet.

## GLM: gemeinsamer Attention-Speicher und längere warme Antworten

Der nächste Schritt nach der korrekten Speicherreservierung beseitigt die
zusätzlichen Attention-Kopien: Die ursprünglichen nativen Projektionsmodule
verwenden nun Zeilenansichten der zusammengefassten Gewichte. Packed Weights,
Scales und Biases teilen dieselben Allokationen. Dadurch werden beim lokalen
GLM **3,4308 GiB** frei, ohne Gewichte umzuwandeln oder die Modellrechnung zu
ändern. Der native ungefusionierte Rechenpfad bleibt möglich; gemischte
Quantisierung ohne native Fusion bleibt unverändert.

Zehn native Textadapter-Tests bestehen. Ein neuer Test misst die tatsächlich
freigegebenen MLX-Allokationen für quantisierte und unquantisierte Projektionen,
prüft bitgleiche Fusionsausgaben und vergleicht den ungefusionierten Pfad.
Die vorhandenen GLM/Qwen-Tests prüfen weiterhin Prefill, Decode und identische
Fortsetzung nach Cache-Neuöffnung.

[Vorher](glm-warm-before.json) und [nachher](glm-warm-after.json): dreimal
„Write me a single sentence about the programming language Rust.“ innerhalb
eines wachsenden Gesprächs, Temperatur 0, Reasoning low, Auto-Expertenbudget,
gruppierte Auswertung, Persistenz und maximal 256 generierte Tokens. Beide
privaten Server wurden nacheinander gestartet und wieder beendet. Sämtliche
drei Antworten sind zwischen den Läufen **wortgleich**, einschließlich gleicher
Prompt-/Ausgabetokenzahlen. Alle Streams enden mit `stop` und `[DONE]`.

| Turn | Gesamt vorher | Gesamt nachher | Kürzer | Experten-Hitrate vorher → nachher | Decode vorher → nachher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 36,90 s | 35,64 s | 3,4 % | 54,8 → 57,1 % | 1,74 → 1,81 tok/s |
| 2 | 31,42 s | 29,26 s | 6,9 % | 60,1 → 63,8 % | 1,59 → 1,73 tok/s |
| 3 | 45,99 s | 43,06 s | 6,4 % | 48,5 → 53,0 % | 1,52 → 1,61 tok/s |

Das Auto-Budget steigt in diesen Läufen von **18,67 auf 21,82 GiB**; nach dem
dritten Turn beträgt die effektive Obergrenze 16,89 bzw. 20,35 GiB. Die erste
Speicherzulassung sieht rund **10,18 statt 13,81 GB** Basisbelegung. Native
Speicherwachen bleiben aktiv. Der verfügbare Systemspeicher und OS-Dateicache
sind nicht fixiert; die Tabelle beschreibt einzelne Vergleichsläufe und keine
statistisch abgesicherte pauschale Beschleunigung.

Der dritte Turn braucht weiterhin mehr Zeit. Wiederverwendete Prompttokens:
0, 62, 74; neu berechnete Prompttokens: 23, 13, 42. Der KV-Cache kann also auch
Tokens aus einer früheren Antwort wiederverwenden, wenn deren Tokenpräfix zum
neu gerenderten Gespräch passt. Er kann das bei veränderter Darstellung, etwa
durch entfernte Reasoning-Teile, nicht pauschal tun. Es werden ausschließlich
passende Tokenpräfixe wiederverwendet; keine Zustände anderer Texte.

Die lokalen Header ergeben für aktive Expertengewichte pro Token, summiert
über alle gerouteten Schichten: GLM **2,46 GiB**, Qwen **1,24 GiB**, DeepSeek
**2,01 GiB**. Das sind Gewichtsgrößen, keine physische SSD-Lesemenge und keine
FLOP-/Latenzprognose. GLMs Gesamtbestand an Expertengewichten beträgt 88,59 GiB,
Qwens 63,28 GiB und DeepSeeks 85,88 GiB. Warmes Caching garantiert daher weder
gleichen Durchsatz zwischen Modellen noch monoton sinkende Turnzeiten.

Reproduktion mit einem neuen Label, ohne anderen geladenen Modellworker:

```sh
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py custom
```

## GLM: Expertenbudget während Decode wiederherstellen

Die Budgetprüfung lief bisher nur bei Aufnahme einer Anfrage und während
Prefill. Ein für dessen temporäre Tensoren verkleinerter Experten-Cache blieb
daher auch nach deren Freigabe während der gesamten Ausgabe klein. Der
gemeinsame Offload-Speicherschutz prüft das Budget jetzt auch nach nativen
Decode-Antworten auf dem MLX-Executor. Das gilt für eine einzelne laufende
Anfrage ohne wartende Anfragen oder parallele Prefills. Tatsächlich belegter
Speicher, native Grenzwerte, Transientreserve, Pins und explizite Obergrenzen
bleiben berücksichtigt. `last_decode_admission` macht die Prüfung sichtbar.

70 Experten-/Textadaptertests bestehen, einschließlich Wiederherstellung nach
Freigabe, verbleibendem Speicherdruck, expliziter Obergrenze, parallelen
Anfragen und unveränderter Rückgabe des nativen Antwort-Handlers. Beide
folgenden Großmodelltests verwenden native Persistenz und vollständiges SSE.

Vergleich mit dem unmittelbar vorherigen Build, erneut dieselben drei Fragen
im wachsenden Gespräch, wortgleiche Antworten und identische Tokenzahlen:

| Turn | Decode vorher | Decode mit Budgeterholung |
| --- | ---: | ---: |
| 1 | 1,81 tok/s | 1,86 tok/s |
| 2 | 1,73 tok/s | 1,74 tok/s |
| 3 | 1,61 tok/s | 1,67 tok/s |

[Messdaten](glm-warm-decode-recovery.json): Im dritten Turn steigt die
effektive Cachegrenze nach Prefill von 21,91 auf 23,08 GB zurück. Die alte
Messung blieb bei 21,85 GB. Das behebt die fehlende Budgeterholung, aber
**nicht den gesamten Decode-Rückgang zwischen unterschiedlichen Antworten**.
Auto-Obergrenzen und OS-Dateicache waren nicht fixiert; kleine Zeitunterschiede
sind keine statistisch abgesicherten Gewinne.

Ein zusätzlicher [Test mit identischer Anfrage ohne Verlauf](glm-warm-identical-context.json)
isoliert das Aufwärmen von verändertem Kontext und Antworttext. Alle drei
Antworten sind wortgleich, jeweils 23 Prompt- und 38 Ausgabetokens:

| Anfrage | Decode | Experten-Hitrate über die Anfrage | Wiederverwendete Prompttokens |
| --- | ---: | ---: | ---: |
| Kalt | 1,87 tok/s | 57,4 % | 0 |
| Warm 1 | 2,00 tok/s | 73,4 % | 22 |
| Warm 2 | 1,97 tok/s | 73,4 % | 22 |

Hier sinkt die Decode-Rate durch Aufwärmen nicht gegenüber kalt. Diese
Messung beweist weder gleiche Kosten unterschiedlicher Antworten noch eine
generelle GLM-Leistung auf DeepSeek-/Qwen-Niveau. Alle sechs Testantworten
waren korrekt und vollständig; beide privaten Testserver wurden beendet.

```sh
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py recovery
python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py identical --same-context
```

## GLM: direkter Persistenzvergleich und Decode-Nachladen

Zwei aufeinanderfolgende Läufe mit unverändertem Build, festen **20 GiB**
Expertenbudget, Gruppierung, Temperatur 0 und Reasoning low. `--fixed-history`
verwendet für jede Folgeanfrage dieselben aufgezeichneten Assistant-Nachrichten,
auch wenn die neue Antwort abweicht. Die Eingaben sind daher in beiden Läufen
identisch (23/75/116 Prompttokens). Ohne `--persistence` bleibt die gewöhnliche
oMLX-Cacheverwaltung bestehen; in diesen drei kurzen Anfragen wurden dabei
tatsächlich **0/0/0** Prompttokens wiederverwendet. Der zur Auslagerung gehörende
Experten-Cache bleibt in beiden Varianten aktiv.

| Runde | Decode ohne Werk-Persistenz | Decode mit Werk-Persistenz | Prefill ohne → mit |
| --- | ---: | ---: | ---: |
| 1 | 1,81 tok/s | 1,82 tok/s | 14,72 → 14,70 s |
| 2 | 1,60 tok/s | 1,66 tok/s | 22,66 → 7,85 s |
| 3 | 1,61 tok/s | 1,63 tok/s | 25,48 → 17,53 s |

[Ohne](glm-warm-persistence-ab-off.json), [mit](glm-warm-persistence-ab-on.json).
Alle sechs Antworten korrekt und vollständig. Die Antworten in Runde 2/3 sind
zwischen den Varianten nicht wortgleich (ohne: 36/36 Ausgabetokens, mit: 37/41).
Der Vergleich liefert deshalb keine isolierte prozentuale Decode-Beschleunigung
durch Persistenz. Er zeigt aber, dass der Rückgang auch ohne Werk-Persistenz
und ohne wiederverwendete Prompttokens auftritt. Die Expertenkapazität erreicht
während Decode in beiden Fällen wieder die festen 20 GiB.

Die Statusaufzeichnung enthält jetzt Zeit- und Zugriffszähler aus der reinen
Decode-Phase. Ausgewertet wird das Fenster zwischen erster und letzter
sekündlicher Stichprobe nach einem Decode-Callback für die aktuelle Anfrage.
Die 42 Expertenaufrufe pro GLM-Token dienen zur Normierung; Stichproben können
innerhalb eines Tokens liegen. Die Zahlen sind entsprechend Näherungen und
umfassen nicht zwingend den ersten/letzten vollständigen Token.

| Ohne Persistenz | Experten-Hitrate im Decode-Fenster | Logisch nachgeladene Gewichte/Token | Lesezeit/Token | Expertenpfad inklusive Vorarbeit/Token |
| --- | ---: | ---: | ---: | ---: |
| Runde 1 | 70,2 % | 0,733 GiB | 0,297 s | 0,558 s |
| Runde 2 | 63,6 % | 0,897 GiB | 0,361 s | 0,632 s |
| Runde 3 | 64,1 % | 0,883 GiB | 0,358 s | 0,631 s |

Mit Persistenz: Hitrate 70,2/66,8/65,2 %, Lesezeit 0,297/0,330/0,346 s pro
Token. Der größere Anteil des zusätzlichen Zeitbedarfs entfällt in beiden
Varianten auf das Nachladen von Experten. Der Messwert `routing_seconds`
enthält auch die Auswertung zuvor aufgeschobener nativer Operationen; er bleibt
ungefähr bei 0,11 s pro Token. Diese Messung stützt daher keine Erklärung,
die den beobachteten Rückgang allein wachsender Attention-Arbeit zuschreibt.
Logische Lesevorgänge können aus dem OS-Dateicache bedient werden; die Tabelle
ist keine Messung physischer SSD-Transfers.

Warum dieser GLM-Checkpoint empfindlich ist: Seine aktiven gepackten
Expertengewichte summieren sich auf 2,46 GiB pro Token, gegenüber 1,24 GiB bei
Qwen und 2,01 GiB bei DeepSeek. GLM verarbeitet 8 Experten pro gerouteter
Schicht mit `hidden_size=4096` und `moe_intermediate_size=2048`; Qwen verwendet
10 mit 2560/640. Bei GLM wird außerdem die native `compile_ffn`-Optimierung
für ausgelagerte Schichten deaktiviert, da dynamische Python-Routen und
Dateizugriffe nicht in diesem kompilierten Graphen ausgeführt werden können.
Das sind Gründe für hohe Grundkosten und größere Empfindlichkeit gegenüber
Fehlzugriffen, kein Beweis für zwangsläufig abnehmende Leistung jeder Sitzung.
Die unterschiedliche Auswahl benötigter Experten und die Cacheverdrängung
sind nun konkretere Optimierungsziele als weiteres Verändern der Persistenz.

```sh
WERK_FLASH_EXPERT_CACHE_MB=20480 python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py persistence-ab-off --no-persistence --fixed-history
WERK_FLASH_EXPERT_CACHE_MB=20480 python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py persistence-ab-on --fixed-history
```

## GLM: geschützter Experten-Cache und paralleles Lesen

Der anschließende [Optimierungsbericht](../../glm-decode-optimization.md)
dokumentiert die aus Krasis, KTransformers und Vates abgeleiteten Verfahren
sowie deren Grenzen für MLX. Keine zusätzliche Runtime wurde eingebunden.

| Runde | LRU-Referenz | Geschützter Cache | Geschützter Cache + parallele Reads |
| --- | ---: | ---: | ---: |
| 1 | 1,82 tok/s | 1,80 tok/s | 2,38 tok/s |
| 2 | 1,66 tok/s | 1,88 tok/s | 2,54 tok/s |
| 3 | 1,63 tok/s | 1,83 tok/s | 2,51 tok/s |

Feste 20 GiB, Persistenz an, identische Eingaben und wortgleiche Antworten.
Rohdaten: [LRU](glm-warm-persistence-ab-on.json),
[segmentierter LRU](glm-warm-retention-slru.json),
[zusätzlich parallele Reads](glm-warm-retention-parallel.json),
[nachfolgender LRU-Gegencheck](glm-warm-retention-baseline-recheck.json).
Die letzten beiden Spalten unterscheiden sich nicht bei den kumulierten
Cache-Zugriffs-/Eviction-Zählern und logischen Lesebytes an den Anfragegrenzen.
Es wurden keine Experten ausgelassen und keine Gewichte umquantisiert.

Alle Streams wurden vollständig abgeschlossen. Der Gegencheck verwendet den
noch nicht aktualisierten installierten Build; vor dessen späterem Austausch
gewinnt der neue Build daher einen Vergleich in beiden zeitlichen Richtungen.
OS-Dateicache und Hintergrundlast bleiben unkontrollierte Einflussgrößen.

Reproduktion mit aktuellem Arbeitsstand:

```sh
WERK_FLASH_EXPERT_CACHE_MB=20480 python3 docs/benchmarks/2026-09-12-flash-offload/reproduce_glm_warm.py new-retention --fixed-history
```

[Abnahme mit Auto-Budget](glm-warm-retention-auto.json): **2,55 / 2,66 / 2,62 tok/s**,
ebenfalls wortgleiche Antworten, vollständiges SSE und native Budgeterholung
nach dem dritten Prefill. Die Auto-Obergrenze beträgt hier 22,26 GiB; dieser
Lauf ist deshalb getrennt vom festen 20-GiB-Vergleich zu bewerten.

## Qwen / DeepSeek: geschützter Experten-Cache (2026-09-15)

Fortsetzung der GLM-Optimierung auf demselben 48-GiB-Mac. Pro Variante ein
frischer Worker, drei Runden mit „Explain ownership and borrowing in Rust in
about 100 words.“, Temperatur 0, Thinking aus, maximal 256 Ausgabetokens und
Persistenz. Die neuen Varianten spielen die Antworten der jeweiligen Baseline
als Verlauf ein. Das Expertenbudget ist in allen Varianten **20 GiB**; Qwens
N-gram-Cache bleibt bei **1 GiB**. Die anderen Modellserver sind beendet, Builds
und native Tests laufen außerhalb der Messungen. Angegeben ist die native
Decode-Rate, ohne Prefill. „Kalt“ bezeichnet den neuen Worker, keinen geleerten
macOS-Dateicache.

| Modell | Runde | Bisheriger LRU (tok/s) | Geschützter Cache (tok/s) |
| --- | ---: | ---: | ---: |
| Qwen3.8 Flash | 1 | 3,820 | 3,892 |
| Qwen3.8 Flash | 2 | 4,040 | 5,485 |
| Qwen3.8 Flash | 3 | 3,385 | 4,485 |
| DeepSeek V4 Flash | 1 | 2,755 | 2,784 |
| DeepSeek V4 Flash | 2 | 2,533 | 2,909 |
| DeepSeek V4 Flash | 3 | 2,328 | 2,579 |

Alle sechs Antworten sind gegenüber der jeweiligen Baseline wortgleich und
enden regulär mit `stop`. Der geschützte Cache verbessert beide Folgerunden:
Qwen um 35,7 / 32,5 %, DeepSeek um 14,9 / 10,8 %. Die erste Runde verändert
sich wenig. Größerer Verlauf und andere Routerentscheidungen können trotzdem
unterschiedliche Raten zwischen den Runden erzeugen.

Rohdaten: [Qwen vorher](qwen-decode-baseline.json),
[Qwen geschützter Cache](qwen-decode-retention.json),
[DeepSeek vorher](deepseek-decode-baseline.json),
[DeepSeek geschützter Cache](deepseek-decode-retention.json).

Reproduktion über [reproduce_flash_decode.py](reproduce_flash_decode.py):
`WERK_FLASH_FAMILY=qwen` oder `deepseek`, `WERK_FLASH_EXPERT_CACHE_MB=20480`,
`WERK_FLASH_BIN` zur Auswahl des Vergleichs-Builds. Zuerst Label `baseline`,
anschließend ein anderes Label mit `--fixed-history` ausführen. Die lokalen
Vergleichs-Binaries liegen unter `/private/tmp/werk-perf-before-20260915` und
`/private/tmp/werk-perf-retention-20260915`. Das Skript startet und beendet seine
eigenen privaten Server; laufende große Modelle vorher regulär beenden.

### Zusätzlich paralleles Nachladen

Maximal vier CPU-Leser laden die bereits angeforderten Experten-Tensoren einer
Gruppe. Die temporären Rohdaten bleiben auf 128 MiB begrenzt; größere Gruppen
fallen auf den synchronen Pfad zurück. DeepSeek verwendet dafür nun denselben
geprüften Reader wie Qwen/GLM und vermeidet seine bisherige zusätzliche
Bytearray-Kopie. Die BF16→FP16-Konvertierung seiner Expert-Metadaten bleibt
unverändert, alle MLX-Operationen bleiben auf dem besitzenden Executor.

| Modell | Runde | LRU vorher | Geschützter Cache | Zusätzlich paralleles Lesen | Gewinn gegenüber vorher |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 Flash | 1 | 3,820 | 3,892 | 4,723 | +23,7 % |
| Qwen3.8 Flash | 2 | 4,040 | 5,485 | 6,257 | +54,9 % |
| Qwen3.8 Flash | 3 | 3,385 | 4,485 | 5,152 | +52,2 % |
| DeepSeek V4 Flash | 1 | 2,755 | 2,784 | 3,684 | +33,7 % |
| DeepSeek V4 Flash | 2 | 2,533 | 2,909 | 4,211 | +66,3 % |
| DeepSeek V4 Flash | 3 | 2,328 | 2,579 | 3,508 | +50,7 % |

Alle Raten in tok/s. Die beiden Folgerunden gemeinsam, als Tokens geteilt durch
summierte Decode-Dauer: Qwen **3,68 → 5,65 tok/s**, DeepSeek **2,43 → 3,83 tok/s**.
Gegenüber der Cache-Variante bleiben Antworten, Treffer-/Miss-Zähler und die
gesamten logischen Lesemengen an allen drei Rundengrenzen exakt gleich. Der
zusätzliche Gewinn kommt daher vom Lesepfad und nicht von weniger generierten
Tokens oder einem größeren Expertenbudget. Logische Reads können aus dem
macOS-Dateicache bedient werden und sind keine Messung physischer SSD-Bytes.

Rohdaten: [Qwen parallel](qwen-decode-parallel.json),
[DeepSeek parallel](deepseek-decode-parallel.json), jeweils mit nativem
Timing, Zählern und Build-SHA256. Beide beginnen mit null wiederverwendeten
Prompt-Tokens und verwenden die erwarteten Prefixe in den Folgerunden.
Während der abschließenden Vergleiche meldete der Nutzer eine Cache-Löschung.
Alle sechs Requests beendeten den Stream regulär, ohne Fehler;
eine zusätzliche Wiederholung der Ausgangsversion dient der Absicherung.

Validierung: **101 Tests bestanden**, einschließlich nativer Qwen-/GLM- und
DeepSeek-Numerik, BF16/FP16-Umwandlung, Budgetverkleinerung mit Pins/Leases,
paralleler Lesefehler, Verwerfen temporärer Puffer und erneuter Verwendung nach
`deactivate()`. Automatische Wahl im Backend; keine neuen ComfyUI-/n8n-Optionen
und keine Änderungen an Tool Calling oder Prompt-Semantik erforderlich.

### Gegenprobe nach der Cache-Löschung

Die ursprüngliche Binary wurde zuletzt erneut mit identischem Referenzverlauf
und unveränderten Budgets gestartet:

| Modell | Runde 1 | Runde 2 | Runde 3 |
| --- | ---: | ---: | ---: |
| Qwen, Ausgangsversion erneut | 3,900 | 4,061 | 3,419 |
| DeepSeek, Ausgangsversion erneut | 2,450 | 2,223 | 2,305 |

Alle Raten in tok/s; alle Antworten wieder wortgleich und ohne Fehler. Qwen
reproduziert die erste Baseline nahezu genau; DeepSeek ist in dieser Gegenprobe
langsamer. Beide Gegenproben bleiben in jeder Runde unter der neuen Variante.
Die oben genannten Gewinne beziehen sich weiterhin auf die erste Baseline,
nicht auf die langsamere DeepSeek-Gegenprobe. Einzelne lokale Vergleiche, kein
universelles Speedup-Versprechen für beliebige Verläufe oder Tool-Schemata.

Rohdaten: [Qwen Gegenprobe](qwen-decode-baseline-recheck.json),
[DeepSeek Gegenprobe](deepseek-decode-baseline-recheck.json).
[Build- und Quelltext-Hashes](flash-decode-builds.json) dokumentieren die drei
Varianten. Der getestete finale Build wurde mit
`cargo install --path . --locked --offline` lokal installiert.
