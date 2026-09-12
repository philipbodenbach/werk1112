# oMLX: Expert-Ausführung und Chat-Messungen

Aktualisierung 2026-09-12: [CLI-/Serve-Parität und automatische Budgetwahl](omlx-chat-serve-parity.md) mit neuen kontrollierten Vergleichsmessungen.

Stand: 2026-09-11. Implementiert sind eine schnellere Expert-Ausführung und
vergleichbare Chat-Messungen. Die frühere degenerierte Antwort ist bisher nicht
als konkreter Rechenfehler reproduziert; diese Änderung ist kein belegter
universeller Qualitätsfix.

## Mehrturn-Speicherfehler und Offload

Die ursprünglichen kurzen Cache-Vergleiche reichten nicht als Freigabe für
fortlaufende Chats: Ein echter WebUI-Verlauf mit 50 Prompt-Tokens und 24 GiB
Expert-Cache lief zunächst, die nächste Nachricht scheiterte mit
`prefill_memory_exceeded` (28,41 GiB belegt plus angeblich 20,65 GiB KV/SDPA).
Der Fehler wurde anschließend über dieselbe öffentliche Werk-API reproduziert.

oMLX 0.6.4 lernt temporäre Prefill-Kosten aus der Änderung des Prozessspeichers.
Der erste Prefill füllt jedoch auch den dauerhaften Werk-Experten-Cache. Dessen
Zuwachs wurde als pro Token wiederkehrender Rechenbedarf gelernt und im nächsten
Turn zusätzlich zum bereits belegten Speicher berechnet. 24,323 GiB Zuwachs bei
49 Prefill-Tokens, skaliert auf 32 Tokens mit dem nativen Sicherheitsfaktor 1,3,
ergeben die beobachteten 20,65 GiB.

Der private Werk-Worker erfasst jetzt den Cachebestand vor und nach genau dem
gemessenen Chunk und trennt dessen dauerhafte Änderung vom nativen Lernsignal.
Noch freie Expert-Cachekapazität und der Streaming-Workspace werden separat
reserviert. Der kalte erste Prompt bleibt damit ebenfalls abgesichert. Native
Speichergrenzen, KV-/Attention-Schätzungen und Ablehnungen bleiben aktiv.

Das konfigurierte Expert-Budget ist eine Obergrenze. Vor der Prompt-Zulassung
und an Prefill-Chunkgrenzen kann der Worker es verringern und ungeschützte
LRU-Experten freigeben; bei mehr Platz darf es wieder bis zur Obergrenze wachsen.
Pins und laufende Berechnungen bleiben geschützt. Diese Operationen laufen auf
dem Executor des betreffenden Modells. Lange Decode-Phasen besitzen weiterhin
die native Speicherüberwachung, aber keine zusätzliche Budgetanpassung pro Token.

Das vollständige Modell muss weiterhin **nicht in den Arbeitsspeicher passen**.
Die rund 85,9 GiB an Expert-Gewichten dieses Checkpoints bleiben auf SSD; ein
kleineres Budget lädt häufiger nach. 24 GiB sind keine Mindestanforderung.
Die Architekturprüfung und die Mindestgröße für einen einzelnen vollständigen
Experten bleiben bestehen. Es wird kein vollständig residentes Modell geladen
und kein Kernel-Speicherlimit erhöht.

### Messung des Fixes

Auf demselben 48-GiB-Mac wurden frische Worker mit 8, 24 und 30 GiB
Cache-Obergrenze gestartet. Jeder Verlauf beginnt mit der ursprünglichen
Rust-Frage, der schon vorhandenen kurzen Antwort und derselben Frage erneut:
50 Prompt-Tokens beim ersten Request. Danach wird jeweils die tatsächlich
erzeugte Antwort angehängt: 3, 5, 7, 9, 11 und 13 Nachrichten. Temperatur 0,
Top-p 0,95, Seed 42, Thinking aus, maximal 96 Completion-Tokens.

| Cache-Obergrenze | Effektiver Cache nach Antworten | Erfolgreiche Turns | Erster Request gesamt | Erster Text bei Folgeturns, Median | Folgeturns gesamt, Median | Max. abgetastete Worker-RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 GiB | 8 GiB | 2/2 | 32,33 s | 6,10 s¹ | 24,86 s¹ | 12,44 GiB |
| 24 GiB | 24 GiB | 6/6 | 25,40 s | 4,29 s | 15,66 s | 28,48 GiB |
| 30 GiB | 28,03–28,39 GiB | 6/6 | 22,75 s | 3,76 s | 12,48 s | 32,59 GiB |

¹ Nur ein Folgeturn beim gezielten Test mit kleinem Budget. Die beiden größeren
Profile vergleichen jeweils dieselben fünf Folgeturns mit je 41 Ausgabetokens;
alle sechs Antworten sind zwischen 24 und 30 GiB exakt identisch. Auch die beiden
8-GiB-Antworten stimmen überein. Sie sind verständliches Englisch ohne die zuvor
beobachtete Degeneration, enthalten aber zusätzlich eine Einleitung. Dies ist
kein allgemeiner Qualitätsnachweis.

Bei 30 GiB Obergrenze verkleinert der Worker den Cache tatsächlich, bevor die
Speichergrenzen erreicht werden. Gegenüber 24 GiB sinkt die mediane Gesamtzeit
der fünf Folgeturns hier um etwa 20 %. Einzelne Folgeturns brauchen 8,49–15,40 s
statt 14,36–18,12 s. Die nativen Präfix-Restores betragen 85, 140, 195, 250 und
305 Tokens. Alle Anfragen endeten mit `stop`, ohne Fehler oder Trunkierung.

Die Swap-Belegung war bereits vor den Tests vorhanden: bei 24 GiB blieb sie bei
1664,19 MiB, beim größeren Profil bei 1656,19 MiB. Das bedeutet keine Zunahme der
Belegung während dieser Läufe, nicht nachgewiesene Abwesenheit jeglicher
Swap-Aktivität. RSS wurde im Sekundentakt abgetastet und ist kein exaktes
Metal-Spitzenmaß. Diese kurzen Verläufe validieren den gemeldeten Fehler; sie
ersetzen keinen Dauertest oder eine Prüfung sehr großer Kontexte bzw. anderer
physischer Rechner mit weniger RAM.

Aktuell größtes hier erfolgreich geprüftes Profil:

```sh
WERK_OMLX_THINKING=0 WERK_OMLX_EXPERT_CACHE_MB=30720 \
  werk --backend omlx serve \
  --model mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose --persistence
```

30 GiB sind ein gewünschtes Maximum; `effective_cache_budget_bytes` zeigt die
darunter liegende tatsächliche Grenze. Andere Programme und Kontexte können
weniger erlauben. Für kleinere Maschinen bleibt beispielsweise `8192` gültig;
auch dann müssen Basisgewichte, KV und Workspace zusätzlich Platz finden.

Rohdaten: [reproduzierter Fehler](benchmarks/2026-09-11-omlx/multiturn-before-memory-fix.json),
[8 GiB](benchmarks/2026-09-11-omlx/multiturn-after-memory-fix-8g.json),
[24 GiB](benchmarks/2026-09-11-omlx/multiturn-after-memory-fix-24g.json),
[30 GiB Obergrenze](benchmarks/2026-09-11-omlx/multiturn-after-memory-fix-30g.json).
Das [lokale Reproduktionsskript](benchmarks/2026-09-11-omlx/multiturn_memory.py)
benutzt die öffentliche Werk-API und liest private Worker-Telemetrie separat.
Auf diesem Rechner kann es den bestehenden WebUI-Schlüssel lesend übernehmen;
alternativ `OPENAI_API_KEY` setzen. Beispiel:
`python3 docs/benchmarks/2026-09-11-omlx/multiturn_memory.py repeat-memory-test`.

Die Regressionstests reproduzieren zusätzlich mit dem tatsächlich installierten
oMLX-Scheduler und seinem Transient-Tracker die falsche 20-GiB-Schätzung und
prüfen die korrigierte Zulassung, echte Speicherablehnungen und die Reclaim-
Buchhaltung. Numerische Expert-/KV-Tests, kleine Budgets, Pins und Leases sowie
die Ausführung von Expert-Aktionen auf dem besitzenden Modell-Executor sind
ebenfalls geprüft. Der Fix bleibt auf den privaten Offload-Worker begrenzt.

Anschließend wurde die ursprüngliche Rust-Frage direkt aus Safari/Open WebUI
gesendet: 22 Ausgabetokens, `stop`, 16,01 s Backend-Gesamtzeit. Der gespeicherte
WebUI-Verlauf enthält die vollständige verständliche Antwort mit `done=true`
und ohne Fehler. Das ist eine echte Anfrage aus WebUI mit Prüfung des gespeicherten
Ergebnisses, keine visuelle Screenshot-Prüfung.
[Beleg](benchmarks/2026-09-11-omlx/memory-webui-validation.json).
Installierte und getestete Release-Binary sind bytegleich (SHA-256
`3626308a751f01cd312f440a0fb2530a4b7ba5b040bbc2039eaf59eb56c4c7e7`).

## Änderung

`WERK_OMLX_EXPERT_CACHE_MB` aktiviert weiterhin den experimentellen Offload.
Innerhalb dieses Pfads ist `WERK_OMLX_EXPERT_EXECUTION=grouped` jetzt Standard.
Die neun Tensoren eines nachgeladenen Experten werden gemeinsam materialisiert;
mehrere Expert-Ausgaben werden vor einer gemeinsamen GPU-Auswertung aufgebaut.
Aktive Gewichte bleiben bis zum Abschluss dieser Auswertung geschützt. Ein
kleiner wiederverwendbarer Allocator-Cache ersetzt das bisherige Leeren bei
jeder Verdrängung. `serial` erlaubt einen Vergleich mit der bisherigen Strategie.
Quantisierung, Routing und Modellgewichte bleiben dieselben.

Verbose-Ausgaben enthalten Treffer, Fehlzugriffe, Verdrängungen, logische
Lesebytes, Materialisierungen, Auswertungen und Laufzeiten. Diese Differenzen
betreffen den gesamten Worker; parallele Anfragen können beitragen. Lesebytes
können aus dem Dateicache des Betriebssystems stammen. Teilzeiten überlappen und
dürfen nicht zu einer vermeintlichen Gesamtdauer addiert werden.

Für alle Chat-Backends unterstützt Werk optional
`stream_options: {"include_usage": true}` mit tatsächlichen Tokenzählern im letzten
SSE-Chunk vor `[DONE]`. Fehlende Cache- oder Phaseninformationen werden dabei
nicht geschätzt. Der [HTTP-Benchmark](../utils/benchmarks/README.md) misst den
öffentlichen Chat-Pfad, erhält echte Gesprächsverläufe und prüft unter anderem
JSON, Rechnen, Sprachen und Abbruchgründe. `werk bench --json --include-output`
enthält Antworten und Diagnostik; native Templates erhalten Originalnachrichten.
Serve bindet außerdem vor dem Laden des Modells den Port. Der zweite Aufruf am
belegten Port wurde geprüft: sofortiger Fehler, kein zweiter Modellstart.

Zusätzlich werden erfolgreiche oMLX-Kompatibilitätsprüfungen nun in einem
geteilten Cache mit maximal 32 Einträgen wiederverwendet. Modellmetadaten,
Shard-Dateistatistiken, Launcher, Umgebung und importierte Laufzeitdateien bzw.
Suchverzeichnisse bilden seine Gültigkeitsprüfung. Änderungen invalidieren den
Eintrag; fehlgeschlagene Prüfungen und unvollständige Abhängigkeitslisten werden
nicht gespeichert. Zuvor startete die HTTP-Prompt-Auswahl für jede Anfrage erneut
Python; eine neue Seed-spezifische Session verursachte weitere Prüfungen.
Die optionale Dateiliste besitzt ein eigenes Größenlimit und einen größeren,
weiterhin begrenzten JSON-Transport. Eine Regression mit mehr als 64 KiB Ausgabe
prüft den im echten Neustart gefundenen Grenzfall. Zu große Inventare deaktivieren
nur die Optimierung.

Unter `--backend auto` wird die hier inkompatible MLX-Installation weiterhin vor
oMLX geprüft. Der MLX-Pfad ist bewusst nicht Teil dieses Probe-Caches. Der direkte
Start mit `--backend omlx` vermeidet diesen bekannten Fehlversuch für das gewählte
DeepSeek-Modell; bei anderen Modellen bleibt die Backend-Wahl modellabhängig.

## Vergleichsmethode

Apple M4 Pro, 48 GiB RAM, oMLX 0.6.4,
`mlx-community/DeepSeek-V4-Flash-2bit-DQ`, Thinking aus. Ausgangsversion: installierte
serielle Expert-Ausführung, 8 GiB Cache. Kandidat: gruppierte Ausführung mit 8,
12, 16 und 24 GiB. Je Konfiguration drei Aufgaben (Rust Englisch, Deutsch, JSON),
jeweils zweimal, sequenziell und mit Temperatur 0, Top-p 0,95, Seed 42, maximal
64 Tokens. Der deutsche Text wird bei diesem Limit abgeschnitten; das ist als
`finish=length` erfasst und zählt nicht als bestandener Qualitätsfall.

Die Kernmessungen gehen direkt an den privaten oMLX-Worker und isolieren dessen
Ausführung. Sie sind keine WebUI-Browsermessung. Die Ausgangsversion hatte den
ersten Rust-Präfix bereits im Cache; neue Worker anfangs nicht. Daher sind die
zweiten Läufe für Reaktionszeit und Gesamtzeit besser vergleichbar. Das ist eine
Pilotmessung mit zwei Läufen je Aufgabe, keine belastbare Perzentilanalyse.
OS-Dateicache und thermischer Zustand wurden nicht zurückgesetzt.

Zweiter Lauf je Aufgabe, Decode-Rate aus der oMLX-Usage:

| Ausführung / Expert-Cache | Rust Tokens/s | Rust erster Text | Rust gesamt | Deutsch Tokens/s | JSON Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bisher / 8 GiB | 1,59 | 2,58 s | 26,47 s | 1,88 | 1,67 |
| Gruppiert / 8 GiB | 2,19 | 1,99 s | 19,38 s | 2,56 | 2,55 |
| Gruppiert / 12 GiB | 2,42 | 1,88 s | 17,60 s | 2,94 | 3,21 |
| Gruppiert / 16 GiB | 2,67 | 1,66 s | 15,87 s | 3,56 | 15,28 |
| Gruppiert / 24 GiB | 14,30 | 0,38 s | 3,04 s | 6,08 | 15,83 |

Bei identischem 8-GiB-Budget verbessert sich Rust um 38 %; die anderen beiden
wiederholten Aufgaben um 36 % bzw. 53 %. Alle 24 Kandidat-Antworten sind exakt
mit ihren jeweiligen Baseline-Antworten identisch. Der große Sprung bei 24 GiB
hat einen anderen zusätzlichen Grund: Im zweiten Rust-Lauf gab es 10.320
Expert-Treffer und **keinen einzigen Fehlzugriff**; alle benötigten Gewichte
passten in den Cache. Der erste Rust-Lauf nach Worker-Neustart benötigte dagegen
23,01 Sekunden und erreichte nur 2,73 Tokens/s. Neue Inhalte, wechselnde Aufgaben
und andere Sampling-Ergebnisse können erneut Nachladen erfordern.

Auch die JSON-Wiederholung passt bereits bei 16 GiB vollständig in den
Expert-Cache. Die deutsche Wiederholung lädt bei 24 GiB weiterhin 1.614-mal nach;
ihre 6,08 Tokens/s sind deshalb aussagekräftiger für einen gemischten Verlauf
als eine pauschale Behauptung von 14–16 Tokens/s. Das 2×-Ziel wurde in diesen
warm wiederholten Aufgaben mit mehr RAM überschritten, **nicht** allein durch
den neuen Code bei konstantem 8-GiB-Budget und nicht allgemein für beliebige Chats.

Beobachtete maximale Worker-RSS bei Abtastung im Sekundentakt: 16,44 GiB mit
12-GiB-Cache, 20,24 GiB mit 16-GiB-Cache, 28,26 GiB mit 24-GiB-Cache. Swap-Belegung
blieb bei allen drei Vergleichen bei 0,25 MiB, die kumulierten Swapouts bei 16.
Der Betriebssystem-Kompressor wuchs dabei; zusätzliche Anwendungen und lange
Kontexte brauchen weiterhin Reserve. Das sind kurze Testläufe und abgetastete
RSS-Werte, kein garantiertes Speichermaximum. Das schnellste gemessene Profil
dieser ersten Pilotmessung verwendete 24 GiB Expert-Cache; 16 GiB lässt mehr Reserve.
Eine automatische globale Budgeterhöhung wurde nicht eingeführt.

Rohdaten: [Baseline](benchmarks/2026-09-11-omlx/baseline.json),
[8 GiB](benchmarks/2026-09-11-omlx/grouped8.json),
[12 GiB](benchmarks/2026-09-11-omlx/grouped12.json),
[16 GiB](benchmarks/2026-09-11-omlx/grouped16.json),
[24 GiB](benchmarks/2026-09-11-omlx/grouped24.json),
[Identität](benchmarks/2026-09-11-omlx/identity.json).

Historischer Startbefehl der 24-GiB-Pilotmessung; aktuelles Profil siehe oben:

```sh
WERK_OMLX_THINKING=0 WERK_OMLX_EXPERT_CACHE_MB=24576 \
  werk --backend omlx serve \
  --model mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose --persistence
```

`--backend omlx` wählt den nachweislich kompatiblen Backend-Pfad direkt. Thinking
und Expert-Budget gelten gleichermaßen für CLI und Server; WebUI sowie werkStation
verwenden den Serverzustand, solange sie keine Request-Overrides senden.

## Öffentliche API: zusätzlicher Latenzvergleich

Gleiche gruppierte Ausführung und 24 GiB Expert-Cache in allen drei Reihen;
identische JSON-Antwort mit 16 Tokens, Temperatur 0, Top-p 0,95, Seed 42.
Je zwei Aufwärmanfragen, danach fünf Messungen. Median der Clientmessungen:

| API-Konfiguration | Erster sichtbarer Text | Gesamtzeit | Gesamtzeit min–max |
| --- | ---: | ---: | ---: |
| Vor Prüf-Cache, Auto | 1,99 s | 3,05 s | 3,03–4,02 s |
| Mit Prüf-Cache, Auto | 1,28 s | 2,32 s | 2,30–2,33 s |
| Mit Prüf-Cache, direkte oMLX-Auswahl | 0,35 s | 1,39 s | 1,38–1,40 s |

Der zweite Schritt verändert nur die Wiederverwendung erfolgreicher Prüfungen;
der dritte vermeidet zusätzlich die bei Auto wiederholte inkompatible
MLX-Prüfung. Zusammen sinken hier die erste Textlatenz um 82 % und die Gesamtzeit
um 54 %. Alle Messanfragen erzeugten dasselbe korrekte JSON. Der erste Kaltlauf
nach Worker-Neustart ist ausdrücklich nicht in diesen Medianen enthalten.

Rohdaten: [vorher](benchmarks/2026-09-11-omlx/api-latency-before.json),
[Auto nachher](benchmarks/2026-09-11-omlx/api-latency-after-auto.json),
[direktes oMLX nachher](benchmarks/2026-09-11-omlx/api-latency-after-omlx.json).

## Öffentliche API und Qualität

Die öffentliche Werk-API wurde zusätzlich mit Temperatur 1 (ohne Seed),
Temperatur 0,6/Seed 42 und sechs festen Gesprächs-/Aufgaben-Turns bei Temperatur 0
geprüft. Das neue SSE-Usage-Format funktioniert; Gesprächsfortsetzungen erhalten
die vorherigen Antworten und nutzen laut Backend-Log den Präfix-Cache. Die API
liefert noch keine strukturierten Cache-/Reasoning-Details. Der Benchmark lässt
seine Decode-Schätzung deshalb standardmäßig offen; eine Schätzung bei fehlenden
Details erfordert ausdrücklich `--estimate-decode-rate` und die Annahme, dass
alle gezählten Tokens sichtbarer Text sind.

Es gibt **keinen pauschalen bestandenen Qualitätstest**: Temperatur 1 produzierte
vor dem angeforderten JSON zusätzlichen Text. Bei der Rechenaufgabe „three boxes
with four pencils each“ kam mit Thinking aus reproduzierbar `104` statt `12`.
Mit demselben Seed und Thinking an kam `12`, allerdings mit 54 Completion-Tokens
(einschließlich unsichtbarem Reasoning) und etwa 20,4 Sekunden statt einer
Ein-Token-Antwort. Dies zeigt einen Qualitäts-/Latenzunterschied an diesem Fall,
beweist aber weder eine allgemeine Qualitätsgarantie noch die Ursache der
historischen leeren/degenerierten Antwort. Temperatur 0,6 bestand das einzelne
JSON-Beispiel, machte die deutsche Antwort jedoch deutlich länger. Sie wurde
deshalb nicht stillschweigend zum Standard erhoben.

Thinking bleibt wie angefordert standardmäßig aus, wenn der Server mit
`WERK_OMLX_THINKING=0` gestartet wird. Für einen bewussten Qualitätsmodus kann
werkStation bereits `werk: {"omlx": {"thinking": true}}` pro Anfrage senden.
Keine generierten Texte werden zur Kaschierung von Modellfehlern umgeschrieben.

Rohdaten: [Sampling-Vergleich](benchmarks/2026-09-11-omlx/public-quality.json),
[Aufgaben und Mehrturn-Verlauf](benchmarks/2026-09-11-omlx/public-fixtures.json),
[Thinking-Vergleich](benchmarks/2026-09-11-omlx/thinking-quality.json).
Das Aufgabenpaket meldet den Rechenfehler korrekt als fehlgeschlagene Prüfung.

Ein neuer Chat wurde anschließend über Safari direkt in Open WebUI geöffnet.
Die ursprüngliche Rust-Frage erzeugte eine vollständige englische Antwort; der
Chatdatensatz enthält `done=true`, einen abgeschlossenen `output_text` und keinen
Fehler. Die leere ältere `content`-Eigenschaft ist hier kein fehlender Text:
WebUI speichert die Ausgabe unter `output[].content[].text`.
Für diesen neuen Rust-Präfix meldete das Backend 0 wiederverwendete Prompt-Tokens,
22 ausgegebene Tokens, 3,00 Tokens/s und 14,65 Sekunden Gesamtzeit. Dieser reale
WebUI-Fall ist kein vollständig warmer Wiederholungsfall. Eine Fensteraufnahme
war in der Umgebung nicht möglich; der Safari-Ablauf und gespeicherte Abschluss
wurden geprüft, die gerenderten Pixel nicht separat.
[Prüfdatensatz](benchmarks/2026-09-11-omlx/webui-validation.json).

Die finale Release-Datei ist als lokales `werk` installiert und bytegleich mit
dem geprüften Build. Der eigene Testserver wurde beendet; Port 11434 ist frei.

## Numerische Tests und verbleibende Grenzen

46 native Tests prüfen unter anderem gemischte 2-/4-Bit-Quantisierung,
BF16-Metadaten, native Hash-Routen, mehrschrittige Logits, Pooling-/Rotating-KV,
Präfix-Speicherung/Wiederherstellung, Evictions, kleine Budgets und Fehler während
des Ladens bzw. der GPU-Auswertung. Die Modellausgabe der deterministischen
Vergleichsfälle stimmt exakt mit der Ausgangsversion überein.
Die numerischen Referenzmodelle sind klein und zufällig initialisiert; sie prüfen
Implementierungstreue, keine Sprachqualität des großen Checkpoints.

Zusätzlich bestehen 55 oMLX-Rust-Tests, 33 Python-Preflight-Tests, 21 Chat-API-Tests, der Regressionstest für
strukturierte Benchmark-Nachrichten und 17 HTTP-Benchmark-Tests. Eine Kollision
in parallelen Testverzeichnissen wurde durch einen eindeutigen Zähler behoben.

Nicht umgesetzt sind neue Gather-Kernels, asynchrones Expert-Prefetching,
Speculative Decoding, eine automatische RAM-Budgetwahl sowie ein universelles
Sampling-/Qualitätsprofil für andere Modelle. Der KV-Cache beschleunigt vor allem
die Verarbeitung bereits bekannter Eingaben; die Token-Ausgabe muss weiterhin
Modellrechnung und nachgeladene Experten durchlaufen. Die Workspace-Angabe ist
eine Planungsreserve; ein striktes transientes Maximum bei sehr großen Prefills
wurde nicht nachgewiesen.
