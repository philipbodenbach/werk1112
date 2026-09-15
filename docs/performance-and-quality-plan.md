**Plan: DeepSeek schneller und zuverlässiger machen, anschließend auf Werk und werkStation übertragen**

Stand: 2026-09-12. CLI und Serve verwenden jetzt dieselbe automatische, hardware- und modellabhängige Experten-Cacheauswahl. Explizite kleine Budgets bleiben erhalten. CLI-Prefill-Raten schließen verifiziert gecachte Tokens aus; Serve protokolliert dieselben Phasen getrennt. Nach dem gemeldeten Mehrturn-Speicherfehler ergänzt die Implementierung eine getrennte Buchhaltung für dauerhaften Expert-Cache und Prefill-Transienten sowie ein unter Speicherdruck verkleinerbares Cachebudget. Kleine Cachebudgets und Modelle oberhalb der RAM-Kapazität bleiben ausdrücklich unterstützt. Die erste Implementierungsstufe ist umgesetzt: gruppierte
Expert-Ausführung, gebündelte Tensor-Materialisierung, geschützte aktive Gruppen,
weniger Allocator-Flushes, Worker-Telemetrie, numerische Mehrschritt-/KV-Tests,
HTTP-Mehrturn-Benchmark und optionale SSE-Tokenzähler. `werk bench` bewahrt jetzt
strukturierte Nachrichten und kann Antworten für Qualitätsprüfungen speichern.
Serve prüft den Port vor dem Laden des Modells. Erfolgreiche oMLX-Prüfungen
werden mit Datei-/Laufzeit-Invalidierung wiederverwendet.

Ausgangspunkt bleibt DeepSeek-V4-Flash-2bit-DQ mit oMLX 0.6.4, deaktiviertem
Thinking und 8 GiB Expert-Cache. Die Schritte unten beschreiben auch die noch
nicht umgesetzten Ausbaustufen; ein universeller Qualitätsfix oder eine allgemeine
2×-Beschleunigung sind damit nicht behauptet. Messwerte und Grenzen stehen im
[Implementierungs- und Messbericht](omlx-performance-validation.md).

Der Kurzpräfixcache funktioniert inzwischen: Die letzten Nutzeranfragen verwenden
94–96 % ihrer Prompt-Tokens wieder. Die Ausgabe bleibt bei ungefähr 1,6 Tokens/s.
Eine unverständliche Antwort lief bis zum Limit von 256 Tokens. Ihre Ursache ist
offen; Sampling, Expert-Ausführung und Cache-Zustände müssen getrennt untersucht
werden. Ein größerer Cache allein beantwortet diese Qualitätsfrage nicht.

**Ziele und Messregeln**

| Bereich | Entwicklungsziel | Nachweis |
| --- | --- | --- |
| Decode | Zunächst mindestens 2× gegenüber einer erneut gemessenen Baseline; bei bestätigten 1,6 Tokens/s also mindestens 3,2 Tokens/s. Danach weiter profilieren. | Gleiche Hardware, Modellversion, Eingaben, Sampling und Speichergrenze; Median mehrerer repräsentativer Läufe. Das ist ein Ziel, keine zugesicherte Beschleunigung. |
| Reaktion | Warmer kurzer Folgeturn möglichst unter 5 Sekunden bis zum ersten sichtbaren Text. | Clientmessung einschließlich Queue und Routing; lange Prompts separat ausweisen. |
| Qualität | Die beobachtete leere/degenerierte Ausgabe erklären und die reproduzierbaren Fehler beheben; keine Verschlechterung durch Optimierungen. | Numerische Referenztests und ein festes Aufgabenpaket bei deterministischem und vorgesehenem produktivem Sampling. |
| Speicher | Expert-, KV-, Staging- und Allocator-Speicher bleiben zusammen innerhalb eines gemessenen Budgets mit Reserve für Betriebssystem und UI. | Spitzenwerte, Speicherdruck und zusätzliche Swap-outs während längerer Läufe erfassen. |
| Integration | CLI, Serve, WebUI und werkStation verwenden nachvollziehbar dieselben effektiven Einstellungen. | Gleicher Request-Inhalt und kontrollierte Unterschiede bei Streaming, Tools und Request-Overrides. |

Zuerst kurze Tests und Microbenchmarks ausführen. Nur erfolgreiche Kandidaten
erhalten längere Modellläufe. Für erste Vergleiche mindestens fünf Messläufe nach
separatem Warmup; belastbare Aussagen über hohe Perzentile benötigen anschließend
eine größere Stichprobe. Anzahl und Streuung immer mit ausgeben. Kürzere Antworten
oder repetitive Ausgaben zählen nicht als Nachweis einer verbesserten Decode-Leistung.

**1. Messbarkeit herstellen und die Qualitätsursache eingrenzen**

Das vorhandene `werk bench` erweitern, statt ein zweites unabhängiges Benchmark-
System aufzubauen. Zusätzlich zum direkten Backend einen HTTP-Mehrturn-Testpfad
bereitstellen, der dieselben strukturierten Nachrichten wie WebUI versendet.
Gerenderte Templates nicht erneut als User-Nachricht templatisieren.

Pro Lauf Modell-/Tokenizer-/Runtime-Identität und effektive Parameter erfassen:
Thinking, Temperatur, Top-p, Seed, Stop/EOS, Tokenlimit und Tools. Geheimnisse und
Gesprächsinhalte gehören nicht in normale Diagnose-Logs. Opt-in-Testartefakte
enthalten nur die dafür ausgewählten Testnachrichten.

Queue, Modellstart, Routing/Probe, Prefill, erstes generiertes Token, erster
sichtbarer Text, Decode und HTTP-Gesamtzeit unterscheiden. Fehlende Messungen
bleiben ausdrücklich unbekannt. Expert-Treffer, Fehlzugriffe, logische Lesebytes,
Lese-/Materialisierungszeit, Synchronisationen und Speicher-Spitzen ergänzen.
Die Backend-Verbindung fragt den eigenen Worker ab; Prozessargumente sind keine
verlässliche Schnittstelle für seine Telemetrie.

Expert-Zähler nach Phase auswerten: Ein Cache-Hit zählt derzeit eine
Expert-Akquisition pro Layer-Aufruf, nicht jede einzelne Token-Route. `pread`-
Bytes können aus dem Dateicache stammen und sind keine gemessenen physischen
SSD-Bytes. Globale Zählerdifferenzen sind bei parallelen Requests nicht eindeutig
einem Request zuzuordnen; dann Batch-/Worker-Metriken kennzeichnen oder echte
Request-Zuordnung implementieren. Instrumentierung darf selbst keine zusätzlichen
GPU-Synchronisationen in den normalen Decode-Pfad einführen.

Die Fehler zunächst mit frischem Worker und der ursprünglichen Nachrichtenfolge
reproduzieren. Cache-Erweiterung an/aus sowie Temperatur 0 und die bisherigen
Produktparameter vergleichen, dabei jeweils nur eine Variable ändern. Ebenso
Chat-Template, wirksames Thinking, EOS und die Antwort mit nur einem Token prüfen.
Temperatur 0 ist eine Diagnoseeinstellung, kein universeller Qualitätsfix.

Die vorhandenen MLX-Referenztests um mehrschrittiges Decode mit KV-Zustand,
Präfix-Restore, wechselnde Chats, Evictions, gemischte Quantisierung und gewichtete
Expert-Ausgaben erweitern. Numerische Abweichungen mit Teacher-Forcing und
Logit-/Layer-Vergleichen untersuchen; freie Textausgabe allein lokalisiert keinen
Rechenfehler. Eine vollständig residente Kopie des großen Modells wird nur
verwendet, wenn sie tatsächlich ins Speicherbudget passt. Sonst kleine residente
Referenzmodelle und ausgewählte reale Expert-/Layer-Berechnungen vergleichen.

Das Aufgabenpaket beginnt mit dem konkreten Rust-Fehler, kurzen Antworten auf
Deutsch und Englisch, JSON, kleinen ausführbar prüfbaren Code-Aufgaben und
Gesprächsfortsetzungen. Wiederholungsschleifen, leere Antworten, falsche Sprache,
vorzeitiges EOS und unerwartetes Erreichen des Tokenlimits separat zählen;
manuelle Qualitätsbewertung ergänzt automatische Prüfungen. Falls der konkrete
Fehler zunächst nicht wiederkehrt, bleibt er als nicht reproduziert offen.

Ergebnis dieses Schritts: reproduzierbare Vergleichsberichte, eine eingegrenzte
Qualitätsursache und ein Profil, das den nächsten Performance-Schritt bestimmt.

**2. Speicherbudget und günstige Laufzeiteinstellungen ausmessen**

Realen RAM, GPU-/Metal-Budget, laufende Prozesse und Speicherdruck erfassen.
Ausgehend von 8 GiB weitere Expert-Budgets testen, beispielsweise 12 und 16 GiB,
aber nur innerhalb der zuvor bestimmten Reserve. Jeden Test-Worker vollständig
beenden, bevor der nächste große Worker startet. Den gemessenen Gewinn gegen
Fehlzugriffe, Spitzenverbrauch und Swap-outs stellen und bei fehlendem Nutzen
oder Speicherdruck nicht weiter erhöhen.

Kontextlimit und Prefill-Chunk-Größe auf den Chat-Einsatz abstimmen und kurze
Requests ebenso wie längere Eingaben messen. Warmes Modell mit neuem Kontext,
echte Präfix-Wiederverwendung und vollständigen Kaltstart getrennt behandeln.
Auch verschiedene Themen testen, damit wiederholte Rust-Sätze keine unrealistisch
günstige Expert-Lokalität vortäuschen.

Einen Parametervergleich für das gewünschte Qualitätsprofil durchführen.
Modell-/Runtime-Versionen für die Baseline festhalten; Änderungen an Runtime oder
Kernels einzeln vergleichen. Fehlende DeepSeek-Indexer-Kernels sind laut lokalem
Log insbesondere für langen Prefill relevant und kein belegter Hauptgrund der
heutigen kurzen Decode-Leistung.

Ergebnis: das schnellste nachgewiesen stabile Profil für die vorhandene Hardware
und den vorhandenen Checkpoint. Dieses Profil ist die Referenz für Codeänderungen.

**3. Synchronisation und Speicherallokation im Expert-Pfad reduzieren**

Der aktuelle Pfad materialisiert jedes der neun Arrays eines nachgeladenen
Experten einzeln, synchronisiert nach jeder Expert-Berechnung und leert bei jeder
Verdrängung den MLX-Allokationscache. Diese Grenzen zuerst einzeln vermessen.

Neun Arrays gemeinsam materialisieren und deren CPU-Puffer bis zur sicheren
Übernahme halten. Wiederverwendbare Puffer und gebündelte Evictions prüfen.
Allocator-Freigaben an ein tatsächliches Speicherbudget koppeln, statt bei jedem
einzelnen Cache-Miss wieder sämtliche wiederverwendbaren Allokationen zu verwerfen.

Für gleichzeitig genutzte Experten eine begrenzte Gruppe mit klarer Besitzdauer
einführen. Gewichte dürfen erst verdrängt werden, wenn ihre GPU-Arbeit beendet
ist. Erst damit Eval-Grenzen von jedem Experten auf eine Gruppe verschieben.
Globale Sperren oder Synchronisationen nicht ersatzlos entfernen. Minimale
Cachegrößen, gepinnte Experten, Fehler und Abbrüche behalten einen korrekten,
begrenzten Ausführungspfad.

Ergebnis: weniger Synchronisationen und Allokationen pro Token bei geprüfter
numerischer Übereinstimmung und unverändert eingehaltenem Gesamtbudget.

**4. Mehrere Experten gemeinsam rechnen und Lesen überlappen**

Einen eigenen Decode-Pfad für einzelne Tokens prüfen. Kompatible aktive Experten
in wiederverwendbaren Slots halten und ihre Projektionen gebündelt ausführen.
MLX bietet dafür quantisierte Matrix-Gathers über Batch-Dimensionen an;
`gather_qmm` ist ein konkreter Kandidat, dessen Nutzen im installierten Runtime-
und Quantisierungsformat gemessen werden muss.
[MLX-Dokumentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)

Geometrie, Bits, Group-Size und Dtype pro Projektion beachten. Gemischte Gate-/Up-
Quantisierung verhindert pauschales Zusammenlegen. Das erneute Kopieren und
Stapeln aller ausgewählten Gewichte bei jedem Token kann den Gewinn aufheben;
deshalb zunächst einen Microbenchmark und dann einen begrenzten Slot-Pool bauen.
Kleine Budgets behalten den seriellen Fallback. GPU-Routing beibehalten, soweit
es zur CPU-gesteuerten Nachfrage nach fehlenden Gewichten passt.

Aufgezeichnete Expert-Zugriffe offline gegen globales LRU, geschützte häufige
Experten und Layer-Quoten vergleichen. Nur nachgewiesen bessere Strategien
integrieren. Bekannte benötigte Experten innerhalb eines Layers über eine
begrenzte Lese-Pipeline vorladen und Lesen mit laufender GPU-Arbeit überlappen.
Experten des nächsten Layers nicht als bereits bekannt voraussetzen.

Asynchrone Auswertung ist technisch verfügbar, ersetzt aber keine sichere
Gewichtslebensdauer, Speicherbegrenzung und Synchronisation an Besitzwechseln.
[MLX-Dokumentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html)

Ergebnis: höhere Decode-Leistung auf mehreren Aufgaben und weniger Wartezeit
durch Expert-Nachladen. Jede Änderung erhält eine getrennte Vergleichsmessung.

**5. Server-Latenz und werkStation-Tauglichkeit verbessern**

Gewichtsfreie Modell-/Runtime-Probes und Routingzeit separat messen. Wiederholte
teure Prüfungen bei unveränderter Identität vermeiden; Invalidierung an Modell,
Tokenizer, Launcher, Runtime und relevante Konfiguration binden. Keine veraltete
Kompatibilitätsentscheidung nach Modell- oder Runtime-Wechsel übernehmen.

Den Listen-Port vor dem großen Modellstart reservieren, damit ein belegter Port
sofort auffällt. Gleichzeitige Anfragen dürfen keine doppelten Modell-Worker
erzeugen. Abbruch muss Generierung, Vorladen und wartende Arbeit beenden.

Interaktive Chats gegenüber Hintergrundaufgaben wie Titel und Tags priorisieren.
Ausstehende Arbeit begrenzen und Queuezeit sichtbar machen. Ein kleineres bereits
vorhandenes Modell kann später Hintergrundaufgaben übernehmen, sofern dafür
Speicher verfügbar ist. Backend-Metriken einheitlich für CLI, HTTP und werkStation
bereitstellen; neue Felder als optional und vorhandene Clients kompatibel halten.

Nach erfolgreichen Einzelnutzer-Tests zwei konkurrierende Chats, längere Verläufe,
Bearbeitung/Verzweigung älterer Nachrichten, Streaming/JSON und Tools prüfen.
Zunächst kurze Kontexte; danach 2K/8K und nur bei ausreichendem Budget längere.

**6. Verbesserungen auf andere Modelle übertragen**

Benchmark, Qualitätsaufgaben, Telemetrie, Speicherplanung, Worker-Lebensdauer und
Server-Scheduling sind gemeinsam nutzbar. Effektive Sampling-/Template-/Thinking-
Einstellungen als versionierte Modell-/Runtime-Profile behandeln; explizite
Nutzerwerte haben Vorrang.

Die Expert-Ausführung bleibt ein geprüfter Adapter: Der heutige Loader erlaubt
ausdrücklich nur DeepSeek V4. Weitere MoE-Architekturen benötigen eigene Prüfungen
für Routing, Aktivierung, Quantisierung und KV-Zustände. Dichte Modelle nutzen
ihren passenden nativen Pfad. Als nächste Kontrollfälle das bereits vorhandene
Gemma und anschließend ein vorhandenes kompatibles MoE-Modell verwenden.

Falls die bestehende Quantisierung nach behobenen Runtime-Fehlern das gewünschte
Qualitätsniveau nicht erreicht, alternative Quantisierungen beziehungsweise
Modelle anhand desselben Aufgabenpakets und der Hardwaregrenzen vergleichen.
Zuerst den aktuellen Checkpoint optimieren; zusätzliche große Downloads gehören
nicht zur anfänglichen Messung.

Speculative Decoding/MTP erst nach stabilem normalen Decode bewerten. Der aktuelle
Expert-Loader lehnt eingebettete Drafter ab; Unterstützung erfordert eigene
Arbeit an Verifikation, Cache-Rollback und Speicherbudget. Nur bei gemessenem
Gesamtgewinn einschließlich zusätzlicher Expert-Zugriffe weiterverfolgen.

**Reihenfolge der Umsetzung und Abschluss**

Messbarkeit und Qualitätsreproduktion beginnen gemeinsam. Danach das stabile
Hardwareprofil bestimmen; Qualitätstests müssen vor Änderungen am Rechenpfad
stehen. Synchronisation/Allokation vor gebündelten Kernels und asynchronem
Vorladen bearbeiten. Die nächste größere Optimierung jeweils aus dem neuen
Profil auswählen. Server-Latenz kann unabhängig davon bearbeitet werden.

Wesentliche bestehende Einstiegspunkte sind `src/cli.rs` (`werk bench`),
`src/backend/omlx.rs` (Worker und Telemetrie), `src/backend/omlx_experts.py`
(Lesen/Cache/Expert-Rechnung), `src/backend/omlx_persistence.py`, deren native
Python-Tests sowie `src/api/chat.rs` und `src/api/state.rs`.

Der erste Abschlussbericht enthält vor/nachher gemessene Decode- und
Reaktionszeiten, Qualitätsfehler und deren Reproduktionsstatus, Speicherspitzen,
das empfohlene Profil und ausdrücklich verbleibende Grenzen. Kleine Gewinne
nicht zu einer ungemessenen Gesamtbeschleunigung multiplizieren. Neue Defaults
erst nach den Vergleichstests übernehmen; währenddessen Kandidaten getrennt
und reversibel halten.
