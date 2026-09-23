**Plan: Anthropic-kompatible Messages-API für Werk**

Stand: 2026-09-23. Status: Messages-Adapter implementiert; Rust-Vertragstests und
offizieller Python-SDK-Test vorhanden. Reale Qwen-/GLM-Abnahme und
Performance-Vergleiche erfolgen durch den Nutzer in seiner Konsole.
Bedienung, Grenzen und Testbefehle: [Anthropic-Clients](integrations/anthropic-clients.md).

Implementierungsentscheidung: Text streamt sofort. Tool-Argumente streamen,
sobald der Name im übergebenen Katalog eindeutig ist. Mehrdeutige Namen und
interleavte Aufrufe werden begrenzt gepuffert; eigene stabile Antwort-IDs
entkoppeln die Ausgabe von fragmentierten Backend-IDs. Der Client gibt diese
Antwort-IDs im folgenden Tool-Ergebnis zurück.

Ziel ist ein zusätzlicher `POST /v1/messages`-Endpunkt auf demselben
`werk serve`-Listener. Text, Streaming und vollständige Zyklen mit vom Client
ausgeführten Tools gehören gemeinsam zur ersten freigegebenen Version.
Der vorhandene OpenAI-Endpunkt bleibt verfügbar. Beide Formate verwenden
dieselbe Modellauflösung, Backend-Auswahl, Worker-Verwaltung und native Caches.
Die erste Abnahme richtet sich an direkte HTTP-Clients und das offizielle
Anthropic-Python-SDK mit lokalen Werk-Modell-IDs. Claude-Code-Kompatibilität
wird gesondert geprüft und erst nach einem echten Integrationstest zugesagt.

**1. Gemeinsame Verarbeitung und Bestandsschutz**

Aus `src/api/chat.rs` die gemeinsame Request-Vorbereitung und die Auswahl
zwischen Backend und bestehender Chat-Session in ein kleines internes
Servicemodul herauslösen. Eingaben/Ergebnisse bleiben die vorhandenen
`GenerateRequest`, `GenerateResponse` und `GenerateStreamEvent`.
HTTP-Deserialisierung, Fehlerdarstellung und Ausgabeformat bleiben je Protokoll
getrennt. Kein HTTP-Aufruf vom neuen Endpunkt zum eigenen OpenAI-Endpunkt.

Die bestehenden OpenAI-Tests zuerst als Referenz verwenden. Nach dem Refactor
müssen gleiche OpenAI-Anfragen dieselben normalisierten Backend-Aufträge,
Tool-Konfigurationen und Entscheidungen zur Wiederverwendung erzeugen.
Tool-Anfragen behalten die bestehende Backend-Route; sie dürfen nicht
versehentlich auf eine Chat-Session ohne Tool-Unterstützung wechseln.

Der Anthropic-Eingang erhält eine explizite Kontextprüfung. Zu große Anfragen
werden abgewiesen; seine Tool-Zyklen dürfen nicht durch das vorhandene
paarweise Kürzen gewöhnlicher Chat-Nachrichten beschädigt werden. Die Prüfung
berücksichtigt auch Tool-Schemas, Argumente und Ergebnisse. Eine Schätzung
für die Zulassung ist keine exakte Tokenzähl-API.

**2. Vertrag und Funktionsumfang festlegen**

| Bereich | Erste Version |
| --- | --- |
| Endpoint | `POST /v1/messages`, JSON und `stream: true` |
| Pflichtfelder | Werk-Modell-ID, `messages`, positives `max_tokens` |
| Nachrichten | `user`/`assistant`, Text als String oder Textblöcke, separates `system` |
| Sampling | `temperature` und `top_p` mit Typ-/Bereichsprüfung |
| Tools | Benutzerdefinierte Tools mit `name`, `description`, `input_schema`; Tool-Historie und Rückgaben |
| Auswahl | `auto`, `none`, `any` und benanntes Tool werden abgebildet; tatsächliche Unterstützung wird am Backend geprüft |
| Authentifizierung | Vorhandene API-Schlüsselprüfung für `x-api-key` und Bearer wiederverwenden; Fehler im Anthropic-Format |
| Version/Header | Unterstützte `anthropic-version` festlegen, validieren und dokumentieren; CORS-Allowlist ergänzen |
| Fehler | Anthropic-Fehlerhülle, passende HTTP-Statuscodes, Request-ID; auch für ungültiges JSON, Auth und Body-Limits |

Für Beta-Header und optionale Felder eine ausdrückliche Kompatibilitätsmatrix
pflegen: unterstützen oder mit verständlicher Fehlermeldung ablehnen.
Insbesondere keine stillschweigenden Zusagen für `strict`, Thinking,
`cache_control`, Server-Tools, Bilder oder Dokumente. Diese Funktionen sowie
Batch-/Files-APIs gehören zunächst nicht zur ersten Version.

Nichtleere benutzerdefinierte `stop_sequences` zunächst ebenfalls klar
ablehnen: Der aktuelle Backendvertrag liefert nicht zuverlässig, welche
Sequenz den Abbruch ausgelöst hat. Unterstützung folgt mit belastbarer
Stop-Metadatenübertragung; die Antwort darf keine Sequenz erfinden.
Leere/fehlende Stopsequenzen funktionieren mit den normalen Modell-Stopps.

**3. Vollständiges Tool-Calling**

Tool-Zyklen folgen dem
[Anthropic-Vertrag für Tool-Aufrufe und Ergebnisse](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).
Werk stellt Aufrufe und Ergebnisse zu; das aufrufende Programm führt die
angebotenen Tools aus.

| Anthropic | Bestehende interne Darstellung |
| --- | --- |
| `tools[].input_schema` | `function.parameters` |
| `tool_use` | Assistant-Tool-Aufruf mit stabiler ID, Name und JSON-Argumenten |
| `tool_result.tool_use_id` | Tool-Nachricht mit passender `tool_call_id` |
| `tool_choice: auto/none` | `auto/none` |
| `tool_choice: any` | `required`, nur bei unterstützendem Backend |
| `tool_choice: tool` | Benannter Funktionsaufruf, nur bei unterstützendem Backend |

Tool-IDs, JSON-Typen und Zuordnung über mehrere Gesprächsrunden erhalten.
Mehrere Tool-Aufrufe und zugehörige Ergebnisse müssen verarbeitet werden.
`is_error` wird als ausdrücklich markiertes fehlgeschlagenes Tool-Ergebnis
an das Modell weitergegeben; die konkrete Textdarstellung wird dokumentiert
und getestet. Fehlende, doppelte oder verwaiste Tool-IDs werden abgewiesen.
Text nach Tool-Ergebnissen innerhalb derselben User-Nachricht bleibt erhalten.

Die vorhandene interne Assistant-Darstellung trennt Text von Tool-Aufrufen.
Für frei verschachtelte Text-/Tool-Blöcke darf dadurch keine unbemerkte
Umordnung entstehen: Die erste Version unterstützt die übliche Reihenfolge
Text gefolgt von Tool-Aufrufen; andere noch nicht verlustfrei darstellbare
Reihenfolgen werden ausdrücklich abgewiesen.

Aktuelle oMLX-Grenzen berücksichtigen: `auto` und `none` sind unterstützt;
erzwungene/benannte Auswahl, explizite Parallelitätssteuerung und
`strict: true` werden derzeit abgelehnt. Anthropic-Optionen wie
`disable_parallel_tool_use` dürfen nicht ungeprüft als ausdrücklicher Wert
an oMLX weitergereicht werden: Weglassen erhält dessen Standardverhalten;
eine geforderte Einschränkung muss unterstützt sein oder abgewiesen werden.
Mehrere zurückgelieferte Aufrufe korrekt zu transportieren ist unabhängig
von der Möglichkeit, ihre Anzahl am Modell zu erzwingen.

Qwen und GLM werden zuerst mit `tool_choice: auto` getestet. Der Endpunkt
meldet Backendgrenzen verständlich, statt Funktionen vorzutäuschen oder
stillschweigend das Backend zu wechseln.

**4. Antworten und Streaming**

Nichtstreamende Antworten enthalten Anthropic-Message-ID, Modell,
Assistant-Rolle, Content-Blöcke, Stop-Grund und Usage. Normales Ende wird
`end_turn`, Tokenlimit wird `max_tokens`, abgeschlossener Tool-Aufruf wird
`tool_use`. Unbekannte Backend-Stop-Gründe nicht pauschal als Erfolg ausgeben.
Tool-Argumente müssen für vollständige `tool_use`-Blöcke gültige JSON-Objekte
sein; ungültige Ausgaben nicht still reparieren.

Für Streaming einen eigenen Zustandsautomaten entsprechend den
[Anthropic-SSE-Ereignissen](https://platform.claude.com/docs/en/build-with-claude/streaming)
einsetzen: `message_start`, Content-Block-Start/Delta/Stop, `message_delta`,
`message_stop`. Er verarbeitet Text-Deltas und `input_json_delta`, stabile
Blockindizes, fragmentierte Tool-IDs/Namen und mehrere Tool-Aufrufe.
Puffer nach Anzahl und Bytes begrenzen; gewöhnliche Textausgabe sofort
weitergeben. Interleavte Backend-Tool-Deltas korrekt den Blöcken zuordnen.

Bei Tokenlimit mitten in Tool-Argumenten das Ende als `max_tokens` darstellen,
ohne einen vollständigen ausführbaren Tool-Aufruf vorzutäuschen. Für den
nichtstreamenden Fall ungültiger Tool-JSON-Ausgabe einen definierten Fehler
verwenden. Fehler, Client-Abbruch und Backpressure müssen sauber propagieren;
auf einen fehlerhaften oder abgebrochenen Stream folgt kein Erfolgsschluss.

Echte Tokenzahlen sind im aktuellen internen Stream erst bei `Done` verfügbar.
Startwerte daher als vorläufig dokumentieren und die Endwerte im
`message_delta` liefern. Das offizielle SDK unterstützt dort auch
[aktualisierte Input-Tokenzahlen](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/types/message_delta_usage.py).
Die SDK-Akkumulation wird ausdrücklich getestet. Native Prompt-Cache-Treffer
nicht ungeprüft in Anthropic-Abrechnungs- oder Cache-TTL-Zusagen umdeuten;
zunächst tatsächliche Gesamt-Input-/Output-Zahlen berichten und keine
zusätzlichen Cache-Abrechnungsfelder behaupten.

**5. Abnahme und Performance**

Zuerst schnelle deterministische Tests mit vorhandenen Mock-Backends:
Request-Abbildung, Systemtexte, Tool-Schemas und -IDs, mehrere Aufrufe,
Tool-Fehler, ungültige Historien, Backendgrenzen und Kontextüberschreitung.
Für Streams vollständige Eventfolgen, fragmentiertes JSON, Unicode,
Tokenlimit, Fehler, Client-Abbruch und Puffergrenzen prüfen.

Danach Integration mit einer festgehaltenen Version des offiziellen
Anthropic-Python-SDK: `messages.create`, `messages.stream` und ein kompletter
Tool-Zyklus vom ersten Aufruf über das Client-Ergebnis bis zur finalen Antwort.
JSON- und Streaming-Varianten müssen dieselbe Semantik liefern.

Für Qwen und GLM jeweils reale Text- und Tool-Zyklen vorsehen, einschließlich
eines größeren Tool-Katalogs und mehrerer aufeinanderfolgender Runden.
Die bestehenden Backend-Capability-Prüfungen bleiben maßgeblich. CPU-Tests
decken zusätzlich die Ablehnung nicht unterstützter Modell-/Backendpaare ab.

Performance in zwei getrennten Vergleichen prüfen:

- Alter versus neuer Build über denselben OpenAI-Endpunkt: keine
  reproduzierbare Verschlechterung durch den gemeinsamen Refactor.
- OpenAI versus Anthropic im neuen Build: semantisch gleiche Requests,
  gleiche normalisierte Backend-Aufträge und gleiche Modell-/Cachebudgets.

ABBA-Reihenfolge, erste und warme Requests, TTFT, Decode-Rate, gesamte
Requestzeit, Cache-Treffer und Worker-Anzahl dokumentieren. Startwerte aus
dem bisherigen 96-Token-/22-GiB-Test dienen nur zur Orientierung; entscheidend
ist der direkte Vergleich unter gleichen Bedingungen. Bei reproduzierbarer
Regression API-Verarbeitung korrigieren oder Refactor enger isolieren.
Decoder, Quantisierung, Cache-Budgets und CUDA/llama.cpp-Inferenz bleiben
außerhalb dieses API-Vorhabens. Große Modelltests werden für den Nutzer
reproduzierbar bereitgestellt; deren Ausführung erfolgt gemäß der zuletzt
vereinbarten Arbeitsweise durch den Nutzer in seiner Konsole.

**6. Dateien, Reihenfolge und Fertigkriterien**

Vorgesehene neue Module: `src/anthropic.rs` für Protokolltypen und
`src/api/anthropic/{mod,request,response,stream}.rs` für den Adapter.
Ein kleines gemeinsames Generierungsmodul unter `src/api/` trägt die
wiederverwendete Vorbereitung/Ausführung. `src/api/router.rs`, Auth-Fehlerpfad
und CORS werden gezielt ergänzt; `src/openai.rs` wird als vorhandene interne
Darstellung weiterverwendet. Neue API-Tests und ein SDK-Testskript ergänzen
`docs/api.md` sowie eine Anleitung für Anthropic-Clients mit Curl-Beispielen.

Implementierung in separat prüfbaren Schritten: zuerst gemeinsamer Kern mit
OpenAI-Parität, danach nichtstreamende Messages samt vollständigem Tool-Zyklus,
anschließend Streaming und Fehlerfälle, zuletzt SDK-Abnahme, Modelltests und
Dokumentation. Tool-Calling bleibt Freigabekriterium der ersten Version.

Fertig ist diese Version, wenn der SDK-Tool-Zyklus mit Qwen und GLM in beiden
Ausgabemodi funktioniert, die vorhandenen OpenAI-Tests bestehen, beide APIs
denselben geladenen Worker wiederverwenden, Abbrüche keine Request-Ressourcen
zurücklassen und keine reproduzierbare Performance-Regression vorliegt.

Erweiterungsstand 2026-09-23: `/v1/messages/count_tokens` mit nativer
Backend-Tokenisierung, Bildblöcke und Live-Tool-Argumente sind implementiert.
Die Backend-Grenzen und Tests stehen in `integrations/anthropic-clients.md`.
Thinking-Signaturen und Anthropic-Cache-Steuerung werden nicht durch lokale
Werk-Parameter nachgebildet. Weitere Anbieterfunktionen bleiben separat. Keine vollständige Anthropic- oder Claude-Code-Parität
allein aufgrund des neuen Messages-Endpunkts behaupten.
