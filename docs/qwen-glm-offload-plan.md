# DeepSeek V4 Flash, Qwen3.8-Flash-Next und GLM-5.3-Flash: Auslagerung und Performance

Stand: 2026-09-13. **Umsetzung und laufende Abnahme auf dem vorhandenen Mac
mit 48 GB Unified Memory.**

## Aktueller Implementierungsstand

- Private native Textloader für Qwen `qwen4_exp` und GLM `glm5_next` sind in
  Probe, Worker, Speicherzulassung und Expertensteuerung angebunden.
- Experten und Qwen-PLE-Zeilen werden separat begrenzt; gemeinsame native
  Speicherzulassung verhindert, dass beide Caches denselben freien RAM verplanen.
- API, CLI-Umgebung, ComfyUI und n8n unterstützen zusätzlich
  `ngram_cache_mb` / `WERK_OMLX_NGRAM_CACHE_MB`. `0` hält Tabellen resident,
  positive Werte begrenzen den Zeilencache; `auto` startet klein und wächst bei
  tatsächlicher Verdrängung bis zur gemeinsamen Speichergrenze. Unveränderte
  Node-Workflows erben weiterhin die Servereinstellung.
- Vor der nativen adaptiven Prefill-Chunkwahl werden verdrängbare Gewichte
  freigegeben. So erzwingt ein zuvor gefüllter Expertencache keine unnötig
  kleinen Chunks. Die Textadapter reservieren zusätzlich eine native
  Transientprognose, mindestens zwei FP32-Vokabularpuffer, für Spitzen innerhalb
  eines Chunks. Verzögerte physische Freigaben zuvor verdrängter Gewichte
  werden nicht als zukünftige Transientreserve angelernt; gleichzeitiges
  reales MLX-Wachstum bleibt sichtbar. Die nativen Speicherprüfungen bleiben aktiv.
- Native SSD-Präfixe erhalten Qwen Arrays/QSA- sowie GLM-Hybridzustände.
  Qwen/GLM verwenden dafür vollständige native Snapshots statt GDN-Sidecars,
  da oMLX 0.6.4 damit kurze exakte Präfixe noch nicht atomar sichern kann.
- Kleine native Modelle: Prefill und zehn Folgetokens gegen residente Modelle;
  Qwen mit jeder unabhängigen Kombination von Experten-/N-Gramm-Auslagerung;
  erzwungene Cacheverdrängung; nach SSD-Neuöffnung exakt gleiche Fortsetzung.
  Echte private oMLX-Worker liefern vollständige HTTP-Streams und speichern Präfixe.
- ComfyUI: 239 Tests bestanden. n8n: 99 Tests, Build und Lint bestanden.
  MLX-Suite: 171 Tests, 4 optionale Tests ausgelassen; bestanden nach Isolation
  der synthetischen Probe-Runtime von zuvor importierten nativen Architekturen.
- Qwen: elf echte CLI-Chatturns mit 8 GiB Experten-/1 GiB N-Gramm-Cache
  erfolgreich; alle elf kurzen Antworten korrekt, letzte Erinnerungsantwort
  `37`. Im letzten Turn wurden 289 von 317 Prompttokens wiederverwendet.
  Serielle Referenz: warmer Median 18,217 s pro Turn; erster Lauf 16,182 s.
  Das Aufgabenpaket umfasst kurze Zahlen-/Sprachaufgaben, keine allgemeine
  Qualitätsbewertung.
- Gebündelte Expertenauswertung: ebenfalls elf korrekte Antworten im CLI und
  über Serve, warme Mediane 13,000 / 12,654 s (2,7 % Unterschied der gesamten
  Turnzeit; kein isolierter Decode-Vergleich). Auto mit effektiv 22.400 GB
  Expertenbudget: elf korrekte Antworten, warmer CLI-Median 4,173 s.
  Unterschiedliche Budgets und unkontrollierter OS-Dateicache erlauben daraus
  keine pauschale Zusage für andere Geräte oder längere Antworten.
- Reeller Qwen-Prozessneustart mit unveränderter Konfiguration: korrekte
  Erinnerungsantwort `37`, 356 von 389 Prompttokens aus persistentem Präfix.
- Der später regulär installierte GLM-Checkpoint wurde tatsächlich ausgeführt.
  Zwei Loaderfehler (Vision-Attribut und affine Forget-Gate-Namensräume) wurden
  korrigiert; 140 native Tests bestanden. Elf CLI-Antworten waren korrekt.
  Mit festem 16-GiB-Expertenbudget bestehen außerdem Gesprächswiederherstellung
  und drei vollständige HTTP-Textstreams. Bei einem weiteren Prozessneustart
  wurden 238 von 265 Prompttokens aus dem nativen SSD-Präfix wiederverwendet.
  Höhere Auto-Budgets führten bei
  Neustart/Serve zu nativen Speicherabbrüchen und sind noch nicht zuverlässig.
  GLM-Tool-Calling wird abgewiesen; das GLM-Template ignoriert `enable_thinking`.
  Diese Textprüfung ist keine Freigabe für Vision/MTP oder die vollständige
  geplante Abnahmematrix. [Messbericht und reproduzierbare Aufrufe](benchmarks/2026-09-12-flash-offload/README.md).

### Checkpointinventar

| Checkpoint | Basisgewichte | Experten | N-Gramm-Tabellen | Ausgeschlossene Gewichte |
| --- | ---: | ---: | ---: | --- |
| PipeNetwork Qwen3.8-Flash-Next mixed 4/8 | 5.349 GB | 67.948 GB | 32.000 GB | Vision; MTP im Inventar nicht vorhanden |
| Vontra GLM-5.3-Flash oQ2-MTP | 9.567 GB | 95.127 GB | keine | Vision 1.127 GB, MTP 4.306 GB |

Dezimale GB, tatsächliche Tensorbytes inklusive Quantisierungsskalen/Biases.
Qwen: vollständige lokale Shards im Modellstore geprüft; 24.576 Experten,
größter Experte 2.765 MB. GLM: alle 22 Tensorheader und Index zur Revision
`5da47563173b3c0b5fb4065ab7676477a7d50c44` per begrenzten Range-Abfragen geprüft;
3.125 Tensoren, 12.096 Experten, größter Experte 7.864 MB. Die vorhandenen
DeepSeek-V4- und GLM-Indizes enthalten keine `ngram`-/`engram`-/PLE-Gewichte;
N-Gramm-Auslagerung ist dort nicht anwendbar. Fehlende Tabellen schalten MoE
nicht ab. Für weitere Checkpoints zählt ihr konkretes Layout.

Mit 8 GiB Experten- und 1 GiB N-Gramm-Budget beträgt Qwens statische
Zulassungsschätzung 16.086 GB einschließlich 1 GiB Workspace. GLM kommt bei
8 GiB Expertenbudget auf 19.231 GB. Das sind keine gemessenen Prozessspitzen;
KV-Zustände und temporäre Aufwände werden zusätzlich dynamisch berücksichtigt.

## Verbindlicher Umfang für alle drei Modelle

MoE-Expertenauslagerung **und** N-Gramm-/Embeddingtabellen-Auslagerung gehören
für **DeepSeek V4 Flash, Qwen Flash und GLM Flash** zum Implementierungs- und
Prüfumfang. Beide Funktionen werden je konkretem Checkpoint, Architekturadapter
und Runtime getrennt ermittelt. N-Gramm-Auslagerung ist nicht grundsätzlich
auf Qwen beschränkt; die bisher bekannte Qwen-PLE-Struktur ist der erste
konkrete Adapterfall.

Für jede der sechs Kombinationen aus Modell und Auslagerungsart festhalten:

- **Vorhanden und unterstützt:** im tatsächlichen Lade-/Inferenzpfad anbinden,
  Speicherbegrenzung und numerische Gleichwertigkeit prüfen und einen realen
  Mehrturn-Lauf nachweisen. Wenn beide Arten unterstützt sind, zusätzlich
  ihren gemeinsamen Betrieb prüfen.
- **Vorhanden, aber noch nicht unterstützt:** konkrete Layout-/Runtime-Lücke
  ausweisen. Eine vorhandene Hilfsfunktion oder ein bestandener isolierter Test
  gilt noch nicht als nutzbare Modellunterstützung.
- **Nicht vorhanden:** anhand der Architektur und des Tensorinventars belegen
  und als nicht anwendbar ausweisen. Fehlende N-Gramm-Tabellen verhindern keine
  unabhängig unterstützte MoE-Auslagerung.

CLI, Serve, ComfyUI und n8n verwenden dafür dieselben effektiven Einstellungen
und Fähigkeiten. Expert- und N-Gramm-Caches bleiben getrennt steuer- und
beobachtbar, berücksichtigen aber denselben verfügbaren Gesamtspeicher.
N-Gramm-Optionen benötigen einen eigenen dokumentierten API-/Node-Vertrag;
das bestehende Expertenbudget wird dafür nicht umgedeutet. Persistenz und
Wiederaufnahme müssen die tatsächlich verwendeten architekturspezifischen
Zustände erhalten. Die Abnahme umfasst alle drei Modelle, auch den bestehenden
DeepSeek-Pfad, und kennzeichnet noch offene Kombinationen ausdrücklich.

## Ziel und Reihenfolge

Dieselbe Iteration wie bei DeepSeek: korrekt laden → stabile erste Antwort →
korrekte Folgeturns → Engpass anhand von Verbose messen → einzeln optimieren →
CLI, Serve und Clients unter gleichen Bedingungen vergleichen.

1. Gemeinsame Metadatenprüfung und Messgrundlage für beide Modelle.
2. Qwen: Experten und N-Gramm-Embeddings gemeinsam auslagern.
3. Qwen: Zustands-/Präfixwiederverwendung, Qualität und Decode optimieren.
4. GLM: geprüfte Infrastruktur übernehmen, eigenen Architekturadapter ergänzen.
5. GLM optimieren; anschließend DeepSeek, alle Clients und Budgetgrenzen prüfen.

Ein Modell muss nicht vollständig in RAM passen. Vollständig im RAM liegen
muss aber das minimale aktive Arbeitsset: unvermeidbare Basisgewichte, laufende
Zustände, aktive Rechengruppe, Staging und Workspace. Dieses Minimum bestimmen
wir zuerst. Reicht es nicht, wird der verbleibende Speicherbedarf konkret
ausgewiesen und der nächste auslagerbare Teil untersucht. Kernel-Limits oder
Speicherprüfungen werden nicht als Ersatz dafür abgeschaltet.

## Kandidaten und gesicherter Ausgangspunkt

| Modell | Kandidat | Dateigröße laut Anbieter | Besonderheiten |
| --- | --- | ---: | --- |
| Qwen3.8-Flash-Next | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` | 106,2 GB | `qwen4_exp`; gemischte Quantisierung, große gehashte N-Gramm-Tabellen; dieser Build enthält einen eigenen `model_file`-Runtime-Port und keinen MTP-Kopf. |
| GLM-5.3-Flash | `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP` | 110,128 GB | `glm5_next`; gemischte Q2/Q4/Q8-Gewichte, hybride Attention, eingebetteter MTP-Kopf; erste Tests ohne MTP. |

Quellen: [Qwen-Modellkarte](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit),
[GLM-Modellkarte](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-oQ2-MTP).
Größen sind Checkpoint-/Dateigrößen, keine gemessenen RAM-Anforderungen.
Der aktuelle Headerabgleich steht oben; die folgenden Abschnitte beschreiben
den vollständigen Plan einschließlich der noch offenen Abnahmekriterien.

Ausgangspunkt vor dieser Erweiterung:

- [Werk-Expertenadapter](../src/backend/omlx_experts.py): ausschließlich native
  affine DeepSeek-V4-Checkpoints; fremde `model_file`-Loader und eingebettete
  spekulative Köpfe werden abgewiesen. Die Architekturprüfung einfach zu lockern
  wäre keine Implementierung für Qwen oder GLM.
- [Kompatibilitätsprobe](../src/backend/omlx_probe.py): Auto-Offload ist an den
  verifizierten DeepSeek-Adapter und oMLX 0.6.4 gebunden.
- Installiertes oMLX 0.6.4 enthält `mlx_vlm_qwen4_exp_compat`,
  `mlx_vlm_glm5_next_compat`, Qwen-PLE-SSD-Konfiguration und eine
  PLE-Residenzschätzung. Vorhandener Quellcode beweist noch nicht, dass Werks
  Worker diese Pfade für die gewählten Checkpoints korrekt nutzt.
- [Kurzpräfix-Persistenz](../src/backend/omlx_persistence.py) hat eine enge
  Cache-Typenliste. Qwens lokal vorhandene `ArraysCache`-/`QSAKVCache`-Zustände
  sind damit nicht automatisch kompatibel.
- Der [DeepSeek-Paritätsbericht](omlx-chat-serve-parity.md) liefert das Verfahren
  und Rohdaten. Auto erreichte dort in einzelnen Folgeturns 5,98/6,43 tok/s
  bei CLI/Serve. Das sind keine Erwartungen oder Zusagen für Qwen/GLM.

## 1. Checkpoint und Runtime vor dem großen Download abgleichen

Zuerst nur Modellkarte, Konfiguration, Tokenizer-/Template-Metadaten, Index und
notwendige Tensorheader beziehen. Modellrevision und Runtime-Dateien mit Hashes
festhalten; Dateistatistiken für Invalidierung ergänzen. Range-Abrufe begrenzen;
ein Server ohne Range-Unterstützung darf nicht unbemerkt einen ganzen Shard laden.

Das Inventar weist reale Bytes getrennt aus: Basis, Experten, N-Gramm-Tabellen,
Vision, MTP, Skalen/Biases sowie notwendiger Konvertierungs-/Staging-Speicher.
Pro Projektion tatsächliche Bitbreite, Gruppengröße, Achsen und Tensorlayout
lesen. Keine Ableitung der Speichergröße nur aus dem Modellnamen.

Qwen zuerst gegen den vorhandenen nativen oMLX-Port prüfen. Dessen Namensräume,
Normkonventionen und PLE-Layout müssen zum PipeNetwork-Build passen. Falls nicht:
gezielt geprüften privaten Loader oder eine nachweislich äquivalente
Layout-Anpassung entwickeln. Änderungen an Normgewichten dürfen nicht doppelt
angewendet werden. Kein pauschales Vertrauen in beliebigen Checkpoint-Code und
keine Veränderung der global installierten Runtime.

GLMs Vision-/MTP-Gewichte und gemischte Quantisierung explizit erfassen. MTP aus
heißt nicht automatisch, dass seine Gewichte nicht geladen werden; beides prüfen.
Texttests benötigen keine Bildinferenz. Eine optionale Vision-Initialisierung
darf später nur dann entfallen, wenn der Textpfad numerisch gleich bleibt.

**Abnahme:** eindeutiger Loaderpfad, dokumentierte Layout-/Versionsanforderungen,
bytebasierte Speicherbilanz und begründeter erster Budgetvorschlag. Erst dann
Checkpoint-Download über den bestehenden Werk-Modellstore, mit Fortsetzung und
Integritätsprüfung; keine zusätzliche BF16-Kopie als Voraussetzung.

## 2. Gemeinsame Speicherverwaltung ohne Verlust kleiner Budgets

Aus dem DeepSeek-Code wiederverwendbare Infrastruktur lösen: sichere
Bytebereichs-Leser, Fingerprints, quantisierte Tensoransichten, Cacheeinträge,
aktive Leases, Statistik und native Speicherzulassung. Architekturadapter
liefern Tensorzuordnung, Routing, Aktivierung und Zustandsverträge.

Ein übergeordneter Speicherverwalter berücksichtigt gemeinsam:

```
Basis + Experten-RAM + PLE-RAM + KV/rekurrente Zustände
      + Staging/in-flight I/O + Workspace + Allocator + Reserve
```

Es gibt keinen separaten Experten- und PLE-Auto-Regler, der jeweils den gesamten
freien Speicher verplant. CPU-/mmap-Seiten und Metal-Allokationen liegen auf
diesem Gerät im selben physischen Speicher; überlappende Messgrößen nicht doppelt
addieren. Logische mmap-Größe ist keine gemessene Residenz.

Auto verwendet gemessenen verfügbaren Systemspeicher, Metal-Grenze und
Checkpointgröße. Kleine explizite Expertenbudgets bleiben Obergrenzen; `0`
behält seine bisherige Bedeutung. Zusätzliche PLE-/Gesamtbudget-Optionen sind
vor Implementierung als eigener Vertrag festzulegen, nicht durch Umdeuten von
`WERK_OMLX_EXPERT_CACHE_MB`. Auf großen Maschinen kann Auto bis zur sinnvollen
Vollresidenz wachsen; keine feste 24-GiB- oder TiB-Pauschalgrenze.

Vor Prefill und an Chunk-Grenzen verdrängbare Caches verkleinern. KV-/Attention-
Bedarf, Cachewachstum und echte temporäre Spitzen getrennt verbuchen, damit der
DeepSeek-Fehler mit fälschlich angelerntem Prefill-Verbrauch nicht wiederkehrt.
Aktiv genutzte Gewichte/Embeddingzeilen erst nach abgeschlossener GPU-Auswertung
freigeben. Spätere Vergrößerung mit Reserve und Hysterese gegen ständiges Wechseln.

**Abnahme:** Tiny-Modelle laufen mit erzwungenen Evictions korrekt; Pins, Abbruch,
Speicherdruck und fehlende Dateien werden sauber behandelt. Simulierte kleine
und TiB-Systeme prüfen Budgetarithmetik; reale 48-GB-Messungen bleiben getrennt.

## 3. Qwen: Experten und N-Gramm-/PLE-Auslagerung

Die hier als „Engram“ bezeichneten Daten sind Qwens N-Gramm-Embeddingtabellen.
Ihre Indizes entstehen aus Tokenhistorie und Hashing; sie sind keine MoE-Routen.
Der [Qwen-Referenzport](https://github.com/PipeNetwork/qwen38-flash-next-mlx)
beschreibt unter anderem Hash-/Segmentgrenzen, Normkonventionen und kausale
Sparse-Attention als empfindliche Korrektheitspunkte.

**Erste lauffähige Stufe:** vorhandenen PLE-SSD-Pfad auf Header-/Layoutkompatibilität,
Startspitzen, quantisierte Zeilenzugriffe und CPU-RAM-Verbrauch prüfen. Wo er
korrekt und begrenzbar ist, anbinden. Falls sein Layout nicht passt oder seine
Residenz nicht steuerbar ist, den fehlenden Teil gezielt im privaten Adapter
ergänzen. Parallel Qwen-spezifisches Expert-Streaming implementieren. Nur PLE
auszulagern genügt als Speicherstrategie nicht ohne Prüfung der übrigen Gewichte.

**Danach PLE-I/O optimieren:** gleiche Zeilen pro Prefill-Chunk deduplizieren,
Abfragen nach Shard/Offset bündeln, inverse Reihenfolge exakt wiederherstellen,
quantisierte Zeilen mit passenden Skalen lesen. Zeilen- gegen kleinen Blockcache
vergleichen; Read-Amplification messen. Tabellen nie vollständig für einen
einzigen Lookup dequantisieren. Standard zunächst einfach und begrenzt.

Tokenbekannte PLE-Zugriffe können Kandidaten für begrenztes Prefetch sein.
Experten-Prefetch kennt zukünftige Routen dagegen nicht zuverlässig. Spekulatives
Lesen erst bei messbarem Nutzen und separat begrenztem Budget zulassen.

**Korrektheit:** gleiche Hashindizes und rekonstruierte Zeilen wie residente
Referenz; EOS-/Segmentreset, Shardgrenzen, wiederholte IDs, Prompt-Chunks und
inkrementelles Decode prüfen. Anschließend Logits und mehrschrittige Zustände
mit/ohne beide Auslagerungen vergleichen.

**Abnahme:** auf 48 GB erst eine kohärente Antwort, dann mindestens zehn
Folgeturns, ohne vollständige Tabellen-/Expertmaterialisierung beim Laden oder
Fortsetzen. Der erste konservative funktionierende Stand wird als Baseline
festgehalten, bevor komplexe Cachepolitik hinzukommt.

## 4. GLM: eigener Experten- und Zustandsadapter

Gemeinsame Infrastruktur übernehmen, aber GLM-Mathematik unverändert erhalten.
Der [GLM-Referenzport](https://github.com/PipeNetwork/glm53-flash-mlx) beschreibt
SwiGLU-Clamping, Routerpräzision, mHC-Datentypen und Norm-Epsilons als
Korrektheitsfallen. Diese Punkte an der tatsächlich installierten Runtime
prüfen; upstream beschriebene Fehler nicht ungeprüft als lokale Fehler ausgeben.

Q2/Q4/Q8-Zuordnung je Projektion auswerten. Gruppen nur bilden, wenn Bitbreiten,
Geometrie und Aktivierung kompatibel sind. Keine global erzwungene 2-bit-
Konfiguration; keine Experten entfernen, um den Speicher zu reduzieren.
Shared Experts und dichte Anfangsschichten in der Basisbilanz berücksichtigen.

KDA-/rekurrente Zustände, sparse MLA/Indexer und Hyper-Connections in kleinen
Referenzmodellen einzeln sowie gemeinsam testen. N-Gramm-Tabellen anhand des
konkreten GLM-Tensorinventars prüfen: bei vorhandenem Layout einen passenden
Adapter anbinden; bei Abwesenheit als nicht anwendbar dokumentieren. Ein
Qwen-artiges PLE-Layout wird dabei nicht vorausgesetzt. Dieselbe Prüfung gilt
für DeepSeek und wird in der gemeinsamen Abnahmematrix festgehalten.

**Abnahme:** erste Antwort und zehn Folgeturns auf 48 GB, kleiner expliziter
Cache ebenso wie Auto, gemessene Speicherobergrenzen, bestandene numerische
Vergleiche. Erst danach MTP separat untersuchen. Laut
[Vontra-Modellkarte](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-oQ2-MTP)
war MTP in einem Vergleich nicht tokenidentisch; deshalb kein Standard für die
erste Qualitäts-/Geschwindigkeitsmessung.

## 5. Korrekte Wiederverwendung und gleiche Clientpfade

Präfixwiederverwendung muss alle architekturspezifischen Zustände umfassen:
Attention, rekurrente/Convolution-Zustände, PLE-Tokenhistorie und Positionszähler.
Eine Erweiterung der Cache-Typenliste ohne Clone-/Save-/Restore-/Rollback-Tests
reicht nicht. Wo die Runtime das nicht sicher kann, Cachemiss statt falscher Antwort.

Unveränderliche Gewichtscaches dürfen im Worker geteilt werden; veränderliche
Gesprächszustände bleiben requestgebunden. Fingerprints binden Modellrevision,
Quantisierung, Runtime-/Adaptercode und Tokenizer/Template. Generierte Tokens
sind nicht zwingend identisch mit dem erneut tokenisierten nächsten Prompt.
Nur nachweislich passende Präfixe wiederverwenden; Historie nicht dafür umschreiben.

CLI und Serve nutzen denselben Adapter und dieselben effektiven Einstellungen.
Vergleiche verwenden frische explizite Sitzungen, identische Nachrichten,
Sampling-/Thinking-/Stop-Einstellungen und zunächst identische manuelle Budgets.
Auto separat vergleichen und das tatsächlich gewählte Budget protokollieren.
„Thinking aus“ nur setzen, wenn für Modell und Template semantisch unterstützt.

HTTP-SSE zunächst direkt testen: erstes sichtbares Zeichen, vollständiger Text,
Finish-Grund, Usage, `[DONE]`, Abbruch und Fehler. Danach Open WebUI sowie
n8n/ComfyUI. Zusätzliche Titel-/Tag-Anfragen getrennt von der Benutzeranfrage
zuordnen. werkStation erhält denselben API-Vertrag und zunächst Vertragstests;
einen echten UI-Nachweis erst mit verfügbarer Chatimplementierung behaupten.

## 6. Performance-Iteration nach bestandener Korrektheitsprüfung

Pro Modell: konservative Streaming-Baseline → Auto → Expert-Gruppengröße →
Materialisierung/Synchronisation → bei vorhandenen N-Gramm-Tabellen
Lesebündelung und Cacheaufteilung
→ Chunkgröße → begrenzte I/O-Überlappung. Je Experiment nur einen Faktor ändern.
Missrate allein entscheidet nicht: eingesparte Ladezeit pro belegtem Byte und
Decode-Gesamtzeit zählen. Ein PLE-Cache mit geringer Wiederverwendung darf klein bleiben.

| Messung | Vorgehen |
| --- | --- |
| Kurzchat | 128/512 Prompt-Tokens, 128 Ausgabetokens; echte EOS separat ausweisen. |
| Prefill-Grenzen | 2.048/4.096/8.192 Tokens, einschließlich Grenzen der modellabhängigen Sparse-Auswahl; nur soweit Speicherzulassung möglich. |
| Mehrturn | Zehn Turns, mindestens drei unabhängige Sitzungen; Verlauf, Cachetreffer und Speicherwachstum erfassen. |
| Budgets | 8 GiB Expertenobergrenze, 16 GiB sofern zulässig, Auto; bei vorhandenen N-Gramm-Tabellen deren Budget zusätzlich ausweisen. Kein erzwungenes 24-GiB-Budget. |
| Cachezustände | Worker-neu, Worker-warm, Präfixhit, Präfixmiss und kontrollierte Cacheverdrängung getrennt. OS-Dateicache nicht als kalt bezeichnen, wenn er nicht kontrolliert wurde. |
| Wiederholungen | Separates Warmup, mindestens fünf gepaarte Läufe je Vergleich; Median und Streuung. Keine belastbare p95-Aussage aus fünf Läufen. |
| Last/Lebenszyklus | Zunächst nur ein großer Worker; dann zwei konkurrierende Chats mit begrenzter Zulassung, Abbruch, Modellwechsel und Neustart. |

Vorhandenes `werk bench`, [HTTP-Benchmarkwerkzeuge](../utils/benchmarks/README.md)
und [Paritätsvergleich](benchmarks/2026-09-12-omlx-parity/compare.py) erweitern.
Rohdaten unter `docs/benchmarks/<datum>-qwen-glm/`: Revisionen, Requests,
effektive Einstellungen, Speicher-/I/O-Werte und ausgewählte Testausgaben.

Verbose trennt Queue, Load, Prefill, Decode, erstes natives Token, ersten
sichtbaren Text und HTTP-Gesamtzeit. Dazu gesamte/gecachte/neue Prompt-Tokens;
Experten- und PLE-Hits/Misses, Evictions, Residenz, effektive Limits, logische
Lesebytes, Read-Amplification, Lade-/Materialisierungs-/Rechenzeiten und
Prefetch-Nutzen. Workerintervalle nicht als requestexakt ausgeben, solange
Parallelität die Zuordnung verhindert. Physische SSD-I/O nur behaupten, wenn
gemessen; OS-Dateicache kann logische Reads bedienen. Überlappende Zeiten nicht summieren.

## 7. Qualitätsnachweis und Abnahmekriterien

Große residente Vergleichsläufe sind auf 48 GB nicht vorausgesetzt. Zuerst
kleine Modelle mit echter Architektur und identischer Quantisierung auf Metal,
dann reale einzelne Layer/Experten/PLE-Zeilen als residente Referenz. Einen
vollständigen residenten Checkpointvergleich erst auf ausreichend großer Hardware
durchführen und als separate Evidenz kennzeichnen. Die erste Streaming-Baseline
allein ist kein unabhängiger Qualitätsbeweis.

Logits, Routing, Zustandsfortschreibung und Fehlergrenzen prüfen. Toleranzen vor
Optimierung anhand Datentyp und Referenz festlegen, nicht nachträglich an Fehler
anpassen. Greedy-Tokenabweichungen untersuchen, besonders bei knappen Logit-/
Routing-Gleichständen; ein einziger flüssiger Satz genügt nicht.

Ein festes Paket deckt Deutsch/Englisch, Rechnen, Fakten aus vorgegebenem Kontext,
JSON, Mehrturn-Erinnerung, Rust/Python mit ausführbaren Tests und später Tools ab.
Deterministische Diagnose und vorgesehenes produktives Sampling getrennt.
Keine zufällige Ausgabelängenverkürzung als höhere Decode-Leistung verkaufen.

| Gate | Erforderlicher Nachweis |
| --- | --- |
| Ausführbarkeit | Beide Checkpoints laufen einzeln auf 48 GB, inklusive Laden und zehn Folgeturns; kein OOM, keine deaktivierte Speicherprüfung, keine entfernten Experten. |
| Kleine Maschinen | Kleine zulässige manuelle Caches funktionieren korrekt und laden fehlende Gewichte nach. Minimales Arbeitsset und getestete Hardware explizit ausweisen. |
| Qualität | Numerische Architektur-/Offload-/Cachetests bestanden; keine reproduzierbare Verschlechterung gegenüber der geprüften Baseline im Aufgabenpaket. |
| Persistenz | Nachweislich wiederverwendete Präfixe oder klarer Unsupported-/Miss-Grund; kein erzwungenes Wiederverwenden unvollständiger Zustände. |
| Chat/Serve | Bei gleichem Budget und Request angestrebte Abweichung der medianen warmen Decode-Rate höchstens 10 %; größere Differenzen erklären/beheben. |
| Performance | Entwicklungsziel mindestens 2× Decode gegenüber der ersten korrekten kleinen-Cache-Baseline und möglichst unter 5 s bis zum sichtbaren Text im warmen Kurzturn. Ziele, keine Zusagen; auch verfehlte Ziele mit gemessenem Engpass dokumentieren. |
| Regression | Bestehende DeepSeek-Expert-/Persistenz-/Qualitätsprüfungen und kontrollierter 8-GiB-/Auto-Vergleich; kein unbeabsichtigter Wechsel des nativen Pfads anderer Modelle. |

**Lieferumfang:** gemeinsamer Speicherverwalter, zwei explizite Architekturadapter,
Qwen-PLE-Anbindung, geprüfte Zustandswiederverwendung, verständliche Diagnose bei
fehlender Unterstützung und reproduzierbarer Abschlussbericht je Modell.
Eine pauschale Beschleunigung oder beliebig kleine Hardware gilt erst dann als
unterstützt, wenn dafür gesonderte Messungen vorliegen.
