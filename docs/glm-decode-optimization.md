# GLM-5.3-Flash: Decode-Optimierung in Werk

Ziel ist der installierte `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP` auf dem
48-GiB-Apple-Silicon-Rechner mit oMLX 0.6.4. Die untersuchten Projekte dienen
als Quellen für Verfahren; keine ihrer Laufzeiten oder Abhängigkeiten wird
installiert oder eingebunden. Stand: 14. September 2026.

## Übertragbare Verfahren

| Quelle | Verfahren | Umsetzung bzw. Eignung für Werk |
| --- | --- | --- |
| [Krasis HCS](https://github.com/brontoguana/krasis/blob/main/ADVANCED.md) | Geschützter heißer Expertenbestand plus flexibler Bereich; demand-first Prefetch und getrennte Hot/Cold-Ausführung | Schutz wiederholt benötigter Experten ist übertragbar. CUDA-Graphen und RAM→VRAM-DMA sind keine direkt nutzbaren Implementierungen für MLX/SSD. |
| [KTransformers GLM-5.3-Flash](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md) | CPU/GPU-Aufteilung der Experten, schichtweiser Prefill und spezialisierte FP8-Kernels | Hilft bei der Trennung von Prefill und Decode. Die dokumentierte Konfiguration verlangt mindestens 350 GB verfügbaren RAM, NVIDIA SM89/SM120 und AVX-512; ihre Leistungswerte sind kein Mac-Zielwert. |
| [Vates](https://github.com/AMOS144/Vates) | Begrenzte MLX-Experten-Pools, paralleles `pread`, prädiktives Vorladen, MTP mit Verifikation | Nahe an unserem SSD-/Unified-Memory-Fall. Parallele notwendige Reads lassen sich ohne Routervorhersage übernehmen. MTP verlangt einen eigenen GLM-Drafter und korrekte Zustandsrücknahme. |
| [SGLang Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) | Draft-Kandidaten erzeugen, gemeinsam prüfen und nur bestätigte Tokens übernehmen | Algorithmisches Vorbild für spätere GLM-MTP-Integration; Unterstützung eines anderen GLM-Modells belegt keine fertige MLX-Implementierung für `glm5_next`. |

## Gemessener Engpass

Der vorherige Persistenzvergleich zeigte den Decode-Rückgang auch ohne
Werk-Persistenz. Bei konstantem 20-GiB-Expertenbudget sank die Trefferrate in
den Decode-Stichproben von rund 70 auf 64 Prozent; die Lesezeit pro Token
stieg von 0,30 auf 0,36 Sekunden. Die anderen im Expertenpfad erfassten Kosten
blieben ungefähr gleich. Details und Grenzen stehen im
[Benchmarkbericht](benchmarks/2026-09-12-flash-offload/README.md).

## Änderung 1: Schutz wiederholt verwendeter Experten

GLM erhält einen segmentierten LRU-Index. Ein erstmaliger Zugriff kommt in
den flexiblen Bereich, ein erneuter Zugriff befördert den Experten in den
geschützten Bereich. Dieser beansprucht höchstens 80 Prozent des bestehenden
Expertenbudgets. Bei Überschreitung wandert sein ältester Eintrag zurück in
den flexiblen Bereich. Neue Themen können den Bestand damit weiterhin ändern.

Es entsteht kein zweiter Gewichtsspeicher. Native Budgetverkleinerungen,
manuelle Evictions, Pins und aktive Leases werden berücksichtigt. Im ersten
GLM-Vergleich behalten Qwen und DeepSeek noch ihre bisherige LRU-Strategie.
Weder Router-Auswahl noch
Quantisierung oder Modellrechnung ändern sich.

Im ersten Vergleich mit festem 20-GiB-Budget, Persistenz und identischem
Verlauf waren alle drei Antworten wortgleich:

| Runde | LRU vorher | Segmentierter LRU | Decode-Hitrate vorher → nachher |
| --- | ---: | ---: | ---: |
| 1 | 1,82 tok/s | 1,80 tok/s | 70,2 → 70,4 % |
| 2 | 1,66 tok/s | 1,88 tok/s | 66,8 → 71,0 % |
| 3 | 1,63 tok/s | 1,83 tok/s | 65,2 → 70,7 % |

[Referenz](benchmarks/2026-09-12-flash-offload/glm-warm-persistence-ab-on.json),
[neue Strategie](benchmarks/2026-09-12-flash-offload/glm-warm-retention-slru.json).
Einzelne Läufe, kein statistisch abgesichertes universelles Speedup-Versprechen.

## Änderung 2: Paralleles Lesen benötigter Experten

Der bestehende GLM-Adapter liest fehlende Tensorbereiche einer bekannten
Expertengruppe mit höchstens vier I/O-Threads. Es werden ausschließlich
tatsächlich angeforderte Bereiche gelesen. Die Gruppenobergrenze bleibt
128 MiB innerhalb der vorhandenen Workspace-Reserve; größere Gruppen verwenden
den bisherigen Pfad. MLX-Operationen laufen weiterhin auf dem besitzenden
Executor. Die Leser prüfen Dateisignaturen, besitzen ihre Dateideskriptoren
und werden auch bei Fehlern vollständig abgewartet.

Das parallelisiert notwendige Reads. Es implementiert noch keine Vorhersage
zukünftiger Routerentscheidungen und kein schichtübergreifendes Prefetch.

Mit gleichem 20-GiB-Budget, Persistenz, Temperatur 0 und wortgleichen Antworten:

| Runde | Ursprünglicher LRU | Geschützter Cache | Zusätzlich paralleles Lesen |
| --- | ---: | ---: | ---: |
| 1 | 1,82 tok/s | 1,80 tok/s | 2,38 tok/s |
| 2 | 1,66 tok/s | 1,88 tok/s | 2,54 tok/s |
| 3 | 1,63 tok/s | 1,83 tok/s | 2,51 tok/s |

[Paralleler Lauf](benchmarks/2026-09-12-flash-offload/glm-warm-retention-parallel.json).
Zwischen den beiden neuen Varianten sind kumulierte Hits, Misses, Evictions
und logische Lesebytes an jeder Anfragegrenze identisch. Im Decode-Fenster
fällt die Lesezeit von etwa 0,29 auf 0,21 Sekunden pro Token. Die wiederholte
[alte Variante nach diesem Lauf](benchmarks/2026-09-12-flash-offload/glm-warm-retention-baseline-recheck.json)
bleibt deutlich langsamer; die Änderung erklärt den Unterschied besser als
ein reiner Effekt der Testreihenfolge. Trotzdem sind dies wenige lokale
Vergleichsläufe und keine Garantie für beliebige Prompts oder Parallelbetrieb.

97 Tests bestehen, darunter native GLM-/Qwen-Rechnung, Cache-Neuöffnung,
Budgetverkleinerung, Pins/Leases, parallele Reads, fehlerhafte Reads,
Dateiaustausch und Fallback bei zu kleinem Staging-Budget. Alle neun Antworten
der drei ersten Vergleichsläufe waren korrekt und vollständig. Die neuen
Verfahren werden anhand der GLM-Architektur gewählt und benötigen keine
zusätzlichen CLI- oder Node-Parameter.

Der abschließende [Auto-Lauf](benchmarks/2026-09-12-flash-offload/glm-warm-retention-auto.json)
erreicht **2,55 / 2,66 / 2,62 tok/s** bei einem oberen Expertenbudget von
23.900.137.860 Bytes (22,26 GiB). Die Antworten bleiben wortgleich. Im dritten
Prefill fällt die Grenze auf 22.097.061.206 Bytes und erholt sich während
Decode wieder bis zur Obergrenze. Auch Auto bleibt damit innerhalb der
nativen Speicherplanung. Dieser Lauf hat mehr Cache als der feste Vergleich
und ist kein zusätzlicher isolierter Speedup-Nachweis.

Die getestete Version wurde lokal mit `cargo install --path . --locked --offline`
installiert; alle privaten Testserver wurden anschließend beendet. Die
anschließenden Qwen-/DeepSeek-Vergleiche stehen im
[Benchmark-Protokoll](benchmarks/2026-09-12-flash-offload/README.md#qwen--deepseek-geschützter-experten-cache-2026-09-15).
Beide Verfahren sind dort inzwischen ebenfalls aktiviert: Qwen nutzt dieselben
Helfer, DeepSeek zusätzlich den gemeinsamen Reader ohne seine frühere
Bytearray-Zwischenkopie. Seine native Konvertierung der Expert-Metadaten bleibt
erhalten. Die gemessenen Gewinne erfordern keine zusätzlichen Node-Optionen.
Die gemeinsame Wiederherstellung des Decode-Budgets aus dem vorherigen
Schritt gilt weiterhin für alle drei Experten-Adapter.

## MTP: lohnender nächster größerer Schritt, kein fertiger Schalter

Der lokale Checkpoint enthält 59 MTP-Tensoren mit insgesamt etwa **4,01 GiB**
gepackten Gewichten. Der installierte native `glm5_next`-Port entfernt diese
beim Laden und implementiert in diesem Pfad keinen MTP-Drafter. Werk kann
Speculative-Konfigurationen an vLLM weiterreichen; daraus folgt kein fertiger
GLM-MTP-Pfad für den lokalen oMLX-Offload-Adapter.

Eine Umsetzung müsste mindestens:

1. GLMs MTP-Block mit seiner tatsächlichen Gewichtsstruktur und Quantisierung
   laden und gegen eine native Referenz prüfen.
2. Draft-/Verify-Zyklen mit zunächst einem, dann mehreren Kandidaten ausführen.
3. KV-, Indexer- und rekurrente Linear-Attention-Zustände bei Ablehnung exakt
   zurücksetzen; der nächste normale Token muss zur Referenz passen.
4. Zusatzspeicher, Expert-Cache-Verlust und zusätzliche Draft-Reads budgetieren.
5. Akzeptanzrate, Verify-Zeit und ausgegebene Tokens/s messen und die Methode
   nur bei tatsächlichem Gewinn einsetzen; Streams und Tool-Aufrufe müssen
   ausschließlich bestätigte Tokens erhalten.

Der mögliche Gewinn liegt darin, ein geladenes Expertengewicht für mehrere
Verify-Tokens zu verwenden. Zusätzliche MTP-Gewichte können zugleich den
Hauptmodell-Cache verkleinern; ein Gewinn ist deshalb vor Messung offen.
