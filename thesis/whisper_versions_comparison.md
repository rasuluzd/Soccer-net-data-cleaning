# Whisper Version Comparison: SoccerNet bundled vs faster-whisper-v3

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**

Both ASR transcripts cover the same audio. The difference is
**which Whisper engine + decoding parameters were used to
produce the raw transcript** — not the cleaning pipeline.

| Track | Engine + parameters |
|---|---|
| **1_asr.json** | SoccerNet bundled — most likely stock OpenAI Whisper (medium or large), default decoding parameters, frozen in the SoccerNet-Echoes release. Schema-1 list format. |
| **1_asr_v3.json** | Our regeneration via Systran/faster-whisper-large-v3 with beam=5, word_timestamps=True, no_speech_threshold=0.95, condition_on_previous_text=False, Q4_K_M int8 quantisation on CPU. Schema-2 dict format with per-word probabilities. |

Both are evaluated against `1_asr_corrected.json` (GOAL human-
annotated ground truth, scored with jiwer + a custom entity-F1).

---

## Results per half

### Half 1 (230 GT segments)

| Track | Segments | WER | Sub/Ins/Del | Entity-F1 | Entity P/R |
|---|---|---|---|---|---|
| SoccerNet bundled | 612 | **46.50 %** | 836/815/523 | **0.711** | 0.81 / 0.64 |
| faster-whisper-v3 | 754 | **36.49 %** | 530/795/381 | **0.749** | 0.85 / 0.67 |

### Half 2 (211 GT segments)

| Track | Segments | WER | Sub/Ins/Del | Entity-F1 | Entity P/R |
|---|---|---|---|---|---|
| SoccerNet bundled | 674 | **30.78 %** | 591/476/383 | **0.829** | 0.87 / 0.79 |
| faster-whisper-v3 | 775 | **34.66 %** | 552/693/388 | **0.773** | 0.88 / 0.69 |

---

## Combined (both halves)

| Track | Corpus WER | Entity-F1 | Entity P / R |
|---|---|---|---|
| **SoccerNet bundled** | **38.61 %** | **0.771** | 0.84 / 0.71 |
| **faster-whisper-v3** | **35.57 %** | **0.761** | 0.87 / 0.68 |

- **WER delta**: -3.04 pp (-7.9 % rel) — faster-whisper-v3 vs SoccerNet bundled
- **Entity-F1 delta**: -0.010 abs (-1.3 % rel)

## Interpretation

If WER goes DOWN (negative delta) with faster-whisper-v3, the regeneration improved transcription quality before our cleaning pipeline ever runs — meaning some of the gain we attribute to our pipeline is actually the better Whisper engine, not the post-processing.

If Entity-F1 goes UP, the larger model + better decoding parameters resolved more named entities correctly. This matters when comparing our cleaning pipeline against the raw input it actually ingests (`1_asr_v3.json`), not against the SoccerNet-bundled output.

**Takeaway for thesis**: report both the Whisper-engine delta and the cleaning-pipeline delta separately so the two effects don't get conflated.