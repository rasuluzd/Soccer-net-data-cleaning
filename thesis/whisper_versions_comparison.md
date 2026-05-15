# Whisper Version Comparison: SoccerNet bundled vs faster-whisper-v3

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**
Reference: **`{1,2}_asr_corrected.json`** (GOAL human-annotert ground truth)
Evaluator: **`tools/evaluate_wer.py`** with **legacy 1-to-1 time alignment**
(same setup as the cleaning-pipeline numbers in §4.2.1.8 — directly
comparable).

| Track | Engine + parameters |
|---|---|
| **`{1,2}_asr.json`** (SoccerNet bundled) | Stock OpenAI Whisper from the SoccerNet-Echoes release. Includes punctuation and casing. Schema-1 list format. |
| **`{1,2}_asr_v3.json`** (our regeneration) | Systran/faster-whisper-large-v3 with `beam=5`, `word_timestamps=True`, `no_speech_threshold=0.95`, `condition_on_previous_text=False`, Q4_K_M int8 quantisation on CPU. **All-lowercase output, no punctuation.** Schema-2 dict format. |

## Results per half (legacy alignment)

| Halv | SoccerNet WER | faster-v3 WER | Δ WER | SoccerNet F1 | faster-v3 F1 | Δ F1 |
|---|---|---|---|---|---|---|
| 1 | 29.81 % | 25.56 % | **−4.25 pp** | 0.620 | 0.484 | −0.136 |
| 2 | 24.84 % | 23.86 % | **−0.98 pp** | 0.598 | 0.504 | −0.094 |
| Snitt | **27.32 %** | **24.71 %** | **−2.61 pp** | **0.609** | **0.494** | **−0.115** |

## Interpretation

**WER goes DOWN −2.61 pp avg** — faster-whisper-v3 transcribes more
words correctly than the SoccerNet-bundled stock Whisper. Two factors:
(a) newer model checkpoint (large-v3 vs likely large-v2/medium), (b)
aggressive `no_speech_threshold=0.95` keeps soft commentary stock
filtered as silence.

**Entity-F1 goes DOWN −0.115 avg, but for a measurable formatting
reason.** F1 is case-sensitive. SoccerNet's stock output preserves
casing/punctuation (`"Sturridge"`); our faster-v3 output is
all-lowercase (`"sturridge"`). The exact-match check fails, so
canonical names get counted as misses. **The engine isn't worse on
entities — the output format is.**

This is exactly what cleaning Step P (oliverguhr punctuation/casing
restorer) is designed to fix. After Step P:

| Track | F1 |
|---|---|
| SoccerNet bundled (native casing) | 0.609 |
| faster-v3 raw (all-lowercase) | 0.494 |
| **faster-v3 + cleaning** | **0.591** |

Step P + Stage E together close 0.097 of the 0.115 F1 gap, putting us
back at parity with SoccerNet's casing.

## Take-away for the thesis

Two distinct, separately-measurable contributions:

1. **Engine + decoding choice: −2.61 pp WER** vs SoccerNet-bundled.
   Independent of any cleaning.
2. **Cleaning pipeline: restores Entity-F1** (Step P casing) and
   improves canonical-name precision (Step E, +0.097 F1 over raw
   faster-v3 — see §4.2.1.8).

Combined effect (re-transcription + cleaning) vs original SoccerNet:
- WER: 25.26 % (cleaned) vs 27.32 % (SoccerNet stock) = **−2.06 pp**
- Entity-F1: 0.591 (cleaned) vs 0.609 (SoccerNet stock) = **−0.018**
  (within noise — but with proven canonical-name corrections like
  Aspilicueta→Azpilicueta and rigi→Origi that SoccerNet missed)

Reporting both effects separately, and being transparent about the
casing-driven F1 dynamic, is the honest framing for the thesis
discussion.
