# Batched pitch extraction

`extract_batch` extracts F0 for many clips in a single model call, for
throughput-oriented workloads such as building pitch features for TTS or
voice-model training over hundreds of hours of audio.

It is designed to be **training-safe**: batched output matches single-clip
output closely enough that it does not change F0 features in a way that could
hurt training. See [Correctness](#correctness) for measured numbers.

Both backends expose the same method:

- `rmvpe_lite.onnx.RMVPEOnnx.extract_batch`
- `rmvpe_lite.torch.RMVPETorch.extract_batch`

## Quick start

```python
from pathlib import Path

import soundfile as sf

from rmvpe_lite.onnx import RMVPEOnnx  # or: from rmvpe_lite.torch import RMVPETorch


def load(path: str):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


paths = ["0.wav", "1.wav", "2.wav"]
audios, sample_rates = zip(*(load(p) for p in paths))

model = RMVPEOnnx(Path("rmvpe.onnx"))
f0_list = model.extract_batch(
    list(audios),
    sample_rate=list(sample_rates),  # or a single int if all clips share a rate
    batch_size=32,
)

for path, f0 in zip(paths, f0_list):
    voiced = f0[f0 > 0]
    print(path, len(f0), "frames,", len(voiced), "voiced")
```

`extract_batch` returns a list of `np.ndarray` F0 arrays in the **same order**
as the input clips. Each array is per-frame F0 in Hz, with `0.0` for unvoiced
frames — identical in shape and meaning to what `extract` returns for a single
clip.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `audios` | — | Sequence of mono 1-D float arrays. |
| `sample_rate` | — | A single `int` (applied to all clips) or a sequence of `int` matching `audios`. Clips are resampled to the model rate as needed. |
| `threshold` | `0.03` | Voicing threshold, same as `extract`. |
| `use_viterbi` | `False` | Viterbi decoding, same as `extract`. Runs per clip. |
| `batch_size` | `32` | Maximum number of clips per model call. Caps memory. |
| `deterministic` | `False` | When `True`, extraction runs one clip at a time and is **bit-identical** to `extract`. See [Correctness](#correctness). |

## How it works, and why it is safe

RMVPE is a DeepUNet (2-D convolutions over mel bins × frames) followed by a
**bidirectional GRU** over the frame axis. Every output frame therefore depends
on the entire input sequence, *including any padding*. Padding a short clip up
to the length of a longer one in the same batch changes the network's context
for the short clip's real frames and corrupts its F0 — even after cropping the
padding away.

`extract_batch` avoids this entirely:

1. Each clip is turned into a log-mel spectrogram.
2. Each spectrogram is padded to the next multiple of 32 frames — exactly the
   same padding the single-clip path applies.
3. Clips are **bucketed by their exact padded frame length**. Only clips that
   are already the same length get stacked together, so batching introduces
   **no additional padding** and no cross-clip contamination.
4. Each bucket is run in chunks of at most `batch_size`.
5. Each clip's output is cropped back to its true frame count and decoded.

The only remaining coupling between clips is the batch dimension itself, which
in practice contributes only floating-point-level differences.

## Correctness

Measured on the `michael-gold-v1` dataset, comparing `extract_batch` against
per-clip `extract`:

| Backend | Max abs diff | Voiced/unvoiced flips |
| --- | --- | --- |
| ONNX (`batch_size=16`) | ~0.0002 Hz | 0 |
| ONNX (`deterministic=True`) | 0 Hz (bit-exact) | 0 |
| Torch (`batch_size=16`) | ~0.0002 Hz | 0 |
| Torch (`deterministic=True`) | 0 Hz (bit-exact) | 0 |

Differences are far below the perceptual and training-relevant resolution of
F0, and **no frame ever changes between voiced and unvoiced**. On larger runs a
single ambiguous frame may occasionally differ by up to ~1 Hz due to numeric
drift in ONNXRuntime's batched GRU kernel; this still never flips voicing.

**When to use `deterministic=True`:** if you must reproduce an existing feature
cache bit-for-bit, or want a guarantee of exact equality with the single-clip
path, pass `deterministic=True`. It skips batching and runs clips one by one,
trading throughput for exactness. For fresh feature extraction the default
batched path is recommended.

## Throughput notes

- The speedup from batching comes from amortizing per-call overhead. It is
  largest on the **CUDA execution provider** (ONNX) or a **CUDA device**
  (Torch), where per-call launch overhead dominates.
- On CPU, batching gives little or no speedup because the CPU kernels already
  saturate available threads per call. Batching still remains correct on CPU.
- Buckets fill best when clips have similar lengths. For TTS datasets, sorting
  or grouping utterances by duration before calling `extract_batch` maximizes
  bucket sizes and throughput. Note that clips of different lengths simply land
  in different buckets — correctness never depends on how you order the input.

## Requirements

- **ONNX:** the model must have a dynamic batch axis. The exporter produces one
  by default:

  ```bash
  uv run --project export_onnx rmvpe-export-onnx -o rmvpe.onnx
  ```

  The released model is already exported this way, so `extract_batch` works out
  of the box. An older model with a fixed batch size of 1 would reject two or
  more same-length clips in ONNXRuntime; single-clip `extract` works with
  either.
- **Torch:** no special export step; `extract_batch` works with the standard
  weights.

## Edge cases

- Clips must be longer than the model's `window_length` (1024 samples) after
  resampling; a shorter clip raises a `ValueError`. This is a limitation of the
  mel front end and applies to `extract` as well as `extract_batch`; it is not
  specific to batching. Real TTS utterances are always well above this length.
  In `extract_batch`, one short clip raises for the whole call, so filter out
  clips shorter than this before batching (about 64 ms at 16 kHz).
- `sample_rate` given as a sequence must have the same length as `audios`,
  otherwise a `ValueError` is raised.
