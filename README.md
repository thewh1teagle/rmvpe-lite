## RMVPE ONNX

ONNXRuntime package for extracting RMVPE F0/pitch features for TTS and voice
model training, without PyTorch at runtime.

Install from GitHub:

```bash
uv pip install git+https://github.com/thewh1teagle/rmvpe-onnx
```

Basic usage:

```python
import soundfile as sf
from rmvpe_onnx import RMVPE

audio, sample_rate = sf.read("audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

model = RMVPE("rmvpe.onnx")
f0 = model.extract(audio, sample_rate=sample_rate)
```

`f0` is a NumPy array of pitch values in Hz. Unvoiced frames are `0.0`.

See [Attribution](docs/attribution.md) for RMVPE code, model, and paper sources.
