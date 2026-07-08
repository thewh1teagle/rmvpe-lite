## RMVPE Lite

Lightweight RMVPE F0/pitch extraction for TTS and voice model training, with
ONNXRuntime and PyTorch backends.

Install the ONNX backend from GitHub:

```bash
uv pip install "rmvpe-lite[onnx] @ git+https://github.com/thewh1teagle/rmvpe-onnx"
```

Or install the PyTorch backend:

```bash
uv pip install "rmvpe-lite[torch] @ git+https://github.com/thewh1teagle/rmvpe-onnx"
```

See the `examples/` folder for usage examples.

`extract()` returns a NumPy array of pitch values in Hz. Unvoiced frames are `0.0`.

### Matching a TTS mel grid

Both backends accept `hop_length` so TTS pipelines can ask RMVPE for F0 on the
same frame grid as their acoustic features:

```python
from rmvpe_lite.torch import RMVPETorch

rmvpe_hop_float = 16000 * tts_hop_length / tts_sample_rate
rmvpe_hop = int(rmvpe_hop_float)
assert rmvpe_hop == rmvpe_hop_float
model = RMVPETorch("rmvpe.safetensors", hop_length=rmvpe_hop)
```

RMVPE resamples audio to 16 kHz internally, so convert your TTS hop with
`16000 * tts_hop_length / tts_sample_rate`. Prefer integer results and keep the
value near the default 160 samples, which is the 10 ms grid the model was trained
on. If the converted hop is not an integer, such as 22,050 Hz audio with hop 256
giving about 185.76, extract with a nearby supported hop and interpolate F0 to
your target mel grid. The same `hop_length` keyword is available on
`RMVPEOnnx`.

See [Attribution](docs/attribution.md) for RMVPE code, model, and paper sources.
