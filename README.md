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

Download the ONNX model from the [model-files-v1.0 release](https://github.com/thewh1teagle/rmvpe-onnx/releases/download/model-files-v1.0/rmvpe.onnx).

See the `examples/` folder for usage examples.

`extract()` returns a NumPy array of pitch values in Hz. Unvoiced frames are `0.0`.

See [Attribution](docs/attribution.md) for RMVPE code, model, and paper sources.
