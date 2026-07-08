from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeModelMeta:
    custom_metadata_map = {
        "sample_rate": "16000",
        "n_mels": "128",
        "window_length": "1024",
        "hop_length": "160",
        "mel_fmin": "30.0",
        "mel_fmax": "8000.0",
        "n_class": "360",
        "cents_const": "1997.3794084376191",
        "center": "true",
        "stft_pad_mode": "reflect",
    }


class _FakeTensor:
    name = "mel"


class _FakeSession:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_modelmeta(self) -> _FakeModelMeta:
        return _FakeModelMeta()

    def get_inputs(self) -> list[_FakeTensor]:
        return [_FakeTensor()]

    def get_outputs(self) -> list[_FakeTensor]:
        return [_FakeTensor()]


def _fake_mel(**kwargs: object) -> np.ndarray:
    return np.ones(
        (int(kwargs["n_mels"]), int(kwargs["n_fft"]) // 2 + 1),
        dtype=np.float32,
    )


fake_onnxruntime = types.ModuleType("onnxruntime")
fake_onnxruntime.InferenceSession = _FakeSession
fake_onnxruntime.SessionOptions = object
sys.modules["onnxruntime"] = fake_onnxruntime

fake_librosa = types.ModuleType("librosa")
fake_librosa.filters = types.SimpleNamespace(mel=_fake_mel)
sys.modules["librosa"] = fake_librosa


class OnnxHopLengthTest(unittest.TestCase):
    def test_constructor_uses_metadata_hop_length_by_default(self) -> None:
        from rmvpe_lite.onnx import RMVPEOnnx

        model = RMVPEOnnx("rmvpe.onnx")

        self.assertEqual(model.config.hop_length, 160)

    def test_constructor_can_override_hop_length(self) -> None:
        from rmvpe_lite.onnx import RMVPEOnnx

        model = RMVPEOnnx("rmvpe.onnx", hop_length=200)

        self.assertEqual(model.config.hop_length, 200)

    def test_constructor_rejects_nonpositive_hop_length(self) -> None:
        from rmvpe_lite.onnx import RMVPEOnnx

        with self.assertRaisesRegex(ValueError, "hop_length must be positive"):
            RMVPEOnnx("rmvpe.onnx", hop_length=0)


if __name__ == "__main__":
    unittest.main()
