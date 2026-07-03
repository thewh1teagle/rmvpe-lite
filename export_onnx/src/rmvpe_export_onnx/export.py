import argparse
import sys
from pathlib import Path

import onnx
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from rmvpe_lite.model.constants import (
    CONST,
    MEL_FMAX,
    MEL_FMIN,
    N_CLASS,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from rmvpe_lite.model.model import E2E0


DEFAULT_REPO_ID = "stylish-tts/pitch_extractor"
DEFAULT_FILENAME = "rmvpe.safetensors"
DEFAULT_HOP_LENGTH = 160


def load_model(weights_path: str | Path, device: torch.device) -> E2E0:
    model = E2E0(4, 1, (2, 2))
    checkpoint = {}
    with safe_open(str(weights_path), framework="pt", device=str(device)) as f:
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def resolve_weights(path: str | None) -> str:
    if path:
        return path
    return hf_hub_download(DEFAULT_REPO_ID, DEFAULT_FILENAME)


def write_metadata(output: Path, *, hop_length: int, opset: int) -> None:
    model = onnx.load(output)
    metadata = {
        "model_type": "rmvpe",
        "input_type": "log_mel",
        "sample_rate": str(SAMPLE_RATE),
        "n_mels": str(N_MELS),
        "window_length": str(WINDOW_LENGTH),
        "hop_length": str(hop_length),
        "mel_fmin": str(MEL_FMIN),
        "mel_fmax": str(MEL_FMAX),
        "n_class": str(N_CLASS),
        "cents_const": str(CONST),
        "center": "true",
        "stft_pad_mode": "reflect",
        "onnx_opset": str(opset),
        "source_repo": DEFAULT_REPO_ID,
        "source_file": DEFAULT_FILENAME,
    }
    for prop in model.metadata_props:
        if prop.key in metadata:
            prop.value = metadata.pop(prop.key)
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, output)


def export_onnx(
    *,
    output: Path,
    weights: str | None,
    frames: int,
    opset: int,
    device_name: str,
    hop_length: int,
) -> None:
    device = torch.device(device_name)
    weights_path = resolve_weights(weights)
    model = load_model(weights_path, device)
    dummy_mel = torch.randn(2, N_MELS, frames, device=device)
    dynamic_axes = {
        "mel": {0: "batch", 2: "frames"},
        "hidden": {0: "batch", 1: "frames"},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_mel,
        output,
        input_names=["mel"],
        output_names=["hidden"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    write_metadata(output, hop_length=hop_length, opset=opset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RMVPE torch weights to ONNX.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rmvpe.onnx"),
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Path to rmvpe.safetensors. Downloads the Stylish TTS weights when omitted.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=128,
        help="Dummy frame count for tracing. Use a multiple of 32.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Torch device for export.")
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop length expected by downstream RMVPE audio inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames % 32 != 0:
        raise SystemExit("--frames must be a multiple of 32")
    export_onnx(
        output=args.output,
        weights=args.weights,
        frames=args.frames,
        opset=args.opset,
        device_name=args.device,
        hop_length=args.hop_length,
    )
    print(f"Exported ONNX model to {args.output}")
