from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .model.constants import SAMPLE_RATE
from .model.inference import RMVPE as _TorchRMVPE
from .model.utils import to_local_average_f0, to_viterbi_f0


class RMVPETorch:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        hop_length: int = 160,
    ) -> None:
        self.device = device
        self.model = _TorchRMVPE(str(model_path), device=device, hop_length=hop_length)

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        threshold: float = 0.03,
        use_viterbi: bool = False,
    ) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError("audio must be a mono 1D array")
        return self.model.infer_from_audio(
            audio.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            device=self.device,
            thred=threshold,
            use_viterbi=use_viterbi,
        )

    def extract_batch(
        self,
        audios: Sequence[np.ndarray],
        *,
        sample_rate: int | Sequence[int],
        threshold: float = 0.03,
        use_viterbi: bool = False,
        batch_size: int = 32,
        deterministic: bool = False,
    ) -> list[np.ndarray]:
        """Extract F0 for multiple clips.

        Clips are bucketed by exact padded mel length to avoid contaminating
        real frames with arbitrary right padding. Set deterministic=True to
        force one-by-one extraction for bit-exact single-item behavior.
        """
        sample_rates = _normalize_sample_rates(sample_rate, len(audios))
        if deterministic or batch_size <= 1:
            return [
                self.extract(
                    audio,
                    sample_rate=audio_sample_rate,
                    threshold=threshold,
                    use_viterbi=use_viterbi,
                )
                for audio, audio_sample_rate in zip(audios, sample_rates)
            ]

        prepared = []
        buckets = defaultdict(list)
        for index, (audio, audio_sample_rate) in enumerate(zip(audios, sample_rates)):
            mel = self._audio_to_mel(audio, audio_sample_rate)
            frame_count = mel.shape[-1]
            mel = _pad_mel_to_multiple(mel, multiple=32)
            prepared.append((mel, frame_count))
            buckets[mel.shape[-1]].append(index)

        results: list[np.ndarray | None] = [None] * len(prepared)
        with torch.no_grad():
            for indices in buckets.values():
                for start in range(0, len(indices), batch_size):
                    chunk = indices[start : start + batch_size]
                    batch = torch.cat([prepared[index][0] for index in chunk], dim=0)
                    hidden = self.model.model(batch)
                    for batch_index, result_index in enumerate(chunk):
                        frame_count = prepared[result_index][1]
                        item_hidden = hidden[
                            batch_index : batch_index + 1, :frame_count, :
                        ]
                        if use_viterbi:
                            result = to_viterbi_f0(item_hidden, thred=threshold)
                        else:
                            result = to_local_average_f0(item_hidden, thred=threshold)
                        results[result_index] = result

        if any(result is None for result in results):
            raise RuntimeError("internal error: missing batch extraction result")
        return results

    def _audio_to_mel(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError("audio must be a mono 1D array")
        audio_tensor = (
            torch.from_numpy(audio.astype(np.float32, copy=False))
            .float()
            .unsqueeze(0)
            .to(self.model.device)
        )
        if sample_rate != SAMPLE_RATE:
            key = str(sample_rate)
            if key not in self.model.resample_kernel:
                self.model.resample_kernel[key] = Resample(
                    sample_rate, SAMPLE_RATE, lowpass_filter_width=128
                ).to(self.model.device)
            audio_tensor = self.model.resample_kernel[key](audio_tensor)
        return self.model.mel_extractor(audio_tensor, center=True)


def _normalize_sample_rates(
    sample_rate: int | Sequence[int], length: int
) -> list[int]:
    if isinstance(sample_rate, int):
        return [sample_rate] * length
    sample_rates = list(sample_rate)
    if len(sample_rates) != length:
        raise ValueError("sample_rate sequence must match audios length")
    return sample_rates


def _pad_mel_to_multiple(mel: torch.Tensor, *, multiple: int) -> torch.Tensor:
    frame_count = mel.shape[-1]
    padded_count = multiple * ((frame_count - 1) // multiple + 1)
    pad_count = padded_count - frame_count
    if pad_count == 0:
        return mel
    mode = "reflect" if pad_count < frame_count else "replicate"
    return F.pad(mel, (0, pad_count), mode=mode)


__all__ = ["RMVPETorch", "SAMPLE_RATE"]
