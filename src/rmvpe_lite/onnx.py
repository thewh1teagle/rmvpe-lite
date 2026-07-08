from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class RMVPEOnnxConfig:
    sample_rate: int
    n_mels: int
    window_length: int
    hop_length: int
    mel_fmin: float
    mel_fmax: float
    n_class: int
    cents_const: float
    center: bool
    stft_pad_mode: str


class RMVPEOnnx:
    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        hop_length: int | None = None,
    ) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.config = _config_from_metadata(
            self.session.get_modelmeta().custom_metadata_map
        )
        if hop_length is not None:
            if hop_length <= 0:
                raise ValueError("hop_length must be positive")
            self.config = replace(self.config, hop_length=hop_length)
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.output_name = self.session.get_outputs()[0].name
        self._mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.window_length,
            n_mels=self.config.n_mels,
            fmin=self.config.mel_fmin,
            fmax=self.config.mel_fmax,
            htk=True,
        ).astype(np.float32)

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        threshold: float = 0.03,
        use_viterbi: bool = False,
    ) -> np.ndarray:
        """Extract F0 for a single clip.

        Raises ValueError if the clip has window_length or fewer samples after
        resampling (about 64 ms at 16 kHz); it is too short to compute an STFT.
        """
        audio = _mono_float32(audio)
        if sample_rate != self.config.sample_rate:
            audio = _resample_audio(audio, sample_rate, self.config.sample_rate)

        mel = self._log_mel(audio)
        frame_count = mel.shape[-1]
        mel = _pad_frames_to_multiple(mel, multiple=32)
        hidden = self.session.run(
            [self.output_name],
            {self.input_name: mel[None, :, :].astype(np.float32, copy=False)},
        )[0]
        hidden = hidden[:, :frame_count, :]

        if use_viterbi:
            return _to_viterbi_f0(hidden, self.config, threshold)
        return _to_local_average_f0(hidden, self.config, threshold)

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

        Raises ValueError if any clip has window_length or fewer samples after
        resampling (about 64 ms at 16 kHz); this fails the whole call, so filter
        out short clips before batching.
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
            audio = _mono_float32(audio)
            if audio_sample_rate != self.config.sample_rate:
                audio = _resample_audio(audio, audio_sample_rate, self.config.sample_rate)

            mel = self._log_mel(audio)
            frame_count = mel.shape[-1]
            mel = _pad_frames_to_multiple(mel, multiple=32)
            prepared.append((mel, frame_count))
            buckets[mel.shape[-1]].append(index)

        results: list[np.ndarray | None] = [None] * len(prepared)
        for indices in buckets.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                batch = np.stack([prepared[index][0] for index in chunk]).astype(
                    np.float32, copy=False
                )
                hidden = self.session.run(
                    [self.output_name],
                    {self.input_name: batch},
                )[0]
                for batch_index, result_index in enumerate(chunk):
                    frame_count = prepared[result_index][1]
                    item_hidden = hidden[batch_index : batch_index + 1, :frame_count, :]
                    if use_viterbi:
                        result = _to_viterbi_f0(item_hidden, self.config, threshold)
                    else:
                        result = _to_local_average_f0(
                            item_hidden, self.config, threshold
                        )
                    results[result_index] = result

        if any(result is None for result in results):
            raise RuntimeError("internal error: missing batch extraction result")
        return results

    def _log_mel(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        if audio.size == 0:
            raise ValueError("audio must contain at least one sample")
        if audio.size <= cfg.window_length:
            raise ValueError(
                "audio is too short after resampling; "
                f"expected more than {cfg.window_length} samples, got {audio.size}"
            )

        magnitude = np.abs(
            librosa.stft(
                audio,
                n_fft=cfg.window_length,
                hop_length=cfg.hop_length,
                win_length=cfg.window_length,
                window="hann",
                center=cfg.center,
                pad_mode=cfg.stft_pad_mode,
            )
        )
        mel = self._mel_basis @ magnitude
        return np.log(np.clip(mel, 1e-5, None)).astype(np.float32, copy=False)


def _config_from_metadata(metadata: dict[str, str]) -> RMVPEOnnxConfig:
    required = {
        "sample_rate",
        "n_mels",
        "window_length",
        "hop_length",
        "mel_fmin",
        "mel_fmax",
        "n_class",
        "cents_const",
        "center",
        "stft_pad_mode",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"ONNX model is missing RMVPE metadata: {joined}")

    return RMVPEOnnxConfig(
        sample_rate=int(metadata["sample_rate"]),
        n_mels=int(metadata["n_mels"]),
        window_length=int(metadata["window_length"]),
        hop_length=int(metadata["hop_length"]),
        mel_fmin=float(metadata["mel_fmin"]),
        mel_fmax=float(metadata["mel_fmax"]),
        n_class=int(metadata["n_class"]),
        cents_const=float(metadata["cents_const"]),
        center=metadata["center"].lower() == "true",
        stft_pad_mode=metadata["stft_pad_mode"],
    )


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1D array")
    return audio.astype(np.float32, copy=False)


def _normalize_sample_rates(
    sample_rate: int | Sequence[int], length: int
) -> list[int]:
    if isinstance(sample_rate, int):
        return [sample_rate] * length
    sample_rates = list(sample_rate)
    if len(sample_rates) != length:
        raise ValueError("sample_rate sequence must match audios length")
    return sample_rates


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample_rate values must be positive")
    if audio.size == 0 or source_rate == target_rate:
        return audio
    return librosa.resample(
        audio, orig_sr=source_rate, target_sr=target_rate, res_type="scipy"
    ).astype(np.float32, copy=False)


def _pad_frames_to_multiple(mel: np.ndarray, *, multiple: int) -> np.ndarray:
    frame_count = mel.shape[-1]
    padded_count = multiple * ((frame_count - 1) // multiple + 1)
    pad_count = padded_count - frame_count
    if pad_count == 0:
        return mel
    mode = "reflect" if frame_count > 1 else "edge"
    return np.pad(mel, ((0, 0), (0, pad_count)), mode=mode)


def _to_local_average_f0(
    hidden: np.ndarray,
    config: RMVPEOnnxConfig,
    threshold: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    hidden = hidden[0]
    bins = np.arange(config.n_class, dtype=np.float32)
    cents_mapping = bins * 20.0 + config.cents_const

    if center is None:
        center = np.argmax(hidden, axis=1)
    f0 = np.zeros(hidden.shape[0], dtype=np.float32)

    for frame_index, center_bin in enumerate(center):
        start = max(0, int(center_bin) - 4)
        end = min(config.n_class, int(center_bin) + 5)
        weights = hidden[frame_index, start:end]
        if weights.size == 0 or float(np.max(hidden[frame_index])) < threshold:
            continue
        weight_sum = float(np.sum(weights))
        if weight_sum == 0:
            continue
        cents = float(np.sum(weights * cents_mapping[start:end]) / weight_sum)
        f0[frame_index] = 10.0 * (2.0 ** (cents / 1200.0))

    return f0


def _to_viterbi_f0(
    hidden: np.ndarray,
    config: RMVPEOnnxConfig,
    threshold: float,
) -> np.ndarray:
    salience = hidden[0].astype(np.float64, copy=False)
    probability = salience.T / np.maximum(salience.T.sum(axis=0, keepdims=True), 1e-12)
    center = librosa.sequence.viterbi(
        probability, _viterbi_transition(config.n_class)
    ).astype(np.int64)
    return _to_local_average_f0(hidden, config, threshold, center=center)


def _viterbi_transition(n_class: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(n_class), np.arange(n_class))
    transition = np.maximum(30 - np.abs(x - y), 0).astype(np.float64)
    return transition / transition.sum(axis=1, keepdims=True)
