# Attribution

This project packages RMVPE for ONNXRuntime inference.

The PyTorch RMVPE export source used here was copied from
[Stylish-TTS/stylish-tts](https://github.com/Stylish-TTS/stylish-tts). Stylish
TTS downloads its RMVPE weights from its Hugging Face repository:
[stylish-tts/pitch_extractor](https://huggingface.co/stylish-tts/pitch_extractor),
specifically `rmvpe.safetensors`.

The Stylish TTS RMVPE implementation appears to be derived from the original
[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE/) code. The commonly used
PyTorch RMVPE checkpoint is available as
[lj1995/VoiceConversionWebUI/rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt).

RMVPE is based on the paper
[RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://arxiv.org/abs/2306.15412).
