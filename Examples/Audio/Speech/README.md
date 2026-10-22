# Speech

Transcribe a short clip of speech with Whisper, through any OAAX-compliant
runtime.

This example deliberately breaks the pattern of the others: it is **not one
inference**. Whisper is two models, and the second one runs in a loop:

1. The **encoder** runs once. It takes the whole clip as a log-mel
   spectrogram (`[1, 80, 3000]`: 80 frequency bands x 30 s in 10 ms steps)
   and returns 1500 audio vectors (`[1, 1500, 384]`).
2. The **decoder** runs once per word piece. Each step it is fed every token
   written so far plus the audio vectors, and returns scores for every
   possible next token at every position (`[1, tokens, 51864]`); the script
   takes the highest score at the last position, appends it, and repeats
   until the model emits `<|endoftext|>`. A sentence is 20-30 steps.

The script keeps the decoder loop as plain as possible: no key/value cache
(the export that reuses past attention states is a separate, more involved
file), no beam search, no timestamps. `transcribe.py` works with any runtime
library; everything about the model comes from `../Models/Whisper-Tiny-English/`:

| File | Purpose |
| --- | --- |
| `encoder.onnx`, `decoder.onnx` | Whisper `tiny.en` (OpenAI, MIT), 8-bit quantised ONNX exports from Hugging Face's `Xenova` org - not checked in, fetched by `download_model.py` |
| `tokenizer.json` | the model's own tokenizer, to turn its word-piece ids back into text - also fetched by `download_model.py` |
| `mel_filters.npy` | the model's mel filterbank (`[201, 80]`), so the spectrogram needs only a numpy FFT - derived by `download_model.py` from OpenAI's own published filterbank |
| `model.json` | which file is which, the start/end tokens, and the audio settings |
| `download_model.py` | fetches/derives the three files above |

The bundled `speech.wav` is utterance `1272-128104-0000` from the LibriSpeech
corpus (Panayotov et al., 2015; CC BY 4.0), dev-clean split, 5.9 s. Its
reference transcript is
*"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"*.

## Setup

```bash
pip install -e ".[examples]"                        # from the repo root: adds the tokenizers package
python ../Models/Whisper-Tiny-English/download_model.py   # fetches the model/tokenizer, derives the mel filterbank
```

None of those files are checked into the repo; the script fetches/derives
them and is safe to re-run, it skips anything already present.

## Run

```bash
python transcribe.py <runtime_library> <model_directory>
```

The model argument is a *directory* here rather than a file, since there are
two model files. For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python transcribe.py ../../../RuntimeDownloads/CPU/libRuntimeLibrary.so ../Models/Whisper-Tiny-English
```

Useful flags:

- `--audio path/to.wav` - transcribe a different clip. Mono, 16-bit PCM, 16 kHz; anything else is refused rather than resampled. Clips longer than 30 s are cut to the first 30 s.
- `--max-tokens 128` - safety stop if the model never emits its end token.
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.

## Model metadata

```json
{
  "name": "Whisper-Tiny-English",
  "encoder": "encoder.onnx",
  "decoder": "decoder.onnx",
  "tokenizer": "tokenizer.json",
  "start_tokens": ["<|startoftranscript|>", "<|notimestamps|>"],
  "end_token": "<|endoftext|>",
  "audio": { "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "chunk_seconds": 30, "mel_filters": "mel_filters.npy" }
}
```

`start_tokens` is the prompt every transcript begins with: "start of
transcript" and "don't emit timestamps". The multilingual Whisper variants
would add a language and a task token here; the English-only `tiny.en` has
neither. The `audio` block drives `audio_utils.log_mel_spectrogram()`: 16 kHz
samples, a 400-sample window every 160 samples (25 ms every 10 ms), 30 s per
chunk, and the filterbank file. The tensor names (`input_features`,
`last_hidden_state`, `input_ids`, `encoder_hidden_states`, `logits`) are the
ones every Hugging Face Whisper export uses, so the script uses them directly.

## Output

```
Audio: speech.wav, 5.9 s
Runtime: OAAX CPU Runtime v2.0.0
Model: Whisper-Tiny-English - encoder encoder.onnx, decoder decoder.onnx

Transcript (22 tokens, 23 decoder runs):
  Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
```

Whisper adds its own casing and punctuation, which is why this differs from
LibriSpeech's all-caps reference while saying the same thing. Note that this
decoder export also returns its attention maps and key/value states on every
step (25 output tensors); the script only reads `logits` and lets the rest go.
