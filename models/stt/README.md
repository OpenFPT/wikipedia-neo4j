# Local Speech-to-Text Models

Put local faster-whisper model directories here.

Recommended default:

```text
models/stt/faster-whisper-small/
  config.json
  tokenizer.json
  model.bin
  vocabulary.*
```

Then set:

```env
STT_MODEL_PATH=models/stt/faster-whisper-small
```

This repository only creates the directory structure. The actual model files are not committed.
