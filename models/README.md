# Model Versioning & Checkpoint Management

## Naming Convention

```
models/
├── text2cypher_adapter/          # Stage 1: QLoRA instruction tuning
│   ├── checkpoint-{step}/        # Intermediate checkpoints
│   └── final/                    # Best checkpoint (by compile rate)
├── text2cypher_dpo/              # Stage 2: DPO alignment
│   ├── checkpoint-{step}/
│   └── final/
├── react_adapter/                # ReAct format adapter
│   ├── checkpoint-{step}/
│   └── final/
├── stt/                          # Local faster-whisper speech-to-text models
│   └── faster-whisper-small/     # Example STT_MODEL_PATH target
└── README.md
```

## Checkpoint Naming

Format: `{model_name}-v{major}.{minor}-{base_model}-{date}`

Examples:
- `text2cypher-v1.0-qwen2.5-7b-20240601`
- `text2cypher-dpo-v1.1-qwen2.5-7b-20240615`
- `react-v1.0-qwen2.5-7b-20240620`

## Version Tracking

| Version | Stage | Base Model | Training Data | Key Metric | Date |
|---------|-------|------------|---------------|------------|------|
| v1.0 | Text2Cypher SFT | Qwen2.5-7B-Instruct | 12K pairs | Compile rate | - |
| v1.1 | Text2Cypher DPO | v1.0 | 24K DPO pairs | 0% fluff | - |
| v1.0 | ReAct Format | Qwen2.5-7B-Instruct | 3K traces | JSON valid % | - |

## Training Hyperparameters (Reference)

```yaml
# QLoRA config
quantization: nf4 (BitsAndBytes)
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
optimizer: paged_adamw_8bit
learning_rate: 2e-4
warmup_ratio: 0.03
max_seq_length: 2048
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 8
```

## Evaluation Criteria

### Text2Cypher Adapter
- **Cypher compilation rate:** >90% (target)
- **Schema accuracy:** correct node labels and relationship types
- **No preamble/fluff:** 0% conversational text in output

### ReAct Adapter
- **JSON action validity:** >95%
- **Tool name accuracy:** correct tool selection
- **Iteration cap compliance:** stops within 6 iterations

## How to Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load adapter
model = PeftModel.from_pretrained(base_model, "models/text2cypher_adapter/final")
```

## .gitignore Note

Model checkpoints are large (>1GB). They are NOT committed to git.
Upload final adapters to HuggingFace Hub instead.

## Local STT Model

Speech-to-text can use a local faster-whisper export by setting:

```env
STT_MODEL_PATH=models/stt/faster-whisper-small
```

Expected directory shape:

```text
models/stt/faster-whisper-small/
  config.json
  tokenizer.json
  model.bin
  vocabulary.*
```

If `STT_MODEL_PATH` is empty, the backend falls back to `STT_MODEL_SIZE` and may try to
download a model at runtime.
