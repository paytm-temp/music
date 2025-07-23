# Hindi LORA Fine-tuning for ACE-Step

This repository contains the necessary files for fine-tuning ACE-Step on Hindi/Hinglish lyrics using LORA.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Directory Structure:
- `acestep/`: Core model files and utilities
  - `models/lyrics_utils/`: Hindi/Hinglish text processing
  - `language_segmentation/`: Language detection
- `config/`: LORA configuration
- `examples/`: Example input parameters

## Training

1. Prepare your Hindi/Hinglish dataset in the following format: 