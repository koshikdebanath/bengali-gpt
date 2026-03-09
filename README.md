# Bengali GPT for Poetry Generation

A lightweight **GPT-style language model trained from scratch in PyTorch** on a collection of Bengali poems.
The project includes a **custom Byte Pair Encoding (BPE) tokenizer**, a **decoder-only transformer architecture**, and training scripts for building Bengali language models.

The goal of this project is to explore **low-resource language modeling for Bengali** using a compact transformer architecture.

---

## Project Overview

This repository contains:

* A GPT-style transformer model implemented in **pure PyTorch**
* A **custom BPE tokenizer** trained on Bengali poetry
* Training scripts for language model training
* A pretrained model checkpoint
* Utilities for generating Bengali poetry

The model learns the statistical structure of Bengali poetry and can generate new poetic text given a prompt.

---

## Model Details

| Parameter       | Value                          |
| --------------- | ------------------------------ |
| Architecture    | GPT (decoder-only transformer) |
| Parameters      | ~12.67M                        |
| Layers          | 6                              |
| Attention Heads | 6                              |
| Embedding Size  | 384                            |
| Context Length  | 256 tokens                     |
| Vocabulary Size | 5000 (BPE)                     |
| Framework       | PyTorch                        |

---

## Dataset

The model was trained on the **Free Bengali Poetry dataset**, which contains public-domain Bengali poems.

Dataset information:

* **Name:** Free Bengali Poetry
* **Poems:** 2,686
* **Language:** Bengali
* **Source:** Kaggle

Dataset link:

https://www.kaggle.com/datasets/truthr/free-bengali-poetry

### Dataset Citation

```
@misc{ritobrata ghosh_2021,
author = {Ritobrata Ghosh},
year = {2021},
title = {Free Bengali Poetry},
publisher = {Kaggle},
address = {Kolkata, India}
}
```

---

## Repository Structure

```
bengali-gpt-poetry/
│
├── train_bengali_gpt_v1.py     # Training script
├── best_model.pt               # Trained model weights 
├── bpe_tokenizer.pkl           # Serialized BPE tokenizer
├── dataset/                    # Optional dataset directory
│
└── README.md
```

**Note**: Becuase of large file size > **25MB** please download the model from here: [Pre-trained-model-HF](https://huggingface.co/koshikdebanath/bengali-gpt-poetry/blob/main/best_model.pt)

---

## Installation

Clone the repository.

```
git clone https://github.com/koshikdebanath/bengali-gpt-poetry.git
cd bengali-gpt-poetry
```

Install dependencies.

```
pip install torch tqdm numpy
```

---

## Training the Model

To train the model from scratch:

```
python train_bengali_gpt_v1.py
```

Training uses:

* **Optimizer:** AdamW
* **Learning Rate Scheduler:** Cosine decay with warmup
* **Mixed Precision Training (fp16)**
* **Sliding token windows (256 tokens)**

The best checkpoint will be saved as:

```
.checkpoints/best_model.pt
```

---

## Using the Pretrained Model

Example code to generate Bengali poetry:

```python
import torch
import pickle

from train_bengali_gpt_v1 import GPT, BPETokenizer, Config

# Load tokenizer
with open("bpe_tokenizer.pkl", "rb") as f:
    tokenizer_data = pickle.load(f)

tokenizer = BPETokenizer()
tokenizer.__dict__.update(tokenizer_data)

# Create model config
config = Config(
    vocab_size=len(tokenizer.vocab),
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

# Load model
model = GPT(config)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

prompt = "আমার সোনার বাংলা"

input_ids = tokenizer.encode(prompt, add_special_tokens=True)

input_tensor = torch.tensor([input_ids], dtype=torch.long)

with torch.no_grad():
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40
    )

print(tokenizer.decode(output_ids[0].tolist()))
```

---

## Example Prompt

Input:

```
আমার সোনার বাংলা
```

Example generated continuation:

```
আমার সোনার বাংলা  
তোমার আকাশ ভরা আলো  
নদীর স্রোতে ভাসে স্বপ্ন  
মাটির গন্ধে জাগে ভালোবাসা
```

*(Generated text will vary each time.)*

---

## Hugging Face Model

The pretrained model is also available on Hugging Face:

https://huggingface.co/koshikdebanath/bengali-gpt-poetry

---

## Future Improvements

Possible directions for extending this project:

* Train on larger Bengali corpora
* Expand vocabulary size
* Support longer context windows
* Convert to Hugging Face Transformers format
* Fine-tune for Bengali story generation or dialogue
* Build a web demo for interactive poetry generation

---

## License

The dataset and derived model follow the **CC BY-SA 4.0 license**.

---

## Acknowledgments

* Bengali poetry dataset compiled by **Ritobrata Ghosh**
* PyTorch open-source community
* Researchers working on low-resource language modeling

---

## Contact

If you are working on **Bengali NLP, language models, or poetry generation**, feel free to connect or collaborate.
