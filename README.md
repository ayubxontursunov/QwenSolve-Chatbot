# 🤖 Qwen-Math Chatbot · Fine-tuned with LoRA on Math Dataset

[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)](https://github.com/your-username)
[![LoRA Fine-tuning](https://img.shields.io/badge/PEFT-LoRA-blue)](https://github.com/huggingface/peft)
[![Model: Qwen 0.6B](https://img.shields.io/badge/Model-Qwen--0.6B-yellow)](https://huggingface.co/Qwen)

> **A lightweight, math-solving chatbot fine-tuned locally using LoRA for rapid inference and educational applications.**

---

## 🧠 Project Highlights

🚀 **Fast Local Training**  
💡 **LoRA Fine-tuning (Low-Rank Adaptation)**  
📊 **Math-focused Chatbot**  
🛠️ **Runs on CPU (No GPU required)**  
💬 **Interactive Terminal Interface**  
📦 **All model files saved locally**

---

## 🚀 Quick Start

### 1. Clone & Install

📥 Clone the project repository from GitHub
```bash
git clone https://github.com/your-username/qwen-math-chatbot.git
```

📂 Navigate into the project directory
```bash
cd qwen-math-chatbot
```
🐍 Create a new virtual environment (recommended)
```bash
python -m venv .venv
```

✅ Activate the virtual environment (for Windows)
```bash
.venv/Scripts/activate
```
📦 Install all required Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training Script
```bash
python main.py
```
Downloads the base Qwen-0.6B model

Loads a 200-sample math dataset

Applies LoRA fine-tuning (~10–20 mins on CPU)

Saves to ./qwen_math_finetuned/

### 3. Launch Chatbot
```bash
python chatbot.py
```

🤖 Qwen Math Chatbot
=============================
> Solve: 2x + 4 = 10

🤖 Assistant: x = 3

## 🛠 Project Structure


├── local_base_model/          # Cached Qwen base model<br>
├── qwen_math_finetuned/      # Fine-tuned adapter via LoRA<br>
├── main.py                   # Training script (LoRA)<br>
├── chatbot.py                # Interactive CLI chatbot<br>
├── requirements.txt          # Required dependencies<br>
└── README.md

## 🧪 Example Questions
- Solve: 3x + 2 = 11

- Derivative of: x^2 + 3x

- Factor: x^2 - 9

- Evaluate: sqrt(144)

- Area of a circle with radius 7

## ⚙️ LoRA Configuration

```python
LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

## 📈 Training Summary


| Property             | Value                  |
|----------------------|------------------------|
| **Base Model**       | Qwen 0.6B              |
| **Epochs**           | 1                      |
| **Max Training Steps** | 50                   |
| **Samples Used**     | 200                    |
| **Device**           | CPU                    |
| **Output Directory** | `./qwen_math_finetuned/` |


## 📚 References
Qwen Language Models

Hugging Face PEFT (LoRA)

Hugging Face Transformers

KaggleHub


## 👨‍💻 Author
Ayyubkhon Tursunov
🎓 Trained in AI, Machine Learning, and Python <br>
📍 Uzbekistan