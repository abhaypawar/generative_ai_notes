## ✅ Native Ollama Models (Run via `ollama run <model>`)
These are included in Ollama's built-in repository:
- **Mistral (7B)** — `ollama run mistral` *(Mistral AI)*
- **Neural Chat (7B)** — `ollama run neural-chat` *(Intel fine‑tuned on Mistral)*
- **Starling (7B)** — `ollama run starling-lm` *(UC Berkeley LLM)*
- **Solar (10.7B)** — `ollama run solar` *(Upstage domain-tuned model)*
- **Llama 2 Uncensored (7B)** — `ollama run llama2-uncensored` *(Meta-based, uncensored variant)*
- **Gemma 2B** — `ollama run gemma:2b` *(Google “lite” variant)*
- **Gemma 7B** — `ollama run gemma:7b`
- **Code Llama (7B)** — `ollama run codellama` *(Meta’s code-specialized Llama)*
- **LLaVA** — `ollama run llava` *(Microsoft multimodal vision + language assistant)*

---

## 🌐 Hugging Face GGUF Models (Run via `ollama run hf.co/...`)
Ollama supports running **any GGUF-formatted LLM on Hugging Face** without manual conversion, offering access to tens of thousands of models :contentReference[oaicite:1]{index=1}:

### Examples of popular GH models:
- `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF[:quant]`
- `hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-GGUF`
- `hf.co/bartowski/Humanish-Llama3-8B-Instruct-GGUF`
- `hf.co/unsloth/DeepSeek-R1-7B-GGUF`
- `hf.co/unsloth/DeepSeek-R1-32B-GGUF`
- `hf.co/unsloth/DeepSeek-R1-671B-GGUF` *(full DeepSeek R1 series)* :contentReference[oaicite:2]{index=2}

### Google Gemma expanded:
The **Gemma 3** family is available and already integrated into Ollama (sizes: 1B, 4B, 12B, 27B) :contentReference[oaicite:3]{index=3}.

### Microsoft Phi‑3 Mini:
Phi‑3 Mini (3.8B) is listed on Hugging Face and also available to run through Ollama :contentReference[oaicite:4]{index=4}.

---

## 📋 Summary Table

| Category            | Model Family               | Available Sizes         | Usage Command (CLI)                                     |
|---------------------|----------------------------|--------------------------|----------------------------------------------------------|
| **Built-in Ollama** | Mistral                    | 7B                       | `ollama run mistral`                                     |
|                     | Neural Chat                | 7B                       | `ollama run neural-chat`                                 |
|                     | Starling                   | 7B                       | `ollama run starling-lm`                                 |
|                     | Solar                      | 10.7B                    | `ollama run solar`                                       |
|                     | Llama 2 Uncensored         | 7B                       | `ollama run llama2-uncensored`                           |
|                     | Gemma (Google)             | 2B / 7B                  | `ollama run gemma:2b` or `gemma:7b`                       |
|                     | Code Llama                 | 7B                       | `ollama run codellama`                                   |
|                     | LLaVA                      | Vision + Language        | `ollama run llava`                                       |
| **Hugging Face GGUF** | Llama 3.x Instruct       | 1B, 3B, 8B+              | `ollama run hf.co/...`                                   |
|                     | DeepSeek R1                | 1.5B, 7B, 8B, 14B, 32B, 70B, 671B | `ollama run deepseek-r1:<size>`                      |
|                     | Gemma 3                    | 1B, 4B, 12B, 27B         | `ollama run hf.co/...Gemma-3.x‑GGUF`                     |
|                     | Phi-3 Mini                 | 3.8B                     | `ollama run hf.co/microsoft/Phi-3-Mini-GGUF`             |
|                     | Any GGUF model on HF Hub   | Varies (0.1B–405B+)      | `ollama run hf.co/{username}/{repo}[:quant]`             |

---

## 🛠 How to Run a Hugging Face GGUF Model with Ollama

```bash
# Default quantization (Q4_K_M is used if available)
ollama run hf.co/<username>/<model-repo>

# Specify a quantization scheme explicitly
ollama run hf.co/<username>/<model-repo>:Q8_0

    Ollama auto-downloads the GGUF file, handles model metadata, and configures inference automatically
    arXiv+15Marc Julian Schwarz+15academy.finxter.com+15
    docs.salad.com+15Reddit+15academy.finxter.com+15
    docs.salad.com+12academy.finxter.com+12Reddit+12
    arXiv
    Reddit
    Reddit
    Gist+1Reddit+1
    Rise-+1docs.salad.com+1
    Reddit
    .

    Models range from tiny (~0.1B) to massive (>70B), depending on hardware compatibility.

📝 Notes & Lookup Tips

    Ollama supports only GGUF format, which is llama.cpp–compatible and ideal for efficient quantized inference
    Marc Julian Schwarz+9Reddit+9Reddit+9
    .

    You can even run private GGUF repos by registering your SSH key on Hugging Face, then pointing ollama run hf.co/yourname/…
    Medium+15Reddit+15Gist+15
    .

    There are 45K+ GGUF models publicly available on Hugging Face—covering text-to-code, instruction variants, RAG, embedding models, etc.
    Medium+9docs.salad.com+9Reddit+9
    .

🧭 TL;DR

    Built-in Ollama models include Mistral, Neural Chat, Starling, Solar, Llama 2 Uncensored, Gemma 2/7B, Code Llama, and LLaVA.

    For any GGUF model on Hugging Face, simply use the ollama run hf.co/... syntax to load and interact locally.

    Some standout Hugging Face models already supported include Llama 3.x instruct variants, DeepSeek R1 series, Gemma 3, and Microsoft’s Phi‑3 Mini.
