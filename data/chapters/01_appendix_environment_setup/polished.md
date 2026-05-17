## Environment Setup and Dependencies

Before writing a single line of model code, you need a working environment. This chapter walks through every library you will install, explains why each one is needed, and gives you concrete guidance on which hardware to expect reasonable runtimes from. Get this right once and the rest of the book runs smoothly.

### Python Libraries Overview

The examples in this book rely on a small, focused set of libraries. Each one earns its place:

- **PyTorch** — the primary deep-learning framework used throughout for model definition, training, and inference.
- **tiktoken** — OpenAI's fast tokenizer library. Implementing Byte-Pair Encoding (BPE) from scratch is relatively complicated, so the tutorials use tiktoken instead [c2487]. At the time of writing, tiktoken has approximately 11,000 stars and 780 forks on its repository [c2489], reflecting broad community adoption.
- **TensorFlow** — needed only for one specific task: loading the original GPT-2 weights. OpenAI originally saved those weights in TensorFlow format, while the lecture-series code uses PyTorch [c3469], so TensorFlow must be installed to bridge the two [c3470].
- **tqdm** — a lightweight progress-bar library installed to track the download progress of the GPT-2 weights [c3471].
- **pandas** — used for data handling in later fine-tuning chapters.
- **Ollama** — a separate application (not a Python package) for running pre-trained LLMs locally; covered in its own section below.

### Installing the Core Libraries

#### Version Requirements

Two libraries have explicit minimum version requirements:

- TensorFlow ≥ 2.15 [c3472]
- tqdm ≥ 4.66 [c3473]

For PyTorch, follow the official installation matrix at pytorch.org and select the build that matches your operating system and CUDA version (if applicable).

#### Tokenizer Setup

Once tiktoken is installed, initializing the GPT-2 tokenizer takes two lines:

```python
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
```
[c2490]

This downloads a small vocabulary file on first run and caches it locally; all subsequent calls are instantaneous.

### Device Selection: CPU, CUDA, and Apple MPS

PyTorch supports three device backends relevant to this book: CPU, CUDA (NVIDIA GPUs), and MPS (Apple silicon) [c2307]. A standard one-liner selects CUDA when available and falls back to CPU otherwise:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
[c2309]

For MacBooks with Apple silicon, the device can instead be set to `torch.device('mps')` for faster execution [c2310]. MPS offloads tensor operations to the GPU cores inside the M-series chip, giving a meaningful speedup over the CPU path without requiring an NVIDIA card.

#### Why Device Transfer Placement Matters

When building data-loading pipelines, *where* you move tensors to the target device is not merely a style choice — it has a real performance consequence. Performing the device transfer inside the custom collate function (outside the training loop) allows it to happen as a background process, preventing it from blocking the GPU during model training [c2308]. The custom collate function therefore includes code to move input and target tensors to the appropriate device (CPU, CUDA, or MPS) at data-loading time [c2306].

### Hardware Expectations and Realistic Runtimes

One of the most common frustrations when starting out is underestimating how long training takes. The table below summarizes concrete timings from the source material so you can plan accordingly.

| Task | Hardware | Time |
|---|---|---|
| Pre-training a small GPT model (1 epoch) | MacBook Air 2020, 8 GB RAM, CPU | ~2 hours [c3136] |
| Pre-training a small GPT model (short run) | MacBook Air 2020, CPU | ~8.8 minutes [c115] |
| Pre-training on i5/i7 or comparable MacBook | CPU | 7–10 minutes [c115] |
| Fine-tuning GPT-2 medium | Author's PC (GPU) | ~4 hours [c642] |
| Running the full evaluation function | MacBook Air M3 | ~1 minute [c688] |

A few points are worth noting. Training for 2 epochs on a CPU-only machine would take 4 to 5 hours and risk crashing the system [c3137] — a strong reason to either use a GPU or keep experimental runs short. Apple silicon is approximately 2× faster than Apple CPU [c3861], so an M-series MacBook is a meaningful upgrade over an Intel-based one for this workbook. The examples were developed and tested on a MacBook Air 2020 with 8 GB of RAM running on CPU [c3135], so the code is deliberately written to be accessible even without dedicated hardware.

### Ollama: Running Large Models Locally

Later chapters explore instruction following and evaluation using larger pre-trained models. For that, Ollama is the recommended tool.

#### What Ollama Is (and Is Not)

Ollama is an efficient application for running large language models on your laptop [c648]. It generates text via LLM inference but does not support training or fine-tuning [c657]: when you use Ollama, the model is already pre-trained and you are performing inference only [c653]. Keep this distinction clear — Ollama is a runtime, not a training framework.

#### Supported Models

Ollama supports several well-known open models out of the box, all runnable on a laptop [c667]:

- **LLAMA 3**, developed by Meta [c649]
- **Phi 3**, developed by Microsoft [c650]
- **Mistral** [c651]
- **Gemma 2** [c652]

#### Installation

Ollama is available for Mac OS, Linux, and Windows [c658]. The Mac OS installer is 177 MB [c659], so the download itself is quick. The larger cost comes from pulling the model weights afterward: the LLAMA 3 model files are 4.7 GB [c664], and downloading the 8-billion-parameter model takes approximately 15 to 20 minutes [c665].

#### Running Ollama

On Mac, start the model with a single command [c661]:

```
ollama run llama3
```
[c660]

On Windows, run `ollama serve` first, then `ollama run llama3` [c663]:

```
ollama serve
```
[c662]

When LLAMA 3 loads successfully, right-hand-side arrows appear in the terminal indicating the model is ready for interaction [c666]. One important operational note: Ollama must remain running in the terminal for any code that depends on it — closing that terminal window will stop Ollama [c668].

#### Interacting via Python

Typing prompts directly into the terminal is fine for quick tests, but the evaluation pipelines in later chapters require programmatic access. Rather than invoking `ollama run` in the terminal each time, you can interact with the model through a Python API [c670]. A short check confirms the session is live by verifying that `ollama_running` equals true [c669].

### Summary

With the libraries installed and your device configured, you are ready to start building. The key points to carry forward:

1. Install tiktoken, TensorFlow (≥ 2.15), tqdm (≥ 4.66), and PyTorch before running any notebook.
2. Use `torch.device('mps')` on Apple silicon for a roughly 2× speedup over CPU; use `torch.device('cuda')` on NVIDIA hardware.
3. Move tensors to the target device inside the collate function, not inside the training loop, to avoid blocking the GPU.
4. Expect multi-hour runtimes for fine-tuning on CPU; plan experiments accordingly.
5. Ollama handles inference for large pre-trained models locally — it does not train or fine-tune.