## Environment Setup: PyTorch, TikToken, TensorFlow, and Ollama

Before writing a single line of model code, you need a working environment. This chapter walks through every tool you will need throughout the book: PyTorch for tensor operations and training, tiktoken for tokenization, TensorFlow (briefly) for loading pre-trained GPT-2 weights, tqdm for download progress tracking, and Ollama for running open-source LLMs locally during evaluation. Getting these pieces in place now will save you from interruptions later.

### Device Selection: CPU, CUDA, and Apple Silicon

One of the first decisions your code must make at runtime is which hardware device to use for computation. PyTorch supports three backends relevant here: CPU, CUDA (for NVIDIA GPUs), and MPS (Metal Performance Shaders, for Macs with Apple Silicon chips) [c2307].

The standard one-liner for selecting between CPU and CUDA is:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
[c2309]

On a MacBook with an Apple Silicon chip, you can go one step further and set the device to `torch.device('mps')` for faster execution [c2310]. Apple Silicon is approximately 2× faster than the Apple CPU for these workloads [c3861], making the extra line well worth it if you have the hardware.

The `device` variable is passed to every tensor and model call so that all computation happens in the same memory space. Mixing devices — for example, keeping your model on the GPU while your data stays on the CPU — is a common source of runtime errors, so establishing a consistent device-selection strategy early pays dividends throughout the project.

#### Moving Data to the Device During Loading

A subtle but important optimization concerns *when* you transfer tensors to the target device. The recommended approach is to perform the transfer inside a custom collate function, which runs as part of the data-loading pipeline rather than inside the training loop [c2306, c2308]. Because the collate function executes in background worker processes, the transfer happens concurrently with the forward and backward passes, preventing it from blocking the GPU during model training [c2308].

### TensorFlow and tqdm: Loading GPT-2 Weights

Later in the book you will load the original GPT-2 weights released by OpenAI. There is a small compatibility wrinkle: OpenAI originally saved those weights using TensorFlow, while the code in this series uses PyTorch [c3469]. The weights therefore cannot be loaded directly with `torch.load`, so TensorFlow must be installed as an intermediary [c3470].

You will need the following minimum versions:

- **TensorFlow** ≥ 2.15 [c3472]
- **tqdm** ≥ 4.66 [c3473]

tqdm is installed specifically to track the download progress of the weight files [c3471], which can be several gigabytes. A progress bar is a small quality-of-life improvement that makes it clear the download has not stalled.

[GAP: explicit pip/conda install commands for PyTorch, tiktoken, TensorFlow, and tqdm are not present in the supplied claims.]

### Ollama: Running LLMs Locally

Once you have trained or fine-tuned a model, you will want to compare its outputs against a strong reference. Ollama is a lightweight application designed to run large language models on a laptop [c648], and it supports several well-known open-source models:

- **Llama 3** by Meta [c649]
- **Phi 3** by Microsoft [c650]
- **Mistral** [c651]
- **Gemma 2** [c652]

One important distinction: Ollama is an inference-only tool [c657]. It works with already pre-trained models and does not support training or fine-tuning [c653, c657]. Think of it as a convenient local server for querying models, not a training framework.

#### Installing Ollama

Ollama is available for Mac OS, Linux, and Windows [c658]. The Mac OS installer weighs in at 177 MB [c659], so the application itself downloads quickly. The model files are a different story — the Llama 3 8-billion-parameter model is 4.7 GB [c664] and takes roughly 15 to 20 minutes to download on a typical connection [c665].

#### Running Ollama on Mac

On Mac, a single terminal command downloads (on first run) and starts the model:

```
ollama run llama3
```
[c660, c661]

When Ollama has successfully loaded Llama 3, right-hand-side arrows appear in the terminal, indicating the model is ready for interaction [c666].

#### Running Ollama on Windows

On Windows the process requires two commands in sequence: first start the server, then launch the model [c663].

```
ollama serve
```
[c662]

Then, in the same or a second terminal:

```
ollama run llama3
```
[c660, c663]

#### Keeping Ollama Running

One operational detail to keep in mind: Ollama must remain running in the terminal for any downstream Python code to reach it [c668]. Closing the terminal window stops the server. A good habit is to open a dedicated terminal tab for Ollama and leave it untouched while you work in other tabs.

#### Talking to Ollama from Python

Typing prompts directly in the terminal is fine for quick checks, but systematic evaluation calls for a programmatic approach. Rather than invoking `ollama run` each time, the recommended method is to interact with the model through an API from Python [c670]. You can verify that the Ollama session is reachable by checking whether `ollama_running` equals `true` in your verification code [c669].

[GAP: the specific Python API calls (e.g., the requests or ollama Python package usage) for interacting with Ollama programmatically are not present in the supplied claims.]

### Summary

With this setup complete, you have everything you need to move forward:

| Tool | Purpose | Key version / note |
|---|---|---|
| PyTorch | Tensors, training, inference | Device: CPU / CUDA / MPS |
| tiktoken | Tokenization | [GAP: version not specified in claims] |
| TensorFlow | Loading GPT-2 weights | ≥ 2.15 [c3472] |
| tqdm | Download progress bars | ≥ 4.66 [c3473] |
| Ollama | Local LLM inference for evaluation | Inference only [c657] |

The next chapter begins building the model itself. Having your device selection sorted out now means you can pass `device` into model and data-loader code without revisiting this setup.