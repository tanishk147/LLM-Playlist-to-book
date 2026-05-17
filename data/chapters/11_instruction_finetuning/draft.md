## Instruction Fine-Tuning: Building a Personal Assistant

By this point in the book, you have built a GPT-style language model from scratch, pre-trained it on raw text, and loaded weights from OpenAI's publicly released GPT-2 checkpoints. The model can complete sentences and generate coherent prose, but it does not yet behave like an assistant. Ask it "What is the capital of France?" and it is just as likely to continue with more questions as it is to answer. The gap between a raw language model and a useful assistant is bridged by **instruction fine-tuning**, and that is the subject of this chapter.

We will walk through the entire pipeline end to end: formatting data, batching it correctly, loading a pre-trained foundation, running the training loop, extracting responses, and evaluating quality—both by hand and automatically using a second LLM as a judge. By the end you will have a working personal assistant built on GPT-2 medium (355 M parameters).

The pipeline breaks naturally into three stages [c587]:

- **Stage 1 – Dataset preparation:** download the instruction dataset, format it with the Alpaca prompt template, tokenize, pad, and wrap everything in PyTorch `DataLoader` objects.
- **Stage 2 – Fine-tuning:** load the pre-trained GPT-2 weights and run the supervised training loop.
- **Stage 3 – Evaluation:** extract model responses, inspect them qualitatively, and score them automatically using a locally running LLaMA-3 model via Ollama.

Let's begin.

---

### What Is Instruction Fine-Tuning?

A **foundation model** is a pre-trained large language model that has been trained on large-scale data and can be fine-tuned for downstream tasks [c169]. The pre-training phase teaches the model grammar, facts, and reasoning patterns, but it does not teach the model to *follow instructions*. That second step is instruction fine-tuning.

**Instruction fine-tuning** (also called supervised instruction fine-tuning) is the process of training a pretrained LLM with a specific dataset to teach it to correctly follow instructions [c1722]. More precisely, it is the approach of providing a large set of instruction-response pairs to train a language model to follow instructions [c181]. Fine-tuning on a specific dataset after loading pre-trained weights requires significantly less computational time than training from scratch [c1733], which is why we always start from a pre-trained checkpoint rather than random weights.

Fine-tuning the LLM on instruction data modifies the weights and parameters of the GPT model to understand instruction input-output pairs from a specific dataset [c1761]. Intuitively, the model already knows *how* to predict the next token; we are now steering it so that the tokens it predicts form helpful responses rather than arbitrary continuations.

Here are three concrete examples of the kind of instruction-response pairs we will train on:

```
Instruction: convert 45 km to meters
Response: 45 km is 45000 meters
```
[c178]

```
Instruction: provide a synonym for bright
Response: a synonym for bright is radiant
```
[c179]

```
Instruction: edit the following sentence to remove all passive voice.
             The song was composed by the artist
Response: the artist composed the song
```
[c180]

These examples illustrate the breadth of tasks a single fine-tuned model can handle: unit conversion, vocabulary, and grammar editing—all from the same model, trained on the same dataset.

---

### Stage 1: Dataset Preparation

#### The Instruction Dataset

The instruction dataset used for fine-tuning in this chapter consists of 1,100 instruction input and output pairs [c588]. For reference, the Stanford Alpaca repository contains a dataset of 52,000 instruction-output pairs [c695]; our 1,100-pair dataset is a smaller, self-contained version suitable for experimentation.

Each instruction-response pair in the dataset contains three keys: `instruction`, `input`, and `output` [c188]. The `instruction` field describes the task, the `input` field provides optional context, and the `output` field contains the expected response.

The fine-tuning workflow includes: fine-tuning the LLM, inspecting the loss, extracting responses, performing evaluation, and scoring [c186]. But before any of that, we need to get the data into a form the model can consume.

Stage 1 specifically involves downloading data, formatting it using the Alpaca prompt format, and creating training, testing, and validation data loaders [c3088].

#### The Alpaca Prompt Format

Raw instruction-input-output triples cannot be fed directly to a language model. We need a consistent text template that tells the model where the instruction ends and where the response should begin. The dominant convention is the **Alpaca prompt format**.

Alpaca prompt style is a specific formatting convention for converting instruction-input-output pairs into prompts for fine-tuning large language models, maintained by Stanford in the Stanford Alpaca repository [c1842]. Stanford Alpaca is a specific prompt format for instruction-following LLaMA models [c193].

The Stanford Alpaca prompt format consists of a template [c196]:

> *Below is an instruction that describes a task paired with an input that provides further context. Write a response that appropriately completes the request.*

followed by the instruction, input, and output fields. In Alpaca prompt style, the instruction and input fields are separated in the prompt template [c199].

There are two main ways to format instruction-input-output datasets: Alpaca prompt style and Phi-3 prompt style [c198]. In Phi-3 prompt style, the instruction and input are fused together in a `user` field, and the output is placed in an `assistant` field [c200]. We will use Alpaca style throughout this chapter because it is more transparent and easier to inspect.

The Alpaca prompt template structures instruction fine-tuning data as [c2291]:

```
Below is an instruction that describes a task paired with an input that
provides further context. Write a response that appropriately completes
the request.

### Instruction:
<instruction text>

### Input:
<input text>

### Response:
<output text>
```

When no `input` context is provided, the `### Input:` section is omitted entirely.

#### Implementing `format_input`

The following function converts a dataset entry into a formatted Alpaca prompt. It handles both the case where an `input` field is present and the case where it is absent.

# Source: [c201]
```python
def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request."
    )
    instruction = entry['instruction']
    input_text = entry.get('input', '')
    if input_text:
        prompt = (
            f"{instruction_text}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}"
        )
    else:
        prompt = (
            f"{instruction_text}\n\n"
            f"Instruction:\n{instruction}"
        )
    return prompt
```

The function returns only the *input* side of the prompt—the instruction and optional context—without the response. During training, the response is appended separately so that the model learns to predict it token by token.

A companion function `formatInput` follows the same logic [c1846]:

# Source: [c1846]
```python
def formatInput(entry):
    # Converts instruction-input-output pairs into alpaca prompt format
    # Takes entry with instruction, input, and output keys
    # Returns formatted prompt with instruction, input, and response sections
```

#### Data Batching: Five Steps

Once the data is formatted, we need to convert it into batches of tensors that PyTorch can process. Batching the dataset means converting multiple data samples into a batch where each sample is represented as a numerical array (row) with uniform dimensions [c1848].

Data batching for instruction fine-tuning consists of five steps [c1875]:

1. **Format** data using the prompt template.
2. **Tokenize** the formatted prompt into token IDs.
3. **Pad** sequences to equal length within each batch.
4. **Create target token IDs** by shifting the input right by one position.
5. **Replace padding tokens** in the targets with `-100`.

Let's examine each step in detail.

#### Step 1: Format

We have already covered this with `format_input`. The Alpaca format is a template used for formatting instruction, input, and output data during instruction fine-tuning [c2289].

#### Step 2: Tokenize

Tokenization is the process of converting data into a numerical representation by converting sentences into token IDs [c1854]. Each formatted prompt string is passed through the GPT-2 tokenizer to produce a sequence of integer token IDs.

#### Step 3: Pad

Tokenized sequences will have different lengths because instructions vary in complexity. Padding is the process of adjusting the length of token ID sequences so that all samples in a batch have the same length [c1861].

The end-of-text token is a special token used for padding sequences; in GPT-2, it has token ID `50256` [c1863]. We append copies of token `50256` to shorter sequences until all sequences in the batch share the same length.

#### Step 4: Create Target Pairs

During instruction fine-tuning, the LLM is trained using next-token prediction, where the entire formatted prompt serves as both input and target pairs [c2292]. Target token IDs are created by shifting the input tokens to the right by one position and padding with an additional token to maintain equal length between input and target sequences [c1868].

Concretely, if the input sequence is `[t₀, t₁, t₂, ..., tₙ]`, the target sequence is `[t₁, t₂, ..., tₙ, <pad>]`. The target tensor is the input tensor shifted to the right by one position [c2323]. Target pairs are created by shifting the input sequence to the right by one token and appending an end-of-text token [c2298].

In code, this shift is expressed as:

# Source: [c1896]
```python
targets = inputs[1:]  # Shift inputs right by one by removing first element
```

#### Step 5: Replace Padding Tokens with -100

Here is a subtle but important detail. After creating the target tensor, the positions that correspond to padding tokens should not contribute to the loss. If we let the model be penalized for failing to predict padding tokens, we are training it to reproduce padding—which is meaningless.

The solution is to replace padding token IDs in the target tensor with the special value `-100`. PyTorch's cross-entropy loss function ignores positions where the target is `-100` by default, so those positions contribute zero gradient.

To mask target token IDs, replace the token IDs of instruction and input portions in the target text with `-100` [c1917]. The custom collate function implements this masking:

# Source: [c1902]
```python
def custom_collate_function(batch, padding_token_id, ignore_index):
    # Get inputs and targets
    # Create mask for all padding token indices
    mask = (target_tensor == padding_token_id)
    # Ignore the first occurrence of padding_token_id
    mask[0] = False
    # Replace all remaining padding tokens with ignore_index (-100)
    target_tensor[mask] = ignore_index
    return input_tensor, target_tensor
```

Notice that the first occurrence of the padding token is *not* replaced. This is intentional: the first end-of-text token marks the boundary between the response and the padding, and we want the model to learn to generate it [c1911].

A custom collate function takes a batch of sequences and requires a padding token ID (`50256`) and target device (`cpu`) as inputs [c1886]. The full collate function iterates through all sequences in the batch, tokenizes them, pads them to equal length, and applies the `-100` masking:

# Source: [c2303]
```python
# Custom collate function for creating batches
# Converts tokens to token IDs, pads sequences to equal length within batch,
# replaces padding tokens (50256) with -100, and creates input and target tensors
```

[FIGURE: Five-step data batching pipeline showing a single instruction-response pair flowing through: (1) Alpaca formatting → (2) tokenization → (3) padding to batch max length → (4) target shift by one → (5) -100 masking of padding positions in target]

#### A Note on Masking the Instruction Tokens

An interesting research question is whether we should also mask the *instruction* tokens in the target—that is, replace the token IDs corresponding to the instruction and input portions with `-100` so the model is only trained to predict the response tokens. Intuitively this seems sensible: why penalize the model for not predicting the instruction it was given?

A recent paper titled *Instruction Tuning With Loss Over Instructions* demonstrated that not masking the instruction tokens actually benefits LLM performance during instruction fine-tuning [c1915]. This is a good reminder that intuitions about what "should" help do not always hold empirically. For this chapter we follow the simpler approach and do not mask instruction tokens, but the custom collate function can be extended to do so.

#### Building the DataLoader

Data loaders are an efficient way to collect different batches sequentially and access them in an iterative manner during model training [c2284].

Once the `InstructionDataset` class and the custom collate function are in place, creating a dataset object is straightforward:

# Source: [c2314]
```python
train_dataset = InstructionDataset(training_data)
```

The `DataLoader` wraps this dataset, calls the custom collate function on each batch, and handles shuffling and multi-process loading automatically.

To ensure tensors are sent to the correct device:

# Source: [c2309]
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

### Stage 2: Loading the Pre-Trained Model and Fine-Tuning

#### Loading GPT-2 Medium (355 M)

We use GPT-2 medium as our foundation model. A pretrained GPT-2 model with 355 million weights was loaded as the foundation for instruction fine-tuning [c591].

The configuration and model selection code is:

# Source: [c1744]
```python
from gpt_download3 import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
```

The `BASE_CONFIG` dictionary specifies parameters shared across all GPT-2 variants: a vocabulary of 50,257 tokens, a context window of 1,024 tokens, no dropout during fine-tuning (`drop_rate: 0.0`), and query-key-value bias enabled (`qkv_bias: True`) [c1744]. The `model_configs` dictionary then provides the variant-specific embedding dimension, number of layers, and number of attention heads.

[FIGURE: GPT-2 model size comparison table showing the four variants (small 124M, medium 355M, large 774M, xl 1558M) with their embedding dimensions, layer counts, and head counts]

#### The Fine-Tuning Training Loop

The dataset processing, batching, and data loaders have already been implemented before the fine-tuning training loop [c3116]. Loss calculation and training functions from pre-training are reused for fine-tuning [c3117].

The instruction fine-tuning process consists of three stages: preparing the dataset, fine-tuning the LLM, and evaluating performance [c3087]. We are now in stage 2.

**Loss function.** Cross-entropy loss is used as the loss function for fine-tuning, the same loss function used during pre-training [c3095]. Cross-entropy loss is conceptually the negative of the logarithm of the probability [c3113]. Cross-entropy loss is used to calculate the loss on a batch of data [c3118].

**Optimizer.** AdamW is the Adam optimizer with weight decay, implemented as `torch.optim.AdamW` [c3123]. The gradient descent update rule is [c3103]:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}$$

where $w_{\text{new}}$ is the updated weight, $w_{\text{old}}$ is the previous weight, $\alpha$ is the learning rate, and $\partial L / \partial w$ is the partial derivative of loss with respect to $w$ [c3103].

**Backward pass.** The backward pass calculates the gradient of the loss, which is the partial derivative of the loss with respect to all model parameters [c3100]. Gradient descent is a gradient-based optimizer used to update model parameters [c3102].

**Evaluation frequency.** Evaluation iterations specifies after how many iterations the loss on the evaluation dataset is calculated [c3130]. Monitoring both training and validation loss during fine-tuning lets you detect overfitting early—if training loss keeps falling while validation loss rises, the model is memorizing the training set rather than generalizing.

[FIGURE: Training loop diagram showing the cycle: forward pass → compute cross-entropy loss → backward pass → AdamW weight update → repeat, with periodic evaluation on validation set]

#### Saving and Loading the Fine-Tuned Model

After training, save the model weights so you do not have to retrain from scratch every time:

# Source: [c640]
```python
torch.save(model.state_dict(), 'gpt2medium355_million_sft.pth')
```

To reload the weights later:

# Source: [c641]
```python
model.load_state_dict(torch.load('gpt2medium355_million_sft.pth'))
```

The suffix `sft` stands for *supervised fine-tuning*, a common convention in the field.

---

### Stage 3: Evaluation

Training a model is only half the work. We also need to know whether the fine-tuned model actually follows instructions well. The evaluation stage has two parts: qualitative inspection (reading the outputs yourself) and quantitative scoring (using a second LLM as an automated judge).

The instruction fine-tuning pipeline covered nine steps: data preparation, fine-tuning, evaluation, response extraction, qualitative evaluation, and quantitative scoring using Ollama [c714].

#### Extracting Responses

The first task is to run the fine-tuned model on the test set and collect its responses. The code iterates through all instruction-input pairs in the test data, calls the generate function on each pair, and appends the model response to the `instruction_data_with_response` file [c636].

For a quick sanity check during development, loop over the test data and select the first three entries, generate output from the fine-tuned LLM, convert token IDs to text, remove the instruction input from the output to isolate the response, and print the instruction, correct response, and model response [c607].

[FIGURE: Response extraction diagram showing: test entry → format_input → model.generate → decode token IDs → strip instruction prefix → isolated response string]

#### Qualitative Evaluation

Human preference comparison is an evaluation method where a human uses their own intuition and understanding to benchmark or compare LLMs [c626]. This is the simplest form of evaluation: read a sample of model outputs side by side with the expected outputs and judge whether the model is on the right track.

Qualitative evaluation is fast and catches obvious failures—a model that always outputs the same phrase, or one that ignores the instruction entirely—but it does not scale to thousands of examples.

#### Automated Evaluation with Ollama and LLaMA-3

For systematic evaluation across the full test set, we use a second, larger LLM as an automated judge. Step 7 of evaluating a fine-tuned LLM involves implementing automated response evaluation using a larger pre-trained LLM [c646].

Method 3 LLM evaluation involves comparing the true expected output against the model response by asking a large language model to assign a score based on this comparison [c627]. The judge LLM receives the instruction, the expected response, and the model's actual response, and returns a numeric score.

**What is Ollama?** Ollama is an efficient application to run large language models on your laptop [c648]. Ollama is a tool for generating text using LLM inference and does not support training or fine-tuning LLMs [c657]. It provides a simple command-line interface and a local HTTP API, making it easy to query a large model like LLaMA-3 from Python without sending data to an external server.

To start a LLaMA-3 session interactively:

# Source: [c660]
```
ollama run llama3
```

To start the Ollama server so it can accept API requests from Python:

# Source: [c662]
```
ollama serve
```

Once the server is running, your Python evaluation script can send HTTP POST requests to `localhost` with the instruction, expected output, and model response, and receive a numeric score in return.

[FIGURE: Automated evaluation pipeline: test entry → fine-tuned model → model response → Ollama/LLaMA-3 judge (receives instruction + expected + model response) → numeric score 0-100]

The full evaluation loop iterates over every entry in the test set, generates a response from the fine-tuned model, and sends the triple (instruction, expected output, model response) to the Ollama judge for scoring [c636]. Aggregating these scores gives a single number that summarizes model quality across the entire test set, making it easy to compare different checkpoints or hyperparameter settings.

---

### Putting It All Together

Let's step back and review the complete pipeline as a numbered sequence [c714]:

1. **Download the dataset** – 1,100 instruction-input-output triples [c588].
2. **Format with Alpaca template** – wrap each entry using `format_input` [c201].
3. **Tokenize** – convert formatted strings to token ID sequences [c1854].
4. **Pad** – equalize sequence lengths within each batch using token `50256` [c1863].
5. **Create target pairs** – shift input right by one, append end-of-text token [c2298].
6. **Mask padding in targets** – replace padding token IDs with `-100` [c1875].
7. **Build DataLoaders** – wrap datasets with the custom collate function [c2284].
8. **Load GPT-2 medium (355 M)** – initialize from pre-trained weights [c591].
9. **Fine-tune** – run the training loop with cross-entropy loss and AdamW [c3095, c3123].
10. **Save checkpoint** – `torch.save(model.state_dict(), ...)` [c640].
11. **Extract responses** – generate on the test set and strip the instruction prefix [c636].
12. **Qualitative evaluation** – read a sample of outputs by hand [c626].
13. **Automated scoring** – query Ollama/LLaMA-3 for numeric scores [c627].

[FIGURE: End-to-end pipeline flowchart with three labeled stages: Stage 1 (steps 1-7, dataset preparation), Stage 2 (steps 8-10, fine-tuning), Stage 3 (steps 11-13, evaluation)]

---

### Improving Fine-Tuning Performance

Once you have a working baseline, there are several levers you can pull to improve quality.

**Use a larger pre-trained model.** To improve fine-tuning performance, consider using a larger pre-trained model which may have greater capacity to capture complex patterns and generate more accurate responses [c698]. The `model_configs` dictionary already includes GPT-2 large (774 M) and GPT-2 XL (1,558 M); switching is a one-line change to `CHOOSE_MODEL` [c1744].

**Experiment with prompt formats.** To improve fine-tuning performance, experiment with different prompts or instruction formats to guide model responses more effectively [c697]. The Phi-3 style, which fuses instruction and input into a single `user` field [c200], is one alternative worth trying.

**Use a larger dataset.** The Stanford Alpaca dataset contains 52,000 instruction-output pairs [c695], compared to the 1,100 pairs used here. Training on more diverse examples generally improves generalization.

[GAP: Specific guidance on number of training epochs, learning rate schedule, and LoRA (Low-Rank Adaptation) hyperparameters for instruction fine-tuning]

---

### Summary

In this chapter we built a complete instruction fine-tuning pipeline on top of a pre-trained GPT-2 medium model. The key ideas are:

- **Instruction fine-tuning** teaches a pre-trained language model to follow instructions by training on instruction-response pairs using supervised learning [c1722, c181].
- **The Alpaca prompt format** provides a consistent text template that separates the instruction, optional input context, and expected response [c1842, c196].
- **Data batching** requires five steps: format, tokenize, pad, shift targets, and mask padding with `-100` [c1875]. The `-100` masking ensures that padding positions do not contribute to the cross-entropy loss [c1902].
- **The training loop** reuses the same cross-entropy loss and AdamW optimizer from pre-training [c3095, c3123]. The only change is the dataset.
- **Evaluation** combines qualitative human inspection [c626] with automated LLM-based scoring via Ollama [c627, c648], giving both interpretable examples and a scalar metric.

The fine-tuned model is a foundation for further work. You can push quality higher by training longer, using more data, switching to a larger base model, or applying parameter-efficient methods like LoRA. The pipeline you have built here is the same one used—at larger scale—to produce the instruction-following models that power modern AI assistants.