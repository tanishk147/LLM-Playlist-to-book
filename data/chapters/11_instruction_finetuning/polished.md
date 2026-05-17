# Instruction Fine-Tuning: Building a Personal Assistant

After pre-training, a language model can complete sentences and generate coherent prose—but it does not yet behave like an assistant. Instruction fine-tuning bridges that gap, transforming a next-token predictor into a model that responds helpfully to user requests.

The pipeline breaks naturally into three stages [c587]:

- **Stage 1 – Dataset Preparation:** download the data, format it with the Alpaca prompt template, and build training, validation, and test DataLoaders.
- **Stage 2 – Fine-Tuning:** load the pre-trained GPT-2 medium (355 M) checkpoint and run the supervised training loop.
- **Stage 3 – Evaluation:** extract responses from the fine-tuned model, inspect them qualitatively, and score them automatically using Ollama.

Let's begin.

---

### What Is Instruction Fine-Tuning?

A **foundation model** is a pre-trained large language model trained on large-scale data that can be fine-tuned for downstream tasks [c169]. The pre-training phase teaches the model grammar, facts, and reasoning patterns, but it does not teach the model to *follow instructions*.

**Instruction fine-tuning** is the process of training a pre-trained LLM on a specific dataset to teach it to correctly follow instructions [c1722]—more precisely, providing a large set of instruction-response pairs so the model learns to respond appropriately [c181]. Because we start from a pre-trained checkpoint rather than random weights, fine-tuning requires significantly less computational time than training from scratch [c1733].

Fine-tuning modifies the weights and parameters of the GPT model so that it understands instruction input-output pairs from the target dataset [c1761]. A few representative examples illustrate the range of tasks a single fine-tuned model can handle:

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

Unit conversion, vocabulary lookup, and grammar editing—all from the same model, trained on the same dataset.

---

### Stage 1: Dataset Preparation

#### The Instruction Dataset

The instruction dataset used in this chapter consists of 1,100 instruction input-output pairs [c588]. For reference, the Stanford Alpaca repository contains 52,000 such pairs [c695]; our smaller dataset is self-contained and suitable for experimentation.

Each entry in the dataset contains three keys: `instruction`, `input`, and `output` [c188]. Before any training can begin, this raw data must be formatted, tokenized, padded, and loaded in batches. Stage 1 covers all of that: downloading the data, formatting it with the Alpaca prompt template, and creating training, validation, and test DataLoaders [c3088].

#### The Alpaca Prompt Format

Raw instruction-input-output triples cannot be fed directly to a language model. A consistent text template is needed to tell the model where the instruction ends and where the response should begin.

**Alpaca prompt style** is a formatting convention for converting instruction-input-output pairs into prompts for fine-tuning large language models, maintained by Stanford in the Stanford Alpaca repository [c1842]. It was originally designed for instruction-following LLaMA models [c193]. The template reads [c196]:

> *Below is an instruction that describes a task paired with an input that provides further context. Write a response that appropriately completes the request.*

followed by the instruction, input, and output fields. Crucially, the instruction and input fields are kept separate in the template [c199].

A second common convention is **Phi-3 prompt style**, in which the instruction and input are fused into a single `user` field and the output is placed in an `assistant` field [c200]. There are therefore two main ways to format instruction-input-output datasets: Alpaca style and Phi-3 style [c198].

The full Alpaca template looks like this [c2291]:

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

#### Implementing `format_input`

The following function converts a dataset entry into a formatted Alpaca prompt [c201]:

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

The function returns only the *input* side of the prompt—the instruction and optional context—without the response. A companion function `formatInput` follows the same logic [c1846]:

```python
def formatInput(entry):
    # Converts instruction-input-output pairs into alpaca prompt format
    # Takes entry with instruction, input, and output keys
    # Returns formatted prompt with instruction, input, and response sections
```

#### Data Batching: Five Steps

Batching the dataset means converting multiple data samples into a batch where each sample is represented as a numerical array (row) with uniform dimensions [c1848]. For instruction fine-tuning, this process has five steps [c1875]:

1. **Format** – apply the Alpaca prompt template.
2. **Tokenize** – convert the formatted text to token ID sequences.
3. **Pad** – equalize sequence lengths within each batch.
4. **Create target pairs** – shift the input token IDs right by one position.
5. **Mask padding** – replace padding token IDs in the targets with `-100`.

Let's examine each step in detail.

#### Step 1: Format

Apply the Alpaca template to each entry using `format_input`, as described above [c2289].

#### Step 2: Tokenize

Tokenization converts the formatted text into a numerical representation by mapping sentences to token IDs [c1854].

#### Step 3: Pad

Because sequences in a batch typically have different lengths, padding adjusts them to a uniform length [c1861]. The end-of-text token—token ID `50256` in GPT-2—serves as the padding symbol [c1863].

#### Step 4: Create Target Pairs

During instruction fine-tuning the model is trained with next-token prediction, so the entire formatted prompt serves as both input and target [c2292]. Target token IDs are created by shifting the input tokens one position to the right and appending an end-of-text token to maintain equal length [c1868, c2298]:

```python
targets = inputs[1:]  # Shift inputs right by one by removing first element
```
[c1896]

#### Step 5: Replace Padding Tokens with −100

Padding positions in the target tensor should not contribute to the loss. We therefore replace them with the special ignore index `−100`, which PyTorch's cross-entropy loss automatically skips [c1917]:

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
[c1902]

Note that the *first* end-of-text token is deliberately preserved: it marks the boundary between the response and the padding, and the model should learn to generate it [c1911].

The custom collate function takes a batch of sequences and requires a padding token ID (`50256`) and a target device (`cpu`) as inputs [c1886]:

```python
# Custom collate function for creating batches
# Converts tokens to token IDs, pads sequences to equal length within batch,
# replaces padding tokens (50256) with -100, and creates input and target tensors
```
[c2303]

#### A Note on Masking the Instruction Tokens

A recent paper titled *Instruction Tuning With Loss Over Instructions* demonstrated that not masking the instruction tokens actually benefits LLM performance during instruction fine-tuning [c1915]. This chapter follows the simpler approach and does not mask instruction tokens, but the custom collate function can be extended to do so if desired.

#### Building the DataLoader

Data loaders provide an efficient way to collect batches sequentially and access them iteratively during training [c2284]:

```python
train_dataset = InstructionDataset(training_data)
```
[c2314]

The `DataLoader` wraps this dataset, calls the custom collate function on each batch, and handles shuffling and multi-process loading automatically:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
[c2309]

---

### Stage 2: Loading the Pre-Trained Model and Fine-Tuning

#### Loading GPT-2 Medium (355 M)

A pre-trained GPT-2 model with 355 million weights serves as the foundation for instruction fine-tuning [c591]. The configuration and model-selection code is [c1744]:

```python
from gpt_download3 import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
```

`BASE_CONFIG` specifies parameters shared across all GPT-2 variants: a vocabulary of 50,257 tokens, a context window of 1,024 tokens, no dropout during fine-tuning (`drop_rate: 0.0`), and query-key-value bias enabled (`qkv_bias: True`). `model_configs` then provides the variant-specific embedding dimension, layer count, and attention-head count [c1744].

[FIGURE: GPT-2 model size comparison table showing the four variants (small 124M, medium 355M, large 774M, xl 1558M) with their embedding dimensions, layer counts, and head counts]

#### The Fine-Tuning Training Loop

The dataset processing, batching, and DataLoaders are all in place before the training loop begins [c3116], and the loss calculation and training functions from pre-training are reused directly [c3117].

**Loss function.** Cross-entropy loss is used for fine-tuning, the same function used during pre-training [c3095]. Conceptually it is the negative logarithm of the predicted probability for the correct token [c3113], and it is applied to each batch of data [c3118].

**Optimizer.** AdamW—the Adam optimizer with weight decay—is used, implemented as `torch.optim.AdamW` [c3123]. The gradient-descent update rule is [c3103]:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}$$

where $w_{\text{new}}$ is the updated weight, $w_{\text{old}}$ is the previous weight, $\alpha$ is the learning rate, and $\partial L / \partial w$ is the partial derivative of the loss with respect to $w$ [c3103].

**Backward pass.** The backward pass computes the gradient of the loss—the partial derivative with respect to every model parameter [c3100]—which the optimizer then uses to update the weights [c3102].

**Evaluation frequency.** The evaluation-iterations setting controls after how many training iterations the loss is computed on the validation set [c3130].

[FIGURE: Training loop diagram showing the cycle: forward pass → compute cross-entropy loss → backward pass → AdamW weight update → repeat, with periodic evaluation on validation set]

#### Saving and Loading the Fine-Tuned Model

Once training is complete, save the model weights [c640]:

```python
torch.save(model.state_dict(), 'gpt2medium355_million_sft.pth')
```

To reload the weights later [c641]:

```python
model.load_state_dict(torch.load('gpt2medium355_million_sft.pth'))
```

The suffix `sft` stands for *supervised fine-tuning*, a common convention in the field.

---

### Stage 3: Evaluation

Saving a checkpoint is not enough—we also need to know whether the fine-tuned model actually follows instructions well. The full evaluation pipeline covers response extraction, qualitative inspection, and automated scoring [c714].

#### Extracting Responses

The evaluation code iterates through all instruction-input pairs in the test set, calls the generate function on each pair, and appends the model response to the `instruction_data_with_response` file [c636]. For a quick sanity check during development, a shorter loop selects the first three test entries, generates output from the fine-tuned model, decodes the token IDs to text, strips the instruction prefix to isolate the response, and prints the instruction, the expected response, and the model's response side by side [c607].

[FIGURE: Response extraction diagram showing: test entry → format_input → model.generate → decode token IDs → strip instruction prefix → isolated response string]

#### Qualitative Evaluation

Human preference comparison is an evaluation method in which a human applies their own intuition and understanding to benchmark or compare LLMs [c626]. Reading a sample of model outputs by hand is often the fastest way to spot systematic errors or surprising successes.

#### Automated Evaluation with Ollama and LLaMA-3

Qualitative inspection does not scale to the full test set, so automated scoring provides a complementary signal. Step 7 of the evaluation workflow involves implementing automated response evaluation using a larger pre-trained LLM [c646]: the true expected output and the model's response are both sent to a judge model, which assigns a numeric score [c627].

**Ollama** is an efficient application for running large language models locally [c648]. It supports LLM inference for text generation but does not support training or fine-tuning [c657]. To pull and run LLaMA-3 [c660]:

```
ollama run llama3
```

If the server is not already running, start it with [c662]:

```
ollama serve
```

[FIGURE: Automated evaluation pipeline: test entry → fine-tuned model → model response → Ollama/LLaMA-3 judge (receives instruction + expected + model response) → numeric score 0-100]

The full evaluation loop iterates over every entry in the test set, generates a response from the fine-tuned model, and sends the triple (instruction, expected output, model response) to the Ollama judge for scoring [c636]. Aggregating these scores yields a single number summarizing model quality across the entire test set, making it straightforward to compare different checkpoints or hyperparameter settings.

---

### Putting It All Together

The complete pipeline, as a numbered sequence [c714]:

1. **Download the dataset** – 1,100 instruction-input-output triples [c588].
2. **Format with the Alpaca template** – wrap each entry using `format_input` [c201].
3. **Tokenize** – convert formatted strings to token ID sequences [c1854].
4. **Pad** – equalize sequence lengths within each batch using token `50256` [c1863].
5. **Create target pairs** – shift input right by one and append an end-of-text token [c2298].
6. **Mask padding in targets** – replace padding token IDs with `−100` [c1875].
7. **Build DataLoaders** – wrap datasets with the custom collate function [c2284].
8. **Load GPT-2 medium (355 M)** – initialize from pre-trained weights [c591].
9. **Fine-tune** – run the training loop with cross-entropy loss and AdamW [c3095, c3123].
10. **Save checkpoint** – `torch.save(model.state_dict(), ...)` [c640].
11. **Extract responses** – generate on the test set and strip the instruction prefix [c636].
12. **Qualitative evaluation** – read a sample of outputs by hand [c626].
13. **Automated scoring** – query Ollama/LLaMA-3 for numeric scores [c627].

[FIGURE: End-to-end pipeline flowchart with three labeled stages: Stage 1 (steps 1–7, dataset preparation), Stage 2 (steps 8–10, fine-tuning), Stage 3 (steps 11–13, evaluation)]

---

### Improving Fine-Tuning Performance

**Use a larger pre-trained model.** A larger model may have greater capacity to capture complex patterns and generate more accurate responses [c698]. The `model_configs` dictionary already includes GPT-2 large (774 M) and GPT-2 XL (1,558 M); switching requires only a one-line change to `CHOOSE_MODEL` [c1744].

**Experiment with prompt formats.** Different prompts or instruction formats can guide model responses more effectively [c697]. The Phi-3 style, which fuses instruction and input into a single `user` field [c200], is one alternative worth trying.

**Use a larger dataset.** The Stanford Alpaca dataset contains 52,000 instruction-output pairs [c695], compared to the 1,100 pairs used here. Training on more diverse examples generally improves generalization.

---

### Summary

This chapter walked through the complete instruction fine-tuning pipeline, from raw data to a scored, evaluated model. The key ideas are:

- **Instruction fine-tuning** teaches a pre-trained language model to follow instructions by training on instruction-response pairs with supervised learning [c1722, c181].
- **The Alpaca prompt format** provides a consistent text template that separates the instruction, optional input context, and expected response [c1842, c196].
- **Data batching** requires five steps: format, tokenize, pad, shift targets, and mask padding with `−100` [c1875]. The `−100` masking ensures that padding positions do not contribute to the cross-entropy loss [c1902].
- **The training loop** reuses the same cross-entropy loss and AdamW optimizer from pre-training [c3095, c3123]; the only change is the dataset.
- **Evaluation** combines qualitative human inspection [c626] with automated LLM-based scoring via Ollama [c627, c648], providing both interpretable examples and a scalar metric.

Quality can be pushed further by training longer, using more data, switching to a larger base model, or applying parameter-efficient methods such as LoRA.