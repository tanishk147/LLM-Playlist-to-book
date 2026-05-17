## Instruction Fine-Tuning: Building a Personal Assistant

 The model can complete sentences and generate coherent prose, but it does not yet behave like an assistant. 

 

The pipeline breaks naturally into three stages [c587]:

Let's begin.

---

### What Is Instruction Fine-Tuning?

A **foundation model** is a pre-trained large language model that has been trained on large-scale data and can be fine-tuned for downstream tasks [c169]. The pre-training phase teaches the model grammar, facts, and reasoning patterns, but it does not teach the model to *follow instructions*. 

**Instruction fine-tuning** (also called supervised instruction fine-tuning) is the process of training a pretrained LLM with a specific dataset to teach it to correctly follow instructions [c1722]. More precisely, it is the approach of providing a large set of instruction-response pairs to train a language model to follow instructions [c181]. Fine-tuning on a specific dataset after loading pre-trained weights requires significantly less computational time than training from scratch [c1733], which is why we always start from a pre-trained checkpoint rather than random weights.

Fine-tuning the LLM on instruction data modifies the weights and parameters of the GPT model to understand instruction input-output pairs from a specific dataset [c1761]. 

```

Response: 45 km is 45000 meters
```
[c178]

```

Response: a synonym for bright is radiant
```
[c179]

```

 The song was composed by the artist
Response: the artist composed the song
```
[c180]

These examples illustrate the breadth of tasks a single fine-tuned model can handle: unit conversion, vocabulary, and grammar editing—all from the same model, trained on the same dataset.

---

### Stage 1: Dataset Preparation

#### The Instruction Dataset

The instruction dataset used for fine-tuning in this chapter consists of 1,100 instruction input and output pairs [c588]. For reference, the Stanford Alpaca repository contains a dataset of 52,000 instruction-output pairs [c695]; our 1,100-pair dataset is a smaller, self-contained version suitable for experimentation.

Each instruction-response pair in the dataset contains three keys: `instruction`, `input`, and `output` [c188]. 

The fine-tuning workflow includes: fine-tuning the LLM, inspecting the loss, extracting responses, performing evaluation, and scoring [c186]. But before any of that, we need to get the data into a form the model can consume.

Stage 1 specifically involves downloading data, formatting it using the Alpaca prompt format, and creating training, testing, and validation data loaders [c3088].

#### The Alpaca Prompt Format

Raw instruction-input-output triples cannot be fed directly to a language model. We need a consistent text template that tells the model where the instruction ends and where the response should begin. 

Alpaca prompt style is a specific formatting convention for converting instruction-input-output pairs into prompts for fine-tuning large language models, maintained by Stanford in the Stanford Alpaca repository [c1842]. Stanford Alpaca is a specific prompt format for instruction-following LLaMA models [c193].

The Stanford Alpaca prompt format consists of a template [c196]:

 Write a response that appropriately completes the request.*

followed by the instruction, input, and output fields. In Alpaca prompt style, the instruction and input fields are separated in the prompt template [c199].

There are two main ways to format instruction-input-output datasets: Alpaca prompt style and Phi-3 prompt style [c198]. In Phi-3 prompt style, the instruction and input are fused together in a `user` field, and the output is placed in an `assistant` field [c200]. 

The Alpaca prompt template structures instruction fine-tuning data as [c2291]:

```

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

The following function converts a dataset entry into a formatted Alpaca prompt. 

# Source: [c201]
```python

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

The function returns only the *input* side of the prompt—the instruction and optional context—without the response. 

A companion function `formatInput` follows the same logic [c1846]:

# Source: [c1846]
```python

 # Converts instruction-input-output pairs into alpaca prompt format
 # Takes entry with instruction, input, and output keys
 # Returns formatted prompt with instruction, input, and response sections
```

#### Data Batching: Five Steps

 Batching the dataset means converting multiple data samples into a batch where each sample is represented as a numerical array (row) with uniform dimensions [c1848].

Data batching for instruction fine-tuning consists of five steps [c1875]:

Let's examine each step in detail.

#### Step 1: Format

 The Alpaca format is a template used for formatting instruction, input, and output data during instruction fine-tuning [c2289].

#### Step 2: Tokenize

Tokenization is the process of converting data into a numerical representation by converting sentences into token IDs [c1854]. 

#### Step 3: Pad

 Padding is the process of adjusting the length of token ID sequences so that all samples in a batch have the same length [c1861].

The end-of-text token is a special token used for padding sequences; in GPT-2, it has token ID `50256` [c1863]. 

#### Step 4: Create Target Pairs

During instruction fine-tuning, the LLM is trained using next-token prediction, where the entire formatted prompt serves as both input and target pairs [c2292]. Target token IDs are created by shifting the input tokens to the right by one position and padding with an additional token to maintain equal length between input and target sequences [c1868].

 The target tensor is the input tensor shifted to the right by one position [c2323]. Target pairs are created by shifting the input sequence to the right by one token and appending an end-of-text token [c2298].

# Source: [c1896]
```python

```

#### Step 5: Replace Padding Tokens with -100

 After creating the target tensor, the positions that correspond to padding tokens should not contribute to the loss. 

 

To mask target token IDs, replace the token IDs of instruction and input portions in the target text with `-100` [c1917]. The custom collate function implements this masking:

# Source: [c1902]
```python

 # Get inputs and targets
 # Create mask for all padding token indices
 mask = (target_tensor == padding_token_id)
 # Ignore the first occurrence of padding_token_id
 mask[0] = False
 # Replace all remaining padding tokens with ignore_index (-100)
 target_tensor[mask] = ignore_index
 return input_tensor, target_tensor
```

 This is intentional: the first end-of-text token marks the boundary between the response and the padding, and we want the model to learn to generate it [c1911].

A custom collate function takes a batch of sequences and requires a padding token ID (`50256`) and target device (`cpu`) as inputs [c1886]. 

# Source: [c2303]
```python

# Converts tokens to token IDs, pads sequences to equal length within batch,
# replaces padding tokens (50256) with -100, and creates input and target tensors
```

#### A Note on Masking the Instruction Tokens

 

A recent paper titled *Instruction Tuning With Loss Over Instructions* demonstrated that not masking the instruction tokens actually benefits LLM performance during instruction fine-tuning [c1915]. For this chapter we follow the simpler approach and do not mask instruction tokens, but the custom collate function can be extended to do so.

#### Building the DataLoader

Data loaders are an efficient way to collect different batches sequentially and access them in an iterative manner during model training [c2284].

# Source: [c2314]
```python

```

The `DataLoader` wraps this dataset, calls the custom collate function on each batch, and handles shuffling and multi-process loading automatically.

# Source: [c2309]
```python

```

---

### Stage 2: Loading the Pre-Trained Model and Fine-Tuning

#### Loading GPT-2 Medium (355 M)

 A pretrained GPT-2 model with 355 million weights was loaded as the foundation for instruction fine-tuning [c591].

The configuration and model selection code is:

# Source: [c1744]
```python

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

The instruction fine-tuning process consists of three stages: preparing the dataset, fine-tuning the LLM, and evaluating performance [c3087]. 

**Loss function.** Cross-entropy loss is used as the loss function for fine-tuning, the same loss function used during pre-training [c3095]. Cross-entropy loss is conceptually the negative of the logarithm of the probability [c3113]. Cross-entropy loss is used to calculate the loss on a batch of data [c3118].

**Optimizer.** AdamW is the Adam optimizer with weight decay, implemented as `torch.optim.AdamW` [c3123]. The gradient descent update rule is [c3103]:

where $w_{\text{new}}$ is the updated weight, $w_{\text{old}}$ is the previous weight, $\alpha$ is the learning rate, and $\partial L / \partial w$ is the partial derivative of loss with respect to $w$ [c3103].

**Backward pass.** The backward pass calculates the gradient of the loss, which is the partial derivative of the loss with respect to all model parameters [c3100]. Gradient descent is a gradient-based optimizer used to update model parameters [c3102].

**Evaluation frequency.** Evaluation iterations specifies after how many iterations the loss on the evaluation dataset is calculated [c3130]. 

[FIGURE: Training loop diagram showing the cycle: forward pass → compute cross-entropy loss → backward pass → AdamW weight update → repeat, with periodic evaluation on validation set]

#### Saving and Loading the Fine-Tuned Model

# Source: [c640]
```python

```

To reload the weights later:

# Source: [c641]
```python

```

The suffix `sft` stands for *supervised fine-tuning*, a common convention in the field.

---

### Stage 3: Evaluation

 We also need to know whether the fine-tuned model actually follows instructions well. 

The instruction fine-tuning pipeline covered nine steps: data preparation, fine-tuning, evaluation, response extraction, qualitative evaluation, and quantitative scoring using Ollama [c714].

#### Extracting Responses

 The code iterates through all instruction-input pairs in the test data, calls the generate function on each pair, and appends the model response to the `instruction_data_with_response` file [c636].

For a quick sanity check during development, loop over the test data and select the first three entries, generate output from the fine-tuned LLM, convert token IDs to text, remove the instruction input from the output to isolate the response, and print the instruction, correct response, and model response [c607].

[FIGURE: Response extraction diagram showing: test entry → format_input → model.generate → decode token IDs → strip instruction prefix → isolated response string]

#### Qualitative Evaluation

Human preference comparison is an evaluation method where a human uses their own intuition and understanding to benchmark or compare LLMs [c626]. 

#### Automated Evaluation with Ollama and LLaMA-3

 Step 7 of evaluating a fine-tuned LLM involves implementing automated response evaluation using a larger pre-trained LLM [c646].

Method 3 LLM evaluation involves comparing the true expected output against the model response by asking a large language model to assign a score based on this comparison [c627]. 

**What is Ollama?** Ollama is an efficient application to run large language models on your laptop [c648]. Ollama is a tool for generating text using LLM inference and does not support training or fine-tuning LLMs [c657]. 

# Source: [c660]
```

```

# Source: [c662]
```

```

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

**Use a larger pre-trained model.** To improve fine-tuning performance, consider using a larger pre-trained model which may have greater capacity to capture complex patterns and generate more accurate responses [c698]. The `model_configs` dictionary already includes GPT-2 large (774 M) and GPT-2 XL (1,558 M); switching is a one-line change to `CHOOSE_MODEL` [c1744].

**Experiment with prompt formats.** To improve fine-tuning performance, experiment with different prompts or instruction formats to guide model responses more effectively [c697]. The Phi-3 style, which fuses instruction and input into a single `user` field [c200], is one alternative worth trying.

**Use a larger dataset.** The Stanford Alpaca dataset contains 52,000 instruction-output pairs [c695], compared to the 1,100 pairs used here. Training on more diverse examples generally improves generalization.

---

### Summary

 The key ideas are:

- **Instruction fine-tuning** teaches a pre-trained language model to follow instructions by training on instruction-response pairs using supervised learning [c1722, c181].
- **The Alpaca prompt format** provides a consistent text template that separates the instruction, optional input context, and expected response [c1842, c196].
- **Data batching** requires five steps: format, tokenize, pad, shift targets, and mask padding with `-100` [c1875]. The `-100` masking ensures that padding positions do not contribute to the cross-entropy loss [c1902].
- **The training loop** reuses the same cross-entropy loss and AdamW optimizer from pre-training [c3095, c3123]. The only change is the dataset.
- **Evaluation** combines qualitative human inspection [c626] with automated LLM-based scoring via Ollama [c627, c648], giving both interpretable examples and a scalar metric.

 You can push quality higher by training longer, using more data, switching to a larger base model, or applying parameter-efficient methods like LoRA.
