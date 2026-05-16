## Instruction Fine-Tuning: Building a Personal Assistant

A pre-trained language model can complete sentences and generate coherent prose, but it cannot reliably *follow instructions*. This chapter closes that gap.

We will walk through the complete instruction fine-tuning pipeline from end to end [c587]. The journey has three stages: preparing the dataset, fine-tuning the model, and evaluating what we built [c3087].

---

### Why Instruction Fine-Tuning?

Instruction fine-tuning is the process of training a pre-trained LLM with a specific dataset to teach it to correctly follow instructions [c1722]. The mechanism is straightforward: collect a large set of instruction-response pairs and use them to update the model's weights [c181]. The result is a model that has been modified to understand instruction input-output pairs from a specific dataset [c1761]. Intuitively, pre-training gives the model a rich internal representation of language; fine-tuning steers that representation toward the behavior you want.

Fine-tuning serves two main purposes: training the LLM to follow instructions and training on domain-specific data relevant to the deploying organization [c177]. To make this concrete, consider a healthcare virtual assistant. Fine-tuning it with domain-specific instructions ensures it understands medical terminology, follows healthcare guidelines, and personalizes responses based on patient history [c176]. More generally, building a personal assistant requires the LLM to follow instructions such as correcting grammar, removing filler words, making text concise, and adjusting tone [c168].

One practical advantage: fine-tuning on a specific dataset after loading pre-trained weights requires significantly less computational time than training from scratch [c1733].

---

### The Three-Stage Pipeline

**Stage 1 — Dataset preparation.** Download the instruction data, format it using the Alpaca prompt template, and create training, validation, and test data loaders [c3088]. Significant time should be spent on batching the dataset and creating data loaders, because correct implementation of this step simplifies all subsequent steps [c2333].

**Stage 2 — Fine-tuning.** Load a pre-trained GPT-2 checkpoint and run the supervised training loop on the instruction data [c3087].

**Stage 3 — Evaluation.** Extract model responses on the test set, perform qualitative inspection, and run automated scoring using a larger LLM [c587].

Across all three stages, the pipeline spans nine steps in total: data preparation, fine-tuning, evaluation, response extraction, qualitative evaluation, and quantitative scoring [c714].

---

### The Instruction Dataset

#### What the Data Looks Like

The instruction dataset used for fine-tuning consists of 1,100 instruction input and output pairs [c588]. Each entry contains three keys: `instruction`, `input`, and `output` [c188].

**Example 1** [c178]:
- *Instruction:* Convert 45 km to meters
- *Response:* 45 km is 45000 meters

**Example 2** [c179]:
- *Instruction:* Provide a synonym for bright
- *Response:* A synonym for bright is radiant

**Example 3** [c180]:
- *Instruction:* Edit the following sentence to remove all passive voice. "The song was composed by the artist."
- *Response:* The artist composed the song.

Notice that Example 3 uses the `input` field to supply the sentence being edited, while Examples 1 and 2 do not. This distinction matters when we format the prompt.

For comparison, the Stanford Alpaca repository contains a much larger dataset of 52,000 instruction-output pairs [c695].

#### Loading and Splitting the Data

The fine-tuning process begins by loading a training dataset consisting of instruction-response pairs [c182]. The test split is reserved for evaluating the model's performance after fine-tuning [c605].

---

### The Alpaca Prompt Format

Raw instruction-response pairs cannot be fed directly to the model. We need a consistent text template that tells the model where the instruction ends and where the response should begin. The dominant convention is the **Alpaca prompt format**: a specific formatting convention for converting instruction-input-output pairs into prompts for fine-tuning large language models, maintained by Stanford in the Stanford Alpaca repository [c1842].

The full template reads [c196]:

> Below is an instruction that describes a task paired with an input that provides further context. Write a response that appropriately completes the request.
>
> **Instruction:**
> {instruction}
>
> **Input:**
> {input}
>
> **Response:**
> {output}

When the `input` field is empty, the `Input:` section is omitted entirely, keeping the prompt clean and avoiding confusion from an empty section [c199].

For contrast, the Phi-3 prompt style takes a different approach: the instruction and input are fused together in a `user` field, and the output is placed in an `assistant` field [c200]. Both styles are valid; we use Alpaca throughout this chapter because it is the more widely documented convention [c2289].

#### Implementing `format_input`

The following function converts a dataset entry into an Alpaca-formatted prompt string [c1846]:

```python
# Source: [c201]
def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task paired with an input that "
        "provides further context. Write a response that appropriately completes the request."
    )
    instruction = entry['instruction']
    input_text = entry.get('input', '')
    if input_text:
        prompt = f"{instruction_text}\n\nInstruction:\n{instruction}\n\nInput:\n{input_text}"
    else:
        prompt = f"{instruction_text}\n\nInstruction:\n{instruction}"
    return prompt
```

The function returns only the *input* side of the prompt—the instruction and any context—without the response. During training, the response is appended separately so the model can learn to predict it token by token [c2291].

---

### Data Batching: From Text to Tensors

Converting raw entries into model-ready batches requires five steps [c2294]:

1. Format each entry using the Alpaca prompt template.
2. Tokenize the formatted string into token IDs.
3. Pad sequences to equal length within each batch.
4. Create target token IDs by shifting the input sequence right by one position.
5. Replace padding tokens in the targets with `-100`.

Let us work through each step.

#### Step 1: Format with the Prompt Template

Each entry in the dataset is passed through `format_input` to produce a fully formatted string [c2291].

#### Step 2: Tokenization

Tokenization is the process of converting data into a numerical representation by converting sentences into token IDs [c1854]. The formatted prompt string is tokenized before being passed to the model.

#### Step 3: Padding to Equal Length

Because sequences in a batch can have different lengths, we must equalize them. Padding is the process of adjusting the length of token ID sequences so that all samples in a batch have the same length [c1861]. The end-of-text token—token ID `50256` in GPT-2—is used as the padding symbol [c1863]. Batching the dataset ultimately means converting multiple data samples into a batch where each sample is represented as a numerical array (row) with uniform dimensions [c1848].

[FIGURE: Diagram showing three sequences of different lengths being padded with token ID 50256 to form a uniform rectangular batch matrix]

#### Step 4: Creating Target Sequences

During instruction fine-tuning, the LLM is trained using next-token prediction, where the entire formatted prompt serves as both input and target pairs [c2292]. The target sequence is the input sequence shifted to the right by one position [c2323]: target token IDs are created by shifting the input tokens right by one position and appending a padding token at the end to maintain equal length between input and target sequences [c1868, c1894].

```python
# Source: [c1896]
targets = inputs[1:]  # Shift inputs right by one by removing first element
```

An end-of-text token is appended to complete the target sequence [c2298].

[FIGURE: Side-by-side illustration of input token sequence and target token sequence, showing the one-position rightward shift with an end-of-text token appended at the end of the target]

#### Step 5: Replacing Padding Tokens with `-100`

Padding tokens in the target sequence must not contribute to the loss. PyTorch's cross-entropy loss function has a built-in `ignore_index` parameter; when a target token ID equals `-100`, that position is excluded from the loss calculation entirely [c1902].

There is one exception: the *first* occurrence of the end-of-text token in the target is kept as-is, because it is a legitimate prediction target—the model should learn to emit end-of-text when it finishes a response [c1902]. Additionally, the token IDs corresponding to the instruction and input portions of the target are also replaced with `-100`, so the loss is computed only over the response tokens [c1917].

#### The Custom Collate Function

A custom collate function takes a batch of sequences and requires a padding token ID (`50256`) and a target device (`cpu`) as inputs [c1886]:

```python
# Source: [c1902]
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

In summary, the collate function converts tokens to token IDs, pads sequences to equal length within the batch, replaces padding tokens (`50256`) with `-100`, and returns input and target tensors ready for training [c2303].

---

### Creating the Data Loaders

Data loaders are an efficient way to collect different batches sequentially and access them in an iterative manner during model training [c2284].

First, select the compute device:

```python
# Source: [c2309]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Then wrap the training data in a dataset object:

```python
# Source: [c2314]
train_dataset = InstructionDataset(training_data)
```

A `DataLoader` wraps the dataset and calls the custom collate function to assemble batches on each iteration. Stage one of instruction fine-tuning focuses entirely on preparing the dataset itself [c2335].

---

### Loading the Pre-Trained Model

With the data pipeline in place, we turn to the model. A pre-trained GPT-2 model with 355 million weights serves as the foundation for instruction fine-tuning [c591]:

```python
# Source: [c1744]
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

Fine-tuning is necessary to improve the model's ability to comprehend and appropriately respond to instruction-following requests [c1760].

---

### The Fine-Tuning Training Loop

#### Loss Function

The loss function used for fine-tuning is cross-entropy loss [c3112]—the same function used during pre-training [c3095]. Conceptually, cross-entropy loss is the negative of the logarithm of the probability assigned to the correct token [c3113], and it is applied to calculate the loss on each batch of data [c3118].

#### Optimizer

The optimizer is **AdamW**, the Adam optimizer with weight decay, implemented as `torch.optim.AdamW` [c3123].

#### Parameter Updates

The backward pass calculates the gradient of the loss—the partial derivative of the loss with respect to all model parameters [c3100]. Gradient descent then uses those gradients to update the parameters [c3102]. The update rule is [c3103]:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}$$

where $w_{\text{new}}$ is the updated weight, $w_{\text{old}}$ is the previous weight, $\alpha$ is the learning rate, and $\partial L / \partial w$ is the partial derivative of the loss with respect to $w$.

#### Monitoring Training

The dataset processing, batching, and data loaders are all in place before the training loop begins [c3116]. During training, we track both training loss and validation loss. The evaluation iterations parameter specifies after how many iterations the loss on the evaluation dataset is calculated [c3130]. Fine-tuning the LLM on instruction data modifies the weights and parameters of the GPT model to understand instruction input-output pairs from the dataset [c1761].

#### Saving and Loading the Fine-Tuned Model

After training completes, save the model's state dictionary so it can be reloaded later without re-running the training loop:

```python
# Source: [c640]
torch.save(model.state_dict(), 'gpt2medium355_million_sft.pth')
```

To reload the weights:

```python
# Source: [c641]
model.load_state_dict(torch.load('gpt2medium355_million_sft.pth'))
```

With the weights saved, we can move on to evaluation without repeating the training run [c707].

---

### Evaluating the Fine-Tuned Model

Evaluation has three sub-steps: extracting model responses, performing qualitative inspection, and running automated quantitative scoring [c587].

#### Step 1: Extracting Responses

After fine-tuning on the training portion of the instruction dataset, the model's performance is evaluated on the test set [c605]. The following procedure loops over the test data, generates output from the fine-tuned LLM, converts token IDs back to text, and isolates the response by removing the instruction input from the output [c607]:

```python
# Source: [c636]
# Iterates through all instruction-input pairs in the test data,
# calls the generate function on each pair, and appends the model response
# to the instruction_data_with_response file.
```

#### Step 2: Qualitative Evaluation

The simplest evaluation method is human preference comparison, where a human uses their own intuition and understanding to benchmark or compare LLMs [c626]. Reading through a sample of model responses gives an immediate sense of whether fine-tuning has produced coherent, instruction-following behavior. To improve performance, you can also experiment with different prompts or instruction formats to guide model responses more effectively [c697].

#### Step 3: Automated LLM-Based Scoring

For scalable evaluation, we use a larger language model as a judge. This approach—Method 3 LLM evaluation—involves comparing the true expected output against the model response by asking a large language model to assign a score based on that comparison [c627]. Implementing this automated evaluation using a larger pre-trained LLM is the seventh step of the evaluation stage [c646]. Aggregating these scores across the test set yields a quantitative measure of fine-tuning quality.

---

### Running Llama 3 Locally with Ollama

To use a larger LLM as a scoring judge without relying on a remote API, we can run one locally. Ollama is an efficient application for running large language models on your laptop [c648], handling model downloading, quantization, and inference in a single lightweight binary.

Importantly, Ollama is a tool for LLM inference only and does not support training or fine-tuning [c657].

#### Starting Ollama

```bash
# Source: [c660]
ollama run llama3
```

To run Ollama as a background server so it can be called from Python:

```bash
# Source: [c662]
ollama serve
```

---

### Putting It All Together

1. **Dataset.** 1,100 instruction-response pairs, each with `instruction`, `input`, and `output` fields [c188, c588].
2. **Formatting.** The Alpaca prompt template wraps each entry into a structured string [c196, c201].
3. **Batching.** Tokenization, padding with token ID `50256`, target shifting, and `-100` masking are handled by the custom collate function [c1875, c1911].
4. **Data loaders.** PyTorch `DataLoader` objects iterate over training, validation, and test splits [c2284].
5. **Model.** GPT-2 medium (355M parameters) loaded from a pre-trained checkpoint [c591, c1744].
6. **Training.** Cross-entropy loss over response tokens only, optimized with AdamW [c3095, c3123].
7. **Saving.** Model weights persisted to disk with `torch.save` [c640].
8. **Response extraction.** The model generates responses for all test entries [c636].
9. **Evaluation.** Qualitative human review [c626] and automated LLM scoring via Ollama [c627, c648].

The instruction fine-tuning pipeline spans nine steps in total [c714], and we have now walked through every one of them.

---

### Summary and What Comes Next

This chapter introduced supervised instruction fine-tuning and showed how to build the entire pipeline from raw data to a scored, evaluated personal assistant [c181]. The key ideas are:

- **Instruction fine-tuning** teaches a pre-trained model to follow instructions by training on instruction-response pairs [c1722].
- **The Alpaca prompt format** provides a consistent template that separates instruction, input, and response [c1842, c196].
- **Data batching** requires careful handling of padding and the `-100` ignore index to ensure the loss is computed only over response tokens [c1875].
- **Cross-entropy loss and AdamW** are the same tools used in pre-training, applied here to the fine-tuning objective [c3095, c3123].
- **Evaluation** combines human judgment and automated LLM scoring for a complete picture of model quality [c626, c627].

The supervised instruction fine-tuning approach covered here is the foundation for more advanced alignment techniques. Natural next steps include experimenting with different instruction formats to improve response quality [c697], scaling to larger datasets such as the full Stanford Alpaca 52,000-entry corpus [c695], and exploring reinforcement learning from human feedback (RLHF) to further align model behavior with human preferences.