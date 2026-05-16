## Instruction Fine-Tuning: Building a Personal Assistant

By this point in the book, you have built a GPT-style language model from scratch, pre-trained it on raw text, and watched it learn to predict the next token. That model can complete sentences and generate coherent prose, but it cannot reliably *follow instructions*. Ask it to "correct the grammar in this sentence" and it is just as likely to continue the sentence as to fix it. This chapter closes that gap.

We will walk through the complete instruction fine-tuning pipeline from end to end [c587]. The journey has three stages: preparing the dataset, fine-tuning the model, and evaluating what we built [c3087]. By the end of the chapter you will have a working personal assistant that can answer questions, edit text, and perform a range of language tasks—all built on top of a GPT-2 medium checkpoint.

---

### Why Instruction Fine-Tuning?

A pre-trained language model is a powerful but undirected engine. It has absorbed enormous amounts of world knowledge, but it has no concept of what a *user* wants. Instruction fine-tuning is the process of training a pre-trained LLM with a specific dataset to teach it to correctly follow instructions [c1722]. The mechanism is straightforward: you collect a large set of instruction-response pairs and use them to update the model's weights [c181].

The result is a model that has been modified to understand instruction input-output pairs from a specific dataset [c1761]. Intuitively, the pre-training gave the model a rich internal representation of language; fine-tuning steers that representation toward the behavior you want.

Fine-tuning serves two main purposes: training the LLM to follow instructions and training on domain-specific data relevant to the deploying organization [c177]. To make this concrete, consider a healthcare virtual assistant. Fine-tuning it with domain-specific instructions ensures it understands medical terminology, follows healthcare guidelines, and personalizes responses based on patient history [c176]. More generally, building a personal assistant requires the LLM to follow instructions such as correcting grammar, removing filler words, making text concise, and adjusting tone [c168].

Fine-tuning is also practical from a compute standpoint. Fine-tuning on a specific dataset after loading pre-trained weights requires significantly less computational time than training from scratch [c1733]. You are not teaching the model language from zero—you are redirecting knowledge it already has.

---

### The Three-Stage Pipeline

Before diving into code, it helps to have the full picture in mind.

[FIGURE: Three-stage instruction fine-tuning pipeline: Stage 1 (dataset preparation: download, format, batch, create data loaders), Stage 2 (load pretrained LLM and run fine-tuning training loop), Stage 3 (evaluation: extract responses, qualitative comparison, automated LLM scoring)]

**Stage 1 — Dataset preparation.** Download the instruction data, format it using the Alpaca prompt template, and create training, validation, and test data loaders [c3088]. Significant time should be spent on batching the dataset and creating data loaders, because correct implementation of this step simplifies all subsequent steps [c2333].

**Stage 2 — Fine-tuning.** Load a pre-trained GPT-2 checkpoint and run the supervised training loop on the instruction data [c3087].

**Stage 3 — Evaluation.** Extract model responses on the test set, perform qualitative inspection, and run automated scoring using a larger LLM [c587].

The instruction fine-tuning pipeline covered in this chapter spans nine steps in total: data preparation, fine-tuning, evaluation, response extraction, qualitative evaluation, and quantitative scoring [c714].

---

### The Instruction Dataset

#### What the Data Looks Like

The instruction dataset used for fine-tuning consists of 1,100 instruction input and output pairs [c588]. Each entry in the dataset contains three keys: `instruction`, `input`, and `output` [c188].

Here are three representative examples drawn from the dataset:

**Example 1** [c178]:
- *Instruction:* Convert 45 km to meters
- *Response:* 45 km is 45000 meters

**Example 2** [c179]:
- *Instruction:* Provide a synonym for bright
- *Response:* A synonym for bright is radiant

**Example 3** [c180]:
- *Instruction:* Edit the following sentence to remove all passive voice. "The song was composed by the artist."
- *Response:* The artist composed the song.

Notice that some entries have a non-empty `input` field (Example 3 passes the sentence to edit), while others do not (Examples 1 and 2). This distinction matters when we format the prompt.

For comparison, the Stanford Alpaca repository contains a much larger dataset of 52,000 instruction-output pairs [c695]. Our 1,100-entry dataset is a smaller, curated subset that is practical for experimentation on a single GPU or even a laptop.

#### Loading and Splitting the Data

The fine-tuning process begins by loading a training dataset consisting of instruction-response pairs [c182]. After loading, the data is split into training, validation, and test portions. The test set is used to evaluate the model's performance after fine-tuning [c605].

---

### The Alpaca Prompt Format

Raw instruction-response pairs cannot be fed directly to the model. We need a consistent text template that tells the model where the instruction ends and where the response should begin. The dominant convention is the **Alpaca prompt format**, a specific formatting convention for converting instruction-input-output pairs into prompts for fine-tuning large language models, maintained by Stanford in the Stanford Alpaca repository [c1842].

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

When the `input` field is empty, the `Input:` section is omitted entirely [c199]. This keeps the prompt clean and avoids confusing the model with an empty section.

For contrast, the Phi-3 prompt style takes a different approach: the instruction and input are fused together in a `user` field, and the output is placed in an `assistant` field [c200]. Both styles are valid; we use Alpaca throughout this chapter because it is the more widely documented convention [c2289].

#### Implementing `format_input`

The following function converts a dataset entry into an Alpaca-formatted prompt string. It handles both the case where `input` is present and where it is absent.

# Source: [c201]
```python
def format_input(entry):
    instruction_text = "Below is an instruction that describes a task paired with an input that provides further context. Write a response that appropriately completes the request."
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

Batching instruction data is more involved than batching plain pre-training text. The five-step process is [c2294]:

1. Format data using the prompt template.
2. Tokenize the formatted data into token IDs.
3. Pad sequences to equal length within each batch.
4. Create target token ID sequences shifted by one position.
5. Replace padding tokens in the targets with `-100`.

Let us work through each step.

#### Step 1: Format with the Prompt Template

We have already seen `format_input`. Each entry in the dataset is passed through this function to produce a fully formatted string [c2291].

#### Step 2: Tokenization

Tokenization is the process of converting data into a numerical representation by converting sentences into token IDs [c1854]. The formatted prompt string is passed through the GPT-2 tokenizer, producing a sequence of integer token IDs.

#### Step 3: Padding to Equal Length

Neural networks require all sequences in a batch to have the same length. Padding is the process of adjusting the length of token ID sequences so that all samples in a batch have the same length [c1861].

The special token used for padding is the **end-of-text token**. In GPT-2, the end-of-text token has token ID `50256` [c1863]. Shorter sequences are extended by appending copies of this token until they match the length of the longest sequence in the batch.

Batching the dataset means converting multiple data samples into a batch where each sample is represented as a numerical array (row) with uniform dimensions [c1848].

[FIGURE: Diagram showing three sequences of different lengths being padded with token ID 50256 to form a uniform rectangular batch matrix]

#### Step 4: Creating Target Sequences

During instruction fine-tuning, the LLM is trained using next-token prediction, where the entire formatted prompt serves as both input and target pairs [c2292]. The target sequence is simply the input sequence shifted to the right by one position [c2323].

Concretely, target token IDs are created by shifting the input tokens to the right by one position and appending a padding token at the end to maintain equal length between input and target sequences [c1868, c1894].

# Source: [c1896]
```python
targets = inputs[1:] # Shift inputs right by one by removing first element
```

Target pairs are created by shifting the input sequence to the right by one token and appending an end-of-text token [c2298].

[FIGURE: Side-by-side illustration of input token sequence and target token sequence, showing the one-position rightward shift with an end-of-text token appended at the end of the target]

#### Step 5: Replacing Padding Tokens with `-100`

Here is a subtle but important detail. After padding, both the input and target tensors contain token ID `50256` in the padded positions. If we compute cross-entropy loss over those positions, we are penalizing the model for not predicting padding—which is meaningless and harmful to training.

The fix is to replace padding token IDs in the *target* tensor with `-100`. PyTorch's cross-entropy loss function has a built-in `ignore_index` parameter; when a target token ID equals `-100`, that position is excluded from the loss calculation entirely [c1902].

There is one exception: the *first* occurrence of the end-of-text token in the target is kept as-is, because it is a legitimate prediction target (the model should learn to emit end-of-text when it finishes a response) [c1902].

Additionally, to mask target token IDs, we replace the token IDs of the instruction and input portions in the target text with `-100` [c1917]. This means the model is only trained to predict the *response* tokens, not to reproduce the instruction it was given. Intuitively, there is no point penalizing the model for not memorizing the prompt verbatim.

#### The Custom Collate Function

All five steps are bundled into a custom collate function that PyTorch's `DataLoader` calls on each batch. A custom collate function takes a batch of sequences and requires a padding token ID (`50256`) and target device (`cpu`) as inputs [c1886].

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

The collate function converts tokens to token IDs, pads sequences to equal length within the batch, replaces padding tokens (`50256`) with `-100`, and returns input and target tensors [c2303].

---

### Creating the Data Loaders

Data loaders are an efficient way to collect different batches sequentially and access them in an iterative manner during model training [c2284]. Once the collate function is in place, creating the data loaders is straightforward.

First, select the compute device:

# Source: [c2309]
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Then wrap the training data in a dataset object:

# Source: [c2314]
```python
train_dataset = InstructionDataset(training_data)
```

The `InstructionDataset` class stores the raw entries and applies `format_input` on demand. The `DataLoader` wraps the dataset and calls the custom collate function to assemble batches.

Stage one of instruction fine-tuning focuses entirely on preparing the dataset itself [c2335]. Once the data loaders are running correctly, the remaining stages become much simpler.

---

### Loading the Pre-Trained Model

With the data pipeline in place, we turn to the model. Rather than training from scratch, we load a pre-trained GPT-2 checkpoint and fine-tune it on the instruction data. A pre-trained GPT-2 model with 355 million weights was loaded as the foundation for instruction fine-tuning [c591].

The following code sets up the model configuration and selects the GPT-2 medium variant:

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

The `download_and_load_gpt2` utility downloads the OpenAI checkpoint and loads the weights into our GPT architecture. Fine-tuning is necessary to improve the model's ability to comprehend and appropriately respond to instruction-following requests [c1760].

---

### The Fine-Tuning Training Loop

#### Loss Function

The loss function used for fine-tuning is cross-entropy loss [c3112], the same loss function used during pre-training [c3095]. Cross-entropy loss is conceptually the negative of the logarithm of the probability [c3113]. For a predicted probability $p$ assigned to the correct token:

$$\mathcal{L} = -\log(p)$$

Cross-entropy loss is used to calculate the loss on a batch of data [c3118]. Because we set `ignore_index=-100` in the collate function, the loss is computed only over the response tokens—not over the instruction, input, or padding positions.

#### Optimizer

The optimizer used for fine-tuning is **AdamW**, the Adam optimizer with weight decay, implemented as `torch.optim.AdamW` [c3123].

#### Parameter Updates

The backward pass calculates the gradient of the loss, which is the partial derivative of the loss with respect to all model parameters [c3100]. Gradient descent is then used to update model parameters [c3102]. The update rule is [c3103]:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

where $w_{\text{new}}$ is the updated weight, $w_{\text{old}}$ is the previous weight, $\alpha$ is the learning rate, and $\frac{\partial \mathcal{L}}{\partial w}$ is the partial derivative of the loss with respect to $w$.

#### Monitoring Training

The dataset processing, batching, and data loaders have already been implemented before the fine-tuning training loop [c3116]. During training, we track both training loss and validation loss. The evaluation iterations parameter specifies after how many iterations the loss on the evaluation dataset is calculated [c3130].

Fine-tuning the LLM on instruction data modifies the weights and parameters of the GPT model to understand instruction input-output pairs from a specific dataset [c1761].

#### Saving and Loading the Fine-Tuned Model

After training completes, save the model's state dictionary so you can reload it later without re-running the training loop.

# Source: [c640]
```python
torch.save(model.state_dict(), 'gpt2medium355_million_sft.pth')
```

To reload the weights:

# Source: [c641]
```python
model.load_state_dict(torch.load('gpt2medium355_million_sft.pth'))
```

The lecture covered fine-tuning a model on an instruction dataset from scratch [c707]. With the weights saved, we can move on to evaluation without repeating the training run.

---

### Evaluating the Fine-Tuned Model

Evaluation has three sub-steps: extracting model responses, performing qualitative inspection, and running automated quantitative scoring [c587].

#### Step 1: Extracting Responses

After fine-tuning the LLM on the training portion of the instruction dataset, the model's performance is evaluated on the testing dataset [c605]. The following procedure loops over the test data, generates output from the fine-tuned LLM, converts token IDs back to text, and isolates the response by removing the instruction input from the output [c607].

# Source: [c636]
```python
# The code iterates through all instruction-input pairs in the test data,
# calls the generate function on each pair, and appends the model response
# to the instruction_data_with_response file.
```

The result is a list of entries, each containing the original instruction, the expected (ground-truth) response, and the model's generated response.

#### Step 2: Qualitative Evaluation

The first evaluation method is simply reading the outputs. Human preference comparison is an evaluation method where a human uses their own intuition and understanding to benchmark or compare LLMs [c626]. This is slow and does not scale, but it is invaluable for catching systematic failure modes that automated metrics miss.

To improve fine-tuning performance, you can experiment with different prompts or instruction formats to guide model responses more effectively [c697].

#### Step 3: Automated LLM-Based Scoring

For scalable evaluation, we use a larger language model as a judge. Method 3 LLM evaluation involves comparing the true expected output against the model response by asking a large language model to assign a score based on this comparison [c627]. Step 7 of evaluating a fine-tuned LLM involves implementing automated response evaluation using a larger pre-trained LLM [c646].

This approach is sometimes called "LLM-as-a-judge." You pass the instruction, the ground-truth response, and the model's response to a capable LLM and ask it to rate the model's answer on a numeric scale. Aggregating these scores across the test set gives a quantitative measure of fine-tuning quality.

---

### Running Llama 3 Locally with Ollama

To run the judge LLM locally without sending data to an external API, we use **Ollama**. Ollama is an efficient application to run large language models on your laptop [c648]. It handles model downloading, quantization, and inference in a single lightweight binary.

Importantly, Ollama is a tool for generating text using LLM inference and does not support training or fine-tuning LLMs [c657]. It is purely an inference engine—perfect for our evaluation use case.

#### Starting Ollama

To pull and run the Llama 3 model:

# Source: [c660]
```
ollama run llama3
```

If you want to run Ollama as a background server (so you can call it from Python):

# Source: [c662]
```
ollama serve
```

Once the server is running, your Python evaluation script can send HTTP requests to the local Ollama endpoint, passing the instruction, ground-truth response, and model response, and receiving a numeric score in return.

---

### Putting It All Together

Let us step back and review the complete pipeline we have built.

[FIGURE: End-to-end instruction fine-tuning pipeline diagram: raw JSON dataset → format_input (Alpaca template) → tokenizer → custom collate function (padding + -100 masking) → DataLoader → GPT-2 medium (355M) → AdamW training loop → saved checkpoint → response extraction → qualitative review → Ollama/Llama 3 scoring]

1. **Dataset.** 1,100 instruction-response pairs, each with `instruction`, `input`, and `output` fields [c188, c588].
2. **Formatting.** The Alpaca prompt template wraps each entry into a structured string [c196, c201].
3. **Batching.** Tokenization, padding with token ID `50256`, target shifting, and `-100` masking are handled by the custom collate function [c1875, c1911].
4. **Data loaders.** PyTorch `DataLoader` objects iterate over training, validation, and test splits [c2284].
5. **Model.** GPT-2 medium (355M parameters) loaded from a pre-trained checkpoint [c591, c1744].
6. **Training.** Cross-entropy loss over response tokens only, optimized with AdamW [c3095, c3123].
7. **Saving.** Model weights persisted to disk with `torch.save` [c640].
8. **Response extraction.** The model generates responses for all test entries [c636].
9. **Evaluation.** Qualitative human review [c626] and automated LLM scoring via Ollama [c627, c648].

The instruction fine-tuning pipeline covered nine steps in total [c714], and we have now walked through every one of them.

---

### Summary and What Comes Next

This chapter introduced supervised instruction fine-tuning [c181] and showed how to build the entire pipeline from raw data to a scored, evaluated personal assistant. The key ideas are:

- **Instruction fine-tuning** teaches a pre-trained model to follow instructions by training on instruction-response pairs [c1722].
- **The Alpaca prompt format** provides a consistent template that separates instruction, input, and response [c1842, c196].
- **Data batching** requires careful handling of padding and the `-100` ignore index to ensure the loss is computed only over response tokens [c1875].
- **Cross-entropy loss and AdamW** are the same tools used in pre-training, applied here to the fine-tuning objective [c3095, c3123].
- **Evaluation** combines human judgment and automated LLM scoring for a complete picture of model quality [c626, c627].

The fine-tuning approach covered here—supervised instruction fine-tuning—is the foundation for more advanced alignment techniques. Future directions include experimenting with different instruction formats to improve response quality [c697], scaling to larger datasets like the full Stanford Alpaca 52,000-entry corpus [c695], and exploring reinforcement learning from human feedback (RLHF) to further align model behavior with human preferences.