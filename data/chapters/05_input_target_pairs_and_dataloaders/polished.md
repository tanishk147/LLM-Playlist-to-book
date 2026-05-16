## Creating Input-Target Pairs and Data Loaders

Before a large language model can learn anything, it needs data in a very specific form: for every sequence it sees, it must know what the correct next token should be. This chapter covers exactly how that transformation happens—from a plain text file, through tokenization, through a sliding window that carves out input-target pairs, and finally into batched PyTorch tensors ready to drive gradient updates.

The pipeline described here sits between the tokenizer (covered in the previous chapter) and the embedding layer (covered next).

---

### 5.1 The Self-Supervised Nature of LLM Training

LLMs sidestep the need for manual annotation entirely. The labels are already hiding inside the text itself [c2719]: the sentence structure provides the supervision signal, so given the words so far, the next word is the target [c2711]. Because input-target pairs are derived from the sentence structure without manual labeling, this approach is called **self-supervised learning** [c2719]. Any large corpus of text—books, web pages, code—can be turned into a training set with no annotation cost whatsoever [c1599].

#### 5.1.1 What the Model Is Actually Learning

During training, the prediction task is always the same: predict the next word that follows the input block [c2720]. The model learns to predict one word at a time [c2712], and the output length is always one token [c2717].

To make this concrete, consider the sentence "LLMs learn to predict." The input-target pairs look like this [c2732]:

- Input: `LLMs` → Target: `learn`
- Input: `LLMs learn` → Target: `predict`
- Input: `LLMs learn to` → Target: `predict` (the next token)

In the first iteration, the input is the first word and the target is the second word [c2713]. In subsequent iterations, the previous target becomes part of the input and the next word becomes the new target [c2714]. Words that come after the target word are masked and not shown to the LLM during training [c2715, c2721].

This structure also makes LLMs **autoregressive**: the output of one iteration becomes the input of the next [c2718]. At inference time, the model generates text one token at a time, feeding each predicted token back as part of the growing input sequence.

[FIGURE: Diagram showing a sentence being split into overlapping input-target pairs, with masking applied to tokens beyond the current target]

---

### 5.2 Input-Target Pairs: The Shift-by-One Pattern

Two variables define every training example: `x` contains the input tokens, and `y` contains the targets—which are simply the inputs shifted by one position [c2729]. Concretely, the input array `[1, 2, 3, 4]` maps to the output array `[2, 3, 4, 5]` [c2731]; the target is always the input shifted one token to the right [c1601].

#### 5.2.1 Context Size

The **context size** is the number of tokens the model looks at simultaneously to predict the next token [c2736, c2738]. With a context size of 4, if the input is tokens 1, 2, 3, 4, the target output is tokens 2, 3, 4, 5 [c2737]. For example:

- `x` consists of the first 4 token IDs: `290, 4920, 2241, 287` [c2739]
- If the input is tokens `290` and `4920`, the output is `2241` [c2742]

The following loop illustrates how pairs are built manually [c2747]:

```python
for i in range(1, context_size + 1):
    context = encoded_data[:i]
    desired = encoded_data[i]
```

When `i` equals 1, the context is the first token ID (`290`) and the desired output is the next token ID (`4920`) [c2748]. When `i` equals 3, the context is the first three token IDs (`290`, `4920`, and `2241`) and the desired output is `287` [c2750].

This highlights a key difference from regression and classification problems, where one input-output pair corresponds to exactly one prediction task [c2755]. In language models, a single input-output pair corresponds to multiple prediction tasks—one per position in the context window [c2756]. For a context size of 4, a single input-target pair therefore contains 4 predictions [c2815]. Tokens on the left of the arrow represent the input to the model; the token on the right represents the target token ID the model is supposed to predict [c2752].

[FIGURE: Table showing context size 4 with four prediction tasks per input-target pair, illustrating the shift-by-one relationship]

---

### 5.3 The Sliding Window Approach

Rather than constructing pairs by hand, we need a systematic way to sweep through the entire text and generate them efficiently. The mechanism is a **sliding window** [c2722, c2767].

#### 5.3.1 Stride

**Stride** determines how many tokens to skip when sliding the context window to create the next input-output pair [c2777]. The sliding window works by advancing the input window by one token position to produce the corresponding output [c2773].

Setting stride equal to context length prevents overlap between consecutive input batches while ensuring no tokens are missed [c2817]. Every token in the corpus can therefore participate in training, and each window covers a completely fresh block of text. The stride used in creating input-target pairs is typically equal to the context size [c1602]; in the example implementation, a context length of 4 tokens and a stride of 4 are used [c2786].

[FIGURE: Sliding window diagram showing stride=1 (heavy overlap) vs stride=context_size (no overlap) over a token sequence]

The shift-by-one relationship is preserved throughout: the output tensor is created by shifting the input tensor by one position [c2768], and each row of the input tensor `X` represents one input context of size equal to the context window [c2769].

---

### 5.4 Building the Dataset Class

A dataset for language modeling must consist of input-output pairs, not just raw tokens [c2774]. PyTorch provides built-in `Dataset` and `DataLoader` classes for efficient data processing [c2766], and we use them directly. The goal is to produce an input tensor containing the text the LLM sees and a target tensor containing what it should predict [c2765]. Because PyTorch optimization procedures require data in tensor format rather than raw arrays [c2763], a tensor—which can be thought of as a multi-dimensional array [c2764]—is the natural container.

#### 5.4.1 The GPTDatasetV1 Class

The `GPTDatasetV1` class is based on PyTorch's `Dataset` class and defines how individual rows are fetched from the dataset [c2789]. It implements a **map-style dataset**, which requires a `__getitem__` method for the `DataLoader` to fetch individual samples [c2780]. The class takes four arguments: text file path, tokenizer, `max_length` (context size), and stride [c2775].

The implementation follows four steps [c2772]:

1. Tokenize the entire text.
2. Use a sliding window to generate input-output pairs.
3. Return the total number of rows in the dataset (via `__len__`).
4. Return a single row from the dataset (via `__getitem__`).

Each row in the dataset contains a number of tokens equal to the context length [c2781], and `__getitem__` returns the input tensor row and target tensor row at a given index [c2779].

#### 5.4.2 Loading the Raw Text

Before constructing the dataset, we read and encode the source text. The following snippet reads a text file and encodes it into token IDs [c2724]:

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```

An encoder converts text into token IDs [c2728]. Tokenization is the first step of the data pre-processing stage [c2707], and input-target pairs must be created before vector embeddings are fed into the LLM training process [c2709]. The resulting input-output pairs `X` and `Y` serve as inputs and outputs for language model training [c2760], with each pair containing multiple prediction tasks equal to the context size [c2771].

---

### 5.5 The DataLoader

With the dataset class in place, a `DataLoader` iterates over it and returns inputs and targets as PyTorch tensors [c2762]. The `create_data_loader` function initializes a tokenizer, creates a dataset instance, and wraps it in a `DataLoader` for batch processing [c2783].

#### 5.5.1 Function Signature and Parameters

The `create_data_loader` function accepts the following arguments: text file, `batch_size` (default 4), `max_length`, `stride`, and `num_workers` [c2784]. The `DataLoader` also allows specification of additional parameters including batch size, maximum length, and stride [c2795].

#### 5.5.2 Key DataLoader Parameters

**Batch size** controls how many input-output pairs are grouped together before the model updates its parameters [c2791]. Data is chunked into batches so that parameter updates occur after processing multiple examples rather than the entire dataset at once [c2791]. Batch size is a trade-off and a hyperparameter to experiment with when training LLMs [c2804]:

```python
data_loader = DataLoader(dataset, batch_size=8)
```

**`drop_last`** is set to `True` to drop the final batch if it is shorter than the specified batch size, preventing loss spikes during training [c2782].

**`num_workers`** enables parallel processing across different CPU threads [c2792, c2821].

#### 5.5.3 How the DataLoader Fetches Data

The `DataLoader` accesses the `__getitem__` method of the dataset class to retrieve input-output pairs for batch processing [c2788], and data is processed in batches to enable parallel computing across multiple CPUs [c2761]. A `DataLoader` with batch size of 1 and context size of 4 produces input-output pairs where the output tensor is the input tensor shifted by one position [c2794]. In the example shown, batch size is set to 8, meaning 8 input-output pairs are processed before each parameter update [c2820].

[FIGURE: Diagram showing DataLoader pulling batches from GPTDatasetV1, with input tensor X and target tensor Y side by side, Y being X shifted by one]

---

### 5.6 Hyperparameter Choices

Several hyperparameters govern the behavior of the data pipeline. Understanding their effects helps you make informed choices when scaling up.

#### 5.6.1 Context Size

Context size represents how many tokens the model attends to at one time when predicting the next word [c2738]. A context size of 4 is relatively small; it is common to train LLMs with context sizes of at least 256 [c2802]. GPT-2 and GPT-3 models typically use a context length of 256 tokens [c2785, c1600]. In the example implementation, `max_length` is set to four, meaning each tensor in a batch contains four token IDs [c2801].

#### 5.6.2 Stride and Overlap

Stride dictates the overlap between consecutive batches [c2817]. Setting stride equal to context length prevents overlap while ensuring no tokens are missed [c2817], and stride is increased to 4 in the example to utilize the dataset fully without skipping any samples [c2810]. A smaller stride increases the number of training examples but also increases correlation between consecutive batches, which can slow convergence.

#### 5.6.3 Batch Size

In the example, batch size is set to 8, meaning 8 input-output pairs are processed before each parameter update [c2820]. Larger batches give more stable gradient estimates but require more memory and may generalize less well [c2804].

---

### 5.7 The Complete Pipeline

Here is the full journey from raw text to batched tensors, tying together everything covered in this chapter.

**Step 1 — Tokenization.** The raw text file is read and encoded into a flat sequence of integer token IDs [c2724, c2728]. Tokenization is the first step of data pre-processing [c2707].

**Step 2 — Sliding window dataset.** `GPTDatasetV1` sweeps a window of width `max_length` across the token sequence, stepping by `stride` each time, and stores each (input, target) pair as a pair of PyTorch tensors [c2772, c2773, c2768].

**Step 3 — DataLoader wrapping.** The dataset is wrapped in a PyTorch `DataLoader` with the desired `batch_size`, `drop_last=True`, and `num_workers` [c2783, c2782, c2792].

**Step 4 — Batch delivery.** Each batch contains `batch_size` rows; each row is a tensor of `max_length` token IDs [c2788, c2781]. The `DataLoader` extracts input-output pairs in batches which are then converted into vector embeddings before being fed into the model [c2822]. A `GPTDatasetV1` class paired with a `DataLoader` generates these input-output pairs in a structured manner [c2816].

[FIGURE: End-to-end pipeline diagram: raw text → tokenizer → token IDs → sliding window → GPTDatasetV1 → DataLoader → batched (X, Y) tensors → embedding layer]

#### 5.7.1 Verifying the Output

The output tensor is the input tensor shifted by one position [c2794], and input-target pairs are sequences where the target is the input shifted by one position [c2814]. With `batch_size=8`, eight such pairs are stacked into a matrix before any gradient update occurs [c2820], and each row of the input tensor `X` represents one input context of size equal to the context window [c2769].

---

### 5.8 Why This Design Works

The self-supervised framing means the entire pre-training corpus is its own label set [c2719, c1599]—any text can be fed in without annotation.

The sliding window with stride equal to context size ensures that every token in the corpus participates in training exactly once per epoch [c2817, c1602].

The shift-by-one target construction means that a single forward pass simultaneously trains the model on `context_size` different prediction tasks [c2756, c2771, c2815], making training highly data-efficient.

The `DataLoader` abstraction handles batching and parallel loading transparently [c2766, c2821], keeping the training loop clean. Finally, the output of this entire pipeline—batched integer tensors—feeds directly into the embedding layer, which converts token IDs into dense vectors before the transformer processes them [c2822].

---

### Summary

- LLMs use **self-supervised learning**: input-target pairs are derived from the text itself, with no manual labels required [c2719].
- The **shift-by-one** pattern defines every target: `y` is `x` shifted right by one token [c2729, c1601].
- **Context size** controls how many tokens the model sees at once; typical values are at least 256 [c2802, c2785].
- The **sliding window** sweeps across the token sequence, generating one input-target pair per step [c2722, c2767].
- **Stride** controls overlap between consecutive windows; stride equal to context size eliminates overlap [c2817, c1602].
- The **`GPTDatasetV1`** class wraps the sliding window logic in a PyTorch `Dataset`, implementing `__getitem__` and `__len__` [c2789, c2780].
- The **`DataLoader`** batches dataset rows into tensors, with `drop_last=True` to avoid ragged final batches [c2762, c2782].
- Each batch of tensors flows next into the embedding layer, completing the bridge from raw text to model input [c2822].