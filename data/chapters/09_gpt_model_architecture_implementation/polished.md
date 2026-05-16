## Assembling the Full GPT Model Architecture

From a handful of raw words to a probability distribution over 50,257 possible next tokens — the GPT pipeline accomplishes this through a clean, fixed sequence of transformations. This chapter walks through every stage of that pipeline, from the configuration dictionary to autoregressive text generation.

---

### The Big Picture: What the GPT Pipeline Looks Like

The journey from raw text to a next-token prediction follows a fixed sequence of transformations [c811]:

1. Token IDs are projected into **token embeddings** — 768-dimensional vectors at GPT-2 scale [c818].
2. **Positional embeddings** are added to the token embeddings to produce **input embeddings** [c930].
3. A **dropout** layer is applied to the input embeddings [c932].
4. The sequence passes through **12 stacked transformer blocks** [c1000].
5. A final **layer normalization** is applied [c943].
6. An **output head** (a linear layer) converts the normalized embeddings into **logits** over the vocabulary [c965].

The result is a tensor of shape `(batch_size, sequence_length, vocabulary_size)` [c994]. Each position in the sequence receives its own probability distribution over the entire vocabulary, and the model's job is to make the distribution at position $t$ peak at the token that actually follows position $t$ in the training data [c917].

---

### The Configuration Dictionary

All architectural hyperparameters are collected in a single dictionary [c438]:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,           # Vocabulary size
    "context_length": 1024,        # Context length
    "emb_dim": 768,                # Embedding dimension
    "n_heads": 12,                 # Number of attention heads
    "n_layers": 12,                # Number of layers
    "drop_rate": 0.1,              # Dropout rate
    "qkv_bias": False              # Query-Key-Value bias
}
```

Each entry deserves a brief note:

- **`vocab_size: 50257`** — the number of tokens in the vocabulary; each token ID is projected into a 768-dimensional space [c379].
- **`context_length: 1024`** — the maximum number of tokens the model can attend to at once.
- **`emb_dim: 768`** — the dimensionality of every token and positional embedding vector [c379].
- **`n_heads: 12`** — the number of parallel attention heads inside each transformer block [c1816].
- **`n_layers: 12`** — the number of transformer blocks stacked sequentially [c888].
- **`drop_rate: 0.1`** — the fraction of embedding elements zeroed out by dropout during training [c932].
- **`qkv_bias: False`** — whether to add a bias term to the query, key, and value projection matrices.

---

### A Dummy Model First: Understanding the Skeleton

Before diving into the real sub-modules, it helps to see the overall structure in isolation. The dummy model below replaces each sub-module with a placeholder, letting you trace the complete data flow without getting lost in implementation details [c911]:

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

Even with dummy sub-modules, this skeleton reveals the complete data flow. Notice that `forward` accepts a batch of integer token IDs, not raw text strings [c990].

---

### Building the Real GPT Model

With the skeleton in mind, we can now examine each component of the full model in detail.

#### Token Embeddings

Token embeddings are projections of token IDs into higher-dimensional vector spaces [c1787]. The embedding layer is the first place where discrete symbols become continuous vectors that neural-network operations can act on [c1813]:

```python
self.pos_emb = nn.Embedding(context_length, embedding_dimension)
```
[c425]

Token IDs are converted to token embeddings before being passed through the rest of the model [c411].

#### Positional Embeddings

A transformer's attention mechanism is, by itself, permutation-invariant: it treats a bag of tokens rather than an ordered sequence. Positional embeddings fix this by injecting information about *where* each token sits [c926]. To index into the positional embedding table, we construct a tensor of position indices [c426]:

```python
pos = torch.arange(sequence_length)
```

The input embedding — the value fed into the first transformer block — is then the element-wise sum of the token embedding and the positional embedding [c930]:

> **Input embedding = token embedding + positional embedding**

This sum produces a single 768-dimensional vector per token that encodes both *what* the token is and *where* it appears [c1790, c1814].

[FIGURE: Diagram showing token ID → token embedding table → 768-dim vector, and position index → positional embedding table → 768-dim vector, both summed to produce the input embedding]

#### Dropout on Input Embeddings

Immediately after forming the input embeddings, a dropout layer is applied [c932]. Dropout randomly sets some elements of every input embedding to zero during training, acting as a regularizer that prevents the model from over-relying on any single embedding dimension and improving generalization [c932].

#### Stacked Transformer Blocks

The transformer block is the key component of the entire GPT model [c920]. The full GPT-2-scale model chains 12 of these blocks sequentially [c1000, c1818]. Each block contains [c1815]:

- A **layer normalization** layer
- A **masked multi-head attention** module
- A **dropout** layer
- **Shortcut (residual) connections**
- A second **layer normalization** layer
- A **feed-forward neural network**
- Another **dropout** layer

**Layer normalization** normalizes the embedding values for each token so that the mean is zero and the variance is one [c963], stabilizing training by preventing activations from growing or shrinking uncontrollably across many layers [c943].

**Masked multi-head attention** converts embedding vectors into context vectors [c945]. A context vector captures both the semantic meaning of a token and how much attention should be given to every other token in the sequence [c947]. The attention mechanism is the driving engine that gives LLMs their power [c1794], and it is where the key, query, and value projections operate [c1816].

**Shortcut connections** add the output of a sub-layer back to its input, providing an alternative gradient-flow path and preventing the vanishing gradient problem that would otherwise make training 12-layer networks extremely difficult [c950].

```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
)
```

`nn.Sequential` passes the output of each block directly as the input to the next, which is exactly the behavior we want [c992].

#### Final Layer Normalization

After the last transformer block, one more layer normalization is applied before the output head [c992]. Like all layer norms in the model, it sets the mean to zero and the variance to one for each token independently [c943].

#### The Output Head

The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. It is the only stage in the entire model where the tensor dimensions change: the shape goes from `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]. Every other stage preserves the embedding dimension.

Logits are the final output matrices from the transformer that contain unnormalized scores for each token in the vocabulary at each sequence position [c432]. They become proper probabilities only after a softmax is applied [c1817].

---

### The Complete Forward Pass

Putting it all together, the GPT forward pass proceeds as follows [c1020]:

1. Accept a batch of input token IDs with shape `(batch_size, sequence_length)` [c990].
2. Look up token embeddings and add positional embeddings to form input embeddings [c930].
3. Apply dropout to the input embeddings [c932].
4. Pass the sequence through 12 transformer blocks via `nn.Sequential` [c992].
5. Apply final layer normalization [c992].
6. Project through the output head to produce logits [c965].

The output is a tensor of shape `(batch_size, sequence_length, vocabulary_size)` [c994]. Remarkably, the core logic of this entire pipeline can be expressed in approximately 8 lines of code [c1019] — a testament to how cleanly the components compose, even though each transformer block is doing substantial work internally.

[FIGURE: Annotated tensor shape diagram showing how shape changes at each stage: (B, T) → (B, T, 768) → (B, T, 768) [×12 blocks] → (B, T, 768) → (B, T, 50257)]

---

### Creating a Model Instance

Once the `GPTModel` class is defined, instantiating it requires only the configuration dictionary [c996]:

```python
model = GPTModel(GPT_CONFIG_124M)
```

At this point the model has random weights [c895], so its logits won't correspond to meaningful text — but the architecture is fully functional. You can already run a forward pass through the 124-million-parameter GPT architecture on a laptop [c897]. The model outputs numerical tensors [c1015] whose logits can be decoded to predict the next token [c916].

---

### Understanding the Output: Logits and Their Shape

Given an input batch of shape `(batch_size, sequence_length)`, the model returns logits of shape `(batch_size, sequence_length, 50257)` [c994]. The logits matrix is produced by passing token embeddings through the output head [c968].

To convert these logits into a next-token prediction, three steps are needed:

1. **Extract the logits for the last position** — only the final position in the sequence dimension is needed to predict the next token [c845].
2. **Apply softmax** to convert the raw scores into a probability distribution.
3. **Select the most probable token** (or sample from the distribution).

These steps are implemented in the text generation section below.

---

### Weight Tying: 163M vs. 124M Parameters

Weight tying is a technique in which the original GPT-2 architecture reuses the token embedding layer's weights in the output layer [c1006]. Both the token embedding layer and the output layer have shape `(50257, 768)` [c1007], so sharing them is structurally natural.

Weight tying reduces the memory footprint from 163 million to 124 million parameters and lowers computational complexity [c1009]. When the output layer parameters are removed from the total count — because they are shared with the token embedding layer — the parameter count becomes exactly 124 million, matching the original GPT-2 model size [c1008].

That said, using separate token embedding and output layers yields better training and model performance than weight tying [c1010], which is why the implementation here keeps them distinct.

[FIGURE: Side-by-side comparison: weight-tied model (124M params, shared embedding/output matrix) vs. untied model (163M params, separate matrices)]

---

### Counting Parameters

To count the total number of parameters, iterate over `model.parameters()` and sum the element counts using `p.numel()` [c1004]:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

The trainable parameters in an LLM include [c1822]:
- Token embeddings
- Positional embeddings
- Layer normalization scale and shift parameters
- Query, key, and value weight matrices in multi-head attention
- Feed-forward network weights
- Final output layer weights

---

### From Logits to Text: Autoregressive Generation

A large language model generates tokens given a sequence of input tokens [c805]. The goal of this section is to convert the GPT model's output tensor into predictions of the next word [c817].

#### The `generate_text_simple` Function

`generate_text_simple` takes a model instance, input token indices (`idx`), a maximum number of new tokens to generate, and a context size [c856]. At each step it:

1. **Crops the input to the context window** [c859]:

```python
idx_cond = idx[:, -context_size:]
```

2. **Runs the forward pass without computing gradients** [c872]:

```python
with torch.no_grad():
    logits = model(idx_cond)
```

3. **Extracts the logits at the final sequence position** [c863], reducing the shape from `(batch_size, sequence_length, vocab_size)` to `(batch_size, vocab_size)`:

```python
logits = logits[:, -1, :]
```

4. **Applies softmax** to produce a proper probability distribution [c865]:

```python
probas = torch.softmax(logits, dim=-1)
```

5. **Selects the most probable token** (greedy decoding) [c867]:

```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```

6. **Appends the new token** to the running sequence and repeats [c869]:

```python
idx = torch.cat((idx, idx_next), dim=1)
```

[FIGURE: Autoregressive generation loop diagram: input tokens → model → logits → softmax → argmax → new token → append to input → repeat]

#### Putting It All Together

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Step 1: Crop to context window
        idx_cond = idx[:, -context_size:]

        # Step 2: Forward pass (no gradients needed)
        with torch.no_grad():
            logits = model(idx_cond)

        # Step 3: Focus on the last time step
        logits = logits[:, -1, :]

        # Step 4: Softmax to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Step 5: Greedy selection
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Step 6: Append and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

#### A Concrete Example

Starting with the input `'hello i am'` and `max_new_tokens=6`, the GPT model generates the complete output `'hello i am a model ready to help'` [c854]. Note that during training, sampling techniques modify the softmax output so the model does not always select the most likely token, introducing variability and creativity into generated text [c878]. Greedy decoding — always picking the argmax — tends to produce repetitive output by comparison.

---

### The Loss Function

To train the model, we need a way to measure how wrong its predictions are. The loss function used in LLM training is cross-entropy loss between the predicted next token and the actual next token [c1821]. Cross-entropy penalizes the model whenever it assigns low probability to the correct token, and the attention mechanism — the driving engine that gives LLMs their power [c1794] — is trained end-to-end through this signal.

---

### What Comes After Pre-Training

Once a model has been pre-trained, it is typically fine-tuned for specific tasks. There are two main flavors [c1827, c1828]:

- **Classification fine-tuning**: the model is trained to classify inputs into categories, such as determining whether an email is spam or not spam [c1827].
- **Instruction fine-tuning**: the model is trained on a dataset of instructions, inputs, and outputs to perform well on instruction-following tasks [c1828].

Fine-tuned models are evaluated using benchmarks such as MMLU (Measuring Massive Multitask Language Understanding), which uses 57 tests to assess LLM performance across diverse domains [c1831]. Other evaluation approaches include human evaluation, where humans compare and rate the outputs of different LLMs [c1832], and LLM-based evaluation, where a powerful large language model judges the outputs of another [c1833].

---

### What We Built From Scratch

The GPT architecture was built completely from scratch without using external libraries like LangChain [c900], with every sub-module implemented and coded from the ground up [c901]. The work covered layer normalization, GELU activation, feed-forward networks, shortcut connections, and transformer blocks [c904].

The complete architecture consists of [c1812]:
- Input tokens converted to embeddings
- Positional embeddings added to token embeddings
- The combined embeddings passed through transformer blocks
- A final layer normalization
- A final neural network (the output head) that produces logits for next-token prediction

Each transformer block contains a normalization layer, masked multi-head attention, dropout, shortcut connections, a second normalization layer, a feed-forward neural network, and another dropout layer [c1815]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models like GPT-4 [c824].

---

### Summary

This chapter assembled the full GPT model from its individual components and demonstrated how to generate text autoregressively. The key takeaways are:

- The GPT forward pass is a linear pipeline: embeddings → dropout → 12 transformer blocks → layer norm → output head → logits [c811, c992].
- The output shape is always `(batch_size, sequence_length, vocab_size)` [c994], where `vocab_size = 50257` for GPT-2.
- The output head is the only stage where the embedding dimension changes [c973].
- Weight tying reduces the parameter count from 163M to 124M by sharing the token embedding and output matrices [c1009], though keeping them separate can improve performance [c1010].
- Autoregressive text generation works by repeatedly feeding the model the current sequence, extracting the last-position logits, applying softmax, selecting a token, and appending it [c856].
- The training loss is cross-entropy between predicted and actual next tokens [c1821].

The model outputs logits that will be converted into tokens and text in subsequent work [c1021].