## Assembling the Full GPT Model Architecture

At this point in the book, you have built every individual piece of the GPT puzzle: token embeddings, positional embeddings, multi-head attention, feed-forward networks, layer normalization, shortcut connections, and full transformer blocks. The goal of this chapter is to snap those pieces together into a single, coherent `GPTModel` class, understand exactly what happens during a forward pass, count the parameters, and finally turn the model's raw numerical output into actual text.

This is the chapter where everything clicks. By the end you will have a working 124-million-parameter GPT-2-scale model running on a laptop, producing text from scratch.

---

### The Big Picture: What the GPT Pipeline Looks Like

Before writing a single line of code, it helps to have a clear mental map of the full pipeline. The journey from raw text to a next-token prediction follows a fixed sequence of transformations [c811]:

1. An input sentence is tokenized into **token IDs**.
2. Token IDs are projected into **token embeddings** (768-dimensional vectors for GPT-2 scale) [c818].
3. **Positional embeddings** are added to the token embeddings to produce **input embeddings** [c930].
4. A **dropout** layer is applied to the input embeddings [c932].
5. The sequence passes through **12 stacked transformer blocks** [c1000].
6. A final **layer normalization** is applied [c943].
7. An **output head** (a linear layer) converts the normalized embeddings into **logits** over the vocabulary [c965].

[FIGURE: Linear pipeline diagram showing: Input Sentence → Token IDs → Token Embeddings (768-dim) → + Positional Embeddings → Dropout → Transformer Block ×12 → Layer Norm → Output Head → Logits (vocab_size)]

The output is a tensor of shape `(batch_size, sequence_length, vocabulary_size)` [c994]. Each position in the sequence gets its own probability distribution over the entire vocabulary, and the model's job is to make the distribution at position $t$ peak at the token that actually follows position $t$ in the training data [c917].

Intuitively, every stage in this pipeline has a clear responsibility: embeddings give the model a rich numerical representation of each token and its position; transformer blocks refine those representations by letting tokens attend to one another; and the output head translates the final representations into vocabulary-scale scores.

---

### The Configuration Dictionary

Rather than hard-coding hyperparameters throughout the model class, GPT implementations typically collect all configuration values into a single dictionary. Here is the configuration for the 124-million-parameter GPT-2 model:

# Source: [c438]
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

- **`vocab_size: 50257`** — the number of unique tokens the model knows about. Every logit vector will have exactly this many entries.
- **`context_length: 1024`** — the maximum number of tokens the model can process at once. This determines the size of the positional embedding table.
- **`emb_dim: 768`** — the width of every embedding vector throughout the model. Each token ID is projected into a 768-dimensional space [c379].
- **`n_heads: 12`** — the number of parallel attention heads inside each transformer block [c1816].
- **`n_layers: 12`** — the number of transformer blocks stacked sequentially [c888].
- **`drop_rate: 0.1`** — the fraction of embedding elements zeroed out by dropout during training [c932].
- **`qkv_bias: False`** — whether to add a bias term to the query, key, and value projection matrices. The original GPT-2 omits these biases.

This dictionary is passed into the model constructor and every sub-module, keeping the code clean and making it trivial to experiment with different scales.

---

### A Dummy Model First: Understanding the Skeleton

Before presenting the full implementation, it is instructive to look at a *dummy* version of the GPT model — one where the transformer blocks and layer normalization are replaced with placeholder classes. This lets you see the overall structure without getting lost in the details of any one sub-module.

# Source: [c911]
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

Even with dummy sub-modules, this skeleton reveals the complete data flow. Notice that the `forward` method takes a batch of integer token IDs, not text strings [c990]. The model is entirely numerical from start to finish.

The dummy model is useful for verifying tensor shapes before committing to the full implementation. Once you confirm that the shapes are correct at every stage, you can swap in the real sub-modules one by one.

---

### Building the Real GPT Model

Now let us walk through each component of the real model in detail.

#### Token Embeddings

Token embeddings are projections of token IDs into higher-dimensional vector spaces [c1787]. In PyTorch, this is a simple lookup table:

# Source: [c425]
```python
self.pos_emb = nn.Embedding(context_length, embedding_dimension)
```

(The same `nn.Embedding` API is used for both token and positional embeddings; only the table dimensions differ.) Each token ID is mapped to a unique 768-dimensional vector. These vectors are learned during training — the model adjusts them so that tokens with similar meanings end up with similar vectors.

Token IDs are converted to token embeddings before being passed through the rest of the model [c411]. The embedding layer is the first place where discrete symbols become continuous vectors that neural network operations can act on [c1813].

#### Positional Embeddings

A transformer's attention mechanism is, by itself, permutation-invariant: it treats a bag of tokens, not a sequence. Positional embeddings fix this by injecting information about *where* each token sits in the sequence [c926].

The positional embedding layer is another `nn.Embedding` table, this time of shape `(context_length, emb_dim)` — one vector per possible position. To index into it, we need a tensor of position indices:

# Source: [c426]
```python
pos = torch.arange(sequence_length) creates a tensor [0, 1, 2, ..., sequence_length-1] to index into the positional embedding matrix.
```

The input embedding — the actual value fed into the first transformer block — is the sum of the token embedding and the positional embedding [c930]:

$$\text{Input Embedding} = \text{Token Embedding} + \text{Positional Embedding}$$

This addition is element-wise and produces a single 768-dimensional vector per token that encodes both *what* the token is and *where* it appears [c1790, c1814].

[FIGURE: Diagram showing token ID → token embedding table → 768-dim vector, and position index → positional embedding table → 768-dim vector, both summed to produce the input embedding]

#### Dropout on Input Embeddings

Immediately after forming the input embeddings, a dropout layer is applied [c932]. Dropout randomly sets some elements of every input embedding to zero during training [c932]. This acts as a regularizer, preventing the model from over-relying on any single embedding dimension and improving generalization.

At inference time, dropout is disabled (PyTorch handles this automatically when you call `model.eval()`), so every element is used.

#### Stacked Transformer Blocks

The transformer block is the key component of the entire GPT model [c920]. The full GPT-2-scale model chains 12 of these blocks together [c1000, c1818].

Each transformer block contains [c1815]:
- A **layer normalization** layer
- A **masked multi-head attention** module
- A **dropout** layer
- **Shortcut (residual) connections**
- A second **layer normalization** layer
- A **feed-forward neural network**
- Another **dropout** layer

Let us briefly revisit what each of these does in context.

**Layer normalization** normalizes the embedding values for each token so that the mean is zero and the variance is one [c963]. This stabilizes training by preventing the activations from growing or shrinking uncontrollably as they pass through many layers [c943].

**Masked multi-head attention** converts embedding vectors into context vectors [c945]. A context vector is an embedding that captures both the semantic meaning of a token and how much attention should be given to all other tokens in the sequence [c947]. The "masked" part ensures that when predicting token $t$, the model cannot see tokens at positions $t+1, t+2, \ldots$ — it can only attend to the past. This is what makes the model autoregressive. The attention mechanism is the driving engine that gives LLMs their power [c1794]; it is where key, query, and value projections operate [c1816].

**Shortcut connections** add the output of a sub-layer back to its input [c950]. This provides an alternative gradient flow path and prevents the vanishing gradient problem that would otherwise make training 12-layer networks extremely difficult [c950].

In PyTorch, the 12 transformer blocks are chained using `nn.Sequential`:

```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
)
```

`nn.Sequential` passes the output of each block directly as the input to the next, which is exactly the behavior we want [c992].

#### Final Layer Normalization

After the last transformer block, one more layer normalization is applied before the output head [c992]. This final normalization ensures that the values entering the output head are well-scaled, regardless of how many transformer blocks were stacked. Layer normalization sets the mean to zero and variance to one for each token independently [c943].

#### The Output Head

The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. In practice, it is a single linear (fully connected) layer with no bias:

```python
self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
```

This layer maps each 768-dimensional token representation to a 50257-dimensional vector — one score per vocabulary token. These scores are the **logits** [c968].

The output head is the only stage in the GPT model where the input dimensions change: the tensor goes from shape `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]. Every other stage preserves the embedding dimension.

Logits are the final output matrices from a transformer model that contain probability scores for each token in the vocabulary at each position in the sequence [c432]. The word "logits" is used because these are *unnormalized* scores — they become proper probabilities only after a softmax is applied [c1817].

---

### The Complete Forward Pass

Putting it all together, the forward pass of the GPT model follows this sequence [c1020]:

1. Accept a batch of input token IDs with shape `(batch_size, sequence_length)` [c990].
2. Look up token embeddings: shape becomes `(batch_size, sequence_length, 768)`.
3. Compute positional embeddings and add them to the token embeddings.
4. Apply dropout to the combined input embeddings.
5. Pass the sequence through all 12 transformer blocks via `nn.Sequential`.
6. Apply final layer normalization.
7. Pass through the output head to produce logits of shape `(batch_size, sequence_length, 50257)`.

The full model forward pass applies dropout to input embeddings, chains 12 transformer blocks using `nn.Sequential`, applies layer normalization, and outputs logits [c992]. The output is a tensor of shape `(batch_size, sequence_length, vocabulary_size)` [c994].

The core of this logic can be expressed in approximately 8 lines of code [c1019], which is a remarkable fact given the complexity of what is happening inside each transformer block.

[FIGURE: Annotated tensor shape diagram showing how shape changes at each stage: (B, T) → (B, T, 768) → (B, T, 768) [×12 blocks] → (B, T, 768) → (B, T, 50257)]

---

### Creating a Model Instance

Once the `GPTModel` class is defined, creating an instance is straightforward — you pass in the configuration dictionary [c996]:

```python
model = GPTModel(GPT_CONFIG_124M)
```

At this point, the model has been initialized with random weights [c895]. It can already perform a forward pass and produce logits; those logits just won't correspond to meaningful text yet because the weights haven't been trained. Nevertheless, the architecture is fully functional, and you can run inference through the 124-million-parameter GPT architecture on a laptop [c897].

The GPT model architecture outputs numerical tensors [c1015]. The model outputs logits that can be decoded to predict the next token [c916].

---

### Understanding the Output: Logits and Their Shape

Let us be precise about what the model returns. Given an input batch of shape `(batch_size, sequence_length)`, the model returns logits of shape `(batch_size, sequence_length, 50257)` [c994].

The logits matrix is the output produced by passing token embeddings through the output head neural network [c968]. Each of the `sequence_length` positions gets its own 50257-dimensional logit vector. The value at position `[b, t, v]` is the model's unnormalized score for "token `v` should come after the first `t` tokens in example `b`."

To convert logits into a next-token prediction, you need to:

1. **Extract the logits for the last position** — since you want to predict the token *after* the current sequence, only the last row of the sequence dimension matters [c845].
2. **Apply softmax** to convert logits into probabilities.
3. **Select the most probable token** (or sample from the distribution).

We will implement this step by step in the text generation section below.

---

### Weight Tying: 163M vs. 124M Parameters

Before counting parameters, there is an important design decision to understand: **weight tying**.

Weight tying is a technique where the original GPT-2 architecture reuses the weights from the token embedding layer in its output layer [c1006]. The intuition is that both layers deal with the same vocabulary: the token embedding layer maps token IDs to vectors, and the output head maps vectors back to token IDs. It makes sense that these two mappings might share the same underlying matrix.

The token embedding layer and output layer in the implemented model both have shape `(50257, 768)` [c1007]. That is $50257 \times 768 = 38{,}597{,}376$ parameters — a substantial chunk of the total.

Weight tying reduces the memory footprint from 163 million to 124 million parameters and reduces computational complexity [c1009]. When output layer parameters are removed from the total (because they are shared with the token embedding layer), the parameter count becomes exactly 124 million, matching the original GPT-2 model size [c1008].

However, using separate token embedding and output layers results in better training and model performance compared to weight tying [c1010]. The implementation in this book therefore keeps them separate, which is why the raw parameter count is 163 million rather than 124 million. When people say "GPT-2 124M," they are referring to the weight-tied count.

[FIGURE: Side-by-side comparison: weight-tied model (124M params, shared embedding/output matrix) vs. untied model (163M params, separate matrices)]

---

### Counting Parameters

To count the total number of parameters in the model, you can iterate over `model.parameters()` and sum the element counts:

# Source: [c1004]
```python
p.numel() - method to count the total number of parameters in model's parameter tensors
```

In practice, this looks like:

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

Each of these contributes to the total. The dominant terms are the attention weight matrices (which appear in every one of the 12 transformer blocks) and the embedding/output matrices.

---

### From Logits to Text: Autoregressive Generation

A large language model generates tokens given a sequence of input tokens [c805]. The process is *autoregressive*: the model generates one token at a time, appends it to the input, and then generates the next token conditioned on the extended sequence.

The goal of this section is to convert the output tensor from the GPT model into predictions of the next word [c817].

#### The `generate_text_simple` Function

The `generate_text_simple` function takes a model (an instance of the GPT model class), input token indices (`idx`), a maximum number of new tokens to generate, and a context size as arguments [c856].

Here is the complete generation loop, broken down step by step.

**Step 1: Crop the context window.**

If the accumulated sequence is longer than the model's context window, we must crop it. The model was trained on sequences of at most `context_length` tokens, so feeding it more would be undefined behavior.

# Source: [c859]
```python
idx_cond = idx[:, -context_size:]
```

This slices the last `context_size` tokens from the current sequence, discarding anything older.

**Step 2: Run the forward pass without gradient tracking.**

During inference, we do not need to compute gradients. Wrapping the forward pass in `torch.no_grad()` saves memory and speeds up computation:

# Source: [c872]
```python
with torch.no_grad():
    logits = model(idx_cond)
```

**Step 3: Extract logits for the last position.**

The model returns logits for every position in the sequence, but we only care about the prediction for the *next* token — the one that comes after the last token in our input. We extract the logits at the final sequence position:

# Source: [c863]
```python
logits = logits[:, -1, :]
```

This reduces the shape from `(batch_size, sequence_length, vocab_size)` to `(batch_size, vocab_size)`.

**Step 4: Convert logits to probabilities.**

Apply softmax along the vocabulary dimension to turn raw scores into a proper probability distribution:

# Source: [c865]
```python
probas = torch.softmax(logits, dim=-1)
```

**Step 5: Select the most probable token.**

In the simplest generation strategy (greedy decoding), we always pick the token with the highest probability:

# Source: [c867]
```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```

`keepdim=True` preserves the sequence dimension so the result has shape `(batch_size, 1)`, which is compatible with the concatenation in the next step.

**Step 6: Append the new token and repeat.**

The newly generated token is appended to the running sequence:

# Source: [c869]
```python
idx = torch.cat((idx, idx_next), dim=1)
```

The loop then returns to Step 1 with the extended sequence. This continues until `max_new_tokens` tokens have been generated.

[FIGURE: Autoregressive generation loop diagram: input tokens → model → logits → softmax → argmax → new token → append to input → repeat]

#### Putting It All Together

Here is the complete generation function assembled from the steps above:

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

Starting with the input `'hello i am'` and setting `max_new_tokens=6`, the GPT model generates the complete output `'hello i am a model ready to help'` [c854]. This is with randomly initialized weights — the output is not yet meaningful English, but the mechanics work correctly. After training on real data, the same function produces coherent text.

Note that in GPT training, sampling techniques will modify the softmax output so the model does not always select the most likely token, introducing variability and creativity in generated text [c878]. Greedy decoding (always picking the argmax) tends to produce repetitive, boring output. Temperature scaling, top-k sampling, and nucleus sampling are common alternatives, but they all operate on the same logit output — only the selection strategy changes.

---

### The Loss Function

To train the model, we need a way to measure how wrong its predictions are. The loss function used in LLM training is cross-entropy loss between the predicted next token and the actual next token [c1821].

Cross-entropy loss penalizes the model when it assigns low probability to the correct token. If the model is very confident and correct, the loss is near zero. If the model is confident but wrong, the loss is large. Minimizing this loss over a large corpus of text is what drives the model to learn useful representations.

The attention mechanism — which is the driving engine that gives LLMs their power [c1794] — is trained end-to-end through this loss signal. Gradients flow back through the output head, through the layer normalization, through all 12 transformer blocks, and all the way back to the embedding layers.

---

### What Comes After Pre-Training

Once the GPT model has been pre-trained on a large text corpus, it can be adapted for specific tasks through fine-tuning. There are two main flavors [c1827, c1828]:

- **Classification fine-tuning**: the model is trained to classify inputs into categories, such as determining whether an email is spam or not spam [c1827].
- **Instruction fine-tuning**: the model is trained on a dataset of instructions, inputs, and outputs to perform well on instruction-following tasks [c1828].

Fine-tuned models are evaluated using benchmarks like MMLU (Measuring Massive Multitask Language Understanding), which uses 57 tests to evaluate LLM performance across diverse domains [c1831]. Other evaluation methods include human evaluation, where humans compare and rate the outputs of different LLMs [c1832], and LLM-based evaluation, where a powerful large language model is used to evaluate the outputs of another LLM [c1833].

---

### What We Built From Scratch

It is worth pausing to appreciate the scope of what has been implemented. The GPT architecture was built completely from scratch without using external libraries like LangChain [c900]. All sub-modules of the GPT architecture were implemented and coded from scratch [c901].

The lecture series covered layer normalization, GELU activation, feed-forward networks, shortcut connections, and transformer blocks [c904]. Each of these components was introduced, explained, and coded before being assembled into the final model.

The complete architecture consists of [c1812]:
- Input tokens converted to embeddings
- Positional embeddings added to token embeddings
- The combined embeddings passed through transformer blocks
- A final layer normalization
- A final neural network (the output head) that produces logits for next-token prediction

The transformer block itself contains a normalization layer, multi-head attention, dropout, shortcut connections, another normalization layer, a feed-forward neural network, and another dropout layer [c1815]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models like GPT-4 [c824].

[FIGURE: Full architecture diagram: token IDs → token embedding → + positional embedding → dropout → [LayerNorm → MultiHeadAttention → Dropout → Residual] × 12 → LayerNorm → Linear(768→50257) → Logits]

---

### Summary

In this chapter, you assembled the complete GPT model from its constituent parts. The key takeaways are:

- The GPT forward pass is a linear pipeline: embeddings → dropout → 12 transformer blocks → layer norm → output head → logits [c811, c992].
- The output shape is always `(batch_size, sequence_length, vocab_size)` [c994], where `vocab_size = 50257` for GPT-2.
- The output head is the only stage where the embedding dimension changes [c973].
- Weight tying reduces the parameter count from 163M to 124M by sharing the token embedding and output matrices [c1009], though keeping them separate can improve performance [c1010].
- Autoregressive text generation works by repeatedly feeding the model the current sequence, extracting the last-position logits, applying softmax, selecting a token, and appending it [c856].
- The loss function is cross-entropy between predicted and actual next tokens [c1821].

The model outputs logits which will be converted into tokens and text outputs in subsequent work [c1021]. With the architecture fully assembled and the generation loop in place, the next step is training — adjusting those 124 million parameters so that the model's predictions actually match real language.