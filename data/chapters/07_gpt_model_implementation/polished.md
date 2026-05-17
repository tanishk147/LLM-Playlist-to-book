## Assembling the Full GPT Model and Text Generation

We will now assemble all previously built components into a single, coherent `GPTModel` class, trace a tensor through the entire forward pass, count the model's parameters, and implement the autoregressive text-generation loop that turns raw logits into readable text.

By the end of this chapter, you will have a working 124-million-parameter GPT-2 model that can run inference on a laptop [c897] and produce text from a prompt — even with randomly initialized weights. Training comes later [c906]; here the focus is on architecture and inference.

---

### The Big Picture: What the GPT Pipeline Does

Text generation in a large language model is an **autoregressive** process: the model generates one token at a time, and each newly generated token is appended to the input context before the next prediction is made [c810]. This continues until a user-specified maximum number of new tokens has been produced [c843].

Given a sequence of token IDs, the model executes the following steps:

1. Converts token IDs into dense token embeddings [c818].
2. Adds positional embeddings to encode each token's position in the sequence [c930].
3. Applies dropout to the combined input embeddings [c814].
4. Passes the result through a stack of transformer blocks [c992].
5. Applies a final layer normalization [c992].
6. Projects the normalized embeddings through an output head to produce logits over the vocabulary [c965].

Concisely: input sentence → token IDs → token embeddings → positional embeddings → input embeddings → dropout → transformer blocks → layer normalization → output head → output tensor [c811]. The output is a tensor of logits [c432], which we will later convert into probabilities and then into token IDs.

---

### The GPT Configuration Object

Rather than hard-coding hyperparameters throughout the model, we collect them in a single dictionary [c438]:

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

Here is what each field means in the context of the full model:

- **`vocab_size: 50257`** — Each token ID is an integer in the range `[0, 50256]` [c438].
- **`context_length: 1024`** — The maximum number of tokens the model can process in a single forward pass; also the size of the positional embedding table [c438].
- **`emb_dim: 768`** — Every token is represented as a 768-dimensional vector. Each token ID in GPT-2 is converted into a vector of 768 dimensions, and the output dimensions are matched to those of the input token embeddings [c379, c818].
- **`n_heads: 12`** — The number of separate query, key, and value matrix sets created in multi-head attention [c390]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models [c824].
- **`n_layers: 12`** — The number of transformer blocks stacked in sequence, chained together using `nn.Sequential` [c1000].
- **`drop_rate: 0.1`** — The fraction of elements randomly zeroed during dropout [c932].
- **`qkv_bias: False`** — Whether to include bias terms in the query, key, and value projections of attention [c438].

---

### A Dummy Model First: Validating the Architecture

Before filling in the details of each sub-module, it is useful to build a skeleton model with dummy internals. This lets us verify that tensor shapes flow correctly through the pipeline before worrying about the internals of each component [c911]:

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

Even with dummy internals, this code captures the complete forward-pass logic. The dummy model's `forward` method takes an input and outputs the next-word prediction [c403]. Token IDs are converted to token embeddings before being passed through the model [c411], and those token embeddings are then added to positional embeddings [c912]. The transformer block is the most important part of the large language model architecture, consisting of multiple aspects linked together [c366], and it is the key component of the entire GPT model [c920].

---

### Building the Real GPT Model

With the structural skeleton validated, we now describe the real `GPTModel` that replaces each dummy sub-module with its full implementation.

#### Token and Positional Embeddings

**Token embedding:** An `nn.Embedding` layer maps each token ID to a dense vector. In GPT-2, this embedding layer is trained to learn token representations that capture semantic meaning [c388].

**Positional embedding:** A second `nn.Embedding` layer encodes position information. A positional embedding is a vector representation of the position of a token within a sequence [c926]:

```python
self.pos_emb = nn.Embedding(context_length, embedding_dimension)
```
[c425]

During the forward pass, position indices are generated to look up in this table:

```python
pos = torch.arange(sequence_length)
```
[c426]

This creates a tensor `[0, 1, 2, ..., sequence_length-1]` to index into the positional embedding matrix [c426].

**Combining the two:** The input embedding for each token is the sum of its token embedding and its positional embedding — token embeddings and positional embeddings are summed to produce input embeddings for each token [c820, c930].

#### Dropout on Input Embeddings

Immediately after combining the embeddings, a dropout layer is applied [c814]. Dropout randomly sets some elements of every input embedding to zero [c932], and the combined embeddings pass through this layer before entering the transformer stack [c448]. This regularization step helps prevent overfitting during training.

#### The Transformer Block Stack

The preprocessing pipeline before the transformer block consists of tokenization, token embedding, positional embedding, input-embedding computation, and dropout [c939]. After that preprocessing, the tensor enters the transformer block stack.

The GPT model forward pass chains 12 transformer blocks using `nn.Sequential`, then applies layer normalization, and finally outputs logits [c992]. Inside each transformer block, the feed-forward neural network uses GELU activation [c369]. Masked multi-head attention converts embedding vectors into context vectors that capture both semantic meaning and attention relationships between tokens [c945]; a context vector is an embedding that encodes both the semantic meaning of a token and how much attention should be given to every other token in the sequence [c947]. Shortcut connections (residual connections) add the output of each layer back to its input, providing an alternative gradient-flow path and preventing the vanishing gradient problem [c950].

#### Layer Normalization

Layer normalization normalizes token embeddings by setting the mean to zero and the variance to one for each token independently [c943, c963]. This normalization is applied both within each transformer block and once more after the final block.

#### The Output Head

The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. It is the only stage in the model where the tensor dimensions change — from `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]. The resulting logits matrix is the output produced by passing token embeddings through this linear layer [c968]. Logits are the final output matrices from a transformer model that contain unnormalized scores for each token in the vocabulary at each position in the sequence [c432].

---

### The Forward Pass in Detail

Let's trace a concrete tensor through the model to make the shapes tangible.

The GPT model receives an input tensor of shape `(batch_size, sequence_length)` containing token IDs [c990]. For a batch of 2 sequences each of length 4:

1. **Token embedding lookup:** Shape becomes `(2, 4, 768)`.
2. **Positional embedding lookup:** Shape is `(4, 768)`, broadcast-added to the token embeddings.
3. **Sum:** Shape remains `(2, 4, 768)` [c930].
4. **Dropout:** Shape unchanged, `(2, 4, 768)`.
5. **12 transformer blocks:** Shape unchanged throughout, `(2, 4, 768)`.
6. **Layer normalization:** Shape unchanged, `(2, 4, 768)`.
7. **Output head (linear):** Shape becomes `(2, 4, 50257)`.

The output of the GPT model is therefore logits with shape `(batch_size, sequence_length, vocabulary_size)` [c994]. The forward method takes a batch of input tokens, computes their embeddings, applies positional embeddings, passes the sequence through transformer blocks, normalizes the final output, and computes logits representing the next token's unnormalized probabilities [c1020]. Notably, the core GPT model implementation can be expressed in approximately 8 lines of code [c1019] — a testament to how cleanly PyTorch's module system composes the sub-components built in earlier chapters.

---

### Parameter Counting and Weight Tying

#### Counting Parameters

The standard idiom for counting parameters is [c1004]:

```python
sum(p.numel() for p in model.parameters())
```

Iterating over `model.parameters()` and summing `p.numel()` for each tensor gives the total parameter count.

#### The Token Embedding and Output Layer

Both the token embedding layer and the output layer have shape `(50257, 768)` [c1007], meaning they contain the same number of parameters.

#### Weight Tying

The original GPT-2 architecture exploits this symmetry through **weight tying**: it reuses the weights from the token embedding layer in the output layer [c1006]. This reduces the memory footprint from 163 million to 124 million parameters and lowers computational complexity [c1009] — when the output layer parameters are excluded from the count, the total becomes exactly 124 million, matching the original GPT-2 model size [c1008].

It is worth noting, however, that using separate token embedding and output layers yields better training and model performance than weight tying [c1010]. The implementation described in this book therefore uses separate layers for clarity and performance; the 124 M figure reflects the original GPT-2 design choice [c1009].

---

### Instantiating the Model

A GPT model instance is created by passing a configuration object to the constructor [c996]:

```python
model = GPTModel(GPT_CONFIG_124M)
```

The full GPT architecture has been implemented and the model instance is initialized with random weights [c895]. Even at this stage, the model can accept an input, run inference through the 124-million-parameter architecture, and produce output on a laptop [c897]. The main task of the model is next-token prediction [c917]: it outputs logits that can be decoded to predict the next token [c916], which will subsequently be converted into tokens and text [c1021].

---

### From Logits to Text: The Generation Pipeline

The goal is to convert the model's output tensor into predictions of the next word [c817]. This section implements the full autoregressive generation loop.

#### The Five-Step Decoding Algorithm

The text generation algorithm consists of five sequential steps [c855]:

1. Examine the output tensor.
2. Extract the last position's vector (logits for the next token).
3. Convert logits to probabilities via softmax.
4. Identify the index of the largest probability value.
5. Append the resulting token ID to the previous input, then repeat until `max_new_tokens` is reached.

[FIGURE: Autoregressive generation loop — at each step, extract last-position logits, softmax → probabilities, argmax → token ID, append to sequence, feed back into model]

#### Implementing `generate_text_simple`

The `generate_text_simple` function takes a model instance, input token indices (`idx`), a maximum number of new tokens to generate, and a context size as arguments [c856]. The key steps are as follows.

First, crop the running sequence to the last `context_size` tokens so it never exceeds the positional embedding table [c859]:

```python
idx_cond = idx[:, -context_size:]
```

Then run the forward pass with gradients disabled to save memory and computation [c872]:

```python
with torch.no_grad():
    logits = model(idx_cond)
```

The model produces logits for every position, but only the last position matters for the next-token prediction [c845]:

```python
logits = logits[:, -1, :]
```
[c863]

Apply softmax to obtain a probability distribution [c874]:

```python
probas = torch.softmax(logits, dim=-1)
```
[c865]

Select the token with the highest probability (greedy decoding):

```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```
[c867]

Finally, append the new token to the running sequence [c807]:

```python
idx = torch.cat((idx, idx_next), dim=1)
```
[c869]

#### The Complete `generate_text_simple` Function

Placing these steps inside a loop gives the full generation function. Text generation proceeds by repeating the token-prediction process until the user-specified maximum number of new tokens is reached [c850, c843]:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Step 1: crop context to fit positional embedding table
        idx_cond = idx[:, -context_size:]

        # Step 2: forward pass, no gradients needed
        with torch.no_grad():
            logits = model(idx_cond)

        # Step 3: focus on the last time step
        logits = logits[:, -1, :]          # (batch, vocab_size)

        # Step 4: logits → probabilities
        probas = torch.softmax(logits, dim=-1)

        # Step 5: greedy selection
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Step 6: append to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

#### Running the Generator

The `generate_text_simple` function is called with parameters: model, inputs, `max_new_tokens=6`, and `context_size=1024` [c890]. Starting from the input `'hello i am'` with `max_new_tokens=6`, the GPT model generates the complete output `'hello i am a model ready to help'` [c854].

---

### Greedy Decoding and Its Limitations

The current implementation always selects the single most probable token at each step — a strategy known as greedy decoding. While simple and deterministic, it tends to produce repetitive, bland text because it never explores lower-probability but potentially more interesting continuations. In GPT training, sampling techniques modify the softmax output so the model does not always select the most likely token, introducing variability and creativity in generated text [c878].

---

### Putting It All Together: A Complete Walkthrough

One final end-to-end walkthrough consolidates everything covered in this chapter.

1. **Configuration.** Define `GPT_CONFIG_124M` with `vocab_size=50257`, `context_length=1024`, `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.1`, `qkv_bias=False` [c438].
2. **Model instantiation.** `model = GPTModel(GPT_CONFIG_124M)` creates a model with randomly initialized weights [c895, c996].
3. **Input preparation.** A tokenizer converts the input string into token IDs. The model receives a tensor of shape `(batch_size, sequence_length)` [c990].
4. **Forward pass.** The model maps the input through token embeddings, positional embeddings, dropout, 12 transformer blocks, layer normalization, and the output head [c811], producing an output of shape `(batch_size, sequence_length, 50257)` [c994].
5. **Generation loop.** `generate_text_simple` runs the forward pass repeatedly, each time extracting the last-position logits [c845], applying softmax [c865], selecting the argmax token [c867], and appending it to the sequence [c869].

The implementation has reached the stage where, given a text input, the model can predict output tokens [c902]. All sub-modules of the GPT architecture were implemented and coded from scratch [c901].

---

### What Comes Next

The model currently outputs logits that will be converted into tokens and text in subsequent steps [c1021]. The next stage of the series focuses on training the GPT-2 model with 124 million parameters [c906]. Once trained, the same `generate_text_simple` function — or a more sophisticated sampling variant — will produce coherent, meaningful text rather than random token sequences.