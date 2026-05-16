## Assembling and Running the Full GPT-2 Model

---

### The Big Picture: What the GPT Pipeline Looks Like

The complete pipeline runs as follows: input sentence → token IDs → token embeddings → positional embeddings → input embeddings → dropout → transformer blocks → layer normalization → output head → output tensor [c811]. Each stage has a well-defined tensor shape, and the only stage where the embedding dimension changes is the very last one [c973].

The fundamental task the model is solving is next-token prediction [c917]. Everything else—the attention heads, the feed-forward layers, the normalization—exists to make that prediction as accurate as possible. The model generates tokens given a sequence of input tokens [c805], and because each newly generated token is fed back in as input, the process is inherently autoregressive [c810].

---

### The GPT Configuration Object

Throughout this chapter we refer to a configuration dictionary that holds all hyperparameters in one place:

- `vocab_size`: 50257 (byte-pair encoding vocabulary) [c922]
- `context_length`: 1024
- `emb_dim`: 768 [c818]
- `n_heads`: 12
- `n_layers`: 12 [c888]
- `drop_rate`: 0.1
- `qkv_bias`: False [c984]

Passing this dictionary into the model constructor makes it easy to swap configurations without touching the model code itself.

---

### Building the DummyGPTModel

```python
# Source: [c911]
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

Let's walk through each component in turn.

---

### Component Deep-Dive

#### Token Embeddings

Token embeddings have dimension 768 [c818], so the embedding table has shape `(vocab_size, emb_dim)` = `(50257, 768)` [c1007].

#### Positional Embeddings

Raw token embeddings carry no information about *where* in the sequence a token appears. A positional embedding is a vector representation of the position of a token within a sequence [c926], and it fills exactly that gap.

#### Combining the Two

The input embedding for each token is the sum of its token embedding and its positional embedding [c930]. Token embeddings are added to positional embeddings in the GPT model forward pass [c912], so every vector entering the transformer simultaneously encodes what the token *is* and where it *sits* in the sequence.

#### Dropout

Immediately after the embeddings are combined, a dropout layer is applied [c814]. Dropout randomly sets some elements of every input embedding to zero [c932], acting as a regularizer during training and preventing the model from becoming overly reliant on any single embedding dimension [c913].

#### The Transformer Blocks

The transformer block is the key component of the entire GPT model [c920]. Twelve such blocks are chained together [c1000] using `nn.Sequential` [c992], and two sub-components do most of the heavy lifting inside each one:

- **Masked multi-head attention** converts embedding vectors into context vectors that capture both semantic meaning and attention relationships between tokens [c945]. A context vector encodes both the semantic meaning of a token and how much attention should be given to every other token in the sequence [c947]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models like GPT-4 [c824].

- **Shortcut (residual) connections** add the output of a layer back to its input, providing an alternative gradient-flow path and preventing the vanishing gradient problem [c950].

The preprocessing pipeline feeding into the transformer blocks consists of tokenization, token embedding, positional embedding, input-embedding computation, and dropout [c939]. Everything after that pipeline lives inside the stacked transformer blocks.

#### Layer Normalization

Layer normalization normalizes token embeddings by setting the mean to zero and the variance to one for each token independently [c943]. Concretely, it subtracts the mean and divides by the square root of the variance, then applies trainable scale and shift parameters [c985]. This normalization is applied at multiple points in the network—including after each transformer block—always operating token-by-token [c951, c963].

#### The Output Head

The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. It is the only stage where the embedding dimension changes—from `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]—and its output, the logits matrix, is produced by passing those token embeddings through this linear layer [c968].

---

### The Full GPT Model Forward Pass

With all components understood individually, the forward pass reads almost like pseudocode. The forward method takes a batch of input tokens, computes their embeddings, adds positional embeddings, passes the sequence through the transformer blocks, normalizes the final output, and returns logits representing the next token's unnormalized probabilities [c1020]. The model input is a tensor of shape `(batch_size, sequence_length)` containing token IDs [c990], and the output is logits with shape `(batch_size, sequence_length, vocabulary_size)` [c994], which can be decoded to predict the next token [c916].

Remarkably, the core GPT model implementation can be expressed in approximately 8 lines of code [c1019]. The complexity lives in the sub-modules, not in the top-level assembly.

---

### Counting Parameters: 124 M vs. 163 M

You can count the total number of parameters in a model's parameter tensors using `p.numel()` [c1004]:

```python
print(f"Total parameters: {total_params:,}")
```

A naïve count yields roughly 163 million parameters. Where does the discrepancy with the well-known 124 million figure come from?

Weight tying is a technique where the original GPT-2 architecture reuses the weights from the token embedding layer in its output layer [c1006]. Both layers have shape `(50257, 768)` [c1007], so each holds approximately 38.6 million parameters. When the output layer's parameters are removed from the total—as weight tying effectively does—the count drops to exactly 124 million, matching the original GPT-2 model size [c1008].

Weight tying reduces the memory footprint from 163 million to 124 million parameters and also reduces computational complexity [c1009]. That said, using separate token embedding and output layers yields better training and model performance than weight tying [c1010]. The assembled model requires 600 megabytes of space on a laptop [c1017]—well within the reach of modern consumer hardware—and can run inference through the full 124-million-parameter architecture on that same machine [c897].

---

### Generating Text: The Autoregressive Loop

To produce a sequence of tokens, we need a function that repeatedly queries the model, appending each new prediction to the growing input.

#### The Five-Step Algorithm

Text generation follows five sequential steps [c855]: examine the output tensor, extract the last vector, convert logits to probabilities, identify the index of the largest value, and append the resulting token ID to the previous inputs—then repeat until the maximum number of new tokens is reached.

LLM text generation is autoregressive: the input context grows with each iteration as newly generated tokens are appended, and generation continues until the maximum token count is reached [c810]. Concretely, the token generated in one iteration becomes part of the input for the next [c807], and the loop runs until the user-specified limit is hit [c808].

[FIGURE: Autoregressive generation loop diagram showing: input tokens [t1, t2, t3] → GPT model → logits → softmax → argmax → new token t4 → append → [t1, t2, t3, t4] → repeat]

#### The `generate_text_simple` Function

`generate_text_simple` takes a model (an instance of the GPT model class), input token indices (`idx`), a maximum number of new tokens to generate, and a context size [c856]. Here is how each piece is built up.

If the accumulated sequence grows longer than the model's context window, it must be cropped before being passed in:

```python
# Source: [c859]
idx_cond = idx[:, -context_size:]
```

During inference we do not need gradients, so the forward call is wrapped in `torch.no_grad()` to save memory and computation:

```python
# Source: [c872]
with torch.no_grad():
    logits = model(idx_cond)
```

The model outputs logits for every position in the sequence, but only the *last* position matters for the next-token prediction [c845]:

```python
# Source: [c863]
logits = logits[:, -1, :]
```

This reduces the shape from `(batch_size, seq_len, vocab_size)` to `(batch_size, vocab_size)`. Softmax is then applied to convert those logits into a probability distribution [c874]—included here to make the full transformation explicit, even though it is technically redundant for argmax selection [c877]:

```python
# Source: [c865]
probas = torch.softmax(logits, dim=-1)
```

The token with the highest probability is selected greedily:

```python
# Source: [c867]
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```

Finally, the new token is appended to the running sequence for the next iteration:

```python
# Source: [c869]
idx = torch.cat((idx, idx_next), dim=1)
```

#### Putting It All Together

Assembling the fragments above into a complete function:

```python
for _ in range(max_new_tokens):
    # Crop context if needed
    idx_cond = idx[:, -context_size:]

    # Forward pass
    with torch.no_grad():
        logits = model(idx_cond)

    # Focus on the last time step
    logits = logits[:, -1, :]

    # Convert to probabilities
    probas = torch.softmax(logits, dim=-1)

    # Greedy selection
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)

    # Append to running sequence
    idx = torch.cat((idx, idx_next), dim=1)

return idx
```

The loop iterates exactly `max_new_tokens` times, producing one new token per iteration [c871].

---

### Running the Model: A Concrete Example

With the model instantiated and the generation function defined, we can run a quick end-to-end test. A GPT model instance is created by passing in a model configuration object [c996], initialized with random weights [c895]. Calling `generate_text_simple` with `max_new_tokens=6` and `context_size=1024` [c890] on the input `'hello i am'` produces the complete output `'hello i am a model ready to help'` [c854].

Of course, the goal is ultimately to convert the output tensor into meaningful next-word predictions [c817]. The model's output logits will be converted into tokens and text in subsequent work [c1021], and GPT model output has the same number of rows as input tokens but a number of columns equal to the vocabulary size (50257) [c828].

---

### Why the Output Is Nonsense (For Now)

The next-token predictions from an untrained model are essentially random, because the parameters have not yet been optimized [c896, c905]. Once training begins, sampling techniques will modify the softmax output so the model does not always select the most likely token, introducing variability and creativity into the generated text [c878]. The next phase of work focuses on training the GPT-2 model with 124 million parameters [c906].

---

### What Was Built from Scratch

The GPT architecture was built entirely from scratch, without relying on external libraries like LangChain [c900]. Every sub-module—layer normalization, GELU activation, feed-forward networks, shortcut connections, and transformer blocks—was implemented and coded from the ground up across six to seven lectures [c901, c903, c904]. The result is a working implementation that, given a text input, can predict output tokens [c902].

---

### Summary

This chapter assembled the complete GPT-2 architecture from its individual components: token and positional embeddings, dropout, 12 stacked transformer blocks, a final layer normalization, and a linear output head. The model has approximately 163 million raw parameters, which reduces to 124 million when weight tying is applied [c1008, c1009]. It fits in 600 MB of RAM and can run inference on a laptop [c1017, c897].

The `generate_text_simple` function implements greedy autoregressive decoding: crop the context, run a forward pass, extract the last-position logits, apply softmax, take argmax, append the new token, and repeat [c855]. The output logits will be converted into tokens and text in subsequent work [c1021], and training the model to produce meaningful outputs is the natural next step [c906].