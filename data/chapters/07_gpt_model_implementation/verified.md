## Assembling the Full GPT Model and Text Generation

 We will assemble these components into a single, coherent `GPTModel` class, trace a tensor through the entire forward pass, count the model's parameters, and finally implement the autoregressive text generation loop that turns raw logits into readable text.

By the end of this chapter, you will have a working 124-million-parameter GPT-2 model that can run inference on a laptop [c897] and produce text from a prompt — even if the weights are random at this stage. Training comes later [c906]; here we focus on architecture and inference.

---

### The Big Picture: What the GPT Pipeline Does

 Text generation in a large language model is an **autoregressive** process: the model generates one token at a time, and each newly generated token is appended to the input context before the next prediction is made [c810]. This continues until a user-specified maximum number of new tokens has been produced [c843].

 Given a sequence of token IDs, the model:

 Converts token IDs into dense token embeddings [c818].
2. Adds positional embeddings to encode each token's position in the sequence [c930].
3. Applies dropout to the combined input embeddings [c814].
4. Passes the result through a stack of transformer blocks [c992].
5. Applies a final layer normalization [c992].
6. Projects the normalized embeddings through an output head to produce logits over the vocabulary [c965].

This pipeline is described concisely as: input sentence → token IDs → token embeddings → add positional embeddings → input embeddings → dropout → transformer block → layer normalization → output head → output tensor [c811]. The output is a tensor of logits [c432], which we will later convert into probabilities and then into token IDs.

---

### The GPT Configuration Object

Rather than hard-coding hyperparameters throughout the model, we collect them in a single dictionary. 

# Source: [c438]
```python

 "vocab_size": 50257, # Vocabulary size
 "context_length": 1024, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}
```

Let's unpack what each field means in the context of the full model:

 Each token ID is an integer in the range `[0, 50256]` [c438].
- **`context_length: 1024`** — The maximum number of tokens the model can process in a single forward pass [c438]. This is also the size of the positional embedding table.
- **`emb_dim: 768`** — Every token is represented as a vector of 768 dimensions [c818]. Each token ID in GPT-2 is converted into a vector of 768 dimensions, and the output is generated such that dimensions are matched to the input token embedding dimensions [c379].
- **`n_heads: 12`** — The number of separate query, key, and value matrix sets created in multi-head attention [c390]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models [c824].
- **`n_layers: 12`** — The number of transformer blocks stacked in sequence [c1000]. The GPT model architecture consists of 12 transformer blocks chained together [c1000].
- **`drop_rate: 0.1`** — The fraction of elements randomly zeroed during dropout [c932].
- **`qkv_bias: False`** — Whether to include bias terms in the query, key, and value projections of attention [c438].

---

### A Dummy Model First: Validating the Architecture

 This lets us verify that tensor shapes flow correctly through the pipeline before we worry about the details of each component.

# Source: [c911]
```python

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

Even with dummy internals, this code captures the complete forward pass logic. The dummy model class has a forward method that takes an input and outputs the next word prediction [c403]. Token IDs are converted to token embeddings before being passed to the GPT model class [c411], and token embeddings are added to positional embeddings in the GPT model forward pass [c912].

 The transformer block is the most important part of the large language model architecture and consists of multiple different aspects linked together [c366], and the transformer block is the key component of the entire GPT model [c920].

---

### Building the Real GPT Model

With the structural skeleton validated, we now describe the real `GPTModel` that replaces the dummy sub-modules with their full implementations.

#### Token and Positional Embeddings

 In GPT-2, the embedding layer is trained to learn token representations that capture semantic meaning [c388].

**Positional embedding:** A second `nn.Embedding` layer encodes position information. Positional embedding is a vector representation of the position of a token within a sequence [c926]. 

# Source: [c425]
```python

```

During the forward pass, we need to generate a sequence of position indices to look up in this table:

# Source: [c426]
```python

```

This creates a tensor `[0, 1, 2, ..., sequence_length-1]` to index into the positional embedding matrix [c426].

**Combining the two:** The input embedding for each token is the sum of its token embedding and its positional embedding [c930]:

Token embeddings and positional embeddings are summed to produce input embeddings for each token [c820]. 

#### Dropout on Input Embeddings

Immediately after combining the embeddings, a dropout layer is applied [c814]. Dropout randomly turns off some elements of every input embedding to zero [c932]. The combined embeddings pass through a dropout layer [c448]. This regularization step helps prevent overfitting during training.

#### The Transformer Block Stack

 The preprocessing pipeline before the transformer block consists of: tokenization, token embedding, positional embedding, input embedding computation, and dropout [c939]. After that preprocessing, the tensor enters the transformer block stack.

The GPT model forward pass applies dropout to input embeddings, chains 12 transformer blocks using `nn.Sequential`, applies layer normalization, and outputs logits [c992]. 

Inside each transformer block, the feed-forward neural network uses GELU activation [c369]. Masked multi-head attention converts embedding vectors into context vectors that capture both semantic meaning and attention relationships between tokens [c945]. A context vector is an embedding that captures both the semantic meaning of a token and how much attention should be given to all other tokens in the sequence [c947].

 A shortcut connection (residual connection) adds the output of a layer back to its input to provide an alternative gradient flow path and prevent the vanishing gradient problem [c950].

#### Layer Normalization

 Layer normalization is a technique that normalizes token embeddings by setting the mean to zero and variance to one for each token independently [c943]. More precisely, layer normalization normalizes embedding values for each token such that the mean of the resultant embedding values is zero and the variance is equal to one [c963].

#### The Output Head

 The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. The logits matrix is the output produced by passing token embeddings through the output head neural network [c968].

The output head is the only stage in the GPT model where the input dimensions change from `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]. 

Logits are the final output matrices from a transformer model that contain probability scores for each token in the vocabulary at each position in the sequence [c432].

---

### The Forward Pass in Detail

Let's trace a concrete tensor through the model to make the shapes concrete.

The GPT model input is a tensor of shape `(batch_size, sequence_length)` containing token IDs [c990]. 

1. **Token embedding lookup:** Shape becomes `(2, 4, 768)`.
2. **Positional embedding lookup:** Shape is `(4, 768)`, broadcast-added to the token embeddings.
3. **Sum:** Shape remains `(2, 4, 768)` [c930].
4. **Dropout:** Shape unchanged, `(2, 4, 768)`.
5. **12 transformer blocks:** Shape unchanged throughout, `(2, 4, 768)`.
6. **Layer normalization:** Shape unchanged, `(2, 4, 768)`.
7. **Output head (linear):** Shape becomes `(2, 4, 50257)`.

The output of the GPT model is logits with shape `(batch_size, sequence_length, vocabulary_size)` [c994]. The GPT model architecture outputs numerical tensors [c1015].

The forward method of the GPT model takes a batch of input tokens, computes their embeddings, applies positional embeddings, passes the sequence through transformer blocks, normalizes the final output, and computes logits representing the next token's unnormalized probabilities [c1020].

The core GPT model implementation can be expressed in approximately 8 lines of code [c1019], which is a testament to how cleanly PyTorch's module system composes the sub-components we built in earlier chapters.

---

### Parameter Counting and Weight Tying

#### Counting Parameters

 The standard idiom is:

# Source: [c1004]
```python

```

Iterating over `model.parameters()` and summing `p.numel()` for each gives the total parameter count.

#### The Token Embedding and Output Layer

Both the token embedding layer and the output layer have shape `(50257, 768)` [c1007]. 

#### Weight Tying

The original GPT-2 architecture exploits this symmetry through a technique called **weight tying**: it reuses the weights from the token embedding layer in its output layer [c1006]. Weight tying reduces memory footprint from 163 million to 124 million parameters and reduces computational complexity [c1009].

When output layer parameters are removed from the total, the parameter count becomes exactly 124 million, matching the original GPT-2 model size [c1008]. 

However, it is worth noting that using separate token embedding and output layers results in better training and model performance compared to weight tying [c1010]. The implementation described in this book uses separate layers for clarity and performance, and the weight-tying reduction to 124M is achieved by the original GPT-2 design choice [c1009].

---

### Instantiating the Model

A GPT model instance is created by passing in a model configuration object [c996]:

```python

```

The full GPT architecture has been implemented and a GPT model instance has been initialized with random weights [c895]. The model can take an input, run inference through the 124 million parameter GPT architecture, and predict outputs on a laptop [c897].

The GPT model forward pass applies dropout to input embeddings, chains 12 transformer blocks using `nn.Sequential`, applies layer normalization, and outputs logits [c992]. The model outputs logits which will be converted into tokens and text outputs [c1021].

The main task of the GPT model is next token prediction [c917], and the GPT model outputs logits that can be decoded to predict the next token [c916].

---

### From Logits to Text: The Generation Pipeline

 The goal is to convert this output tensor into predictions of the next word [c817]. This section implements the full autoregressive generation loop.

#### The Five-Step Decoding Algorithm

The text generation algorithm consists of five sequential steps [c855]:

[FIGURE: Autoregressive generation loop — at each step, extract last-position logits, softmax → probabilities, argmax → token ID, append to sequence, feed back into model]

#### Implementing `generate_text_simple`

The `generate_text_simple` function takes a model (instance of GPT model class), input token indices (`idx`), maximum number of new tokens to generate, and context size as arguments [c856].

# Source: [c859]
```python

```

This slices the last `context_size` tokens from the current sequence.

# Source: [c872]
```python

 logits = model(idx_cond)
```

Using `torch.no_grad()` during inference saves memory and computation by disabling gradient bookkeeping.

The model outputs logits for every position in the sequence, but we only care about the prediction for the *next* token, which corresponds to the last position:

# Source: [c863]
```python

```

To generate text from GPT output, extract the last row of the output tensor to obtain logits for the next token prediction [c845].

In the `generate_text` function, softmax is applied to convert logits into a probability distribution before finding the index with the highest value using argmax [c874]:

# Source: [c865]
```python

```

# Source: [c867]
```python

```

# Source: [c869]
```python

```

In autoregressive text generation, the token generated in the previous iteration is appended to the input tokens for the next iteration [c807]. LLM text generation is an autoregressive process where the input context grows with each iteration as newly generated tokens are appended, and generation continues until the maximum number of new tokens is reached [c810].

#### The Complete `generate_text_simple` Function

Putting all six steps together inside a loop gives us the full generation function. Text generation proceeds sequentially by repeating the token prediction process multiple iterations until reaching a user-specified maximum number of new tokens [c850]. Text generation continues by repeating the prediction process until reaching the maximum number of new tokens [c843].

```python

 for _ in range(max_new_tokens):
 # Step 1: crop context to fit positional embedding table
 idx_cond = idx[:, -context_size:]
 
 # Step 2: forward pass, no gradients needed
 with torch.no_grad():
 logits = model(idx_cond)
 
 # Step 3: focus on the last time step
 logits = logits[:, -1, :] # (batch, vocab_size)
 
 # Step 4: logits → probabilities
 probas = torch.softmax(logits, dim=-1)
 
 # Step 5: greedy selection
 idx_next = torch.argmax(probas, dim=-1, keepdim=True)
 
 # Step 6: append to running sequence
 idx = torch.cat((idx, idx_next), dim=1)
 
 return idx
```

#### Running the Generator

The `generate_text_sample` function is called with parameters: model, inputs, `max_new_tokens=6`, and `context_size=1024` [c890]. Starting with input `'hello i am'` and `max_new_tokens=6`, the GPT model generates the complete output `'hello i am a model ready to help'` [c854]. 

---

### Greedy Decoding and Its Limitations

 

In GPT training, sampling techniques will modify the softmax output so the model does not always select the most likely token, introducing variability and creativity in generated text [c878]. Greedy decoding tends to produce repetitive, bland text because it never explores lower-probability but potentially more interesting continuations. 

---

### Putting It All Together: A Complete Walkthrough

Let's do one final end-to-end walkthrough to consolidate everything.

 Configuration.** We define `GPT_CONFIG_124M` with `vocab_size=50257`, `context_length=1024`, `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.1`, `qkv_bias=False` [c438].

 Model instantiation.** `model = GPTModel(GPT_CONFIG_124M)` creates a model with randomly initialized weights [c895, c996].

 Input preparation.** A tokenizer converts the input string into a list of token IDs. The GPT model input is a tensor of shape `(batch_size, sequence_length)` containing token IDs [c990].

 Forward pass.** The model maps the input through token embeddings, positional embeddings, dropout, 12 transformer blocks, layer normalization, and the output head [c811]. The output has shape `(batch_size, sequence_length, 50257)` [c994].

 Generation loop.** `generate_text_simple` runs the forward pass repeatedly, each time extracting the last-position logits [c845], applying softmax [c865], selecting the argmax token [c867], and appending it to the sequence [c869].

 

The implementation reached the stage where given a text input, the model can predict output tokens [c902]. All sub-modules of the GPT architecture were implemented and coded from scratch in this lecture series [c901]. Six to seven lectures were conducted covering the complete GPT architecture module [c903].

---

### What Comes Next

 The model outputs logits which will be converted into tokens and text outputs in a subsequent step [c1021], and the next set of lectures will focus on training the GPT-2 model with 124 million parameters [c906].

 Once trained, the same `generate_text_simple` function — or a more sophisticated sampling variant — will produce coherent, meaningful text rather than random token sequences.
