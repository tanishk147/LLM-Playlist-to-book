## Assembling and Running the Full GPT-2 Model

 

---

### The Big Picture: What the GPT Pipeline Looks Like

The complete pipeline is: input sentence → token IDs → token embeddings → add positional embeddings → input embeddings → dropout → transformer block → layer normalization → output head → output tensor [c811]. Each stage has a well-defined tensor shape, and the only stage where the embedding dimension changes is the very last one [c973].

The fundamental task the model is solving is next token prediction [c917]. Everything else—the attention heads, the feed-forward layers, the normalization—exists to make that prediction as accurate as possible. The model generates tokens given a sequence of input tokens [c805], and because each newly generated token is fed back in as input, the process is inherently autoregressive [c810].

---

### The GPT Configuration Object

Throughout this chapter we will refer to a configuration dictionary that holds all the hyperparameters in one place. 

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

 Let's walk through each component.

---

### Component Deep-Dive

#### Token Embeddings

 Token embeddings are of dimension 768 [c818]. The embedding table has shape `(vocab_size, emb_dim)` = `(50257, 768)` [c1007].

#### Positional Embeddings

Raw token embeddings carry no information about *where* in the sequence a token appears. Positional embedding is a vector representation of the position of a token within a sequence [c926]. 

#### Combining the Two

The input embedding for each token is the sum of its token embedding and its positional embedding [c930]:

Token embeddings are added to positional embeddings in the GPT model forward pass [c912]. Intuitively, this means every vector entering the transformer carries two kinds of information simultaneously: what the token *is* and where it *sits* in the sequence.

#### Dropout

Immediately after the embeddings are combined, a dropout layer is applied [c814]. Dropout randomly turns off some elements of every input embedding to zero [c932]. This acts as a regularizer during training, preventing the model from becoming overly reliant on any single embedding dimension. Dropout is applied after combining token and positional embeddings [c913].

#### The Transformer Blocks

The transformer block is the key component of the entire GPT model [c920]. The GPT model architecture consists of 12 transformer blocks chained together [c1000], and the forward pass chains them using `nn.Sequential` [c992].

Inside each transformer block, the two most important sub-components are:

- **Masked multi-head attention**, which converts embedding vectors into context vectors that capture both semantic meaning and attention relationships between tokens [c945]. A context vector is an embedding that captures both the semantic meaning of a token and how much attention should be given to all other tokens in the sequence [c947]. Multi-head attention is a key architectural component that enables coherent and meaningful outputs in modern GPT models like GPT-4 [c824].

- **Shortcut (residual) connections**, which add the output of a layer back to its input to provide an alternative gradient flow path and prevent the vanishing gradient problem [c950].

The preprocessing pipeline before the transformer block consists of: tokenization, token embedding, positional embedding, input embedding computation, and dropout [c939]. Everything after that pipeline lives inside the stacked transformer blocks.

#### Layer Normalization

 Layer normalization is a technique that normalizes token embeddings by setting the mean to zero and variance to one for each token independently [c943]. More precisely, it subtracts the mean and divides by the square root of variance to normalize token embeddings to mean 0 and standard deviation 1, with trainable scale and shift parameters [c985]. The second layer normalization in the transformer block normalizes embeddings by setting mean to 0 and variance to 1, applied to each token independently [c951]. Layer normalization normalizes embedding values for each token such that the mean of the resultant embedding values is zero and the variance is equal to one [c963].

#### The Output Head

The output head is a neural network at the final stage of the GPT model that transforms token embeddings into logits for vocabulary prediction [c965]. 

The output head is the only stage in the GPT model where the input dimensions change—from `(batch_size, num_tokens, 768)` to `(batch_size, num_tokens, 50257)` [c973]. The logits matrix is the output produced by passing token embeddings through the output head neural network [c968].

---

### The Full GPT Model Forward Pass

With all components understood individually, the forward pass of the complete model reads almost like pseudocode. The forward method of the GPT model takes a batch of input tokens, computes their embeddings, applies positional embeddings, passes the sequence through transformer blocks, normalizes the final output, and computes logits representing the next token's unnormalized probabilities [c1020].

The GPT model forward pass applies dropout to input embeddings, chains 12 transformer blocks using `nn.Sequential`, applies layer normalization, and outputs logits [c992]. The model input is a tensor of shape `(batch_size, sequence_length)` containing token IDs [c990], and the output is logits with shape `(batch_size, sequence_length, vocabulary_size)` [c994].

 The model outputs logits that can be decoded to predict the next token [c916].

Remarkably, the core GPT model implementation can be expressed in approximately 8 lines of code [c1019]. The complexity lives in the sub-modules, not in the top-level assembly.

---

### Counting Parameters: 124 M vs. 163 M

 You can count the total number of parameters in a model's parameter tensors using `p.numel()` [c1004]:

```python

print(f"Total parameters: {total_params:,}")
```

 Where does the discrepancy come from?

 Weight tying is a technique where the original GPT-2 architecture reuses the weights from the token embedding layer in its output layer [c1006]. The token embedding layer and output layer in the implemented model both have shape `(50257, 768)` [c1007], so they each hold 50257 × 768 ≈ 38.6 million parameters. When output layer parameters are removed from the total, the parameter count becomes exactly 124 million, matching the original GPT-2 model size [c1008].

Weight tying reduces memory footprint from 163 million to 124 million parameters and reduces computational complexity [c1009]. However, using separate token embedding and output layers results in better training and model performance compared to weight tying [c1010]. 

The assembled model requires 600 megabytes of space on a laptop [c1017], which is well within the reach of modern consumer hardware. The model can take an input, run inference through the 124 million parameter GPT architecture, and predict outputs on a laptop [c897].

---

### Generating Text: The Autoregressive Loop

 We need a function that repeatedly queries the model to produce a sequence of tokens. 

#### The Five-Step Algorithm

The text generation algorithm consists of five sequential steps [c855]:

 

LLM text generation is an autoregressive process where the input context grows with each iteration as newly generated tokens are appended, and generation continues until the maximum number of new tokens is reached [c810]. In autoregressive text generation, the token generated in the previous iteration is appended to the input tokens for the next iteration [c807]. The iterative generation process continues until a specified maximum number of new tokens has been generated [c808].

[FIGURE: Autoregressive generation loop diagram showing: input tokens [t1, t2, t3] → GPT model → logits → softmax → argmax → new token t4 → append → [t1, t2, t3, t4] → repeat]

#### The `generate_text_simple` Function

The `generate_text_simple` function takes a model (instance of GPT model class), input token indices (`idx`), maximum number of new tokens to generate, and context size as arguments [c856].

Let's build it step by step, citing each code fragment as it appears.

If the accumulated sequence grows longer than the model's context window, we must crop it before passing it in:

# Source: [c859]
```python

```

During inference we do not need gradients, so we wrap the forward call in `torch.no_grad()` to save memory and computation:

# Source: [c872]
```python

 logits = model(idx_cond)
```

The model outputs logits for every position in the sequence, but we only care about the *last* position because that is where the next token prediction lives [c845]:

# Source: [c863]
```python

```

This reduces the shape from `(batch_size, seq_len, vocab_size)` to `(batch_size, vocab_size)`.

# Source: [c865]
```python

```

Softmax is applied to convert logits into a probability distribution before finding the index with the highest value using argmax [c874]. Softmax is used in the current code to show the full process of transforming logits to probabilities, providing additional intuition even though it is technically redundant for argmax selection [c877]. 

# Source: [c867]
```python

```

# Source: [c869]
```python

```

Append the predicted token ID to the previous inputs for the next iteration to continue text generation [c842]. Text generation proceeds sequentially by repeating the token prediction process multiple iterations until reaching a user-specified maximum number of new tokens [c850]. Text generation continues by repeating the prediction process until reaching the maximum number of new tokens [c843].

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

The token generation loop iterates a number of times equal to the maximum number of new tokens parameter, generating one new token per iteration [c871].

---

### Running the Model: A Concrete Example

With the model instantiated and the generation function defined, we can run a quick end-to-end test. A GPT model instance is created by passing in a model configuration object [c996]. The full GPT architecture has been implemented and a GPT model instance has been initialized with random weights [c895].

The `generate_text_simple` function is called with parameters: model, inputs, `max_new_tokens=6`, and `context_size=1024` [c890].

Starting with input `'hello i am'` and `max_new_tokens=6`, the GPT model generates the complete output `'hello i am a model ready to help'` [c854].

The goal of this lecture is to convert the output tensor from the GPT model into predictions of the next word [c817]. GPT model output has the same number of rows as input tokens but number of columns equals vocabulary size (50257) [c828]. The model outputs logits which will be converted into tokens and text outputs in a subsequent lecture [c1021].

---

### Why the Output Is Nonsense (For Now)

The next token predictions from the untrained model produce random text because the model parameters have not been trained [c896]. The GPT architecture implementation has not yet undergone training [c905]. 

In GPT training, sampling techniques will modify the softmax output so the model does not always select the most likely token, introducing variability and creativity in generated text [c878]. 

The next set of lectures will focus on training the GPT-2 model with 124 million parameters [c906].

---

### What Was Built from Scratch

 The GPT architecture was built completely from scratch without using external libraries like Langchain [c900]. All sub-modules of the GPT architecture were implemented and coded from scratch in this lecture series [c901]. Six to seven lectures were conducted covering the complete GPT architecture module [c903].

The topics covered along the way include layer normalization, GELU activation, feed-forward networks, shortcut connections, and transformer blocks [c904]. The implementation reached the stage where given a text input, the model can predict output tokens [c902].

---

### Summary

 

The model has approximately 163 million raw parameters, which reduces to 124 million when weight tying is applied [c1008, c1009]. It fits in 600 MB of RAM [c1017] and can run inference on a laptop [c897].

The `generate_text_simple` function implements greedy autoregressive decoding: crop the context, run a forward pass, extract the last-position logits, apply softmax, take argmax, append the new token, and repeat [c855]. The model outputs logits which will be converted into tokens and text outputs in subsequent work [c1021], and training the model to produce meaningful outputs is the natural next step [c906].
