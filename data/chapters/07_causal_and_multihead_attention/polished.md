## Causal Self-Attention and Multi-Head Attention

Self-attention, as introduced in earlier chapters, lets every token attend to every other token in the sequence—exactly what you want for tasks like sentiment classification, where the full context is available at inference time. Language modeling is different: when a model generates the next word, it has not yet seen the words that follow the current position. Allowing the model to peek at future tokens during training would be cheating, because the model would learn to rely on information it cannot possibly have at generation time.

The solution is **causal attention**, sometimes called **masked attention** [c2617]. This chapter builds causal attention from the ground up, explains why masking must happen *before* the softmax, adds dropout for regularization, and then scales the whole mechanism up to **multi-head attention**—the version that actually runs inside GPT.

---

### The Problem with Unrestricted Self-Attention

Recall the standard self-attention pipeline: compute queries, keys, and values from the input; form an attention score matrix; apply softmax to get attention weights; and finally compute context vectors as a weighted sum of the values [c1938]. In this unrestricted form, every token can attend to every other token, including tokens that appear later in the sequence.

Causal attention is a special form of self-attention [c2618] that restricts the model to only consider the previous and current inputs when processing any given token [c2619]. Concretely, it is designed to prevent future tokens from influencing past tokens [c2640]. In GPT-like large language models, this is achieved by masking out future tokens when computing attention scores [c2622].

[FIGURE: A 4×4 attention score matrix with the upper triangle (above the diagonal) shaded in grey to indicate masked positions, and the lower triangle plus diagonal left open]

---

### Building the Causal Mask

The causal mask is the set of attention weights above the diagonal that are masked out [c2627]. To understand the geometry: an upper triangular matrix has all elements below the diagonal set to zero [c2630], while a lower triangular matrix has all elements above the diagonal set to zero [c2631]. The causal mask is upper-triangular, covering exactly the positions where token $j > i$—that is, where token $j$ lies in the future relative to token $i$.

#### A Simple Lower-Triangular Mask

The most direct approach is to create a lower-triangular matrix of ones and multiply it element-wise with the attention weights:

```python
mask_simple = torch.tril(torch.ones(context_length, context_length))
```
[c2632]

After computing raw attention weights (before any normalization), apply the mask:

```python
masked_attn_weights = attn_weights * mask_simple
```
[c2636]

This zeros out all upper-triangular positions, but each row no longer sums to one, so renormalization is required:

```python
mask_simple_normalized = masked_attn_weights / masked_attn_weights.sum(dim=1, keepdim=True)
```
[c2638]

Intuitively this works, but it has a subtle flaw.

#### Why You Must Mask Before Softmax

Applying softmax first and then zeroing out future positions creates a data leakage problem [c2643]. The softmax has already "seen" the future token scores when computing the exponentials and the normalization denominator; zeroing those positions afterward and renormalizing does not produce the same result as if the future tokens had never participated in the softmax at all.

The correct approach is to apply an upper-triangular mask of negative infinity to the raw attention *scores* before softmax [c2645]. Because $e^{-\infty} = 0$, those positions contribute nothing to the softmax denominator, and the remaining weights automatically sum to one without any manual renormalization [c2653].

The preferred implementation creates an upper-triangular mask (ones above the diagonal) and fills those positions with $-\infty$:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```
[c2649]

```python
attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))
```
[c2650]

Softmax is then applied in the usual way:

```python
attention_weights = torch.softmax(attention_scores / sqrt(d_k), dim=-1)
```
[c2651]

This causal masking mechanism prevents information leakage from future tokens while ensuring that every row of attention weights sums to one [c2652].

---

### Adding Dropout to Attention Weights

Regularization is important in large models. Dropout is a deep learning technique where neurons in different layers are randomly switched to zero during training [c2654]. In causal attention specifically, dropout randomly zeros out attention weights while preserving the causal masking of future tokens [c2661].

For example, `torch.nn.Dropout(p=0.5)` creates a dropout layer that randomly zeroes elements with probability 0.5 and scales the remaining elements by $1/(1-0.5) = 2$ [c2670]. This scaling ensures the expected value of the output is unchanged, so the same model can be used at inference time without adjustment.

---

### The CausalSelfAttention Class

With masking and dropout in hand, we can write a complete causal self-attention class. The CausalSelfAttention class incorporates causal attention and dropout into the self-attention mechanism, with the same structure as the base self-attention class except for the masking and dropout additions [c2672].

The input has shape `(batch_size, num_tokens, input_dimensions)` [c1941], and the weight matrices $W_Q$, $W_K$, and $W_V$ are trainable parameters learned during LLM training [c2608]. The core of the forward pass—computing scores, masking, softmax, dropout, and context vectors—looks like this:

```python
attn_scores = queries @ keys.transpose(1, 2)
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
context_vec = attn_weights @ values
```
[c2687]

Notice the mask slicing `[:num_tokens, :num_tokens]`: it ensures the causal mask is applied only up to the number of tokens in the batch, handling cases where the batch is shorter than the maximum supported context size [c2696].

An input embedding is a vector representation of a token that encodes semantic meaning but does not carry information about how other words in the sentence relate to it [c2603]. The context vectors produced by causal attention remedy this—each one aggregates information from all *previous* tokens (and the token itself), weighted by learned attention scores.

---

### From Single-Head to Multi-Head Attention

A single causal attention head produces one set of context vectors. Multi-head attention divides the attention mechanism into multiple heads, each operating independently [c1924, c1952], and is called "multi-head" because it aggregates the outputs of those independent heads [c1962].

The motivation is representational richness: different heads can learn to attend to different aspects of the input, and their outputs are combined to form a richer context representation.

#### Naive Wrapper Implementation

The straightforward approach is to create multiple instances of the causal self-attention mechanism, each with its own weights, and concatenate their outputs [c1964]. In the forward method, `torch.cat` with `dimension=-1` concatenates outputs along the column dimension [c1966]. A causal attention module is applied to compute context vectors for each head separately, and the results are then concatenated [c1118], with the output dimension $D_{out}$ of each head [c1969] chosen so that the concatenated result has the desired total dimensionality.

While conceptually clean, this approach is less efficient because it requires a separate matrix multiplication for each head at the start of the forward pass [c1117].

---

### The Efficient Weight-Split Implementation

The production-grade approach—used in GPT—performs a single large matrix multiplication and then *splits* the result across heads. This is more computationally efficient than maintaining separate classes for a multi-head attention wrapper and causal attention [c1119].

The key insight is that instead of $h$ separate small weight matrices, a single weight matrix of dimension $d_{in} \times d_{out}$ is used, where:

$$D_{out} = \text{head\_dimension} \times \text{number\_of\_heads}$$ [c1124]

Equivalently:

$$\text{head\_dim} = \frac{d_{out}}{\text{num\_heads}}$$ [c1147]

The head dimension is the dimension of the representation within each individual attention head [c1146]. For a concrete example, with $d_{out} = 6$ and $\text{num\_heads} = 2$:

$$\text{head\_dim} = d_{out} / \text{num\_heads} = 6 / 2 = 3$$ [c1163]

The number of attention heads is the count of parallel attention mechanisms that operate on the output dimension [c1143].

#### Initializing the Layers

The `__init__` method creates three linear projections—one each for queries, keys, and values—plus an output projection, a dropout layer, and a causal mask buffer:

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
```
[c1152]

```python
self.out_proj = nn.Linear(d_out, d_out)
self.dropout = nn.Dropout(dropout)
self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
```
[c1127]

The mask is registered as a buffer rather than a parameter so that it moves to the correct device automatically but is not updated by the optimizer.

#### The Forward Method: Step by Step

The forward method takes an input tensor with three dimensions: batch size, number of tokens, and input dimension $D_{in}$ [c1134]. $D_{in}$ represents the dimensionality of the vector embedding for each token [c1135].

**Step 1 — Project to queries, keys, and values.**

```python
keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
queries = self.W_query(x)
values = self.W_value(x)
```
[c1128]

In the resulting matrices of shape `(batch_size, num_tokens, d_out)`, each row corresponds to one token represented as a $d_{out}$-dimensional vector [c1160].

**Step 2 — Reshape to expose the head dimension.**

The last dimension ($d_{out}$) is unrolled into the number of heads and the head dimension [c1165], using the relationship $d_{out} = \text{head\_dimension} \times \text{num\_heads}$ [c1167]:

```python
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)
```
[c1178]

Each tensor now has shape `(batch_size, num_tokens, num_heads, head_dim)`.

**Step 3 — Transpose to bring the head dimension before the token dimension.**

```python
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
```
[c1180]

After transposing, the shape becomes `(batch_size, num_heads, num_tokens, head_dim)`. This layout lets PyTorch's batch matrix multiplication operate independently on each head.

**Step 4 — Compute attention scores.**

Attention scores are computed by multiplying queries with the transpose of keys along dimensions 2 and 3 [c1186]:

```python
attn_scores = queries @ keys.transpose(2, 3)  # Compute scaled dot-product attention scores for each head
```
[c1205]

Each element in an attention score matrix row encodes how much a particular token relates to the query token—that is, how much importance should be paid to it when processing the query [c1934].

**Step 5 — Apply the causal mask.**

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```
[c1206]

The causal attention mechanism ensures that only attention scores with tokens that come before (and including) the current token survive; all scores involving future tokens are zeroed out [c1197].

**Step 6 — Scale, softmax, and dropout.**

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
```
[c1209, c1133]

Attention weights differ from attention scores in that each row of attention weights sums to one, whereas attention scores do not have this property [c1228].

**Step 7 — Compute context vectors and project.**

The context vector matrix is computed by multiplying attention weights with the values matrix [c1213]. Each row of the result represents the context vector for a particular token, with dimensions equal to `head_dim` [c1218]:

```python
context_vec = (attn_weights @ values).transpose(1, 2)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
context_vec = self.out_proj(context_vec)
```
[c1225]

The `.transpose(1, 2)` swaps the head and token dimensions back, and `.view(b, num_tokens, self.d_out)` concatenates the head outputs by merging the last two dimensions. The output projection `self.out_proj` then mixes information across heads with a learned linear transformation. The resulting $d_{out}$ is the dimension of the context embedding vector produced for each token [c1140].

[FIGURE: Data flow diagram showing input tensor (b, T, d_in) → three linear projections → reshape and transpose to (b, num_heads, T, head_dim) → scaled dot-product attention per head → transpose and reshape back to (b, T, d_out) → output projection]

---

### A Working Example

To verify the implementation, consider a batch of two sequences, each with three tokens of embedding dimension 6:

```python
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
```
[c1138]

We then instantiate the multi-head attention module and run a forward pass:

```python
batch_size, context_length, d_in = batch.shape
d_out = 6
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs.shape)
```
[c1233]

With `d_out = 6` and `num_heads = 2`, the head dimension is $6 / 2 = 3$ [c1163]. The output shape is `(batch_size, num_tokens, 6)`, matching the input embedding dimension—a common design choice that makes it straightforward to stack transformer blocks.

---

### Comparing the Two Implementations

| Property | Wrapper (separate heads) | Weight-split (efficient) |
|---|---|---|
| Number of QKV projections | One per head | One shared projection |
| Matrix multiplications at start | $h$ separate | 1 large |
| Conceptual clarity | High | Moderate |
| Computational efficiency | Lower [c1117] | Higher [c1119] |

The weight-split implementation is what you will find in production GPT code. The wrapper is useful for building intuition, but the efficient version is what we carry forward into the full model.

---

### Summary

This chapter traced the full path from unrestricted self-attention to the efficient multi-head causal attention used in GPT:

1. **Causal masking**: Future token positions are filled with $-\infty$ before softmax [c2645], preventing data leakage [c2643] and ensuring attention weight rows sum to one [c2652].
2. **Mask geometry**: An upper-triangular mask created with `torch.triu(..., diagonal=1)` identifies future positions [c2649].
3. **Dropout**: Applied to attention weights after softmax to regularize training [c2661].
4. **Multi-head attention**: Multiple independent attention heads [c1924] are implemented efficiently by projecting to a large $d_{out}$, reshaping to expose the head dimension, running batched attention, and reshaping back [c1165, c1167].
5. **Output projection**: A final linear layer mixes information across heads [c1225].

The attention series covered simplified self-attention, self-attention, causal attention, and multi-head attention [c359]. With multi-head causal attention complete, we have the core computational primitive of the transformer. The next step is to embed this module inside a full transformer block, add feed-forward layers and layer normalization, and stack those blocks into a complete GPT model.