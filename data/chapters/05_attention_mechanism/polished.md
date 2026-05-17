## The Attention Mechanism: From Simplified to Multi-Head

Understanding how language models weigh the importance of different words is the central challenge this chapter addresses. We build the attention mechanism from the ground up—starting with the historical motivation for why older sequence models struggled, then constructing a simplified, weight-free version of self-attention to build intuition, before arriving at the full multi-head causal attention used in modern LLMs.

---

### Why Attention? A Brief History

#### Recurrent Neural Networks and Their Limits

Recurrent Neural Networks are designed to work with sequential data by maintaining a hidden state that captures information about previous inputs [c1535]—indeed, the hidden state is the key innovation that enables RNNs to process sequential data at all [c1538].

The encoder–decoder architecture built on top of RNNs works roughly as follows: an encoder reads the input sequence token by token, updating its hidden state at each step. The final hidden state from the encoder is called the context vector [c1504], and it is handed to the decoder to generate the output sequence. Because the hidden state captures memory of previous inputs [c1502], the entire source sequence must be compressed into this single vector—a significant bottleneck.

Loss of context is the problem that arises when an RNN decoder struggles to capture longer dependencies and contextual information because it relies on only that one final hidden state [c1517]. Long-term dependencies in sentences—complex structures where multiple clauses or phrases are connected—make it especially difficult for language models to identify relationships between distant words [c1481].

LSTMs alleviate the vanishing gradient problem by maintaining both a long-term memory route and a short-term memory route [c1541], but even LSTMs still funnel everything through a single context vector at the end of the encoder, leaving the bottleneck intact.

#### The Bahdanau Attention Breakthrough

The key insight that broke this bottleneck came from Bahdanau et al. Rather than forcing the decoder to rely solely on a single summary vector, the Bahdanau attention mechanism allows the decoder to selectively access different parts of the input sequence at each decoding step [c1525]. Concretely, when decoding a particular part of the output, the decoder has access to all input tokens and decides how much attention to give to each one [c1557].

The crucial property that makes this work is *dynamic focus*: the ability of the decoder to selectively choose which inputs to attend to, and how much weight to assign each input, at every decoding step [c1533][c1544]. Instead of one fixed summary, the decoder receives a weighted combination of *all* encoder states, with the weights recomputed fresh at every step.

#### From Attention to Self-Attention

Traditional attention operates across two sequences—an input and an output—determining which parts of the output are more related to which parts of the input [c1558]. Self-attention takes this idea one step further: it looks at a single sequence and examines how different positions within that *same* sequence relate to each other [c1559].

More precisely, self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. The term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions within a single input sequence [c1553].

---

### Simplified Self-Attention (No Trainable Weights)

Before introducing the full machinery of queries, keys, and values, it is instructive to build a simplified version of self-attention with no learnable parameters at all [c2158]. This is the purest and most basic form of the attention technique [c1487], and working through it carefully makes the full version much easier to understand.

#### Tokens, Embeddings, and the Goal

Throughout this chapter, $x_i$ denotes the vector representation of the $i$-th token, where $x_1$ is the first token, $x_2$ is the second, and so on [c2176]. An input embedding is a vector representation of a token that encodes semantic meaning but does not carry information about how other words in the sentence relate to it [c2603]. Vector embeddings capture semantic meaning such that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181].

The goal of self-attention is to produce a *context vector* for each token: an enriched embedding that combines contributions from all input embeddings weighted by their corresponding attention weights [c2236]. A context vector for a query contains information about both the query token itself and all other input elements in the sequence [c2184], and is denoted by $z$ [c2179].

#### Attention Scores

The first step is to compute *attention scores*—intermediate values that quantify how much importance should be paid to each input word relative to a query word [c2187]. A *query* is simply the element or token currently being examined [c2183].

For each query token, we compute its attention score against every other token in the sequence. The natural tool for measuring similarity between two vectors is the dot product: $a \cdot b = \sum(a_i \times b_i) = |a||b|\cos\theta$, where $\theta$ is the angle between the vectors [c2193]. A higher dot product means the two vectors point in more similar directions, which we interpret as higher relevance.

#### Normalizing to Attention Weights

Raw dot products are not yet directly usable—they need to be normalized. Normalization transforms attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207].

Softmax normalization achieves this by taking the exponent of every element and dividing by the sum of all exponents:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$ [c2216]

Naive softmax attention weights are thus computed by exponentiating each attention score and dividing by the sum of exponents [c2224]. In PyTorch, `torch.nn.Softmax(dim=None)` applies this function to rescale tensor elements to the range $[0, 1]$ with a sum of 1 [c2229]; the `dim` parameter specifies the dimension along which normalization is computed [c2265].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. Arranged into a matrix, each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

#### Computing the Context Vector

Once we have attention weights, the context vector is computed as a weighted sum over all input embeddings:

$$z_i = \sum_j \text{attention\_weight}_j \times \text{embedding\_vector}_j$$ [c2246]

---

### Scaled Dot-Product Self-Attention with Trainable Weights

The simplified version above has no learnable parameters. Self-attention introduces trainable weights, which form the basis of the actual mechanism used in LLMs [c1488].

#### The Query, Key, and Value Matrices

The self-attention mechanism is implemented using three trainable weight matrices: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) [c1350]. Each plays a distinct role:

- **Query** is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- **Key** represents items in the input sequence and is used to match against the query [c1470].
- **Value** represents the actual content or representation of the input items themselves [c1471].

$W_Q$, $W_K$, and $W_V$ are trainable weight matrices whose parameters are learned from data during LLM training [c2608]. In PyTorch, `torch.nn.Parameter` is a Tensor subclass that marks tensors as module parameters, automatically adding them to the module's parameter list and exposing them through the `parameters()` iterator [c1367]. These weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368].

#### Computing Queries, Keys, and Values

Given an input `x`, the three projections are computed as straightforward matrix multiplications [c1443]:

```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

Or equivalently, using named weight matrices [c1373]:

```python
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

#### Scaling by $\sqrt{d_k}$

Attention scores are computed as queries multiplied by the transpose of keys: $Q \times K^T$ [c1933]. However, as the key dimension $d_k$ grows, dot products grow in magnitude, which can push the softmax into regions with very small gradients. The solution is to scale by $\sqrt{d_k}$ before applying softmax [c1403]:

```python
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```
[c1406]

In compact form [c1445]:

```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

#### Computing the Final Context Vectors

Context vectors are computed by multiplying attention weights by the values matrix [c1938]. An attention weight between two tokens (e.g., between "journey" and "your") indicates how much the model should attend to the second token when processing the first as the query [c1465]:

```python
context_vector = attention_weights @ values
```
[c1448]

Each element in an attention score matrix row encodes how much a particular token relates to the query token—that is, how much importance should be paid to that token when processing the query [c1934].

---

### Causal (Masked) Attention

The self-attention mechanism described so far allows every token to attend to every other token in the sequence, including future ones. For language modeling, where the model must predict the next token given only the tokens seen so far, this look-ahead would be a form of cheating. Causal attention solves this problem.

#### The Concept of Causal Attention

Causal attention is a special form of self-attention [c2618], also called masked attention [c2617]. It restricts the model to only consider previous and current inputs when processing any given token [c2619]—more precisely, it is designed to prevent future tokens from influencing past tokens [c2640].

The causal attention mask is the set of attention weights above the diagonal that are zeroed out [c2627]. Intuitively, if we lay out the attention weight matrix with query tokens as rows and key tokens as columns, position $(i, j)$ represents how much token $i$ attends to token $j$. For causal attention, all positions where $j > i$—everything above the diagonal—must be masked.

[FIGURE: Attention weight matrix with the upper triangle (above diagonal) shaded/masked, showing that token i can only attend to tokens 1 through i]

#### Triangular Matrices

To implement the mask, we rely on two types of triangular matrices:

- An **upper triangular matrix** has all elements below the diagonal set to zero [c2630].
- A **lower triangular matrix** has all elements above the diagonal set to zero [c2631].

For causal masking we want a lower triangular mask—ones on and below the diagonal, zeros above. PyTorch provides `torch.tril` for exactly this purpose [c2632]:

```python
mask = torch.tril(torch.ones(context_length, context_length))
```

#### Approach 1: Multiply and Renormalize

One straightforward approach is to multiply the attention weights element-wise by the mask [c2636]:

```python
masked_attn_weights = attn_weights * mask_simple
```

Because the masked rows no longer sum to one, we renormalize [c2638]:

```python
mask_simple_normalized = masked_attn_weights / masked_attn_weights.sum(dim=1, keepdim=True)
```

#### Approach 2: Mask Before Softmax (Preferred)

A cleaner and more numerically stable approach is to mask *before* applying softmax. Masking in transformers sets attention scores for future tokens to large negative values, making their influence in the softmax calculation effectively zero [c2653].

First, construct the upper triangular mask [c2649]:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

Then fill those positions with $-\infty$ [c2650]:

```python
attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))
```

Then apply softmax as usual [c2651]:

```python
attention_weights = torch.softmax(attention_scores / sqrt(d_k), dim=-1)
```

Because $e^{-\infty} = 0$, the masked positions contribute exactly zero weight after softmax—no separate renormalization step is needed.

#### Dropout in Attention

In practice, dropout is also applied to the attention weights after the softmax. Dropout is a deep learning technique where neurons in different layers are randomly switched to zero during training [c2654], preventing the model from over-relying on any particular attention pattern. `torch.nn.Dropout(p=0.5)` creates a dropout layer that randomly zeroes elements with probability 0.5 and scales the remaining elements by $1/(1-0.5) = 2$ [c2670].

#### Putting It Together: The CausalAttention Forward Pass

The input `x` to the causal attention class has shape `(batch_size, num_tokens, input_dimensions)` [c1941]. A complete forward pass looks like this [c2687]:

```python
attn_scores = queries @ keys.transpose(1, 2)
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
context_vec = attn_weights @ values
```

Notice the slice `[:num_tokens, :num_tokens]` applied to the mask. This ensures the causal mask is created only up to the number of tokens $T$ in the current batch, gracefully handling cases where the batch is smaller than the maximum supported context size [c2696].

---

### Multi-Head Attention

In practice, LLMs use *multi-head attention*, which runs several attention mechanisms in parallel and combines their outputs, allowing the model to capture different types of relationships simultaneously.

#### What Is Multi-Head Attention?

Multi-head attention refers to dividing the attention mechanism into multiple heads, each operating independently [c1952]. It is called "multi-head" because it aggregates the outputs of these multiple independent attention heads [c1962]. More concretely, it involves creating multiple instances of the self-attention mechanism, each with its own weight matrices, and then combining their outputs [c1964].

[FIGURE: Multi-head attention diagram showing input being processed by H parallel attention heads, each with its own W_Q, W_K, W_V matrices, with outputs concatenated and projected]

#### The Wrapper Approach

The simplest implementation wraps several causal attention modules and concatenates their outputs. In the forward method, `torch.cat` is used with `dimension=-1` to concatenate outputs along the columns [c1966], combining the context vectors from all heads into a single, wider vector. $D_{out}$ is the output dimension of each individual attention head [c1969].

#### The Efficient Weight-Split Approach

A more computationally efficient alternative avoids running separate modules entirely. All heads' queries, keys, and values can be computed in one batched matrix multiply, then reshaped and transposed to obtain the per-head tensors—achieving the same result with significantly less overhead.

[FIGURE: Diagram contrasting the wrapper approach (H separate attention modules) with the weight-split approach (one large projection followed by reshape into H heads)]

---

### The Mathematics of Scaling by $\sqrt{d_k}$

As $d_k$ grows, dot products grow in magnitude proportionally to $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ brings the dot products back to unit variance, keeping the softmax in a well-behaved regime regardless of the embedding dimension. Attention weights are therefore computed by scaling attention scores by the square root of the key embedding dimension before applying softmax [c1403].

---

### Putting It All Together

Let us trace through the complete flow of a single forward pass through a multi-head causal attention module, from input to output:

1. **Input**: `x` has shape `(batch_size, num_tokens, input_dimensions)` [c1941].
2. **Linear projections**: Each token embedding is projected into query, key, and value spaces using learned weight matrices $W_Q$, $W_K$, $W_V$ [c1350].
3. **Attention scores**: Queries are multiplied by the transpose of keys: $Q \times K^T$ [c1933]. Each element in the resulting matrix encodes how much a particular token relates to the query token [c1934].
4. **Scaling**: Scores are divided by $\sqrt{d_k}$ to prevent softmax saturation [c1403].
5. **Causal masking**: Future positions are filled with $-\infty$ [c2653] so that after softmax they contribute zero weight.
6. **Softmax**: Scores are converted to attention weights that sum to one [c2216].
7. **Dropout**: Attention weights are randomly zeroed during training for regularization [c2654].
8. **Weighted sum**: Attention weights are multiplied by the values matrix to produce context vectors [c1938].
9. **Multi-head combination**: Context vectors from all heads are concatenated [c1966] and projected to the output dimension.

---

### Summary

- **Motivation**: RNNs compress entire sequences into a single context vector, losing information about long-range dependencies [c1517]. The Bahdanau attention mechanism solved this by giving the decoder dynamic access to all encoder states at every decoding step [c1525].
- **Self-attention**: Rather than attending across encoder and decoder, self-attention allows each position in a single sequence to attend to all other positions in that same sequence [c1548].
- **Simplified self-attention**: Without trainable weights, attention scores are raw dot products between embeddings, normalized by softmax to produce attention weights, which are then used to compute weighted sums of the input embeddings [c2158].
- **Scaled dot-product attention**: Adding trainable $W_Q$, $W_K$, $W_V$ matrices [c1350] and scaling by $\sqrt{d_k}$ [c1403] gives the model the ability to learn what to attend to.
- **Causal masking**: Setting future attention scores to $-\infty$ before softmax [c2653] ensures the model cannot look ahead, making it suitable for autoregressive language modeling [c2640].
- **Dropout**: Randomly zeroing attention weights during training [c2654] provides regularization against over-reliance on specific attention patterns.
- **Multi-head attention**: Running multiple independent attention heads [c1952] and concatenating their outputs [c1966] allows the model to attend to different types of relationships simultaneously.