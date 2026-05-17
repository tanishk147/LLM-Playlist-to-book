## The Attention Mechanism: From Simplified to Multi-Head

 

This chapter builds the attention mechanism from the ground up. We start with the historical motivation—why older sequence models struggled and what researchers did about it—then construct a simplified, weight-free version of self-attention to build intuition. 

---

### Why Attention? A Brief History

#### Recurrent Neural Networks and Their Limits

 RNNs are designed to work with sequential data by maintaining a hidden state that captures information about previous inputs [c1535]. The hidden state is, in fact, the key innovation of RNNs that enables them to process sequential data at all [c1538].

The encoder–decoder architecture built on top of RNNs works roughly as follows: an encoder reads the input sequence token by token, updating its hidden state at each step. The final hidden state from the encoder is called the context vector [c1504], and it is handed to the decoder to generate the output sequence. The hidden state in an RNN captures memory of previous inputs in the sequence [c1502].

 Loss of context is the problem that occurs when an RNN decoder struggles to capture longer dependencies and contextual information because it relies on only one final hidden state [c1517]. Long-term dependencies in sentences—complex sentence structures where multiple clauses or phrases are connected—make it difficult for language models to identify relationships between distant words [c1481].

 LSTMs solve the vanishing gradient problem by maintaining both a long-term memory route and a short-term memory route [c1541]. But even LSTMs still funnel everything through a single context vector at the end of the encoder, which remains a bottleneck.

#### The Bahdanau Attention Breakthrough

The key insight that broke the bottleneck came from Bahdanau et al. In attention mechanisms, when decoding a particular part, the decoder has access to all of the input tokens and decides how much attention to give to each input [c1557]. The Bahdanau attention mechanism allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525].

The crucial property that makes this work is *dynamic focus*: the ability of the decoder to selectively choose which inputs to focus on and how much attention to give to each input at every decoding step [c1533]. The Bahdanau attention mechanism allows the decoder to have access to each input state during decoding and selectively decide which inputs to give more attention to [c1544].

 Instead of one summary, the decoder gets a weighted combination of *all* encoder states, with the weights recomputed fresh at every decoding step.

#### From Attention to Self-Attention

Traditional attention looks at one input sequence and one output sequence, determining which parts of the output sequence are more related to which parts of the input sequence [c1558]. Self-attention takes this idea one step further: it looks at one sequence and examines how different parts of that *same* sequence are related with respect to each other [c1559].

More precisely, self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. The term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions in a single input sequence [c1553].

---

### Simplified Self-Attention (No Trainable Weights)

Before introducing the full machinery of queries, keys, and values, it is instructive to build a simplified version of self-attention that has no learnable parameters at all [c2158]. This is the purest and most basic form of the attention technique [c1487], and working through it carefully will make the full version much easier to understand.

#### Tokens, Embeddings, and the Goal

 In the notation we will use throughout this chapter, $x_i$ denotes the vector representation of the $i$-th token, where $x_1$ is the first token, $x_2$ is the second token, and so on [c2176]. An input embedding is a vector representation of a token that encodes semantic meaning but does not carry information about how other words in the sentence relate to it [c2603].

Vector embeddings capture semantic meaning, such that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181]. The goal of self-attention is to produce a *context vector* for each token: an enriched embedding vector that combines contributions from all input embedding vectors weighted by their corresponding attention weights [c2236].

A context vector for a query contains information about both the query token itself and all other input elements in the sequence [c2184]. The context vector, denoted by $z$, is derived from attention weights and input vectors [c2179].

#### Attention Scores

The first step is to compute *attention scores*—intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187].

A *query* is the element or token being examined [c2183]. For each query token, we compute its attention score against every other token in the sequence. The natural tool for measuring similarity between two vectors is the dot product: the dot product of two vectors $a$ and $b$ is calculated as $a \cdot b = \sum(a_i \times b_i)$, which equals $|a||b|\cos\theta$, where $\theta$ is the angle between the vectors [c2193]. A higher dot product means the two vectors point in more similar directions, which we interpret as higher relevance.

#### Normalizing to Attention Weights

 Normalization is the process of transforming attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207].

 Softmax normalization takes the exponent of every element and divides by the summation of all exponents:

[c2216]

Naive softmax attention weights are computed by taking the exponent of each attention score and dividing by the summation of the exponents [c2224]. In PyTorch, `torch.nn.Softmax(dim=None)` applies the Softmax function to rescale n-dimensional input tensor elements to the range $[0, 1]$ and sum to 1 [c2229]. The `dim` parameter in `torch.softmax` specifies the dimension of the input tensor along which the normalization will be computed [c2265].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. An attention weight matrix is a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

#### Computing the Context Vector

[c2246]

 

---

### Scaled Dot-Product Self-Attention with Trainable Weights

 

 Self-attention introduces trainable weights which form the basis of the actual mechanism used in LLMs [c1488].

#### The Query, Key, and Value Matrices

The self-attention mechanism is implemented using three trainable weight matrices: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) [c1350]. 

More precisely:

- **Query** in the attention mechanism is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- **Key** in the attention mechanism represents items in the input sequence and is used to match with the query [c1470].
- **Value** in the attention mechanism represents the actual content or representation of the input items themselves [c1471].

$W_Q$, $W_K$, and $W_V$ are trainable weight matrices whose parameters need to be trained based on input data as part of LLM training [c2608]. In PyTorch, `torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters, which are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368].

#### Computing Queries, Keys, and Values

# Source: [c1443]
```python

queries = x @ W_query
values = x @ W_value
```

Or equivalently, using named weight matrices:

# Source: [c1373]
```python

values = inputs @ W_v
queries = inputs @ W_q
```

 

#### Scaling by $\sqrt{d_k}$

Attention scores are computed as queries multiplied by keys transpose: $Q \times K^T$ [c1933]. 

Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]:

# Source: [c1406]
```python

attention_scores_scaled = attention_scores / (d_k ** 0.5) # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1) # Apply softmax over columns
```

In a more compact form:

# Source: [c1445]
```python

attention_weights = softmax(attention_scores, dim=-1)
```

#### Computing the Final Context Vectors

Context vectors are computed by multiplying attention weights with the values matrix [c1938]. An attention weight between two tokens (e.g., between "journey" and "your") indicates how much the model should attend to the second token when processing the first token as the query [c1465].

# Source: [c1448]
```python

```

Each element in an attention score matrix row encodes how much a particular token relates to the query token, or how much importance should be paid to that token when processing the query [c1934]. 

---

### Causal (Masked) Attention

The self-attention mechanism described so far allows every token to attend to every other token in the sequence—both past and future. But for language modeling, where the model must predict the next token given only the tokens seen so far, allowing the model to "look ahead" at future tokens would be cheating.

#### The Concept of Causal Attention

Causal attention is a special form of self-attention [c2618], also called masked attention [c2617]. Causal attention restricts the model to only consider the previous and current inputs in a sequence when processing any given token [c2619]. More precisely, causal attention is designed to prevent future tokens from influencing past tokens [c2640].

The causal attention mask is the set of attention weights above the diagonal that are masked out [c2627]. Intuitively, if we lay out the attention weight matrix with query tokens as rows and key tokens as columns, then position $(i, j)$ represents how much token $i$ attends to token $j$. For causal attention, we want to zero out all positions where $j > i$—that is, all positions above the diagonal.

[FIGURE: Attention weight matrix with the upper triangle (above diagonal) shaded/masked, showing that token i can only attend to tokens 1 through i]

#### Triangular Matrices

To implement the mask, we need to understand two types of triangular matrices:

- An **upper triangular matrix** has all elements below the diagonal set to zero [c2630].
- A **lower triangular matrix** has all elements above the diagonal set to zero [c2631].

For causal masking, we want a lower triangular mask (ones on and below the diagonal, zeros above). PyTorch provides `torch.tril` for this:

# Source: [c2632]
```python

```

#### Approach 1: Multiply and Renormalize

# Source: [c2636]
```python

```

After masking, the rows no longer sum to one, so we renormalize:

# Source: [c2638]
```python

```

#### Approach 2: Mask Before Softmax (Preferred)

 Masking in transformers sets attention scores for future tokens to large negative values, making their influence in softmax calculation effectively zero [c2653]. 

# Source: [c2649]
```python

```

Then we fill those positions with $-\infty$:

# Source: [c2650]
```python

```

Then apply softmax as usual:

# Source: [c2651]
```python

```

#### Dropout in Attention

In practice, we also apply dropout to the attention weights after the softmax. Dropout is a deep learning technique where neurons in different layers are randomly switched to zero during training [c2654]. This regularization prevents the model from over-relying on any particular attention pattern.

`torch.nn.Dropout(p=0.5)` creates a dropout layer that randomly zeroes elements with probability 0.5 and scales remaining elements by $1/(1-0.5) = 2$ [c2670]. 

#### Putting It Together: The CausalAttention Forward Pass

The input `x` to the causal attention class has shape `(batch_size, num_tokens, input_dimensions)` [c1941]. 

# Source: [c2687]
```python

attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
context_vec = attn_weights @ values
```

Notice the slice `[:num_tokens, :num_tokens]` applied to the mask. The mask slicing `[:, :T]` ensures the causal mask is created only up to the number of tokens $T$ in the batch, handling cases where batch tokens are smaller than the supported context size [c2696]. 

---

### Multi-Head Attention

 In practice, LLMs use *multi-head attention*, which runs several attention mechanisms in parallel and combines their outputs. 

#### What Is Multi-Head Attention?

Multi-head attention refers to dividing the attention mechanism into multiple heads, where each head operates independently [c1952]. Multi-head attention is called "multi-head" because it aggregates the output of multiple independent attention heads [c1962]. More concretely, multi-head attention involves creating multiple instances of the self-attention mechanism, each with its own weights, and then combining their outputs [c1964].

[FIGURE: Multi-head attention diagram showing input being processed by H parallel attention heads, each with its own W_Q, W_K, W_V matrices, with outputs concatenated and projected]

#### The Wrapper Approach

 

In the forward method, `torch.cat` is used with `dimension=-1` to concatenate outputs along the columns [c1966]. This concatenation combines the context vectors from all heads into a single, wider vector. $D_{out}$ is the output dimension of each attention head [c1969].

 

#### The Efficient Weight-Split Approach

 

 This means we can compute all heads' queries, keys, and values in one batched matrix multiply, then reshape and transpose to get the per-head tensors.

[FIGURE: Diagram contrasting the wrapper approach (H separate attention modules) with the weight-split approach (one large projection followed by reshape into H heads)]

 

---

### The Mathematics of Scaling by $\sqrt{d_k}$

 

As $d_k$ grows, the dot products grow in magnitude proportionally to $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ brings the dot products back to unit variance, keeping the softmax in a well-behaved regime regardless of the embedding dimension.

Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]. 

---

### Putting It All Together

Let us trace through the complete flow of a single forward pass through a multi-head causal attention module, from input to output:

 The input `x` has shape `(batch_size, num_tokens, input_dimensions)` [c1941].

2. **Linear projections**: Each token embedding is projected into query, key, and value spaces using learned weight matrices $W_Q$, $W_K$, $W_V$ [c1350].

3. **Attention scores**: Queries are multiplied by the transpose of keys: $Q \times K^T$ [c1933]. Each element in the resulting matrix encodes how much a particular token relates to the query token [c1934].

4. **Scaling**: Scores are divided by $\sqrt{d_k}$ to prevent saturation [c1403].

5. **Causal masking**: Future positions are filled with $-\infty$ [c2653], so that after softmax they contribute zero weight.

6. **Softmax**: Scores are converted to attention weights that sum to one [c2216].

7. **Dropout**: Attention weights are randomly zeroed during training for regularization [c2654].

8. **Weighted sum**: Attention weights are multiplied by the values matrix to produce context vectors [c1938].

9. **Multi-head combination**: Context vectors from all heads are concatenated [c1966] and projected to the output dimension.

---

### Summary

- **Motivation**: RNNs compress entire sequences into a single context vector, losing information about long-range dependencies [c1517]. The Bahdanau attention mechanism solved this by giving the decoder dynamic access to all encoder states [c1525].

- **Self-attention**: Rather than attending across encoder and decoder, self-attention allows each position in a single sequence to attend to all other positions in that same sequence [c1548].

- **Simplified self-attention**: Without trainable weights, attention scores are raw dot products between embeddings, normalized by softmax to produce attention weights, which are then used to compute weighted sums of the input embeddings [c2158].

- **Scaled dot-product attention**: Adding trainable $W_Q$, $W_K$, $W_V$ matrices [c1350] and scaling by $\sqrt{d_k}$ [c1403] gives the model the ability to learn what to attend to.

- **Causal masking**: Setting future attention scores to $-\infty$ before softmax [c2653] ensures the model cannot look ahead, making it suitable for autoregressive language modeling [c2640].

- **Dropout**: Randomly zeroing attention weights during training [c2654] provides regularization.

- **Multi-head attention**: Running multiple independent attention heads [c1952] and concatenating their outputs [c1966] allows the model to attend to different types of relationships simultaneously.
