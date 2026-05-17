## The Attention Mechanism: From Simplified to Multi-Head

Attention is the engine at the heart of every modern large language model. Without it, transformers would not exist, and GPT-style models would be impossible. Yet the idea is surprisingly approachable: at its core, attention is just a principled way of asking "when I am processing this token, which other tokens in the sequence should I care about, and how much?"

This chapter builds the attention mechanism from the ground up. We start with the historical motivation—why older sequence models struggled and what researchers did about it—then construct a simplified, weight-free version of self-attention to build intuition. From there we add trainable weight matrices (the Query, Key, and Value projections), introduce causal masking and dropout to make the mechanism suitable for autoregressive language modeling, and finally assemble the full multi-head attention module that appears in real LLM architectures. Every piece of code in this chapter is runnable PyTorch; every equation is the real thing.

---

### Why Attention? A Brief History

To appreciate why attention was invented, it helps to understand what came before it.

#### Recurrent Neural Networks and Their Limits

Before transformers, the dominant approach to sequence modeling was the Recurrent Neural Network (RNN). RNNs are designed to work with sequential data by maintaining a hidden state that captures information about previous inputs [c1535]. The hidden state is, in fact, the key innovation of RNNs that enables them to process sequential data at all [c1538].

The encoder–decoder architecture built on top of RNNs works roughly as follows: an encoder reads the input sequence token by token, updating its hidden state at each step. The final hidden state from the encoder is called the context vector [c1504], and it is handed to the decoder to generate the output sequence. The hidden state in an RNN captures memory of previous inputs in the sequence [c1502].

The problem is that compressing an entire input sequence into a single fixed-size vector is a lossy operation. Loss of context is the problem that occurs when an RNN decoder struggles to capture longer dependencies and contextual information because it relies on only one final hidden state [c1517]. Long-term dependencies in sentences—complex sentence structures where multiple clauses or phrases are connected—make it difficult for language models to identify relationships between distant words [c1481].

LSTMs (Long Short-Term Memory networks) were one response to this problem. LSTMs solve the vanishing gradient problem by maintaining both a long-term memory route and a short-term memory route [c1541]. But even LSTMs still funnel everything through a single context vector at the end of the encoder, which remains a bottleneck.

#### The Bahdanau Attention Breakthrough

The key insight that broke the bottleneck came from Bahdanau et al. In attention mechanisms, when decoding a particular part, the decoder has access to all of the input tokens and decides how much attention to give to each input [c1557]. The Bahdanau attention mechanism allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525].

The crucial property that makes this work is *dynamic focus*: the ability of the decoder to selectively choose which inputs to focus on and how much attention to give to each input at every decoding step [c1533]. The Bahdanau attention mechanism allows the decoder to have access to each input state during decoding and selectively decide which inputs to give more attention to [c1544].

This is a fundamentally different paradigm from the single-context-vector approach. Instead of one summary, the decoder gets a weighted combination of *all* encoder states, with the weights recomputed fresh at every decoding step.

#### From Attention to Self-Attention

Traditional attention looks at one input sequence and one output sequence, determining which parts of the output sequence are more related to which parts of the input sequence [c1558]. Self-attention takes this idea one step further: it looks at one sequence and examines how different parts of that *same* sequence are related with respect to each other [c1559].

More precisely, self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. The term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions in a single input sequence [c1553].

Self-attention is the foundation of the transformer architecture and, by extension, of every LLM we will build in this book.

---

### Simplified Self-Attention (No Trainable Weights)

Before introducing the full machinery of queries, keys, and values, it is instructive to build a simplified version of self-attention that has no learnable parameters at all [c2158]. This is the purest and most basic form of the attention technique [c1487], and working through it carefully will make the full version much easier to understand.

#### Tokens, Embeddings, and the Goal

Recall that each token in the input is represented as a vector. In the notation we will use throughout this chapter, $x_i$ denotes the vector representation of the $i$-th token, where $x_1$ is the first token, $x_2$ is the second token, and so on [c2176]. An input embedding is a vector representation of a token that encodes semantic meaning but does not carry information about how other words in the sentence relate to it [c2603].

Vector embeddings capture semantic meaning, such that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181]. But the raw embedding for a token is context-free—it says nothing about how the word is being used in this particular sentence. The goal of self-attention is to produce a *context vector* for each token: an enriched embedding vector that combines contributions from all input embedding vectors weighted by their corresponding attention weights [c2236].

A context vector for a query contains information about both the query token itself and all other input elements in the sequence [c2184]. The context vector, denoted by $z$, is derived from attention weights and input vectors [c2179].

#### Attention Scores

The first step is to compute *attention scores*—intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187].

A *query* is the element or token being examined [c2183]. For each query token, we compute its attention score against every other token in the sequence. The natural tool for measuring similarity between two vectors is the dot product: the dot product of two vectors $a$ and $b$ is calculated as $a \cdot b = \sum(a_i \times b_i)$, which equals $|a||b|\cos\theta$, where $\theta$ is the angle between the vectors [c2193]. A higher dot product means the two vectors point in more similar directions, which we interpret as higher relevance.

#### Normalizing to Attention Weights

Raw dot products are not directly usable as weights because they can be arbitrarily large or small and do not sum to one. Normalization is the process of transforming attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207].

The standard normalization function is softmax. Softmax normalization takes the exponent of every element and divides by the summation of all exponents:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

[c2216]

Naive softmax attention weights are computed by taking the exponent of each attention score and dividing by the summation of the exponents [c2224]. In PyTorch, `torch.nn.Softmax(dim=None)` applies the Softmax function to rescale n-dimensional input tensor elements to the range $[0, 1]$ and sum to 1 [c2229]. The `dim` parameter in `torch.softmax` specifies the dimension of the input tensor along which the normalization will be computed [c2265].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. An attention weight matrix is a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

#### Computing the Context Vector

Once we have attention weights, the context vector for token $i$ is:

$$Z_i = \sum_j \text{attention\_weight}_j \times \text{embedding\_vector}_j$$

[c2246]

Intuitively, this is a weighted average of all the input embeddings, where the weights reflect how relevant each token is to the query. Tokens that are highly relevant get a large weight and contribute more to the context vector; irrelevant tokens contribute almost nothing.

---

### Scaled Dot-Product Self-Attention with Trainable Weights

The simplified version above is useful for building intuition, but it has a critical limitation: there are no learnable parameters. Every token's relationship to every other token is determined entirely by the raw similarity of their embeddings, with no room for the model to learn task-specific notions of relevance.

The solution is to introduce trainable weight matrices. Self-attention introduces trainable weights which form the basis of the actual mechanism used in LLMs [c1488].

#### The Query, Key, and Value Matrices

The self-attention mechanism is implemented using three trainable weight matrices: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) [c1350]. These names come from an analogy to database retrieval: a query is like a search query, a key is like a database index, and a value is the actual content retrieved.

More precisely:

- **Query** in the attention mechanism is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- **Key** in the attention mechanism represents items in the input sequence and is used to match with the query [c1470].
- **Value** in the attention mechanism represents the actual content or representation of the input items themselves [c1471].

$W_Q$, $W_K$, and $W_V$ are trainable weight matrices whose parameters need to be trained based on input data as part of LLM training [c2608]. In PyTorch, `torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters, which are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368].

#### Computing Queries, Keys, and Values

Given an input matrix $x$ (where each row is a token embedding), we project into the query, key, and value spaces:

# Source: [c1443]
```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

Or equivalently, using named weight matrices:

# Source: [c1373]
```python
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

Each of these projections is a learned linear transformation. The model can learn, for example, that certain dimensions of the embedding are more useful for determining relevance (keys and queries) versus for constructing the output representation (values).

#### Scaling by $\sqrt{d_k}$

Attention scores are computed as queries multiplied by keys transpose: $Q \times K^T$ [c1933]. However, raw dot products can grow large in magnitude when the embedding dimension is high, which pushes the softmax into regions where its gradient is very small. The fix is to scale by the square root of the key embedding dimension.

Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]:

# Source: [c1406]
```python
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```

In a more compact form:

# Source: [c1445]
```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

#### Computing the Final Context Vectors

Context vectors are computed by multiplying attention weights with the values matrix [c1938]. An attention weight between two tokens (e.g., between "journey" and "your") indicates how much the model should attend to the second token when processing the first token as the query [c1465].

# Source: [c1448]
```python
context_vector = attention_weights @ values
```

Each element in an attention score matrix row encodes how much a particular token relates to the query token, or how much importance should be paid to that token when processing the query [c1934]. After multiplying by the values matrix, each row of the result is the context vector for the corresponding query token—a weighted blend of all value vectors, shaped by what the model has learned to find relevant.

[FIGURE: Diagram showing the full scaled dot-product attention computation: input embeddings → three linear projections (W_Q, W_K, W_V) → Q×K^T → scale by sqrt(d_k) → softmax → multiply by V → context vectors]

---

### Causal (Masked) Attention

The self-attention mechanism described so far allows every token to attend to every other token in the sequence—both past and future. This is fine for tasks like sentence classification, where the entire input is available at once. But for language modeling, where the model must predict the next token given only the tokens seen so far, allowing the model to "look ahead" at future tokens would be cheating.

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
mask_simple = torch.tril(torch.ones(context_length, context_length))
```

#### Approach 1: Multiply and Renormalize

One straightforward approach is to multiply the attention weights element-wise by the mask (zeroing out future positions) and then renormalize:

# Source: [c2636]
```python
masked_attn_weights = attn_weights * mask_simple
```

After masking, the rows no longer sum to one, so we renormalize:

# Source: [c2638]
```python
mask_simple_normalized = masked_attn_weights / masked_attn_weights.sum(dim=1, keepdim=True)
```

This approach works but is slightly inefficient because it requires an extra division step after the softmax.

#### Approach 2: Mask Before Softmax (Preferred)

A cleaner approach is to apply the mask *before* the softmax, by setting future positions to $-\infty$. Masking in transformers sets attention scores for future tokens to large negative values, making their influence in softmax calculation effectively zero [c2653]. When softmax exponentiates $-\infty$, it produces exactly zero, so the masked positions contribute nothing to the weighted sum—and the remaining weights automatically sum to one.

We use `torch.triu` (upper triangular) with `diagonal=1` to create a mask of the positions we want to zero out:

# Source: [c2649]
```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

Then we fill those positions with $-\infty$:

# Source: [c2650]
```python
attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))
```

Then apply softmax as usual:

# Source: [c2651]
```python
attention_weights = torch.softmax(attention_scores / sqrt(d_k), dim=-1)
```

The `diagonal=1` argument to `torch.triu` means we keep the main diagonal (a token can attend to itself) and zero out everything strictly above it.

#### Dropout in Attention

In practice, we also apply dropout to the attention weights after the softmax. Dropout is a deep learning technique where neurons in different layers are randomly switched to zero during training [c2654]. This regularization prevents the model from over-relying on any particular attention pattern.

`torch.nn.Dropout(p=0.5)` creates a dropout layer that randomly zeroes elements with probability 0.5 and scales remaining elements by $1/(1-0.5) = 2$ [c2670]. The scaling ensures that the expected value of the output is the same whether dropout is active or not.

#### Putting It Together: The CausalAttention Forward Pass

The input `x` to the causal attention class has shape `(batch_size, num_tokens, input_dimensions)` [c1941]. Here is the core of the forward computation, combining scaled dot-product attention, causal masking, and dropout:

# Source: [c2687]
```python
attn_scores = queries @ keys.transpose(1, 2)
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
context_vec = attn_weights @ values
```

Notice the slice `[:num_tokens, :num_tokens]` applied to the mask. The mask slicing `[:, :T]` ensures the causal mask is created only up to the number of tokens $T$ in the batch, handling cases where batch tokens are smaller than the supported context size [c2696]. This is important because the mask is typically pre-allocated for the maximum context length, but individual batches may be shorter.

[FIGURE: Side-by-side comparison of full self-attention (all tokens attend to all tokens) vs. causal attention (each token attends only to itself and previous tokens), showing the masked upper triangle]

---

### Multi-Head Attention

So far we have built a single attention mechanism. In practice, LLMs use *multi-head attention*, which runs several attention mechanisms in parallel and combines their outputs. The intuition is that different heads can learn to attend to different types of relationships simultaneously—one head might focus on syntactic dependencies, another on semantic similarity, and so on.

#### What Is Multi-Head Attention?

Multi-head attention refers to dividing the attention mechanism into multiple heads, where each head operates independently [c1952]. Multi-head attention is called "multi-head" because it aggregates the output of multiple independent attention heads [c1962]. More concretely, multi-head attention involves creating multiple instances of the self-attention mechanism, each with its own weights, and then combining their outputs [c1964].

[FIGURE: Multi-head attention diagram showing input being processed by H parallel attention heads, each with its own W_Q, W_K, W_V matrices, with outputs concatenated and projected]

#### The Wrapper Approach

The simplest way to implement multi-head attention is to create multiple instances of the causal attention module and run them in parallel. This is sometimes called the "wrapper" approach because it wraps several single-head attention modules.

In the forward method, `torch.cat` is used with `dimension=-1` to concatenate outputs along the columns [c1966]. This concatenation combines the context vectors from all heads into a single, wider vector. $D_{out}$ is the output dimension of each attention head [c1969].

Intuitively, if each head produces a context vector of dimension $D_{out}$, and we have $H$ heads, the concatenated output has dimension $H \times D_{out}$. This concatenated vector is then typically projected back down to the model dimension with a learned linear layer.

#### The Efficient Weight-Split Approach

The wrapper approach is conceptually clean but computationally inefficient: it runs $H$ separate matrix multiplications sequentially (or requires explicit parallelism). A more efficient approach is to perform a single large matrix multiplication and then split the result across heads.

The key insight is that concatenating $H$ separate $W_Q$ matrices of shape $(D_{in}, D_{out})$ is mathematically equivalent to a single $W_Q$ matrix of shape $(D_{in}, H \times D_{out})$, followed by a reshape. This means we can compute all heads' queries, keys, and values in one batched matrix multiply, then reshape and transpose to get the per-head tensors.

[FIGURE: Diagram contrasting the wrapper approach (H separate attention modules) with the weight-split approach (one large projection followed by reshape into H heads)]

The weight-split approach is what is used in production implementations. It achieves the same mathematical result as the wrapper approach but with significantly better hardware utilization, because modern GPUs are highly optimized for large matrix multiplications.

---

### The Mathematics of Scaling by $\sqrt{d_k}$

It is worth pausing to understand *why* we scale by $\sqrt{d_k}$ rather than some other constant.

Consider the dot product of two random vectors $q$ and $k$, each of dimension $d_k$, with components drawn independently from a distribution with mean 0 and variance 1. The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ independent random variables, each with mean 0 and variance 1. By the properties of variance, the sum has variance $d_k$, and therefore standard deviation $\sqrt{d_k}$.

As $d_k$ grows, the dot products grow in magnitude proportionally to $\sqrt{d_k}$. Large dot products push the softmax into its saturated regime, where the gradient is nearly zero and learning slows dramatically. Dividing by $\sqrt{d_k}$ brings the dot products back to unit variance, keeping the softmax in a well-behaved regime regardless of the embedding dimension.

Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]. This is not an arbitrary choice—it is a principled normalization that keeps the attention mechanism numerically stable across a wide range of model sizes.

---

### Putting It All Together

Let us trace through the complete flow of a single forward pass through a multi-head causal attention module, from input to output:

1. **Input**: A batch of token sequences, each token represented as an embedding vector. The input `x` has shape `(batch_size, num_tokens, input_dimensions)` [c1941].

2. **Linear projections**: Each token embedding is projected into query, key, and value spaces using learned weight matrices $W_Q$, $W_K$, $W_V$ [c1350].

3. **Attention scores**: Queries are multiplied by the transpose of keys: $Q \times K^T$ [c1933]. Each element in the resulting matrix encodes how much a particular token relates to the query token [c1934].

4. **Scaling**: Scores are divided by $\sqrt{d_k}$ to prevent saturation [c1403].

5. **Causal masking**: Future positions are filled with $-\infty$ [c2653], so that after softmax they contribute zero weight.

6. **Softmax**: Scores are converted to attention weights that sum to one [c2216].

7. **Dropout**: Attention weights are randomly zeroed during training for regularization [c2654].

8. **Weighted sum**: Attention weights are multiplied by the values matrix to produce context vectors [c1938].

9. **Multi-head combination**: Context vectors from all heads are concatenated [c1966] and projected to the output dimension.

The result is a set of context vectors—one per token—that encode not just the token's own meaning but its meaning *in context*, informed by all the tokens it is allowed to attend to.

[FIGURE: End-to-end flow diagram of multi-head causal attention: input embeddings → Q/K/V projections → scaled dot-product attention with causal mask → dropout → concatenate heads → output context vectors]

---

### Summary

This chapter has built the attention mechanism from first principles:

- **Motivation**: RNNs compress entire sequences into a single context vector, losing information about long-range dependencies [c1517]. The Bahdanau attention mechanism solved this by giving the decoder dynamic access to all encoder states [c1525].

- **Self-attention**: Rather than attending across encoder and decoder, self-attention allows each position in a single sequence to attend to all other positions in that same sequence [c1548].

- **Simplified self-attention**: Without trainable weights, attention scores are raw dot products between embeddings, normalized by softmax to produce attention weights, which are then used to compute weighted sums of the input embeddings [c2158].

- **Scaled dot-product attention**: Adding trainable $W_Q$, $W_K$, $W_V$ matrices [c1350] and scaling by $\sqrt{d_k}$ [c1403] gives the model the ability to learn what to attend to.

- **Causal masking**: Setting future attention scores to $-\infty$ before softmax [c2653] ensures the model cannot look ahead, making it suitable for autoregressive language modeling [c2640].

- **Dropout**: Randomly zeroing attention weights during training [c2654] provides regularization.

- **Multi-head attention**: Running multiple independent attention heads [c1952] and concatenating their outputs [c1966] allows the model to attend to different types of relationships simultaneously.

In the next chapter, we will use this multi-head attention module as a building block to assemble the full transformer architecture, adding layer normalization, feed-forward networks, and residual connections to create a complete GPT-style model.