## Attention Mechanisms: From Simplified to Scaled Dot-Product

The attention mechanism is the engine at the heart of every modern large language model. Before you can build one, you need to understand what attention actually computes, why it is designed the way it is, and how to implement it cleanly in PyTorch. This chapter builds that understanding from the ground up.

We start with the simplest possible version of attention—no trainable weights, just dot products and normalization—and then layer in the machinery that makes real LLMs work: trainable query, key, and value matrices, scaling by the square root of the key dimension, and a clean class-based implementation. By the end of the chapter you will have a working `SelfAttention` class that you can drop into a transformer architecture.

---

### The Goal: Turning Embeddings into Context Vectors

Before writing a single line of code, it is worth being precise about what attention is supposed to accomplish.

Every token in your input sequence starts life as an embedding vector—a fixed-size array of floats that encodes something about the token's meaning [c2155]. The problem with raw embeddings is that they are *context-free*: the embedding for the word "bank" is the same whether the surrounding words are about rivers or about finance. Attention fixes this.

The goal of the attention mechanism—whether simplified or the full multi-head version used in GPT—is to convert those context-free embedding vectors into *context vectors* [c2175]. A context vector for a given token is an enriched representation that combines information from all other tokens in the sequence, weighted by how relevant each one is [c2236]. Intuitively, the model learns to "look around" the sequence and blend in the most useful information before making any prediction.

Formally, the context vector $z_i$ is defined as [c2246]:

$$Z_i = \sum_{j} \text{attention\_weight}_j \cdot \text{embedding\_vector}_j$$

In the notation used throughout this chapter, $x_i$ denotes the vector representation of the $i$-th token, where $x_1$ is the first token, $x_2$ is the second token, and so on [c2176]. The context vector for token $i$ is written $z_i$ [c2179].

[FIGURE: Diagram showing a sequence of input token embeddings x1…x6 on the left, arrows weighted by attention weights pointing into a summation box, and a single context vector z2 emerging on the right]

---

### Simplified Self-Attention: No Trainable Weights

We begin with a deliberately stripped-down version of attention that has no learnable parameters [c2158]. This variant will not be used in a real LLM, but it exposes the core computation without the extra machinery of weight matrices. Once you understand this version, adding trainable weights is a natural extension.

#### Attention Scores via Dot Product

The first quantity we need is an *attention score*. Attention scores are intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187].

Pick a token to focus on—call it the *query*. In the running example used throughout this chapter, the word "journey" ($x_2$) plays the role of the query [c2183]. We want to compute, for every other token in the sequence, how relevant that token is to "journey."

The mathematical operation that measures this relevance is the **dot product** [c2192]. Attention scores are calculated using the dot product between a query vector and each input vector [c2197]. The dot product quantifies how much two vectors are aligned; a higher dot product means higher alignment [c2198]. In self-attention mechanisms, the dot product determines the extent to which elements of a sequence attend to one another [c2199]. Higher dot product values correspond to higher similarity and higher attention scores between two elements [c2200].

Why does the dot product capture semantic relevance? Vector embeddings are trained so that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181]. When two vectors are aligned (small angle between them), their dot product is maximum, indicating high attention [c1382]. Geometrically, the dot product of two vectors $a$ and $b$ equals $a \cdot b = \sum_i a_i b_i = |a||b|\cos\theta$, where $\theta$ is the angle between them [c2193]. So the dot product is large when vectors point in similar directions and small (or negative) when they point in different directions.

In the simplified mechanism, attention scores are calculated using the dot product directly [c1342], and attention weights are then obtained through normalization [c1343].

#### Normalization: From Scores to Weights

Raw dot-product scores are not yet useful as weights—they can be any real number, positive or negative, and they do not tell you how to *allocate* attention across the sequence. We need to normalize them.

Normalization is the process of transforming attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207]. The primary goal of normalization is to obtain attention weights that sum up to one [c2208]. Once normalized, attention weights are interpretable as probability distributions over tokens [c1395].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. Concretely, an attention weight between two tokens—say, between "journey" and "your"—indicates how much the model should attend to the second token when processing the first token as the query [c1465].

The standard normalization function is **softmax**. Softmax normalization takes the exponent of every element and divides by the summation of all exponents [c2216]:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Naive softmax attention weights are computed by taking the exponent of each attention score and dividing by the summation of the exponents [c2224]. In PyTorch, this is available as:

# Source: [c2229]
```python
torch.nn.Softmax(dim=None)  # applies Softmax to rescale elements to [0,1] summing to 1
```

The `dim` parameter in `torch.softmax` specifies the dimension of the input tensor along which the normalization will be computed [c2265]. When working with a 2-D attention score matrix, you normalize along `dim=-1` (the last axis), so that each *row* of scores sums to one—each row corresponds to one query token's attention distribution over all key tokens.

#### Computing the Context Vector

With attention weights in hand, computing the context vector is straightforward. The context vector for a query token is computed by multiplying each input embedding vector by its corresponding attention weight and then summing all the scaled vectors [c2237]. This is exactly the weighted sum formula introduced earlier.

To make this concrete, consider computing the context vector for "journey." Each value vector is scaled by its corresponding attention weight: the "your" vector is multiplied by 0.15, the "journey" vector by 0.22, the "begins" vector by 0.2199, the "width" vector by 0.13, the "one" vector by 0.09, and the "step" vector by 0.18 [c1429]. Summing these scaled vectors produces $z_2$, the context vector for "journey."

A context vector $z_2$ for a query contains information about both the query token itself and all other input elements in the sequence [c2184]. This is the enrichment that raw embeddings lack.

#### The Attention Weight Matrix

When we compute context vectors for *all* tokens simultaneously, the attention weights form a matrix. An attention weight matrix is a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

[FIGURE: 6×6 attention weight matrix heatmap, rows labeled with query tokens (your, journey, begins, with, one, step), columns labeled with key tokens, cell values representing attention weights, each row summing to 1.0]

When computing the matrix product of attention weights and inputs, each element in the output row is the dot product of the corresponding attention weight row with each column of the input matrix [c2274]. The matrix multiplication operation scales each input vector by its corresponding attention weight and sums them, which is equivalent to the manual context vector calculation process [c2275]. This means we can compute all context vectors at once with a single matrix multiply—a key efficiency advantage.

---

### Introducing Trainable Weights: The Full Self-Attention Mechanism

The simplified mechanism has a fundamental limitation. Using only the dot product to compute attention weights means attention is based solely on semantic similarity, which fails to capture contextual importance when semantically unrelated words are important in context [c2278]. For example, a preposition might be critical for understanding a sentence even though it is semantically distant from the noun being processed.

The solution is to introduce **trainable weight matrices** that allow the model to learn attention patterns beyond semantic similarity, enabling it to assign high attention to contextually important words even when they are not semantically related to the query [c2279]. Trainable weights allow the attention mechanism to capture long-range dependencies by learning contextual patterns beyond semantic embedding space [c2282].

#### Query, Key, and Value Matrices

The self-attention mechanism is implemented using three trainable weight matrices: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) [c1350]. Self-attention requires all three: $W_Q$ (query weight matrix), $W_K$ (key weight matrix), and $W_V$ (value weight matrix) [c1459]. These matrices are optimized when the large language model is trained [c1348].

The three matrices play distinct conceptual roles:

- **Query** ($W_Q$): The query is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- **Key** ($W_K$): The key represents items in the input sequence and is used to match with the query [c1470].
- **Value** ($W_V$): The value represents the actual content or representation of the input items themselves [c1471].

The first step of the attention mechanism is converting input embeddings into key, query, and value vectors through linear transformations [c1376]. Concretely:

# Source: [c1373]
```python
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

Or equivalently, in a class-based implementation:

# Source: [c1443]
```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

Each of these is a simple matrix multiplication. The weight matrices project the raw embeddings into a new space where the dot product between a query and a key is more meaningful than the dot product between the raw embeddings themselves.

[FIGURE: Diagram showing input embeddings x1…x6 being projected through three separate weight matrices W_Q, W_K, W_V to produce query vectors q1…q6, key vectors k1…k6, and value vectors v1…v6]

#### Why Three Matrices Instead of One?

Intuitively, separating the input into three different projections gives the model more flexibility. The query projection learns "what am I looking for?", the key projection learns "what do I offer to other queries?", and the value projection learns "what information do I actually contribute to the output?" These three roles are distinct, and using a single shared matrix would conflate them.

#### Initializing the Weight Matrices in PyTorch

`torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters, which are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368]:

# Source: [c1368]
```python
# Query, Key, and Value weight matrices initialized as nn.Parameter
# with random values and shape (D_in, D_out)
```

[GAP: Verbatim code block for the __init__ method showing W_q, W_k, W_v initialization as nn.Parameter with torch.rand]

An alternative—and more common—approach is to use `nn.Linear` layers instead of `nn.Parameter` to initialize query, key, and value weight matrices, with bias set to false [c1452]. Using `nn.Linear` for self-attention implementation is more common practice when dealing with LLMs [c1457]. The `nn.Linear` approach is preferred because it integrates cleanly with PyTorch's weight initialization conventions and makes it easy to swap in different initialization schemes.

---

### Scaled Dot-Product Attention

With query, key, and value vectors in hand, the attention score computation looks almost the same as in the simplified version—but with one critical addition: **scaling**.

#### The Scaling Problem

Recall that in the simplified mechanism, attention scores were just raw dot products. When we introduce learned projections, the dot products can become very large in magnitude, especially as the embedding dimension grows. This creates a problem.

The dot product of two random vectors increases the variance; without scaling, the variance of the dot product scales proportionally with the dimension of the query and key vectors [c1414]. For a $d$-dimensional query and key vector sampled from a normal distribution, the variance of their dot product before scaling is approximately equal to $d$ [c1415].

Why does high variance matter? Scaling attention scores by the square root of dimension prevents the softmax distribution from becoming too sharp, which would make the model overly confident in a single key and destabilize learning [c1413]. When the softmax input has very large values, the function saturates—one entry gets nearly all the probability mass and the gradients through the other entries vanish. This makes training slow and unstable.

#### The Fix: Divide by $\sqrt{d_k}$

The solution is simple: divide the dot product by the square root of the key embedding dimension [c1410]. Dividing the dot product by the square root of dimension keeps the variance of the attention scores close to 1, regardless of the dimension of the query and key vectors [c1416]. Dividing attention scores by the square root of the key embedding dimension ensures that values in the attention score matrix remain small and that the variance of the dot product between keys and queries stays close to one [c1446].

Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]. The scaling factor is the square root of the embedding dimension of the keys [c1404]. Attention scores are divided by the square root of the key dimension [c1420].

This scaling is one of the reasons the mechanism is called **scaled dot-product attention** [c1400]. The full name—scaled dot-product attention—is also used interchangeably with "self-attention" [c1331, c1344].

#### Computing Scaled Attention Weights

Here is the scaling and softmax step in code:

# Source: [c1406]
```python
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```

Or in the class-based form:

# Source: [c1445]
```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

The attention weights matrix for all queries is a 6×6 matrix computed by applying the scaling and softmax operation to the full 6×6 attention scores matrix [c1409]. For a query matrix of shape 6 by 2, computing attention scores for the second query ("journey") involves taking the dot product of that query row with all key rows to produce 6 attention scores [c1385].

#### Computing the Final Context Vectors

Once we have attention weights, the context vector computation is the same as in the simplified case—a weighted sum over the *value* vectors (not the raw input embeddings):

# Source: [c1448]
```python
context_vector = attention_weights @ values
```

The context vector for a specific token can be computed by taking that token's row from the attention weights matrix (1×6) and multiplying it by the values matrix (6×2) to produce a 1×2 vector [c1435].

[FIGURE: End-to-end diagram of scaled dot-product attention: inputs → W_Q/W_K/W_V projections → dot product → divide by sqrt(d_k) → softmax → weighted sum over values → context vectors]

---

### Why Attention Weights Are Probability Distributions

It is worth pausing to appreciate what the softmax normalization achieves. After softmax, each row of the attention weight matrix sums to one [c1336]. This means we can read each row as a probability distribution over the input tokens: "given that I am processing token $i$, what fraction of my attention goes to token $j$?" [c1395].

This interpretability is not just aesthetic. Normalization of attention scores serves two purposes: making attention interpretable and helping backpropagation by keeping scales consistent between 0 and 1 [c1397]. Consistent scales mean that gradients flowing back through the softmax are well-behaved, which is important for stable training.

---

### Putting It All Together: The SelfAttention Class

Now we can assemble everything into a reusable PyTorch module. Self-attention is organized as a Python class derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439].

The complete forward pass of scaled dot-product attention proceeds as follows:

1. Project inputs into queries, keys, and values using the three weight matrices.
2. Compute raw attention scores as the dot product of queries and keys.
3. Scale the scores by $1/\sqrt{d_k}$.
4. Apply softmax to obtain attention weights.
5. Compute context vectors as a weighted sum of value vectors.

Let us trace through each step with explicit code.

**Step 1 – Linear projections:**

# Source: [c1443]
```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

**Step 2 & 3 – Scaled attention scores:**

# Source: [c1445]
```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

**Step 4 – Context vectors:**

# Source: [c1448]
```python
context_vector = attention_weights @ values
```

The self-attention mechanism used in the original transformer architecture, GPT models, and most other popular large language models follows exactly this pattern [c1345].

#### A Note on `nn.Linear` vs. `nn.Parameter`

As mentioned earlier, the preferred implementation uses `nn.Linear` with `bias=False` rather than raw `nn.Parameter` tensors [c1452, c1457]. The two are mathematically equivalent—`nn.Linear(d_in, d_out, bias=False)` applies the transformation $xW^T$, which is the same as `x @ W.T` for a parameter matrix $W$ of shape `(d_out, d_in)`. The `nn.Linear` version is preferred in practice because PyTorch's default initialization (Kaiming uniform) is better suited to deep networks than the uniform random initialization you would get from `torch.rand`.

[GAP: Complete verbatim code listing for SelfAttentionVersion1 class (nn.Module subclass) showing __init__ with nn.Parameter initialization and forward method]

[GAP: Complete verbatim code listing for SelfAttentionVersion2 class using nn.Linear instead of nn.Parameter]

---

### Mathematical Justification for the $\sqrt{d_k}$ Scaling

This section gives a more careful look at why dividing by $\sqrt{d_k}$ is the right choice—not just a heuristic.

Suppose the query vector $q$ and key vector $k$ each have $d_k$ components, and each component is drawn independently from a standard normal distribution with mean 0 and variance 1. The dot product is:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Each term $q_i k_i$ has mean 0 and variance 1 (since $\text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) = 1$ for independent zero-mean variables). The sum of $d_k$ such terms has variance $d_k$ [c1415]. So the standard deviation of the dot product grows as $\sqrt{d_k}$.

Dividing by $\sqrt{d_k}$ rescales the dot product to have variance approximately 1, regardless of the embedding dimension [c1416]. This keeps the softmax inputs in a regime where the function is neither saturated (all probability on one token) nor completely flat (uniform attention over all tokens). The variance of the dot product between keys and queries stays close to one after this division [c1446].

To see why saturation is harmful: if one attention score is much larger than the others, softmax assigns nearly all weight to that one token. The gradients for all other tokens are essentially zero, and the model cannot learn from them. Scaling prevents this collapse.

---

### The Query/Key/Value Abstraction: A Database Analogy

The query/key/value terminology is borrowed from information retrieval, and the analogy is instructive.

Think of the input sequence as a soft database. Each token contributes a **key** (an index that describes what it contains) and a **value** (the actual content). When processing a particular token, you form a **query** (what you are looking for) and compare it against all keys to determine how much of each value to retrieve.

- **Query**: the current token the model is focusing on, analogous to a search query in a database [c1469].
- **Key**: represents items in the input sequence, used to match with the query [c1470].
- **Value**: represents the actual content or representation of the input items themselves [c1471].

The dot product between a query and a key measures how well they match. Softmax converts these match scores into a probability distribution. The context vector is then the expected value under that distribution—a weighted blend of all the values in the database.

The crucial insight is that the query, key, and value projections are *learned separately*. This means the model can learn to ask different questions (queries) than it answers (values), and to index its memory differently (keys) than it stores it (values). This flexibility is what makes the mechanism so powerful.

---

### Limitations of the Current Design and What Comes Next

The self-attention class we have built in this chapter is a solid foundation, but it has one important limitation for language modeling: it can see the entire sequence, including tokens that come *after* the current position. In a language model that generates text left-to-right, this is a form of cheating—the model should not be able to use future tokens to predict the current one.

Causal attention modifies the self-attention mechanism to prevent the model from accessing future information in the sequence [c1474]. This is achieved by masking out the upper triangle of the attention weight matrix before applying softmax, so that each token can only attend to itself and earlier tokens. We will cover causal (masked) attention in the next chapter.

Additionally, the single-head attention we have built here is typically extended to **multi-head attention**, where multiple attention heads run in parallel, each learning to attend to different aspects of the input. Both simplified attention mechanisms and complex multi-head attention used in modern large language models share the same fundamental goal: converting embedding vectors into context vectors [c2175].

---

### Summary

This chapter built the attention mechanism from the ground up. Here is a concise recap of the key ideas:

| Concept | Definition |
|---|---|
| Context vector | Weighted sum of input embeddings, weights given by attention [c2246] |
| Attention score | Dot product between query and key vectors [c2197] |
| Attention weight | Normalized attention score; each row sums to 1 [c1395] |
| Softmax | $e^{x_i} / \sum_j e^{x_j}$; converts scores to weights [c2216] |
| Scaling | Divide by $\sqrt{d_k}$ to keep variance ≈ 1 [c1416] |
| $W_Q, W_K, W_V$ | Trainable projections learned during LLM training [c1350, c1348] |

The mechanism earns its full name—**scaled dot-product attention**—from two of these steps: the dot product that computes raw scores, and the $\sqrt{d_k}$ scaling that stabilizes training [c1400, c1331].

With this foundation in place, you are ready to tackle causal masking, multi-head attention, and eventually the full transformer block. Each of those builds directly on what you have learned here.