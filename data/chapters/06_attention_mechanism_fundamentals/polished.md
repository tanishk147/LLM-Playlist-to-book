## Attention Mechanisms: From Simplified to Scaled Dot-Product

Understanding how language models weigh the importance of every word in a sentence is the key to understanding transformers. This chapter builds that understanding from the ground up.

We start with the simplest possible version of attention—no trainable weights, just dot products and normalization—and then layer in the machinery that makes real LLMs work: trainable query, key, and value matrices, scaling by the square root of the key dimension, and a clean class-based implementation.

---

### The Goal: Turning Embeddings into Context Vectors

Every token in your input sequence starts life as an embedding vector—a fixed-size array of floats that encodes something about the token's meaning [c2155]. On its own, that embedding is context-free: the word "bank" looks the same whether it appears next to "river" or "loan." Attention fixes this.

The goal of the attention mechanism—whether simplified or the full multi-head version used in GPT—is to convert those context-free embedding vectors into *context vectors* [c2175]. A context vector for a given token is an enriched representation that combines information from all other tokens in the sequence, weighted by how relevant each one is [c2236]. Intuitively, the model learns to "look around" the sequence and blend in the most useful information before making any prediction.

Formally, the context vector $z_i$ is defined as [c2246]:

$$z_i = \sum_j \text{attention\_weight}_{ij} \cdot x_j$$

In the notation used throughout this chapter, $x_i$ denotes the vector representation of the $i$-th token, where $x_1$ is the first token, $x_2$ is the second token, and so on [c2176]. The context vector for token $i$ is written $z_i$ [c2179].

---

### Simplified Self-Attention: No Trainable Weights

We begin with a deliberately stripped-down version of attention that has no learnable parameters [c2158]. This lets us focus on the core mathematical ideas before adding the complexity of learned projections.

#### Attention Scores via Dot Product

Attention scores are intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187].

Pick a token to focus on—call it the *query*. In the running example used throughout this chapter, the word "journey" ($x_2$) plays the role of the query [c2183]. To compute its context vector, we need to measure how relevant every other token in the sequence is to it.

The mathematical operation that measures this relevance is the **dot product** [c2192]. Attention scores are calculated using the dot product between a query vector and each input vector [c2197]. The dot product quantifies how much two vectors are aligned; a higher dot product means higher alignment [c2198], and in self-attention mechanisms, this alignment determines the extent to which elements of a sequence attend to one another [c2199]. Higher dot product values therefore correspond to higher similarity and higher attention scores between two elements [c2200].

Why does the dot product capture semantic relevance? Vector embeddings are trained so that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181]. When two vectors are aligned (small angle between them), their dot product is maximum, indicating high attention [c1382]. Geometrically, the dot product of two vectors $a$ and $b$ equals $a \cdot b = \sum_i a_i b_i = |a||b|\cos\theta$, where $\theta$ is the angle between them [c2193].

In the simplified mechanism, attention scores are calculated using the dot product directly [c1342], and attention weights are then obtained through normalization [c1343].

#### Normalization: From Scores to Weights

Raw dot-product scores can take any value, making them hard to interpret and compare. We need to normalize them.

Normalization is the process of transforming attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207]. The primary goal of normalization is to obtain attention weights that sum up to one [c2208], at which point they are interpretable as probability distributions over tokens [c1395].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. Concretely, an attention weight between two tokens—say, between "journey" and "your"—indicates how much the model should attend to the second token when processing the first token as the query [c1465].

The standard normalization function is **softmax**, which takes the exponent of every element and divides by the sum of all exponents [c2216]:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Naive softmax attention weights are thus computed by taking the exponent of each attention score and dividing by the summation of the exponents [c2224]. In PyTorch, `torch.nn.Softmax(dim=None)` applies this function to rescale input tensor elements to the range $[0, 1]$ with a sum of 1 [c2229]. The `dim` parameter specifies the dimension of the input tensor along which the normalization will be computed [c2265].

#### Computing the Context Vector

Once we have attention weights, the context vector is straightforward: multiply each input embedding vector by its corresponding attention weight and sum all the scaled vectors [c2237].

To make this concrete, consider computing the context vector for "journey." Each value vector is scaled by its corresponding attention weight: the "your" vector is multiplied by 0.15, the "journey" vector by 0.22, the "begins" vector by 0.2199, the "width" vector by 0.13, the "one" vector by 0.09, and the "step" vector by 0.18 [c1429]. The resulting context vector $z_2$ contains information about both the query token itself and all other input elements in the sequence [c2184].

#### The Attention Weight Matrix

When we compute context vectors for *all* tokens simultaneously, the attention weights form a matrix. An attention weight matrix is a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

When computing the matrix product of attention weights and inputs, each element in the output row is the dot product of the corresponding attention weight row with each column of the input matrix [c2274]. This matrix multiplication scales each input vector by its corresponding attention weight and sums them—exactly equivalent to the manual context vector calculation described above [c2275]. The payoff is that we can compute all context vectors at once with a single matrix multiply, a key efficiency advantage.

---

### Introducing Trainable Weights: The Full Self-Attention Mechanism

The simplified mechanism has a fundamental limitation: using only the dot product to compute attention weights means attention is based solely on semantic similarity, which fails to capture contextual importance when semantically unrelated words are important in context [c2278].

The solution is to introduce **trainable weight matrices** that allow the model to learn attention patterns beyond semantic similarity, enabling it to assign high attention to contextually important words even when they are not semantically related to the query [c2279]. These learned weights also allow the attention mechanism to capture long-range dependencies by learning contextual patterns beyond the semantic embedding space [c2282].

#### Query, Key, and Value Matrices

The self-attention mechanism is implemented using three trainable weight matrices: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) [c1350]. All three are required [c1459], and all are optimized when the large language model is trained [c1348].

The three matrices play distinct conceptual roles:

- **Query** ($W_Q$): Analogous to a search query in a database, it represents the current token the model is focusing on [c1469].
- **Key** ($W_K$): Represents items in the input sequence and is used to match with the query [c1470].
- **Value** ($W_V$): Represents the actual content or representation of the input items themselves [c1471].

The first step of the attention mechanism is converting input embeddings into key, query, and value vectors through linear transformations [c1376]:

```python
# Source: [c1373]
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

Or equivalently, in a class-based implementation [c1443]:

```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

[FIGURE: Diagram showing input embeddings x1…x6 being projected through three separate weight matrices W_Q, W_K, W_V to produce query vectors q1…q6, key vectors k1…k6, and value vectors v1…v6]

#### Why Three Matrices Instead of One?

Separating the input into three different projections gives the model substantially more flexibility. The query and key projections can learn to measure relevance in a space that is different from the raw embedding space, while the value projection can learn to extract the most useful information to pass forward. This decoupling is what allows the model to attend to contextually important tokens even when they are not semantically similar to the query.

#### Initializing the Weight Matrices in PyTorch

`torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters; they are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368]:

```python
# Source: [c1368]
self.W_query = nn.Parameter(torch.rand(D_in, D_out))
self.W_key   = nn.Parameter(torch.rand(D_in, D_out))
self.W_value = nn.Parameter(torch.rand(D_in, D_out))
# with random values and shape (D_in, D_out)
```

An alternative—and more common—approach is to use `nn.Linear` layers with `bias=False` instead of raw `nn.Parameter` tensors [c1452], and this is the preferred practice when building LLMs [c1457].

---

### Scaled Dot-Product Attention

With query, key, and value vectors in hand, the attention score computation looks almost the same as in the simplified version—but with one critical addition: **scaling**.

#### The Scaling Problem

When we introduce learned projections, dot products can become very large in magnitude, especially as the embedding dimension grows. The dot product of two random vectors increases in variance; without scaling, that variance grows proportionally with the dimension of the query and key vectors [c1414]. For a $d$-dimensional query and key vector sampled from a normal distribution, the variance of their dot product before scaling is approximately equal to $d$ [c1415].

High variance is problematic because it causes the softmax distribution to become very sharp, making the model overly confident in a single key and destabilizing learning [c1413]. The result is slow, unstable training.

#### The Fix: Divide by $\sqrt{d_k}$

The solution is simple: divide the dot product by the square root of the key embedding dimension [c1410]. This division keeps the variance of the attention scores close to 1, regardless of the embedding dimension [c1416], ensuring that values in the attention score matrix remain small and that training stays stable [c1446].

Attention weights are therefore computed by scaling attention scores by $\sqrt{d_k}$ and then applying softmax [c1403]. The scaling factor is specifically the square root of the embedding dimension of the keys [c1404], and attention scores are divided by exactly this quantity [c1420].

This scaling step is one of the reasons the mechanism is called **scaled dot-product attention** [c1400]—a name used interchangeably with "self-attention" [c1331, c1344].

#### Computing Scaled Attention Weights

```python
# Source: [c1406]
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```

Or in the class-based form [c1445]:

```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

The attention weights matrix for all queries is a 6×6 matrix computed by applying the scaling and softmax operation to the full 6×6 attention scores matrix [c1409]. For a query matrix of shape 6×2, computing attention scores for the second query ("journey") involves taking the dot product of that query row with all key rows to produce 6 attention scores [c1385].

#### Computing the Final Context Vectors

With attention weights in hand, the context vectors are obtained by a single matrix multiply [c1448]:

```python
context_vector = attention_weights @ values
```

The context vector for a specific token can also be computed by taking that token's row from the attention weights matrix (1×6) and multiplying it by the values matrix (6×2) to produce a 1×2 vector [c1435].

[FIGURE: End-to-end diagram of scaled dot-product attention: inputs → W_Q/W_K/W_V projections → dot product → divide by sqrt(d_k) → softmax → weighted sum over values → context vectors]

---

### Why Attention Weights Are Probability Distributions

After softmax, each row of the attention weight matrix sums to one [c1336]. This means we can read each row as a probability distribution over the input tokens: "given that I am processing token $i$, what fraction of my attention goes to token $j$?" [c1395]. This interpretability is not just a nice property—it is by design.

Normalization of attention scores serves two purposes: making attention interpretable and helping backpropagation by keeping scales consistent between 0 and 1 [c1397]. Both benefits compound as the model grows deeper.

---

### Putting It All Together: The SelfAttention Class

We can now assemble everything into a reusable PyTorch module. Self-attention is organized as a Python class derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439].

The complete forward pass of scaled dot-product attention proceeds in three steps: (1) project inputs into query, key, and value vectors; (2) compute scaled attention weights; and (3) compute context vectors as a weighted sum of value vectors.

**Step 1 — Project inputs** [c1443]:

```python
keys    = x @ W_key
queries = x @ W_query
values  = x @ W_value
```

**Step 2 — Compute scaled attention weights** [c1445]:

```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

**Step 3 — Compute context vectors** [c1448]:

```python
context_vector = attention_weights @ values
```

The self-attention mechanism used in the original transformer architecture, GPT models, and most other popular large language models follows exactly this pattern [c1345].

#### A Note on `nn.Linear` vs. `nn.Parameter`

As mentioned earlier, the preferred implementation uses `nn.Linear` with `bias=False` rather than raw `nn.Parameter` tensors [c1452, c1457]. The two approaches are mathematically equivalent, but `nn.Linear` integrates more cleanly with PyTorch's module ecosystem and is the convention in production LLM code.

---

### Mathematical Justification for the $\sqrt{d_k}$ Scaling

If the entries of the query and key vectors are drawn independently from a standard normal distribution, each term $q_i k_i$ has mean 0 and variance 1. The dot product is the sum of $d_k$ such terms, so its variance is $d_k$ [c1415] and its standard deviation grows as $\sqrt{d_k}$.

Dividing by $\sqrt{d_k}$ rescales the dot product to have variance approximately 1, regardless of the embedding dimension [c1416], keeping the variance of the dot product between keys and queries close to one after division [c1446]. Without this correction, large dot products would push the softmax into saturation—producing near-zero gradients for all but the highest-scoring key—and scaling prevents this collapse.

---

### The Query/Key/Value Abstraction: A Database Analogy

Think of the input sequence as a soft database. Each token contributes a **key** (an index describing what it contains) and a **value** (the actual content). When processing a token, the model issues a **query** and retrieves a weighted blend of all values, where the weights are determined by how well each key matches the query.

- **Query**: the current token the model is focusing on, analogous to a search query in a database [c1469].
- **Key**: represents items in the input sequence, used to match with the query [c1470].
- **Value**: represents the actual content or representation of the input items themselves [c1471].

The dot product between a query and a key measures how well they match; softmax converts these match scores into a probability distribution; and the weighted sum over values produces the context vector. Crucially, because the three roles are handled by separate learned projections, the model can learn to ask different questions (queries) than it answers (values), and to index its memory differently (keys) than it stores it (values). This flexibility is what makes the mechanism so powerful.

---

### Limitations of the Current Design and What Comes Next

The self-attention mechanism described here attends to all tokens in the sequence equally—including future tokens. For autoregressive language models, this is problematic: during generation, the model should not be able to "see" tokens that have not yet been produced.

Causal attention modifies the self-attention mechanism to prevent the model from accessing future information in the sequence [c1474]. We will cover causal (masked) attention in the next chapter. Beyond masking, the other major extension is **multi-head attention**, which runs several attention mechanisms in parallel and concatenates their outputs—but both simplified attention and full multi-head attention share the same fundamental goal of converting embedding vectors into context vectors [c2175].

---

### Summary

This chapter built the attention mechanism from the ground up, starting from a parameter-free dot-product baseline and arriving at the full scaled dot-product attention used in modern LLMs.

| Concept | Definition |
|---|---|
| Context vector | Weighted sum of input embeddings, weights given by attention [c2246] |
| Attention score | Dot product between query and key vectors [c2197] |
| Attention weight | Normalized attention score; each row sums to 1 [c1395] |
| Softmax | $e^{x_i} / \sum_j e^{x_j}$; converts scores to weights [c2216] |
| Scaling | Divide by $\sqrt{d_k}$ to keep variance ≈ 1 [c1416] |
| $W_Q, W_K, W_V$ | Trainable projections learned during LLM training [c1350, c1348] |

The mechanism earns its full name—**scaled dot-product attention**—from two of these steps: the dot product that computes raw scores, and the $\sqrt{d_k}$ scaling that stabilizes training [c1400, c1331].