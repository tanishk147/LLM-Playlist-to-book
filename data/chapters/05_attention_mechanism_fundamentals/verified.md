## The Attention Mechanism: From RNNs to Self-Attention

 Before we can implement it properly, we need to understand *why* it exists—what problems it solves, and what came before it. Along the way, we will build a working implementation of simplified self-attention from scratch in Python, step by step.

---

### The Problem: Sequential Data and Long-Range Dependencies

 The meaning of a word depends on the words around it, sometimes on words that appeared many sentences earlier. 

#### Recurrent Neural Networks

The first serious attempt to handle sequential data with neural networks was the Recurrent Neural Network (RNN) [c1535]. RNNs process a sequence one token at a time, and at each step they maintain a **hidden state**—a vector that accumulates information about everything the network has seen so far [c1502]. The hidden state is the key innovation of RNNs: it is what allows them to, in principle, remember earlier parts of a sequence when processing later parts [c1538].

[FIGURE: Diagram of an RNN unrolled over time, showing input tokens x_1, x_2, ..., x_n feeding into hidden states h_1, h_2, ..., h_n in sequence, with the final hidden state h_n passed to a decoder]

In sequence-to-sequence tasks like machine translation, an RNN encoder reads the entire source sentence and compresses it into a single vector—the **context vector**—which is the final hidden state of the encoder [c1504]. 

This design works reasonably well for short sequences, but it runs into a fundamental wall as sequences grow longer.

#### The Vanishing Gradient and Loss of Context

RNNs struggle with what are called **long-term dependencies**: complex sentence structures where multiple clauses or phrases are connected, making it difficult for the model to identify relationships between distant words [c1481]. 

Long Short-Term Memory networks (LSTMs) were designed to address this by maintaining both a long-term memory route and a short-term memory route [c1541]. LSTMs are a meaningful improvement, but both RNNs and LSTMs still had problems with respect to longer context [c1542]. The fundamental bottleneck remains: the entire source sequence must be compressed into a single fixed-size context vector. When the source sentence is long, that single vector simply cannot carry all the relevant information forward [c1517]. The decoder, relying on only one final hidden state, struggles to capture longer dependencies and contextual information [c1517].

 

---

### Bahdanau Attention: A First Solution

The breakthrough came in 2014, when Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio published "Neural Machine Translation by Jointly Learning to Align and Translate" [c1523]. The **Bahdanau attention mechanism** allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525].

The key insight is **dynamic focus**: the attention mechanism allows the decoder to selectively choose which inputs to focus on and how much attention to give to each input at every decoding step [c1533]. Instead of a single context vector computed once at the end of encoding, the decoder computes a fresh, weighted combination of all encoder hidden states at every generation step [c1544]. When decoding a particular part of the output, the decoder has access to all of the input tokens and decides how much attention to give to each one [c1557].

 Traditional attention looks at one input sequence and one output sequence, determining which parts of the output sequence are more related to which parts of the input sequence [c1558]. 

---

### From Attention to Self-Attention

The Transformer architecture, introduced in 2017, was built with the self-attention mechanism at its core, which was itself inspired by the Bahdanau attention mechanism [c1534]. 

Traditional attention operates *between* two sequences—for example, between a source sentence and a target sentence during translation [c1549]. **Self-attention**, by contrast, looks at one sequence and examines how different parts of that same sequence are related with respect to each other [c1559]. In self-attention, the term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions in a single input sequence [c1553]. Self-attention learns the relationship between various parts of the input itself, whereas traditional attention mechanisms focus on relationships between elements of two different sequences [c1554].

More formally: self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. 

Self-attention is used in the original Transformer architecture, GPT models, and most other popular large language models [c1345]. It is also called **scaled dot-product attention** [c1331, c1344].

---

### Simplified Self-Attention: The Core Idea

We will build up to the full self-attention mechanism in stages. The first stage is a **simplified self-attention** mechanism—the purest and most basic form of the attention technique [c1487]. This simplified version does not include learnable parameters [c2158]. It is a variant of self-attention without trainable weights [c2158]. 

At a high level, self-attention is a mechanism that transforms an input vector (a token embedding) into a **context vector** through attention weights computed from all input vectors [c2155]. The context vector for a given token is an enriched embedding that combines contributions from all input embedding vectors, weighted by their corresponding attention weights [c2236].

This chapter covers the mathematical foundations of attention and implements a simplified version of the attention mechanism from scratch in Python [c2157].

#### Notation

We will use the following notation throughout. The input sequence consists of token embeddings $x_1, x_2, \ldots, x_n$, where $x_i$ denotes the vector representation of the $i$-th token [c2176]. For a concrete example, consider the sentence "Your journey starts here." The word "journey" would be represented as $x_2$.

Vector embeddings capture semantic meaning, such that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181]. 

---

### Step 1: Computing Attention Scores

The first step in self-attention is to compute **attention scores**—intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187].

 A query is the element or token being examined; for example, the word "journey" ($x_2$) is the query when we are computing the context vector for that token [c2183]. The attention score between the query and each other token tells us how much the query should attend to that token.

Attention scores are calculated using the **dot product** between a query vector and each input vector [c2197]. Recall that the dot product of two vectors $a$ and $b$ is:

where $\theta$ is the angle between the vectors [c2193]. In self-attention mechanisms, the dot product determines the extent to which elements of a sequence attend to one another [c2199]. 

In the simplified attention mechanism, attention scores are calculated using the dot product [c1342]. Both the attention scores and the resulting attention weights encode information about how much the query vector and the input embedding vector are related to each other [c1335].

[FIGURE: Diagram showing query vector x_2 ("journey") computing dot products with all input vectors x_1 through x_n, producing a row of raw attention scores omega_21 through omega_2n]

---

### Step 2: Normalizing to Attention Weights

 We want to know, as a fraction, how much attention the query pays to each token. 

Normalization is the process of transforming attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207]. The primary goal of normalization is to obtain attention weights that sum up to one [c2208]. Normalized attention scores are called **attention weights** [c1332]. Attention weights are a normalized version of attention scores, and each row of the attention weight matrix sums to one, making them interpretable as probability distributions over tokens [c1395].

Normalization of attention scores serves two purposes: making attention interpretable and helping backpropagation by keeping scales consistent between 0 and 1 [c1397].

#### Softmax Normalization

 Softmax normalization takes the exponent of every element and divides by the summation of all exponents [c2216]:

Naive softmax attention weights are computed by taking the exponent of each attention score and dividing by the summation of the exponents [c2224]. Softmax attention weights are always positive because they result from the exponent operation [c2226].

In PyTorch, softmax is available as `torch.nn.Softmax` [c2229]:

# Source: [c2229]
```python

```

The `dim` parameter in `torch.softmax` specifies the dimension of the input tensor along which the normalization will be computed [c2265]. 

In the simplified attention mechanism, attention weights are obtained through normalization [c1343].

---

### Step 3: Computing Context Vectors

 

[c2246]

That is, context vectors are computed as a weighted sum over the input vectors [c1333]. The context vector, denoted by $z$, is derived from attention weights and input vectors [c2179]. For a query token like "journey" ($x_2$), the context vector $z_2$ contains information about both the query token itself and all other input elements in the sequence [c2184].

Attention weights determine how much attention to give to each input token when computing the context vector for any embedding vector [c2178]. An attention weight between two tokens—for example, between "journey" and "your"—indicates how much the model should attend to the second token when processing the first token as the query [c1465].

[FIGURE: Diagram showing attention weights alpha_21 through alpha_2n multiplying input vectors x_1 through x_n and summing to produce context vector z_2 for the query "journey"]

---

### Putting It Together: A Complete Walkthrough

Let us now trace through the full simplified self-attention computation for a single query token, then generalize to all tokens simultaneously.

#### The Attention Weight Matrix

When we compute context vectors for every token in the sequence, we produce an **attention weight matrix**: a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

[FIGURE: Heatmap of an attention weight matrix with tokens as both row labels (queries) and column labels (keys); each cell shows the attention weight; rows sum to 1.0]

---

### Implementing Simplified Self-Attention in Python

 The implementation follows directly from the three steps above: compute dot-product attention scores, normalize with softmax, and compute the weighted sum.

#### Computing Keys, Queries, and Values (Simplified)

 The input embeddings themselves serve directly as queries, keys, and values. We compute attention scores as dot products between input vectors, normalize, and sum.

The following code shows how to compute scaled attention scores and weights:

# Source: [c1406]
```python

attention_scores_scaled = attention_scores / (d_k ** 0.5) # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1) # Apply softmax over columns
```

# Source: [c1448]
```python

```

We will revisit the scaling factor $\sqrt{d_k}$ in detail when we introduce trainable weights. For now, note that attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403].

---

### Introducing Trainable Weights

 Using only the dot product to compute attention weights means attention is based solely on semantic similarity, which fails to capture contextual importance when semantically unrelated words are important in context [c2278].

Consider a sentence where a grammatically important function word (like "not") needs to strongly influence the interpretation of a semantically distant content word. 

 Trainable weights in attention allow the model to learn attention patterns beyond semantic similarity, enabling it to assign high attention to contextually important words even when they are not semantically related to the query [c2279]. Trainable weights allow the attention mechanism to capture long-range dependencies by learning contextual patterns beyond the semantic embedding space [c2282].

Self-attention introduces trainable weights which form the basis of the actual mechanism used in LLMs [c1488]. These trainable weight matrices are optimized when the large language model is trained [c1348].

#### The Three Weight Matrices

The self-attention mechanism is implemented using three trainable weight matrices: **Query** ($W_Q$), **Key** ($W_K$), and **Value** ($W_V$) [c1350]. Self-attention requires all three: $W_Q$ (query weight matrix), $W_K$ (key weight matrix), and $W_V$ (value weight matrix) [c1459].

The intuition behind these three roles maps onto a familiar analogy:

- The **Query** is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- The **Key** represents items in the input sequence and is used to match with the query [c1470].
- The **Value** represents the actual content or representation of the input items themselves [c1471].

#### Projecting Inputs Through Weight Matrices

With trainable weight matrices, the computation changes. Instead of using the raw input embeddings as queries, keys, and values, we project the inputs through the learned matrices:

# Source: [c1373]
```python

values = inputs @ W_v
queries = inputs @ W_q
```

Or equivalently, in a class-based implementation:

# Source: [c1443]
```python

queries = x @ W_query
values = x @ W_value
```

# Source: [c1445]
```python

attention_weights = softmax(attention_scores, dim=-1)
```

# Source: [c1448]
```python

```

#### Initializing Weight Matrices in PyTorch

In PyTorch, `torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters, which are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. 

Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368]:

# Source: [c1368]
```python

torch.nn.Parameter(torch.rand(D_in, D_out))
```

#### Organizing Self-Attention as a PyTorch Class

Self-attention is organized as a Python class (`SelfAttentionVersion1`) derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439]. 

---

### Why Scaling by $\sqrt{d_k}$?

You may have noticed that in both the simplified and trainable versions, we divide the attention scores by the square root of the key embedding dimension before applying softmax [c1403, c1406]. This is the "scaled" part of "scaled dot-product attention" [c1331].

 

What we do know is that attention weights sum to one after softmax normalization [c1422], and the scaling helps maintain this property in a numerically stable way across different embedding dimensions.

---

### The Road Ahead

 To summarize the conceptual journey:

1. **RNNs** process sequences with a hidden state but struggle with long-range dependencies [c1535, c1502].
2. **LSTMs** improve on RNNs with dual memory routes but still bottleneck at a single context vector [c1541, c1542].
3. **Bahdanau attention** (2014) allows the decoder to attend to all encoder states at each step, introducing dynamic focus [c1523, c1525, c1533].
4. **Self-attention** generalizes this to a single sequence, allowing every position to attend to every other position [c1548, c1553].
5. **Simplified self-attention** (no trainable weights) demonstrates the core dot-product-normalize-aggregate pattern [c1487, c2158].
6. **Full self-attention** (with $W_Q$, $W_K$, $W_V$) adds learnable projections that allow the model to go beyond raw semantic similarity [c1350, c2279].

The lecture series will go on to cover causal attention and multi-head attention, with mathematical formulations and code implementations in Python [c1560]. **Causal attention** modifies the self-attention mechanism to prevent the model from accessing future information in the sequence [c1474]—a critical property for autoregressive language models like GPT, which must predict the next token without "cheating" by looking ahead.

In the next chapter, we will implement the full self-attention mechanism with trainable weights, add the causal mask, and then stack multiple attention heads to build multi-head attention—the complete building block of the Transformer.
