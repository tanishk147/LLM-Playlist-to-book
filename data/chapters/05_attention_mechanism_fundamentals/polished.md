## The Attention Mechanism: From RNNs to Self-Attention

To understand self-attention, we need to understand *why* it exists—what problems it solves and what came before it. Along the way, we will build a working implementation of simplified self-attention from scratch in Python, step by step.

---

### The Problem: Sequential Data and Long-Range Dependencies

The meaning of a word depends on the words around it—sometimes on words that appeared many sentences earlier.

#### Recurrent Neural Networks

The first serious attempt to handle sequential data with neural networks was the Recurrent Neural Network (RNN) [c1535]. RNNs process a sequence one token at a time, maintaining at each step a **hidden state**—a vector that accumulates information about everything the network has seen so far [c1502]. This hidden state is the key innovation of RNNs: it is what allows them, in principle, to remember earlier parts of a sequence when processing later parts [c1538].

[FIGURE: Diagram of an RNN unrolled over time, showing input tokens x_1, x_2, ..., x_n feeding into hidden states h_1, h_2, ..., h_n in sequence, with the final hidden state h_n passed to a decoder]

In sequence-to-sequence tasks like machine translation, an RNN encoder reads the entire source sentence and compresses it into a single vector—the **context vector**—which is the final hidden state of the encoder [c1504]. This design works reasonably well for short sequences, but it runs into a fundamental wall as sequences grow longer.

#### The Vanishing Gradient and Loss of Context

RNNs struggle with **long-term dependencies**: complex sentence structures where multiple clauses or phrases are connected, making it difficult for the model to identify relationships between distant words [c1481].

Long Short-Term Memory networks (LSTMs) were designed to address this by maintaining both a long-term memory route and a short-term memory route [c1541]. LSTMs are a meaningful improvement, but both RNNs and LSTMs still had problems with longer context [c1542]. The fundamental bottleneck remains: the entire source sequence must be compressed into a single fixed-size context vector. When the source sentence is long, that single vector simply cannot carry all the relevant information forward. The decoder, relying on only one final hidden state, struggles to capture longer dependencies and contextual information [c1517].

---

### Bahdanau Attention: A First Solution

The breakthrough came in 2014, when Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio published "Neural Machine Translation by Jointly Learning to Align and Translate" [c1523]. Their **Bahdanau attention mechanism** allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525].

The key insight is **dynamic focus**: instead of a single context vector computed once at the end of encoding, the decoder computes a fresh, weighted combination of all encoder hidden states at every generation step [c1544]. The attention mechanism allows the decoder to selectively choose which inputs to focus on and how much attention to give to each one at every decoding step [c1533]. When decoding a particular part of the output, the decoder has access to all input tokens and decides how much attention to give to each [c1557]. Traditional attention thus looks at one input sequence and one output sequence, determining which parts of the output are more related to which parts of the input [c1558].

---

### From Attention to Self-Attention

The Transformer architecture, introduced in 2017, was built with the self-attention mechanism at its core, itself inspired by the Bahdanau attention mechanism [c1534].

Traditional attention operates *between* two sequences—for example, between a source sentence and a target sentence during translation [c1549]. **Self-attention**, by contrast, looks at one sequence and examines how different parts of that same sequence relate to each other [c1559]. The term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions within a single input sequence [c1553], learning the relationships between various parts of the input itself rather than between elements of two different sequences [c1554].

More formally, self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. It is used in the original Transformer architecture, GPT models, and most other popular large language models [c1345], and is also called **scaled dot-product attention** [c1331, c1344].

---

### Simplified Self-Attention: The Core Idea

We will build up to the full self-attention mechanism in stages. The first stage is a **simplified self-attention** mechanism—the purest and most basic form of the attention technique—which does not include learnable parameters [c1487, c2158].

At a high level, self-attention transforms an input vector (a token embedding) into a **context vector** through attention weights computed from all input vectors [c2155]. The context vector for a given token is an enriched embedding that combines contributions from all input embedding vectors, weighted by their corresponding attention weights [c2236]. This chapter covers the mathematical foundations of this process and implements it from scratch in Python [c2157].

#### Notation

The input sequence consists of token embeddings $x_1, x_2, \ldots, x_n$, where $x_i$ denotes the vector representation of the $i$-th token [c2176]. For a concrete example, consider the sentence "Your journey starts here." The word "journey" is represented as $x_2$. Vector embeddings capture semantic meaning, such that semantically related words like "journey" and "starts" are positioned closer to each other in vector space [c2181].

---

### Step 1: Computing Attention Scores

The first step in self-attention is to compute **attention scores**—intermediate values that quantify how much importance or attention should be paid to each input word relative to a query word [c2187]. A **query** is the element or token being examined; for example, "journey" ($x_2$) is the query when we are computing the context vector for that token [c2183].

Attention scores are calculated using the **dot product** between a query vector and each input vector [c2197]. Recall that the dot product of two vectors $a$ and $b$ is:

$$a \cdot b = \sum_i (a_i \cdot b_i) = |a||b|\cos\theta$$

where $\theta$ is the angle between the vectors [c2193]. In self-attention, the dot product determines the extent to which elements of a sequence attend to one another [c2199], and both the scores and the resulting attention weights encode information about how much the query vector and each input embedding are related [c1335].

[FIGURE: Diagram showing query vector x_2 ("journey") computing dot products with all input vectors x_1 through x_n, producing a row of raw attention scores omega_21 through omega_2n]

---

### Step 2: Normalizing to Attention Weights

Raw attention scores are useful, but what we really want is a distribution: how much attention, as a fraction, does the query pay to each token?

**Normalization** transforms attention scores so that they sum to one, enabling interpretable statements about attention allocation as percentages [c2207]. The primary goal is to obtain attention weights that sum to one [c2208]. These normalized scores are called **attention weights** [c1332]—a normalized version of attention scores where each row sums to one, making them interpretable as probability distributions over tokens [c1395]. Normalization also serves a second purpose: it helps backpropagation by keeping scales consistent between 0 and 1 [c1397].

#### Softmax Normalization

The standard normalization function used in attention is softmax, which takes the exponent of every element and divides by the sum of all exponents [c2216]:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Softmax attention weights are always positive because they result from the exponent operation [c2226]. In PyTorch, softmax is available as `torch.nn.Softmax` [c2229], and the `dim` parameter specifies the dimension along which the normalization is computed [c2265].

In the simplified attention mechanism, attention weights are obtained through this normalization [c1343].

---

### Step 3: Computing Context Vectors

With attention weights in hand, the context vector for a query token is simply a weighted sum of all input vectors [c2246, c1333]. Formally:

$$z_i = \sum_j \alpha_{ij} \cdot x_j$$

The context vector, denoted $z$, is derived from attention weights and input vectors [c2179]. For the query token "journey" ($x_2$), the context vector $z_2$ contains information about both the query token itself and all other input elements in the sequence [c2184]. Attention weights determine how much attention to give to each input token when computing this vector [c2178]: for instance, the attention weight between "journey" and "your" indicates how much the model should attend to "your" when processing "journey" as the query [c1465].

[FIGURE: Diagram showing attention weights alpha_21 through alpha_2n multiplying input vectors x_1 through x_n and summing to produce context vector z_2 for the query "journey"]

---

### Putting It Together: A Complete Walkthrough

When we compute context vectors for every token in the sequence simultaneously, we produce an **attention weight matrix**: a matrix where each row represents the attention weights for one particular query word, and each column represents the attention weight between that query and a key word [c2250].

[FIGURE: Heatmap of an attention weight matrix with tokens as both row labels (queries) and column labels (keys); each cell shows the attention weight; rows sum to 1.0]

---

### Implementing Simplified Self-Attention in Python

The implementation follows directly from the three steps above: compute dot-product attention scores, normalize with softmax, and compute the weighted sum. In the simplified version, the input embeddings themselves serve directly as queries, keys, and values.

The following code shows how to compute scaled attention scores and weights:

```python
# Source: [c1406]
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```

```python
# Source: [c1448]
context_vector = attention_weights @ values
```

We will revisit the scaling factor $\sqrt{d_k}$ shortly. For now, note that attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403].

---

### Introducing Trainable Weights

The simplified mechanism has a fundamental limitation: using only the dot product to compute attention weights means attention is based solely on semantic similarity, which fails to capture contextual importance when semantically unrelated words are critical in context [c2278]. Consider a sentence where a grammatically important function word like "not" needs to strongly influence the interpretation of a semantically distant content word—raw dot-product similarity would miss this relationship entirely.

Trainable weights solve this problem by allowing the model to learn attention patterns beyond semantic similarity, enabling it to assign high attention to contextually important words even when they are not semantically related to the query [c2279]. More broadly, trainable weights allow the attention mechanism to capture long-range dependencies by learning contextual patterns beyond the semantic embedding space [c2282]. These weight matrices are optimized during training of the large language model [c1348].

#### The Three Weight Matrices

The self-attention mechanism is implemented using three trainable weight matrices: **Query** ($W_Q$), **Key** ($W_K$), and **Value** ($W_V$) [c1350, c1459]. Their roles map onto a familiar analogy:

- The **Query** is analogous to a search query in a database and represents the current token the model is focusing on [c1469].
- The **Key** represents items in the input sequence and is used to match with the query [c1470].
- The **Value** represents the actual content or representation of the input items themselves [c1471].

#### Projecting Inputs Through Weight Matrices

With trainable weight matrices, instead of using raw input embeddings as queries, keys, and values, we project the inputs through the learned matrices:

```python
# Source: [c1373]
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

Or equivalently, in a class-based implementation:

```python
# Source: [c1443]
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

```python
# Source: [c1445]
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

```python
# Source: [c1448]
context_vector = attention_weights @ values
```

#### Initializing Weight Matrices in PyTorch

In PyTorch, `torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters; they are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. The Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368]:

```python
# Source: [c1368]
torch.nn.Parameter(torch.rand(D_in, D_out))
```

Self-attention is organized as a Python class (`SelfAttentionVersion1`) derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439].

---

### Why Scale by $\sqrt{d_k}$?

In both the simplified and trainable versions, attention scores are divided by the square root of the key embedding dimension before applying softmax [c1403, c1406]—this is the "scaled" part of "scaled dot-product attention" [c1331]. Attention weights sum to one after softmax normalization [c1422], and the scaling helps maintain numerical stability across different embedding dimensions.

---

### The Road Ahead

To summarize the conceptual journey covered in this chapter:

1. **RNNs** process sequences with a hidden state but struggle with long-range dependencies [c1535, c1502].
2. **LSTMs** improve on RNNs with dual memory routes but still bottleneck at a single context vector [c1541, c1542].
3. **Bahdanau attention** (2014) allows the decoder to attend to all encoder states at each step, introducing dynamic focus [c1523, c1525, c1533].
4. **Self-attention** generalizes this to a single sequence, allowing every position to attend to every other position [c1548, c1553].
5. **Simplified self-attention** (no trainable weights) demonstrates the core dot-product → normalize → aggregate pattern [c1487, c2158].
6. **Full self-attention** (with $W_Q$, $W_K$, $W_V$) adds learnable projections that allow the model to go beyond raw semantic similarity [c1350, c2279].

The lecture series will go on to cover causal attention and multi-head attention, with mathematical formulations and code implementations in Python [c1560]. **Causal attention** modifies the self-attention mechanism to prevent the model from accessing future information in the sequence [c1474]—a critical property for autoregressive language models like GPT, which must predict the next token without looking ahead. In the next chapter, we will implement the full self-attention mechanism with trainable weights, add the causal mask, and stack multiple attention heads to build the complete Transformer building block.