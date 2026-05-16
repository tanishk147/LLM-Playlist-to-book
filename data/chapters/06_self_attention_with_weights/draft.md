## Self-Attention with Trainable Weights: Keys, Queries, and Values

The previous chapter introduced a simplified attention mechanism that could compute context vectors from raw input embeddings. It worked, but it had a fundamental limitation: there were no trainable weights [c1341]. Everything was fixed. The model had no way to *learn* what to attend to. This chapter fixes that by introducing the full self-attention mechanism—the one actually used in transformers, GPT models, and most other popular large language models [c1345].

The mechanism goes by a few names. You will see it called **self-attention**, and you will also see it called **scaled dot-product attention** [c1331, c1344]. Both names refer to the same thing, and by the end of this chapter you will understand exactly why each part of that second name is earned.

The core idea is to introduce three trainable weight matrices—a Query matrix $W_Q$, a Key matrix $W_K$, and a Value matrix $W_V$—that transform raw input embeddings into richer representations before any attention computation happens [c1350]. These matrices are initialized randomly and then optimized during training [c1361], which means the model learns, through gradient descent, how to produce context vectors that are actually useful for the task at hand [c1349].

---

### Why Trainable Weights?

In the simplified mechanism, attention scores were calculated using a direct dot product between input vectors [c1342], and attention weights were obtained by normalizing those scores [c1343]. The problem is that the input embeddings themselves are doing double duty: they are simultaneously acting as the "question being asked" and the "answer being given." There is no separation of concerns.

The trainable weight matrices solve this by projecting each input embedding into three distinct roles [c1376]:

- **Query**: what the current token is looking for [c1469]
- **Key**: what each token in the sequence offers as a match [c1470]
- **Value**: the actual content that gets aggregated into the output [c1471]

Intuitively, this is analogous to a database search. The query is the search term, the keys are the index entries, and the values are the records you retrieve. The attention scores measure how well each key matches the query, and the context vector is a weighted blend of the values, weighted by those match scores [c1465].

The weight matrices are what make this mechanism powerful. Because they are trainable parameters optimized during LLM training [c1462], the model can learn to ask the right questions (queries), expose the right features for matching (keys), and encode the right information for aggregation (values) [c1348].

---

### The Three Weight Matrices

Self-attention requires exactly three trainable weight matrices: $W_Q$, $W_K$, and $W_V$ [c1459]. Each input embedding vector is multiplied with all three of these matrices separately to produce the corresponding query, key, and value vectors [c1377].

[FIGURE: Diagram showing a single input embedding vector being projected through three separate weight matrices WQ, WK, WV to produce three output vectors: query, key, value]

In a concrete example, the input embeddings live in a 3-dimensional space, and the weight matrices project them into a 2-dimensional space for keys, queries, and values [c1375]. The Key weight matrix $W_K$, for instance, has dimensions $3 \times 2$, transforming each 3-dimensional input embedding into a 2-dimensional key vector [c1355].

The projection is a simple matrix multiplication. For the word "journey" (the second input vector), multiplying the 3-dimensional input vector by the $3 \times 2$ Query weight matrix produces a 2-dimensional query vector with values approximately $[0.4306, 1.4551]$ [c1371].

When all six input tokens are projected through the key, query, and value weight matrices, the resulting matrices each have shape $(6, 2)$: six rows for the six input tokens, and two columns for the embedding dimension [c1374].

The keys, queries, and values are obtained by multiplying input embeddings with these separate trainable weight matrices [c1372]:

# Source: [c1373]
```python
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

The input dimension of these weight matrices must match the embedding vector dimension, because a dot product is computed between them [c1460].

---

### Computing Attention Scores

With queries and keys in hand, computing attention scores follows the same dot-product logic as before—but now the dot products are between *learned* query and key vectors rather than raw embeddings.

Each query vector has an attention score with every key vector in the sequence [c1384]. For a query matrix of shape $6 \times 2$, computing attention scores for the second query (the token "journey") involves taking the dot product of that query row with all key rows to produce 6 attention scores [c1385].

The attention scores encode information about how much the query vector and each key vector are related to each other [c1335]. Geometrically, when two vectors are aligned (small angle between them), their dot product is maximum, indicating high attention [c1382]. The model learns to align query and key vectors for tokens that should attend to each other.

However, raw attention scores do not sum to one and are not directly interpretable as attention percentages [c1396]. We need to normalize them—but before normalization, there is an important scaling step.

---

### Scaling by $\sqrt{d_k}$: Why It Matters

Before applying softmax, the attention scores are divided by the square root of the key embedding dimension [c1420]. This is not an arbitrary choice. It is one of the reasons the mechanism is called *scaled* dot-product attention [c1400].

The scaling factor is $\sqrt{d_k}$, where $d_k$ is the embedding dimension of the keys [c1404]. In the running example, the key embedding dimension is 2, so attention scores are divided by $\sqrt{2}$ [c1405].

To understand why this matters, consider what happens to the dot product as the dimension grows. For a $d$-dimensional query and key vector sampled from a normal distribution, the variance of their dot product before scaling is approximately equal to $d$ [c1415]. In other words, the dot product of two random vectors increases in variance as the dimension increases [c1414]. Without scaling, the variance of the dot product scales proportionally with the dimension of the query and key vectors [c1414].

Why does high variance cause problems? When softmax receives large input values, its output becomes *peaky*—the highest value receives disproportionately high probability compared to all other values [c1412]. This means the model becomes overly confident in a single key, which destabilizes learning [c1413].

Dividing by $\sqrt{d_k}$ keeps the variance of the attention scores close to 1, regardless of the dimension of the query and key vectors [c1416]. This is important for stable learning and proper backpropagation, preventing the learning process from becoming unstable [c1417].

Dividing attention scores by the square root of the key embedding dimension ensures that values in the attention score matrix remain small and that the variance of the dot product between keys and queries stays close to one [c1446].

[FIGURE: Two softmax curves side by side: one with unscaled large inputs (peaky, near one-hot) and one with scaled inputs (smoother distribution), illustrating the effect of scaling]

---

### From Scores to Attention Weights

After scaling, softmax is applied to convert the scores into proper probability distributions. Attention weights are computed by scaling attention scores by the square root of the key embedding dimension, then applying softmax [c1403]:

$$\text{attention\_weights} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

The code for this operation is:

# Source: [c1406]
```python
d_k = keys.shape[-1]  # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1)  # Apply softmax over columns
```

The result is a matrix of **attention weights**: a normalized version of the attention scores [c1332]. Attention weights are normalized attention scores where each row sums to one, making them interpretable as probability distributions over tokens [c1395]. After softmax normalization, attention weights sum to one [c1422], and all values lie between zero and one with positive values [c1398].

Normalization of attention scores serves two purposes: making attention interpretable and helping backpropagation by keeping scales consistent between 0 and 1 [c1397].

For the full sequence of six tokens, the attention weights matrix has dimensions $6 \times 6$ [c1426]. The attention weights matrix for all queries is a $6 \times 6$ matrix computed by applying the scaling and softmax operation to the full $6 \times 6$ attention scores matrix [c1409].

An important caveat: attention scores are meaningless until the weight matrices are trained [c1390]. Their values only become semantically significant after training, when the model has learned to produce meaningful query and key vectors.

---

### Computing Context Vectors

The final step is to compute context vectors as a weighted sum over the value vectors [c1333]. The attention weight between two tokens indicates how much the model should attend to the second token when processing the first token as the query [c1465]. The attention scores and the original input embedding values are used together to compute the final context vector [c1473].

Concretely, to compute the context vector for the token "journey," each value vector is scaled by its corresponding attention weight and the results are summed [c1430]. The attention weights for "journey" in the example are: 0.15 for "your," 0.2264 for "journey," 0.2199 for "begins," 0.13 for "width," 0.09 for "one," and 0.18 for "step" [c1428].

Each value vector is scaled by its corresponding attention weight: the "your" vector is multiplied by 0.15, the "journey" vector by 0.22, the "begins" vector by 0.2199, the "width" vector by 0.13, the "one" vector by 0.09, and the "step" vector by 0.18 [c1429]. The context vector is then computed by adding together all the scaled value vectors [c1430].

The context vector for the token "journey" (second row) computed as attention weights for "journey" multiplied by the values matrix equals $[0.3061, 0.8210]$ [c1436].

[FIGURE: Diagram showing six value vectors, each scaled by its attention weight, being summed to produce the context vector for 'journey']

In matrix form, the context vector for a specific token can be computed by taking that token's row from the attention weights matrix ($1 \times 6$) and multiplying it by the values matrix ($6 \times 2$) to produce a $1 \times 2$ vector [c1435]. The context vector can be understood through either a scaling-based graphical approach (multiplying individual value vectors by their attention weights and summing) or through matrix multiplication, both yielding identical results [c1437].

In code:

# Source: [c1448]
```python
context_vector = attention_weights @ values
```

---

### Putting It All Together: The Forward Pass

Let us now trace the complete forward pass of scaled dot-product attention from input embeddings to context vectors. The steps are:

1. Project input embeddings into queries, keys, and values via the three weight matrices.
2. Compute attention scores as dot products between queries and keys.
3. Scale by $\sqrt{d_k}$.
4. Apply softmax to get attention weights.
5. Multiply attention weights by values to get context vectors.

In code, steps 1 through 3 look like this:

# Source: [c1443]
```python
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

Then scaling and softmax:

# Source: [c1445]
```python
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

And finally the context vectors:

# Source: [c1448]
```python
context_vector = attention_weights @ values
```

The objective is to compute the context vector for every given input token using these trainable weight matrices [c1347].

---

### Implementing SelfAttentionV1 in PyTorch

Now we package this logic into a proper PyTorch class. Self-attention is organized as a Python class derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439].

The class is called `SelfAttentionVersion1`. It inherits from `nn.Module`, which is a fundamental building block of PyTorch models providing necessary functionalities for model layer creation and management [c1449].

#### Initializing the Weight Matrices

The `__init__` constructor initializes query, key, and value weight matrices randomly with dimensions of `d_in` rows and `d_out` columns [c1442].

`torch.nn.Parameter` is a Tensor subclass used to mark tensors as module parameters, which are automatically added to the module's parameter list and appear in the `parameters()` iterator [c1367]. Query, Key, and Value weight matrices are initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368]:

The weight matrices are trainable parameters that are initialized randomly and optimized during training [c1361]. The attention weights themselves are initially randomly initialized and are optimized during LLM training [c1432].

#### The Forward Method

The forward method of `SelfAttentionVersion1` takes an input embedding vector `x` and returns a context vector matrix where each row corresponds to the context vector for one input token [c1450].

When instantiating `SelfAttentionVersion1` with `input_dimension=3` and `output_dimension=2`, passing six input embedding vectors produces a context vector matrix with dimensions 6 rows by 2 columns, where each row is the context vector for one input token [c1451].

---

### Implementing SelfAttentionV2: Using `nn.Linear`

The `nn.Parameter` approach works, but there is a more idiomatic way to implement the weight matrices. A better design decision is to use `nn.Linear` layers instead of `nn.Parameter` to initialize query, key, and value weight matrices in self-attention, with bias set to `False` [c1452].

Using `nn.Linear` for self-attention implementation is more common practice when dealing with LLMs [c1457]. The `nn.Linear` layer internally stores its weight matrix and handles initialization in a principled way, and it integrates cleanly with PyTorch's optimizer and parameter tracking infrastructure.

[FIGURE: Side-by-side comparison of SelfAttentionV1 (using nn.Parameter) and SelfAttentionV2 (using nn.Linear with bias=False) showing equivalent structure]

The two implementations are functionally equivalent—both produce the same context vectors given the same weights—but `nn.Linear` is the preferred approach for production LLM code.

---

### Dimensions and Design Constraints

A few important constraints govern the shapes of the weight matrices.

The input dimension of the query, key, and value weight matrices must match the embedding vector dimension, because a dot product is computed between them [c1460]. If your input embeddings are 3-dimensional, the weight matrices must have 3 rows.

The output dimension (number of columns) is a hyperparameter you choose. In the running example, input embeddings are projected from a 3-dimensional embedding space into a 2-dimensional space for keys, values, and queries [c1375]. This projection is what allows the model to learn a compressed or transformed representation of each token in each of the three roles.

In the previous simplified attention mechanism, the query was the token itself without a separate query vector [c1379]. The current approach replaces this with separate learned vectors, which is strictly more expressive: the model can learn to ask questions (queries) that are very different from the raw token representation.

---

### Interpreting the Mechanism

It is worth pausing to build intuition for what these three matrices are actually doing.

The **query** represents the current token the model is focusing on [c1469]. When computing the context vector for "journey," the query vector for "journey" is the search term being issued.

The **key** represents items in the input sequence and is used to match with the query [c1470]. Each token's key vector is like a label that says "here is what I offer as a match."

The **value** represents the actual content or representation of the input items themselves [c1471]. Once the model has decided how much to attend to each token (via the attention weights), it aggregates the value vectors—not the key vectors—into the context vector.

This separation is subtle but important. The key and value for the same token can be very different vectors, because they serve different purposes. The key is optimized to be *found* by relevant queries; the value is optimized to *contribute useful information* to the context vector.

[FIGURE: Diagram illustrating the query-key-value analogy: a query vector searching through key vectors to find matches, then retrieving the corresponding value vectors and blending them]

---

### The Full Picture

Let us summarize the complete scaled dot-product attention computation:

1. **Linear projections**: Multiply input embeddings by $W_Q$, $W_K$, $W_V$ to get queries $Q$, keys $K$, values $V$ [c1376].
2. **Attention scores**: Compute $QK^\top$ via dot products [c1342].
3. **Scaling**: Divide by $\sqrt{d_k}$ to keep variance close to 1 [c1410, c1416].
4. **Softmax**: Apply softmax row-wise to get attention weights that sum to one [c1403, c1422].
5. **Context vectors**: Multiply attention weights by $V$ to produce the output [c1333].

The dot product between query and key vectors is scaled by dividing by the square root of the key dimension [c1410]. This keeps the variance of the attention scores close to 1, regardless of the dimension of the query and key vectors [c1416], which is important for stable learning and proper backpropagation [c1417].

The self-attention mechanism is used in the original transformer architecture, GPT models, and most other popular large language models [c1339]. Understanding it at this level of detail—from the matrix shapes to the scaling rationale—is essential for everything that follows.

---

### What Comes Next

The self-attention mechanism developed in this chapter processes all tokens in a sequence symmetrically: every token can attend to every other token, including future tokens. For language modeling, this is a problem. When predicting the next word, the model should not be allowed to peek at words that come later in the sequence.

Causal attention modifies the self-attention mechanism to prevent the model from accessing future information in the sequence [c1474]. That modification—masking out future positions—is the subject of the next chapter.