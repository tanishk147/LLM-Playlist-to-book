## Self-Attention with Trainable Weights: Keys, Queries, and Values

The previous chapter introduced a simplified attention mechanism that computes context vectors directly from raw input embeddings. It worked, but it had a fundamental limitation: there were no trainable weights [c1341], so the model had no way to *learn* what to attend to. This chapter fixes that by introducing the full self-attention mechanism—the one actually used in transformers, GPT models, and most other popular large language models [c1345].

You will see this mechanism called both **self-attention** and **scaled dot-product attention** [c1331, c1344]. The core idea is to introduce three trainable weight matrices—a Query matrix $W_Q$, a Key matrix $W_K$, and a Value matrix $W_V$—that transform raw input embeddings into richer representations before any attention computation happens [c1350]. These matrices are initialized randomly and optimized during training [c1361], allowing the model to learn, through gradient descent, how to produce context vectors that are genuinely useful for the task at hand [c1349].

---

### Why Trainable Weights?

In the simplified mechanism, attention scores were calculated using a direct dot product between input vectors [c1342], and attention weights were obtained by normalizing those scores [c1343]. The model had no parameters to adjust, so it could not improve its attention patterns through training.

The trainable weight matrices solve this by projecting each input embedding into three distinct roles [c1376]:

- **Query**: what the current token is looking for [c1469]
- **Key**: what each token in the sequence offers as a match [c1470]
- **Value**: the actual content that gets aggregated into the output [c1471]

The attention scores measure how well each key matches the query, and the context vector is a weighted blend of the values, weighted by those match scores [c1465]. Because the weight matrices are trainable parameters optimized during LLM training [c1462], the model can learn to ask the right questions (queries), expose the right features for matching (keys), and encode the right information for aggregation (values) [c1348].

---

### The Three Weight Matrices

Self-attention requires exactly three trainable weight matrices: $W_Q$, $W_K$, and $W_V$ [c1459]. Each input embedding vector is multiplied with all three matrices separately to produce the corresponding query, key, and value vectors [c1377].

[FIGURE: Diagram showing a single input embedding vector being projected through three separate weight matrices WQ, WK, WV to produce three output vectors: query, key, value]

In a concrete example, the input embeddings live in a 3-dimensional space, and the weight matrices project them into a 2-dimensional space for keys, queries, and values [c1375]. The Key weight matrix $W_K$, for instance, has dimensions $3 \times 2$, transforming each 3-dimensional input embedding into a 2-dimensional key vector [c1355]. For the word "journey" (the second input vector), multiplying the 3-dimensional input vector by the $3 \times 2$ Query weight matrix produces a 2-dimensional query vector with values approximately $[0.4306, 1.4551]$ [c1371].

When all six input tokens are projected through the key, query, and value weight matrices, the resulting matrices each have shape $(6, 2)$: six rows for the six input tokens and two columns for the embedding dimension [c1374]. The input dimension of these weight matrices must match the embedding vector dimension, because a dot product is computed between them [c1460].

The keys, queries, and values are obtained by multiplying input embeddings with these separate trainable weight matrices [c1372]:

```python
# Source: [c1373]
keys = inputs @ W_k
values = inputs @ W_v
queries = inputs @ W_q
```

---

### Computing Attention Scores

Each query vector has an attention score with every key vector in the sequence [c1384]. For a query matrix of shape $6 \times 2$, computing attention scores for the second query (the token "journey") involves taking the dot product of that query row with all key rows to produce 6 attention scores [c1385].

The attention scores encode how much the query vector and each key vector are related to each other [c1335]. Geometrically, when two vectors are aligned (small angle between them), their dot product is maximum, indicating high attention [c1382]. The model learns to align query and key vectors for tokens that should attend strongly to each other. However, raw attention scores do not sum to one and are not directly interpretable as attention percentages [c1396].

---

### Scaling by $\sqrt{d_k}$: Why It Matters

Before applying softmax, the attention scores are divided by the square root of the key embedding dimension [c1420]—which is one of the reasons the mechanism is called *scaled* dot-product attention [c1400]. The scaling factor is $\sqrt{d_k}$, where $d_k$ is the embedding dimension of the keys [c1404]. In the running example, the key embedding dimension is 2, so attention scores are divided by $\sqrt{2}$ [c1405].

To understand why this matters, consider what happens to the dot product as the dimension grows. For a $d$-dimensional query and key vector sampled from a normal distribution, the variance of their dot product before scaling is approximately equal to $d$ [c1415]; without scaling, this variance grows proportionally with the dimension [c1414]. When softmax receives large input values, its output becomes *peaky*—the highest value receives disproportionately high probability compared to all others [c1412]—making the model overly confident in a single key and destabilizing learning [c1413]. Dividing by $\sqrt{d_k}$ keeps the variance of the attention scores close to 1 regardless of the query and key dimension [c1416], which is essential for stable learning and proper backpropagation [c1417, c1446].

[FIGURE: Two softmax curves side by side: one with unscaled large inputs (peaky, near one-hot) and one with scaled inputs (smoother distribution), illustrating the effect of scaling]

---

### From Scores to Attention Weights

Attention weights are computed by scaling the attention scores by $\sqrt{d_k}$ and then applying softmax [c1403]:

```python
# Source: [c1406]
d_k = keys.shape[-1]                                        # Extract key embedding dimension from last axis
attention_scores_scaled = attention_scores / (d_k ** 0.5)  # Scale by sqrt(d_k)
attention_weights = softmax(attention_scores_scaled, dim=-1) # Apply softmax over columns
```

The result is a matrix of **attention weights**: a normalized version of the attention scores [c1332] where each row sums to one, making the weights interpretable as probability distributions over tokens [c1395]. All values lie between zero and one [c1398], and this normalization serves two purposes: making attention interpretable and helping backpropagation by keeping scales consistent [c1397].

For the full sequence of six tokens, the attention weights matrix has dimensions $6 \times 6$ [c1426], computed by applying the scaling and softmax operation to the full $6 \times 6$ attention scores matrix [c1409]. One important caveat: attention scores are meaningless until the weight matrices are trained; their values only become semantically significant after training [c1390].

---

### Computing Context Vectors

The final step is to compute context vectors as a weighted sum over the value vectors [c1333]. The attention weight between two tokens indicates how much the model should attend to the second token when processing the first as the query [c1465], and the attention scores and value representations are used together to produce the final context vector [c1473].

Concretely, to compute the context vector for "journey," each value vector is scaled by its corresponding attention weight and the results are summed [c1430]. The attention weights for "journey" in the example are: 0.15 for "your," 0.2264 for "journey," 0.2199 for "begins," 0.13 for "width," 0.09 for "one," and 0.18 for "step" [c1428]. Accordingly, the "your" vector is multiplied by 0.15, the "journey" vector by 0.22, the "begins" vector by 0.2199, the "width" vector by 0.13, the "one" vector by 0.09, and the "step" vector by 0.18 [c1429], and these scaled vectors are then added together [c1430]. The resulting context vector for "journey" equals $[0.3061, 0.8210]$ [c1436].

[FIGURE: Diagram showing six value vectors, each scaled by its attention weight, being summed to produce the context vector for 'journey']

In matrix form, the context vector for a specific token is obtained by taking that token's row from the attention weights matrix ($1 \times 6$) and multiplying it by the values matrix ($6 \times 2$) to produce a $1 \times 2$ vector [c1435]. Whether computed graphically—by scaling individual value vectors and summing—or via matrix multiplication, both approaches yield identical results [c1437]:

```python
# Source: [c1448]
context_vector = attention_weights @ values
```

---

### Putting It All Together: The Forward Pass

The complete forward pass of scaled dot-product attention proceeds as follows. First, project the inputs into queries, keys, and values [c1443]:

```python
# Source: [c1443]
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
```

Then scale the attention scores and apply softmax [c1445]:

```python
# Source: [c1445]
attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
attention_weights = softmax(attention_scores, dim=-1)
```

Finally, compute the context vectors [c1448]:

```python
# Source: [c1448]
context_vector = attention_weights @ values
```

The objective throughout is to compute a context vector for every input token using these trainable weight matrices [c1347].

---

### Implementing SelfAttentionV1 in PyTorch

With the forward pass understood, we can package this logic into a proper PyTorch class. Self-attention is organized as a class derived from `nn.Module` to enable reusable instantiation and integration into larger LLM implementations [c1439]; `nn.Module` is a fundamental building block of PyTorch models, providing the necessary infrastructure for layer creation and parameter management [c1449].

#### Initializing the Weight Matrices

The `__init__` constructor initializes query, key, and value weight matrices randomly with dimensions `d_in` rows by `d_out` columns [c1442]. `torch.nn.Parameter` is a Tensor subclass that marks tensors as module parameters, automatically adding them to the module's parameter list and exposing them through the `parameters()` iterator [c1367]. The three weight matrices are therefore initialized as `torch.nn.Parameter` objects with random values and shape `(D_in, D_out)` [c1368], and they are optimized during training [c1361, c1432].

#### The Forward Method

The forward method takes an input embedding matrix `x` and returns a context vector matrix where each row corresponds to the context vector for one input token [c1450]. When instantiating `SelfAttentionVersion1` with `input_dimension=3` and `output_dimension=2` and passing six input embedding vectors, the output is a context vector matrix with dimensions $6 \times 2$ [c1451].

---

### Implementing SelfAttentionV2: Using `nn.Linear`

A cleaner design is to replace `nn.Parameter` with `nn.Linear` layers (with `bias=False`) to initialize the query, key, and value weight matrices [c1452]. Using `nn.Linear` is more common practice when implementing self-attention for LLMs [c1457], as it handles weight initialization in a principled way and integrates cleanly with PyTorch's optimizer and parameter tracking infrastructure.

[FIGURE: Side-by-side comparison of SelfAttentionV1 (using nn.Parameter) and SelfAttentionV2 (using nn.Linear with bias=False) showing equivalent structure]

---

### Interpreting the Mechanism

It is worth pausing to build intuition for what the three matrices actually do. The **query** represents the current token the model is focusing on—analogous to a search query in a database [c1469]. The **key** represents items in the input sequence and is used to match against the query [c1470]. The **value** represents the actual content or representation of those input items, which gets blended into the output [c1471]. Importantly, the key and value for the same token can be very different vectors, because they serve entirely different purposes.

In contrast to the earlier simplified mechanism, where the query was simply the token's own embedding [c1379], the learned query, key, and value projections allow the model to develop specialized representations for each role.

[FIGURE: Diagram illustrating the query-key-value analogy: a query vector searching through key vectors to find matches, then retrieving the corresponding value vectors and blending them]

---

### The Full Picture

To summarize, scaled dot-product attention proceeds in five steps:

1. **Linear projections**: Multiply input embeddings by $W_Q$, $W_K$, $W_V$ to get queries $Q$, keys $K$, and values $V$ [c1376].
2. **Attention scores**: Compute $QK^\top$ via dot products [c1342].
3. **Scaling**: Divide by $\sqrt{d_k}$ to keep variance close to 1 [c1410, c1416].
4. **Softmax**: Apply softmax row-wise to obtain attention weights that sum to one [c1403, c1422].
5. **Context vectors**: Multiply attention weights by $V$ to produce the output [c1333].

This mechanism is used in the original transformer architecture, GPT models, and most other popular large language models [c1339]. Understanding it at this level of detail—from the matrix shapes to the scaling rationale—is essential for everything that follows.

---

### What Comes Next

The self-attention mechanism developed here processes all tokens symmetrically: every token can attend to every other token, including those that appear later in the sequence. For language modeling, however, the model should not be allowed to peek at future tokens when predicting the next word. Causal attention modifies the self-attention mechanism to prevent exactly that, by masking out future positions [c1474]—and that is the subject of the next chapter.