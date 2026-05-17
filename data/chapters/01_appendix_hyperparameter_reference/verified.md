## GPT Model Configuration Reference

Before writing a single line of model code, it pays to have all the numbers in one place. This chapter is that place. We cover the full family of GPT-2 model sizes, the specific configuration dictionary used throughout this series, and the training hyperparameters that appear in the data-loading and optimization code. Treat this chapter as a reference you can flip back to whenever a number looks unfamiliar.

---

### The GPT_CONFIG_124M Dictionary

Every component of the model we build—the embedding layer, the transformer blocks, the attention heads, the output projection—is parameterized by a single Python dictionary. Here is the canonical version used throughout this series:

# Source: [c438]
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,           # Vocabulary size
    "context_length": 1024,        # Context length
    "emb_dim": 768,                # Embedding dimension
    "n_heads": 12,                 # Number of attention heads
    "n_layers": 12,                # Number of layers
    "drop_rate": 0.1,              # Dropout rate
    "qkv_bias": False              # Query-Key-Value bias
}
```

This dictionary encodes the architecture of GPT-2 small, the 124-million-parameter variant [c2841, c3496]. Every key in it corresponds to a design decision that has downstream consequences for memory, compute, and model quality. The sections below unpack each field in turn.

---

### Hyperparameter Definitions

#### `vocab_size` — Vocabulary Size

The vocabulary size determines how many distinct tokens the model can recognize and generate [c3847]. GPT-2 uses a vocabulary of **50,257 tokens** [c526, c529, c1670, c1740]. This number is not arbitrary: it is the output of the Byte-Pair Encoding (BPE) tokenizer that OpenAI trained for GPT-2, and it must match exactly between the tokenizer and the model's embedding table.

#### `context_length` — Maximum Sequence Length

The context length sets the maximum number of tokens the model can attend to at once [c3847]. For GPT-2 small, the full context length is **1,024 tokens** [c3490, c3536].

Intuitively, a longer context means the model can "see" more of the document when predicting the next token, but it also means the attention matrices grow quadratically. For this reason, the tutorials in this series reduce the context length to **256 tokens** during training runs, so that the code can execute on a laptop without running out of memory [c494, c537, c1671, c3830]. The production value remains 1,024; 256 is purely a pedagogical convenience.

#### `emb_dim` — Embedding Dimension

The embedding dimension is the size of the vector space into which each token ID is converted [c3492]. For GPT-2 small, every token is represented as a **768-dimensional vector** [c531, c1673, c3492, c3536]. This same dimensionality flows through every transformer block: the residual stream, the attention projections, and the feed-forward layers all operate in this 768-dimensional space.

#### `n_heads` — Number of Attention Heads

`n_heads` is the number of attention heads present in each transformer block [c3494]. GPT-2 small uses **12 attention heads** [c532, c1674, c3536]. Each head operates on a slice of the embedding dimension (768 / 12 = 64 dimensions per head), allowing the model to attend to different aspects of the input in parallel.

#### `n_layers` — Number of Transformer Blocks

`n_layers` is the number of transformer blocks stacked in the model [c3495]. GPT-2 small has **12 transformer blocks** [c533, c888, c1675, c3536]. Each block contains a multi-head self-attention sublayer followed by a feed-forward sublayer, with layer normalization and residual connections around each.

#### `drop_rate` — Dropout Rate

The dropout rate is a regularization parameter [c498]. GPT-2 small sets it to **0.1** [c534, c1676]. During training, dropout randomly zeroes out 10% of activations, which helps prevent overfitting. When experimenting or debugging, it is common to set dropout to zero [c498] so that the model's behavior is fully deterministic and easier to inspect.

#### `qkv_bias` — Query-Key-Value Bias

This boolean controls whether the linear projections that produce the query, key, and value matrices include a bias term [c3847]. The GPT-2 small configuration sets this to **`False`** [c499, c535, c1677]: the weight matrices are initialized without bias terms. This is a deliberate design decision that reduces the number of parameters slightly and matches the original GPT-2 architecture [c499, c1677].

> **Note on the loaded weights variant.** When loading pretrained OpenAI GPT-2 weights, `qkv_bias` must be set to **`True`** [c1742], because the released checkpoints include those bias terms. The `False` default is appropriate when training from scratch.

---

### The Full GPT-2 Model Family

GPT-2 was released in four sizes. The table below summarizes their architectural hyperparameters.

[FIGURE: Table with five columns — Model, Parameters, Embedding Dim, Layers, Attention Heads, Context Length — and four rows for GPT-2 small/medium/large/XL]

#### GPT-2 Small (124M)

| Hyperparameter | Value |
|---|---|
| Parameters | 124M [c2841] |
| Vocabulary size | 50,257 [c1740] |
| Context length | 1,024 [c3490] |
| Embedding dimension | 768 [c3536] |
| Transformer layers | 12 [c3536] |
| Attention heads | 12 [c3536] |
| Dropout rate | 0.1 [c534] |
| QKV bias | False [c535] |

This is the model we build and train from scratch throughout the series [c3496].

#### GPT-2 Medium (345M)

| Hyperparameter | Value |
|---|---|
| Parameters | 345M [c3434] |
| Context length | 1,024 [c1737] |
| Embedding dimension | 1,024 [c3537] |
| Transformer layers | 24 [c1738, c3537] |
| Attention heads | 16 [c3537] |

Compared to GPT-2 small, medium doubles both the embedding dimension and the number of layers, roughly tripling the parameter count [c3434].

#### GPT-2 Large (774M)

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 1,280 [c3538] |
| Transformer layers | 36 [c3538] |
| Attention heads | 20 [c3538] |

#### GPT-2 XL (1558M)

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 1,600 [c3539] |
| Transformer layers | 48 [c3539] |
| Attention heads | 25 [c3539] |

The pattern across the family is clear: larger models increase all three of embedding dimension, layer count, and head count together. The vocabulary size of 50,257 and the context length of 1,024 remain constant across all four variants [c526, c1737].

---

### Training Hyperparameters

Beyond the model architecture, several hyperparameters govern how data is fed to the model during training. These appear in the data-loader construction code used throughout the series.

#### Context Size During Training

As noted above, the tutorials use a context length of **256 tokens** rather than the full 1,024, purely to reduce computational requirements [c537, c3830]. This value is used consistently as `max_length` in both the training and validation data loaders.

#### Data Loader Configuration

The training data loader is constructed with the following settings [c3834]:

| Parameter | Value |
|---|---|
| `batch_size` | 2 |
| `max_length` | 256 |
| `stride` | 256 |
| `drop_last` | True |
| `shuffle` | True |
| `num_workers` | 0 |

The validation data loader uses the same `batch_size` and sequence parameters but without shuffling [c3836]:

| Parameter | Value |
|---|---|
| `batch_size` | 2 |
| `max_length` | 256 |
| `stride` | 256 |

Setting `stride` equal to `max_length` means there is no overlap between consecutive windows: each chunk of 256 tokens is drawn from a non-overlapping position in the text. Setting `drop_last=True` ensures that every batch has exactly the same size, which avoids shape surprises during training.

[GAP: Learning rate and weight decay values are referenced in the chapter summary but no claims in the supplied set specify their numeric values. These should be added when the corresponding claims are available.]

---

### Quick-Reference Summary

The table below consolidates everything into a single lookup.

| Key | GPT-2 Small (series default) | Notes |
|---|---|---|
| `vocab_size` | 50,257 [c529] | Fixed across all GPT-2 sizes |
| `context_length` | 1,024 (full) / 256 (tutorial) [c3490, c537] | Reduced for laptop execution |
| `emb_dim` | 768 [c531] | Vector size per token |
| `n_heads` | 12 [c532] | Heads per transformer block |
| `n_layers` | 12 [c533] | Number of transformer blocks |
| `drop_rate` | 0.1 [c534] | Set to 0.0 when debugging |
| `qkv_bias` | False [c535] | True when loading OAI weights |
| Training `batch_size` | 2 [c3834] | |
| Training `max_length` | 256 [c3834] | |
| Training `stride` | 256 [c3834] | Non-overlapping windows |

With these numbers in hand, every subsequent chapter has a concrete anchor. When the code instantiates an embedding layer of size 50,257 × 768, or loops over 12 transformer blocks, or clips sequences to 256 tokens, you can trace each choice back to this reference.