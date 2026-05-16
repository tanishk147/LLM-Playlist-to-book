## GPT Model Configuration Reference and Hyperparameter Glossary

This chapter serves as a practical reference for every hyperparameter and architectural constant that appears throughout the book. Before diving into implementation, it is worth having a single place to look up what each number means, where it comes from, and why it was chosen. We cover the full GPT-2 model family, the canonical `GPT_CONFIG_124M` dictionary used in code examples, and the training hyperparameters you will encounter when running the model.

### The GPT-2 Model Family

OpenAI released GPT-2 in four sizes. Understanding these differences helps you reason about the trade-off between model capacity and computational cost.

All four GPT-2 variants share the same tokenizer and therefore the same vocabulary. The vocabulary size is **50,257 tokens** [c526, c529, c1670, c1740]. This number is not arbitrary — it reflects the byte-pair encoding (BPE) vocabulary used by GPT-2's tokenizer, which balances coverage of English text against embedding table size.

The base (small) model supports a context length of **1,024 tokens** [c3490], meaning it can attend to up to 1,024 tokens at once during a forward pass [c1737]. GPT-2 medium shares this same 1,024-token context window [c1737].

#### GPT-2 Small (124M)

The smallest member of the family is the one you will build and train in this book. Its defining characteristics are [c3536]:

| Hyperparameter | Value |
|---|---|
| Parameters | 124M [c2841] |
| Vocabulary size | 50,257 [c1740] |
| Context length | 1,024 [c3490] |
| Embedding dimension | 768 [c3536] |
| Transformer layers | 12 [c3536] |
| Attention heads | 12 [c3536] |

The embedding dimension is the size of the vector space into which each token ID is converted; for GPT-2 small, every token is represented as a 768-dimensional vector [c3492]. These embedding vectors are not fixed — they are trained during model training, with learnable parameters corresponding to each vector [c3493]. 

#### GPT-2 Medium (345M / 355M)

GPT-2 medium scales up the embedding dimension and doubles the depth [c3537]:

| Hyperparameter | Value |
|---|---|
| Parameters | 345M [c3434] |
| Context length | 1,024 [c1737] |
| Embedding dimension | 1,024 [c3537] |
| Transformer layers | 24 [c3537, c1738] |
| Attention heads | 16 [c3537] |

#### GPT-2 Large (774M)

| Hyperparameter | Value |
|---|---|
| Parameters | 774M |
| Embedding dimension | 1,280 [c3538] |
| Transformer layers | 36 [c3538] |
| Attention heads | 20 [c3538] |

#### GPT-2 XL (1558M)

| Hyperparameter | Value |
|---|---|
| Parameters | 1,558M |
| Embedding dimension | 1,600 [c3539] |
| Transformer layers | 48 [c3539] |
| Attention heads | 25 [c3539] |

The pattern across all four sizes is clear: as the model grows, the embedding dimension, layer count, and head count all increase together. 

---

### The `GPT_CONFIG_124M` Dictionary

 The canonical version for the 124M model is reproduced below exactly as it appears in the tutorial code.

```python
GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size
 "context_length": 1024, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}
```

Each key maps directly to one of the architectural constants described in the previous section. A few entries deserve special attention.

#### `drop_rate`

The dropout rate is set to `0.1` in the full-scale configuration [c534, c1676]. This is a regularization parameter — during training, 10% of activations are randomly zeroed out to reduce overfitting. When running experiments on a laptop or in a resource-constrained environment, this value can be set to zero [c498], which disables dropout entirely and slightly speeds up computation.

#### `qkv_bias`

The query, key, and value weight matrices are initialized **without** bias terms — `qkv_bias` is `False` [c499, c535, c1677]. This is a deliberate design choice that matches the original GPT-2 architecture. 

---

### Practical Adjustments for Tutorial Environments

The full 1,024-token context length is computationally expensive, particularly for self-attention, whose memory cost scales quadratically with sequence length. For the hands-on exercises in this book, the context length is reduced to **256 tokens** [c537, c1671] so that training can run on a standard laptop without a GPU [c537]. This reduction does not change the model architecture in any fundamental way — it simply limits how much context the model can see at once.

The tutorial configuration therefore looks like this in practice:

| Key | Full GPT-2 Small | Tutorial Value |
|---|---|---|
| `vocab_size` | 50,257 [c529] | 50,257 |
| `context_length` | 1,024 [c3490] | 256 [c494, c537] |
| `emb_dim` | 768 [c531] | 768 [c1673] |
| `n_heads` | 12 [c532] | 12 [c1674] |
| `n_layers` | 12 [c533] | 12 [c1675] |
| `drop_rate` | 0.1 [c534] | 0.1 [c1676] |
| `qkv_bias` | False [c535] | False [c1677] |

---

### Training Hyperparameters

Beyond the model architecture, several hyperparameters govern the training loop itself. This section catalogs the values used in the book's training examples.

#### Optimizer Settings

The AdamW optimizer is used with the following settings [c3125, c3126]:

- **Learning rate:** `0.00005` (5 × 10⁻⁵) [c3125]
- **Weight decay:** `0.1` [c3126]

For reference, an Adam optimizer variant is also mentioned with a learning rate of `5e-4` and weight decay of `0.1` [c3569]. Weight decay acts as an L2 regularization term, penalizing large parameter values and helping prevent overfitting.

#### Batch Size and Data Loading

The batch size is set to **8**, meaning each training step processes 8 samples simultaneously [c3107, c2316]. On the data loading side, several additional settings are used:

- **Number of workers:** `0`, which disables parallel data loading [c2392]. This avoids multiprocessing overhead and is the safest default on most platforms.
- **`drop_last`:** `True`, meaning the final batch is discarded if it contains fewer samples than the full batch size [c2393]. This keeps batch shapes consistent throughout training.
- **Shuffle:** `False` for the initial data loader configuration [c3313]. For language modeling, maintaining sequence order can matter depending on how the dataset is constructed.
- **Stride:** `4`, meaning after creating one input sequence, the next sequence starts 4 positions ahead in the text [c3312]. A stride smaller than the context length creates overlapping windows, giving the model more training examples from the same corpus.

#### Dataset Split

The dataset is divided into three splits [c590]:

| Split | Fraction |
|---|---|
| Training | 85% [c590] |
| Test | 10% [c590, c604] |
| Validation | 5% [c590] |

#### Epochs and Evaluation Frequency

The number of training epochs is set to **1** for the basic training run [c3128]. To achieve better fine-tuning results, this should be increased to at least **2 epochs** [c613].

During training, results are printed after every **5 batches** [c3129], and evaluation loss is calculated at the same frequency — every 5 batches [c3131]. This gives you a reasonably granular view of training progress without the overhead of evaluating after every single step.

#### Text Generation Hyperparameters

When generating text from a trained model, two additional hyperparameters come into play:

- **Temperature:** `1.5` [c3557]. Higher temperature makes the output distribution flatter, increasing diversity at the cost of coherence.
- **Top-K sampling:** `K = 50` [c3558]. This restricts token selection to the 50 most likely candidates at each step, preventing the model from sampling from the long tail of very unlikely tokens.

#### Output Embedding Dimension (Auxiliary Tasks)

For certain implementation variants, the output embedding dimension is set to **256** [c3302]. This is distinct from the main model's 768-dimensional token embeddings and applies to specific projection layers in auxiliary tasks.

---

### Quick-Reference Summary

The table below consolidates every hyperparameter discussed in this chapter for the 124M tutorial configuration.

| Category | Hyperparameter | Value |
|---|---|---|
| Architecture | `vocab_size` | 50,257 [c526] |
| Architecture | `context_length` (tutorial) | 256 [c537] |
| Architecture | `context_length` (full) | 1,024 [c3490] |
| Architecture | `emb_dim` | 768 [c531] |
| Architecture | `n_heads` | 12 [c532] |
| Architecture | `n_layers` | 12 [c533] |
| Architecture | `qkv_bias` | False [c535] |
| Regularization | `drop_rate` | 0.1 [c534] |
| Optimizer | Learning rate | 5 × 10⁻⁵ [c3125] |
| Optimizer | Weight decay | 0.1 [c3126] |
| Data | Batch size | 8 [c3107] |
| Data | Stride | 4 [c3312] |
| Data | Train / Test / Val split | 85% / 10% / 5% [c590] |
| Training | Epochs | 1 (min. 2 for fine-tuning) [c3128, c613] |
| Training | Eval frequency | Every 5 batches [c3129] |
| Generation | Temperature | 1.5 [c3557] |
| Generation | Top-K | 50 [c3558] |

Keep this chapter bookmarked. As you work through the implementation chapters, you will frequently need to look up why a particular number was chosen or what a configuration key controls. Every value here has a specific purpose, and understanding that purpose is the first step toward confidently modifying the architecture for your own experiments.
