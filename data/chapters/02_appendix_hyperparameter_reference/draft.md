## GPT-2 Model Configurations and Key Hyperparameters

Before writing a single line of model code, it pays to have a clear map of the territory: what are the exact numbers that define a GPT-2 model, and which training knobs will we be turning throughout this book? This chapter answers both questions. Think of it as a reference you can flip back to whenever you need to double-check a dimension, a layer count, or a learning rate.

### The GPT-2 Model Family

GPT-2 was released in four sizes. Each size shares the same fundamental architecture—transformer blocks stacked on top of a token embedding layer—but differs in how wide and how deep that stack is. Understanding the relationship between these sizes gives you intuition for how scaling works in practice.

The smallest member of the family is **GPT-2 small**, which has 124 million parameters [c2841]. It uses an embedding dimension of 768, 12 transformer layers, and 12 attention heads [c3536]. Its context length—the maximum number of tokens the model can attend to at once—is 1024 [c3490]. This is the model we will build from scratch in the early chapters of this book, and the hyperparameter values shown throughout those chapters correspond to this configuration [c3496].

Moving up in scale, **GPT-2 medium** has 345 million parameters [c3434] (sometimes cited as 355 million [c1748, c3537]). It widens and deepens the architecture considerably: the embedding dimension grows to 1024, the number of layers doubles to 24, and the number of attention heads increases to 16 [c3537]. Its context size remains 1024 tokens [c1737]. As a general pattern, as GPT-2 model complexity increases, the number of transformer blocks also increases [c3499].

**GPT-2 large** reaches 774 million parameters with an embedding dimension of 1280, 36 layers, and 20 attention heads [c3538]. Finally, **GPT-2 XL** tops out at 1558 million parameters, using an embedding dimension of 1600, 48 layers, and 25 attention heads [c3539].

The table below summarizes the full family at a glance.

[FIGURE: Table with five columns—Model, Parameters, Embedding Dim, Layers, Attention Heads, Context Length—and four rows for GPT-2 small (124M, 768, 12, 12, 1024), GPT-2 medium (345M, 1024, 24, 16, 1024), GPT-2 large (774M, 1280, 36, 20, 1024), GPT-2 XL (1558M, 1600, 48, 25, 1024)]

### The Baseline Configuration Dictionary

Throughout this book we represent a model's configuration as a plain Python dictionary. Here is the canonical configuration for GPT-2 small (124M) that we will use as our starting point:

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

Every key in this dictionary corresponds to a meaningful architectural choice. We will walk through each one in turn.

### Dissecting the Configuration Keys

#### `vocab_size`

The vocabulary size is 50257 tokens [c526, c529]. This is the number of unique tokens that the model's embedding layer must handle—one learned vector per token. Intuitively, this is the size of the model's "alphabet," and every input token ID must fall within the range `[0, vocab_size - 1]`.

#### `context_length`

The context length is 1024 for the full GPT-2 small model [c3490]. This is the maximum sequence length the model can process in a single forward pass; tokens beyond this limit simply cannot be attended to. The full specification of a GPT model configuration requires this value to be set explicitly [c3847].

For the hands-on tutorials in this book, we reduce the context length to 256 tokens [c537, c1671]. This reduction is deliberate: it cuts memory and compute requirements enough that the training examples can run on a laptop [c537]. Functionally the model is identical; only the maximum sequence length changes [c494].

#### `emb_dim`

The embedding dimension is 768 [c531, c1673]. This is the size of the vector space into which each token ID is projected; for GPT-2 small, every token is represented as a 768-dimensional vector [c3492]. The same dimensionality flows through every subsequent layer—attention, feed-forward, and layer normalization—so this single number has a large influence on the model's total parameter count.

#### `n_heads`

`n_heads` is the number of attention heads present in each transformer block [c3494]. For GPT-2 small this is 12 [c532, c1674]. Multi-head attention splits the embedding dimension across heads, so each head operates on a $768 / 12 = 64$-dimensional subspace. Intuitively, different heads can learn to attend to different kinds of relationships in the input simultaneously.

#### `n_layers`

`n_layers` is the number of transformer blocks stacked in the model [c3495]. GPT-2 small uses 12 [c533, c888, c1675]. Each block contains its own multi-head self-attention sublayer and feed-forward sublayer, so doubling `n_layers` roughly doubles the depth—and a significant portion of the parameter count—of the network.

#### `drop_rate`

The dropout rate is 0.1 in the standard GPT-2 small configuration [c534, c1676]. Dropout is a regularization technique; setting it to 0.1 means that during training, 10% of activations are randomly zeroed out at each dropout layer. When experimenting or debugging, it is common to set this to zero [c498], which disables regularization and makes the forward pass fully deterministic.

#### `qkv_bias`

The query, key, and value weight matrices are initialized without bias terms—`qkv_bias` is `False` [c499, c535, c1677]. This is a deliberate design decision that matches the original GPT-2 implementation. Omitting these biases slightly reduces the parameter count and has been found to work well in practice.

### Variations Across Model Sizes

It is worth noting that not all GPT-2 configurations use exactly the same defaults. When loading pretrained GPT-2 weights (as opposed to training from scratch), the base configuration sets the dropout rate to 0.0 [c1741] and sets `qkv_bias` to `True` [c1742]. These differences matter when you are trying to match a pretrained checkpoint exactly. For instruction fine-tuning experiments later in the book, GPT-2 medium (355M) is preferred over the 124M model because the smaller model does not perform well on instruction-following tasks [c1748].

### Training Hyperparameters

Beyond the model architecture, several training hyperparameters appear repeatedly in the code examples throughout this book. They are collected here for easy reference.

**Optimizer settings.** We use the AdamW optimizer with a learning rate of `0.00005` (5e-5) [c3125] and a weight decay of `0.1` [c3126]. In some earlier experiments the Adam optimizer is used with a learning rate of `5e-4` and the same weight decay of `0.1` [c3569].

**Training duration.** For the pretraining demonstrations, the number of epochs is set to 1 [c3128]. This is intentionally short—the goal is to verify that the training loop works correctly, not to produce a fully converged model.

**Evaluation cadence.** Results are printed after every 5 batches [c3129], and the evaluation loss is calculated over 5 batches at each evaluation step [c3131]. This gives a reasonable signal about training progress without spending too much time on evaluation.

**Data loader settings.** The training data loader is created with a batch size of 2, a maximum sequence length of 256 (matching the reduced context length), a stride of 256, `drop_last=True`, `shuffle=True`, and `num_workers=0` [c3834]. The validation data loader uses the same batch size and sequence length but without shuffling [c3836].

**Text generation settings.** When generating text to inspect model quality, a temperature of 1.5 is used [c3557], and top-K sampling with $K = 50$ restricts token selection to the 50 most likely candidates at each step [c3558].

### Summary

The table below consolidates the key hyperparameters used in this book's tutorial setting alongside the full GPT-2 small specification for comparison.

[FIGURE: Two-column comparison table: left column "Full GPT-2 Small", right column "Tutorial Setting". Rows: vocab_size (50257 / 50257), context_length (1024 / 256), emb_dim (768 / 768), n_heads (12 / 12), n_layers (12 / 12), drop_rate (0.1 / 0.0 or 0.1), qkv_bias (False / False), learning rate (5e-5 / 5e-5), weight decay (0.1 / 0.1), batch size (— / 2), epochs (— / 1)]

With these numbers firmly in hand, we are ready to start building the model itself. Every architectural choice made in the coming chapters traces back to one of the values defined here.