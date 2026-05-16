## Decoding Strategies, Model Checkpointing, and Loading Pre-Trained GPT-2 Weights

Every time a language model produces a word, it must follow some rule for choosing which token comes next. The choice of that rule — the *decoding strategy* — has an enormous effect on the quality and diversity of the generated text [c750]. This chapter covers three complementary topics: the decoding strategies that control text generation, the checkpointing machinery that saves and restores training state, and the process of loading OpenAI's publicly released GPT-2 weights into a custom PyTorch model.

---

### Decoding Strategies

#### Greedy Decoding

The simplest possible strategy is *greedy decoding*: always pick the token with the highest probability [c751]. In PyTorch terms, this is just `torch.argmax` applied to the probability vector [c3045]. While straightforward, greedy decoding causes the model to effectively memorize passages rather than generate fresh text [c797].

#### Multinomial Sampling

A more flexible approach draws the next token *probabilistically*, selecting it in proportion to its probability score rather than always taking the maximum — a technique called *multinomial sampling* [c754].

The multinomial distribution models $k$ mutually exclusive outcomes, each with a corresponding probability, across $n$ independent trials [c3038]. Sampling from such a distribution means drawing values where the outcome is not predetermined; each draw follows the probability scores of the outcomes [c3037].

PyTorch's `torch.multinomial` function implements this directly. The following examples illustrate its behavior:

```python
# Source: [c3039]
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

The weights act as unnormalized probabilities. Attempting to draw more samples than there are non-zero entries without replacement raises an error:

```python
# Source: [c3040]
torch.multinomial(weights, 5)  # ERROR without replacement
# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

Enabling replacement allows repeated draws:

```python
# Source: [c3041]
torch.multinomial(weights, 4, replacement=True)
# Returns: tensor([2, 1, 1, 1])
```

In practice, the `generate` function samples *with replacement* across the vocabulary at each step, drawing exactly one token per step, with each token selected proportional to its probability score [c3049].

#### Temperature Scaling

Multinomial sampling introduces diversity, but you often want to control *how much* diversity. Temperature scaling is a technique used in large language models for precisely this purpose: controlling randomness and diversity in generated text [c3025]. It works by dividing the logits by a temperature value $T > 0$ before applying softmax to convert them into probabilities [c753, c3058]. Temperature is therefore a parameter that controls the entropy of the probability distribution over tokens — lower values sharpen the distribution toward the most likely tokens, while higher values flatten it [c3074].

Temperature scaling adjusts this sharpness to either focus or spread probability mass, helping prevent overfitting through multinomial sampling [c801]. The effect is visible in practice:

```python
# Source: [c3061]
next_token_logits_2 = next_token_logits / 0.1
probabilities = softmax(next_token_logits_2)
```

```python
# Source: [c3063]
next_token_logits_3 = next_token_logits / 5
probabilities = softmax(next_token_logits_3)
```

A low temperature (e.g., 0.1) concentrates probability on the top tokens; a high temperature (e.g., 5) spreads it more evenly, making the model more creative — and more random.

#### Top-K Sampling

Even with temperature scaling, multinomial sampling can occasionally select a very low-probability token that makes no linguistic sense in context. *Top-k sampling* addresses this by restricting the candidate set before sampling [c760].

Top-k sampling selects only the $k$ tokens with the highest logits for the next token prediction [c769], excluding all others from consideration entirely [c762]. This prevents random or low-probability tokens from ever becoming the next token [c802].

The implementation relies on two PyTorch operations. First, find the top-$k$ logit values and their indices:

```python
# Source: [c771]
torch.topk(logits, k=3)  # returns the top 3 values and their indices
```

Then replace every logit that did *not* make the top-$k$ cut with negative infinity, so that softmax assigns those positions zero probability:

```python
# Source: [c773]
torch.where(condition, logits, -inf)  # replaces non-top-k logits with -inf
```

[FIGURE: Diagram showing the top-k filtering step: a bar chart of logits for all vocabulary tokens, with the top-k bars highlighted and the rest replaced by −∞ before softmax is applied]

#### Combining Top-K Sampling and Temperature Scaling

Top-k sampling and temperature scaling are most powerful when used together. Temperature scaling alone can still occasionally produce nonsense tokens; top-k sampling alone does not control the sharpness of the distribution within the candidate set. Integrating the two ensures both creativity in token selection and prevention of random low-probability tokens from being selected [c780].

The complete decoding workflow is [c804]:

1. Obtain the logit tensor from the GPT model.
2. Apply top-k filtering.
3. Replace non-top-k logits with negative infinity.
4. Divide the remaining logits by the temperature $T$.
5. Apply softmax to obtain probabilities.
6. Sample from the multinomial distribution to select the next token.

[FIGURE: Pipeline diagram showing the six-step decoding workflow: GPT logits → top-k filter → −∞ masking → divide by T → softmax → multinomial sample → next token]

Combining these strategies reduces overfitting in text generation [c724] and prevents the model from memorizing passages when predicting next words [c797]. This is precisely why ChatGPT produces different output each time you ask the same question — it relies on decoding strategies like top-k sampling and temperature scaling rather than memorizing user input [c799].

#### The `generate` Function

The `generate` function encapsulates this entire pipeline. It takes as parameters a GPT model instance, the maximum number of new tokens to generate, the context size, a temperature value, a top-k value, and an end-of-sequence token ID [c784]. Internally, it applies operations in sequence: top-k filtering, temperature scaling, softmax, and multinomial sampling [c781]. This replaces the naive maximum-probability decoding used in earlier versions, improving both quality and diversity [c791, c798].

A representative call looks like this:

```python
# Source: [c3556]
generate(model, input_token_ids=['every', 'effort', 'moves', 'you'],
         max_new_tokens=25, temperature=1.5, top_k=50)
```

---

### Model Checkpointing: Saving and Loading Weights

Training a large language model takes significant time and compute, making it essential to save progress periodically. Loading and saving model weights is especially important when dealing with large models [c719], and doing so correctly saves both memory and time [c720]. Establishing this checkpointing foundation is also a prerequisite for working with pre-trained weights from OpenAI [c729].

#### PyTorch Model State

Every `torch.nn.Module` exposes its learnable parameters through `model.parameters()` [c733]. For saving and restoring, the more useful interface is `model.state_dict()`, which returns a dictionary mapping each layer to its parameter tensors [c731]. The two main PyTorch functions for saving and loading these parameters are `torch.save()` and `model.load_state_dict()` [c730].

#### Saving the Optimizer State

Saving model parameters alone is not sufficient if you intend to resume training, because the optimizer must also be restored to its previous state. The optimizer state dictionary stores both hyperparameters (such as learning rate and weight decay) and historical data used by the optimizer (such as past gradient values and squared gradient values) [c742]. `optimizer.state_dict()` returns this state as a dictionary with two entries: `'state'` (holding the current optimization state per parameter) and `'param_groups'` (a list of all parameter groups) [c746].

It is therefore recommended to save the optimizer state alongside the model parameters when checkpointing [c741]. For large language models in particular, losing this state means having to restart training from scratch — a costly outcome [c749].

#### Restoring from a Checkpoint

When loading a saved checkpoint, `optimizer.load_state_dict()` restores both the optimizer's hyperparameters and its history of gradients and squared gradients [c748]. The complete restoration workflow is:

```python
# Source: [c747]
checkpoint = torch.load('model_and_optimizer.pth')
model = GPTModel()
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()
```

---

### Loading Pre-Trained GPT-2 Weights

Rather than training a model from scratch, you can load OpenAI's publicly released GPT-2 weights directly into the custom GPT model class and use it for text generation immediately [c3453, c3468]. There is one complication: OpenAI originally saved GPT-2 weights using TensorFlow, while this series uses PyTorch [c3469]. Pre-processing steps are therefore required after downloading the weights before they can be integrated with the GPT architecture [c3476].

#### Downloading the GPT-2 Files

The `download_and_load_gpt2` function handles the download:

```python
# Source: [c3462]
def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = ["checkpoint", "encoder.json", "params.json",
                 "model.ckpt.data-00000-of-00001", "model.ckpt.index",
                 "model.ckpt.meta", "vocab.bpe"]
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)
```

To download and load the 124M-parameter model:

```python
# Source: [c3479]
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

#### What Gets Downloaded

- **`checkpoint`** — contains the path where all current parameters and weights of the GPT-2 model are stored [c3481].
- **`encoder.json`** — a vocabulary mapping tokens to their corresponding token IDs [c3483].
- **`vocab.bpe`** — a list of byte-pair encoded token merges, ordered by merge frequency with the highest-probability merges at the top [c3485]. Byte-pair encoding is a sub-word tokenization scheme that merges the most frequently occurring pairs of tokens into single tokens [c3486].
- **`hparams.json`** — contains all hyperparameter settings and configuration values for the GPT-2 model [c3488].
- **`model.ckpt.*`** — the TensorFlow checkpoint files containing the actual weight tensors.

#### Understanding the `settings` and `params` Dictionaries

`download_and_load_gpt2` returns two dictionaries. The `settings` dictionary contains the same hyperparameters as `hparams.json`: vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks [c3503]. Key fields include:

- **Embedding dimension** — the size of the vector space into which each token ID is converted; for GPT-2 small, each token is represented as a 768-dimensional vector [c3492].
- **N_heads** — the number of attention heads in each transformer block [c3494].
- **N_layers** — the number of transformer blocks in the model [c3495].

The `params` dictionary has five top-level keys [c3505]:

| Key | Contents |
|-----|----------|
| `WTE` | Token embedding parameters — converts input token IDs into token embeddings [c3506] |
| `WPE` | Positional embedding parameters — adds positional information to token embeddings [c3507] |
| `blocks` | All trainable weights within the transformer blocks, including attention layers, feed-forward networks, output projections, and layer normalization parameters [c3511] |
| `G` | Final layer normalization scale parameters [c3505] |
| `B` | Final layer normalization shift parameters [c3505] |

One important structural difference from the custom model: in GPT-2, the query, key, and value weight matrices in the attention layer are fused into a single large matrix called `C_ATTN` [c3513], whereas the custom model stores them separately. The weight-loading code must therefore split `C_ATTN` appropriately. Layer normalization in GPT-2 also includes trainable scale (`G`) and shift (`B`) parameters applied after mean subtraction and variance normalization [c3517].

#### GPT-2 Model Variants

OpenAI released GPT-2 in four sizes. The following configuration covers all of them:

```python
# Source: [c3533]
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
```

The goal is to replace the random weight initializations in the GPT model with the downloaded GPT-2 parameters from the `params` dictionary [c3541].

#### Loading TensorFlow Weights into PyTorch

`load_gpt2_params_from_tf_ckpt` iterates over every variable in the TensorFlow checkpoint, extracts its array, and places it into the correct position in a nested Python dictionary that mirrors the structure of the `params` dict:

```python
# Source: [c3521]
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
```

#### Shape Validation During Assignment

To catch mismatches between the downloaded weights and the custom model's parameter shapes, a small helper function validates shapes before assignment:

```python
# Source: [c3544]
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return right
```

#### Assigning Feed-Forward Weights

Feed-forward neural network weights and biases for both the fully connected layer and the projection layer are assigned from the downloaded GPT-2 values [c3550].

[FIGURE: Diagram mapping GPT-2 params dictionary keys (WTE, WPE, blocks[i], G, B) to the corresponding layers in the custom PyTorch GPTModel class]

#### Running Inference with Pre-Trained Weights

The full download-and-load sequence is:

```python
# Source: [c3524]
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

Without GPT-2 weights, the model generates incoherent output [c3561]. Once the pre-trained weights are loaded, inference is fast because no training is required [c3562], and the output is immediately more coherent than that of the untrained model [c3564]. The next step is to use these pre-trained weights directly rather than training from scratch [c726].

---

### Why Build Your Own Instead of Using the API?

Owning the model code and the weights opens up possibilities that a hosted API cannot offer. The custom-built GPT model allows exploration of hyperparameter tuning and architectural modifications — changing the number of layers, the embedding dimension, the attention mechanism, the decoding strategy, or any other component — that are simply not possible with ChatGPT [c3567].

---

### Summary

**Decoding strategies** control how the model selects the next token from its probability distribution. Greedy decoding is simple but leads to repetitive, memorized output [c3045, c797]. Multinomial sampling introduces diversity by drawing from the full probability distribution [c754, c3049]. Temperature scaling sharpens or flattens that distribution by dividing logits by a temperature parameter before softmax [c753, c3058, c3074]. Top-k sampling restricts the candidate pool to the $k$ most likely tokens, preventing low-probability nonsense from being selected [c760, c802]. Combining the two gives creative yet coherent generation [c780, c804].

**Model checkpointing** preserves training state so that work is never lost. PyTorch's `model.state_dict()` captures all learnable parameters [c731], and `optimizer.state_dict()` captures the optimizer's hyperparameters and gradient history [c742, c746]. Saving both together and restoring them with `model.load_state_dict()` and `optimizer.load_state_dict()` allows training to resume seamlessly [c747, c748, c749].

**Loading pre-trained GPT-2 weights** bridges the gap between a randomly initialized model and a capable text generator. The weights are stored in TensorFlow format and must be parsed, restructured, and shape-validated before being assigned to the PyTorch model [c3469, c3476, c3521, c3544]. Once loaded, the model generates coherent text immediately, without any additional training [c3562, c3564].