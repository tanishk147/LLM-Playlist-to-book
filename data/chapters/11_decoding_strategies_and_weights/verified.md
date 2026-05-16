## Decoding Strategies, Model Checkpointing, and Loading Pre-Trained GPT-2 Weights

 

The chapter covers three largely independent but complementary topics:

---

### Decoding Strategies

 The choice of that rule — the *decoding strategy* — has an enormous effect on the quality and diversity of the generated text [c750].

#### Greedy Decoding

The simplest possible strategy is *greedy decoding*: always pick the token with the highest probability [c751]. In PyTorch terms, this is just `torch.argmax` applied to the probability vector [c3045].

 More importantly, greedy decoding causes the model to effectively memorize passages rather than generate fresh text [c797].

#### Multinomial Sampling

 This is called *multinomial sampling* [c754].

The multinomial distribution models $k$ mutually exclusive outcomes, each with a corresponding probability, where $n$ independent trials are conducted to predict outcomes [c3038]. Sampling from such a distribution means drawing values where the outcome is not predetermined; samples are drawn according to the probability scores of the outcomes [c3037]. 

PyTorch's `torch.multinomial` function implements this directly. The following examples illustrate its behavior:

# Source: [c3039]
```python

torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

The weights act as unnormalized probabilities. 

# Source: [c3040]
```python

# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

Enabling replacement allows repeated draws:

# Source: [c3041]
```python

# Returns: tensor([2, 1, 1, 1])
```

In practice, the `generate` function samples *with replacement* across the vocabulary at each step, drawing exactly one token per step. The multinomial function samples the next token proportional to its probability score [c3049].

#### Temperature Scaling

Multinomial sampling introduces diversity, but you often want to control *how much* diversity. 

Temperature scaling is a technique used in large language models to control randomness and diversity in generated text [c3025]. Concretely, it works by dividing the logits by a temperature value before applying softmax to convert them into probabilities [c753]. Temperature is a parameter that controls the entropy of the probability distribution over tokens [c3074].

where $T > 0$ is the temperature [c3058].

 The model becomes more creative (and more random).

Temperature scaling adjusts the sharpness of probability distributions to either sharpen or flatten them, helping prevent overfitting through multinomial sampling [c801].

# Source: [c3061]
```python

probabilities = softmax(next_token_logits_2)
```

# Source: [c3063]
```python

probabilities = softmax(next_token_logits_3)
```

#### Top-K Sampling

Even with temperature scaling, multinomial sampling can occasionally select a very low-probability token — one that makes no linguistic sense in context. *Top-k sampling* addresses this by restricting the candidate set before sampling [c760].

Top-k sampling selects only the top $k$ tokens with the highest logits for the next token prediction [c769]. All other tokens are excluded from consideration entirely [c762]. This prevents random or low-probability tokens from ever becoming the next token [c802].

The implementation relies on two PyTorch operations. First, find the top-$k$ logit values and their indices:

# Source: [c771]
```python

```

Then, replace every logit that did *not* make the top-$k$ cut with negative infinity, so that softmax assigns them zero probability:

# Source: [c773]
```python

```

[FIGURE: Diagram showing the top-k filtering step: a bar chart of logits for all vocabulary tokens, with the top-k bars highlighted and the rest replaced by −∞ before softmax is applied]

#### Combining Top-K Sampling and Temperature Scaling

 Temperature scaling without top-k can still occasionally produce nonsense tokens. Integrating temperature scaling with top-k sampling ensures both creativity in token selection and prevention of random low-probability tokens from being selected [c780].

The complete decoding workflow is [c804]:

 Sample from the multinomial distribution to select the next token.

[FIGURE: Pipeline diagram showing the six-step decoding workflow: GPT logits → top-k filter → −∞ masking → divide by T → softmax → multinomial sample → next token]

Text generation strategies including temperature scaling and top-k sampling can be integrated together to reduce overfitting in text generation [c724]. Probabilistic decoding strategies prevent the GPT model from memorizing passages when predicting next words [c797]. This is precisely why ChatGPT produces new output each time you ask the same question — it uses decoding strategies like top-k sampling and temperature scaling rather than memorizing user input [c799].

#### The `generate` Function

The `generate` function encapsulates this entire pipeline. It takes as parameters a GPT model instance, the maximum number of new tokens to generate, the context size, a temperature value, a top-k value, and an end-of-sequence token ID [c784]. The token generation pipeline applies operations in sequence: top-k filtering, temperature scaling, softmax, and multinomial sampling [c781].

The author uses this new `generate` function with top-k and temperature scaling instead of naive maximum probability decoding to improve token generation quality [c791]. Top-k sampling and temperature scaling are decoding strategies that help avoid overfitting [c798].

A representative call looks like this:

# Source: [c3556]
```python

```

---

### Model Checkpointing: Saving and Loading Weights

Training a large language model takes significant time and compute. Loading and saving model weights is especially important when dealing with large models like the large language model built in this series [c719]. Saving and loading model weights helps save memory and time [c720].

Building the foundation for weight saving and loading is also necessary for working with pre-trained weights from OpenAI [c729].

#### PyTorch Model State

 `model.parameters()` contains all learnable parameters of any `torch.nn.Module` [c733]. `model.state_dict()` is a dictionary mapping each layer of a PyTorch model to its parameter tensors [c731]. 

The two main PyTorch functions for saving and loading model parameters are `torch.save()` and `model.load_state_dict()` [c730].

#### Saving the Optimizer State

 When you resume training, the optimizer also needs to be restored to its previous state. The optimizer state dictionary stores both hyperparameters (such as learning rate and weight decay) and historical data used by the optimizer (such as past gradient values and squared gradient values) [c742].

`optimizer.state_dict()` returns the state of the optimizer as a dictionary containing two entries: `'state'` (a dict holding current optimization state per parameter) and `'param_groups'` (a list containing all parameter groups) [c746]. It is recommended to save the optimizer state in addition to model parameters when checkpointing training [c741].

Saving and loading model and optimizer states is especially important for training large language models to avoid losing progress and having to restart from scratch [c749].

#### Restoring from a Checkpoint

When loading a saved checkpoint, the optimizer state dictionary restores both the optimizer parameters and the history of gradients and squared gradients [c748]. The following code shows the complete restoration workflow:

# Source: [c747]
```python

model = GPTModel()
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()
```

 

---

### Loading Pre-Trained GPT-2 Weights

 OpenAI has publicly released the weights for several sizes of GPT-2, and loading those weights into your custom model gives you a powerful text generator immediately [c3453]. The lecture will integrate the custom GPT model class with publicly released GPT-2 OpenAI weights [c3453]. The GPT-2 weights will be loaded into a GPT model class and used for text generation [c3468].

There is one complication: OpenAI originally saved GPT-2 weights using TensorFlow, while the code in this series uses PyTorch [c3469]. Pre-processing steps are required after downloading GPT-2 weights before they can be integrated with the GPT architecture [c3476].

#### Downloading the GPT-2 Files

The `download_and_load_gpt2` function handles the download. 

# Source: [c3462]
```python

 allowed_sizes = ("124M", "355M", "774M", "1558M")
 if model_size not in allowed_sizes:
 raise ValueError(f"Model size not in {allowed_sizes}")
 model_dir = os.path.join(models_dir, model_size)
 base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
 filenames = ["checkpoint", "encoder.json", "params.json", "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"]
 os.makedirs(model_dir, exist_ok=True)
 for filename in filenames:
 file_url = os.path.join(base_url, model_size, filename)
 file_path = os.path.join(model_dir, filename)
 download_file(file_url, file_path)
```

To download and load the 124M parameter model:

# Source: [c3479]
```python

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

#### What Gets Downloaded

- **`checkpoint`** — contains the path where all current parameters and weights of the GPT-2 model are stored [c3481].
- **`encoder.json`** — a vocabulary mapping tokens to their corresponding token IDs [c3483].
- **`vocab.bpe`** — a list of byte-pair encoded token merges, ordered by merge frequency with the highest probability merges at the top [c3485]. Byte-pair encoding is a sub-word tokenization scheme that merges the most frequently occurring pairs of tokens into single tokens [c3486].
- **`hparams.json`** — contains all the hyperparameter settings and configuration values for the GPT-2 model [c3488].
- **`model.ckpt.*`** — the TensorFlow checkpoint files containing the actual weight tensors.

#### Understanding the `settings` and `params` Dictionaries

`download_and_load_gpt2` returns two dictionaries. The `settings` dictionary contains the same hyperparameters as those in `hparams.json`: vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks [c3503].

- **Embedding dimension** — the size of the vector space into which each token ID is converted; for GPT-2 small, each token is represented as a 768-dimensional vector [c3492].
- **N_heads** — the number of attention heads present in each transformer block [c3494].
- **N_layers** — the number of transformer blocks in the model [c3495].

The `params` dictionary for GPT-2 has five top-level keys [c3505]:

| Key | Contents |
|-----|----------|
| `WTE` | Token embedding parameters — converts input token IDs into token embeddings [c3506] |
| `WPE` | Positional embedding parameters — adds positional information to token embeddings [c3507] |
| `blocks` | All trainable weights within the transformer blocks, including attention layers, feed-forward networks, output projections, and layer normalization parameters [c3511] |
| `G` | Final layer normalization scale parameters [c3505] |
| `B` | Final layer normalization shift parameters [c3505] |

One important structural difference from the custom model: in GPT-2, the query, key, and value weight matrices in the attention layer are fused into a single large matrix called `C_ATTN` [c3513]. Your custom model stores them separately, so the weight-loading code must split `C_ATTN` appropriately.

Layer normalization in GPT-2 includes trainable scale (`G`) and shift (`B`) parameters applied after mean subtraction and variance normalization [c3517].

#### GPT-2 Model Variants

 The following configuration table covers all of them:

# Source: [c3533]
```python

 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
```

The goal is to replace random weight initializations in the GPT model with downloaded GPT-2 parameters from the `params` dictionary [c3541].

#### Loading TensorFlow Weights into PyTorch

 `load_gpt2_params_from_tf_ckpt` iterates over every variable in the TensorFlow checkpoint, extracts its array, and places it into the correct position in a nested Python dictionary that mirrors the structure of the `params` dict:

# Source: [c3521]
```python

 # Initialize parameters dictionary with empty blocks for each layer
 params = {"blocks": [{} for _ in range(settings["n_layer"])]}
 
 # Iterate over each variable in the checkpoint
 for name, _ in tf.train.list_variables(ckpt_path):
 # Load the variable and remove singleton dimensions
 variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
 
 # Process the variable name to extract relevant parts
 variable_name_parts = name.split("/")[1:] # Skip the 'model/' prefix
 
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

 A small helper function guards against this:

# Source: [c3544]
```python

 if left.shape != right.shape:
 raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
 return right
```

 

#### Assigning Feed-Forward Weights

Feed-forward neural network weights and biases for both the fully connected layer and projection layer are assigned from the downloaded GPT-2 values [c3550]. 

[FIGURE: Diagram mapping GPT-2 params dictionary keys (WTE, WPE, blocks[i], G, B) to the corresponding layers in the custom PyTorch GPTModel class]

#### Running Inference with Pre-Trained Weights

 The full download-and-load sequence is:

# Source: [c3524]
```python

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

Without GPT-2 weights, the model generates incoherent output [c3561]. With pre-trained GPT-2 weights loaded, inference execution is fast because the model does not require training [c3562]. Output from the pre-trained GPT-2 model is more coherent than output from the untrained model [c3564].

The next lecture will load pre-trained weights from OpenAI into the model instead of training from scratch [c726].

---

### Why Build Your Own Instead of Using the API?

 The custom-built GPT model allows exploration of hyperparameter tuning and architectural modifications that are not possible with ChatGPT [c3567]. When you own the model code and the weights, you can change the number of layers, the embedding dimension, the attention mechanism, the decoding strategy, or any other component. 

---

### Summary

**Decoding strategies** control how the model selects the next token from its probability distribution. Greedy decoding is simple but leads to repetitive, memorized output [c3045, c797]. Multinomial sampling introduces diversity by drawing from the full probability distribution [c754, c3049]. Temperature scaling sharpens or flattens that distribution by dividing logits by a temperature parameter before softmax [c753, c3058, c3074]. Top-k sampling restricts the candidate pool to the $k$ most likely tokens, preventing low-probability nonsense from being selected [c760, c802]. Combining top-k sampling with temperature scaling gives the best of both worlds: creative but coherent generation [c780, c804].

 PyTorch's `model.state_dict()` captures all learnable parameters [c731], and `optimizer.state_dict()` captures the optimizer's hyperparameters and gradient history [c742, c746]. Saving both together and restoring them with `model.load_state_dict()` and `optimizer.load_state_dict()` allows training to resume seamlessly [c747, c748, c749].

 The weights are stored in TensorFlow format and must be parsed, restructured, and shape-validated before being assigned to your PyTorch model [c3469, c3476, c3521, c3544]. Once loaded, the model generates coherent text immediately, without any training [c3562, c3564].
