## Decoding Strategies, Model Persistence, and Loading Pre-Trained GPT-2 Weights

By this point in the book, you have a working GPT model that can generate text. The problem is that the text it generates is probably not very good—either it repeats itself, or it sounds robotic, or it is outright incoherent. This chapter addresses three interconnected topics that move the model from "technically functional" to "practically useful."

First, we look at *decoding strategies*: the algorithms that decide which token to pick at each generation step. The naive approach has real problems, and we will replace it with temperature scaling and top-k sampling. Second, we look at *model persistence*: how to save and reload both model weights and optimizer state so that long training runs are not lost. Third, we put everything together by loading OpenAI's publicly released GPT-2 weights into our custom model class, verifying that the result produces coherent text.

---

### Decoding Strategies: Controlling Randomness in Generation

The simplest possible decoding strategy is *greedy decoding*: at every step, pick the token with the highest probability [c3045]. In PyTorch terms, this is a call to `torch.argmax`. Greedy decoding is deterministic—given the same prompt, you always get the same output—and it tends to produce repetitive, low-diversity text. More importantly for training, a model that always picks the maximum-probability token can effectively memorize passages from its training data [c797].

LLM decoding strategies are techniques used to control randomness and improve quality when generating the next token in sequence [c750]. The naive strategy selects the generated token corresponding to the largest probability score among all tokens in the vocabulary [c751]. 

#### Temperature Scaling

Temperature scaling is a technique used in large language models to control randomness and diversity in generated text [c3025]. 

where $T$ is the temperature parameter. Temperature controls the entropy of the probability distribution over tokens [c3074]. Intuitively, dividing by a small $T$ (say, 0.1) makes the largest logit relatively much larger than the others, sharpening the distribution toward a near-deterministic choice. Dividing by a large $T$ (say, 5.0) compresses the differences between logits, flattening the distribution so that many tokens have similar probability.

Temperature scaling adjusts the sharpness of probability distributions to either sharpen or flatten them, helping prevent overfitting through multinomial sampling [c801].

# Source: [c3061]
```python
next_token_logits_2 = next_token_logits / 0.1
probabilities = softmax(next_token_logits_2)
```

And high-temperature scaling:

# Source: [c3063]
```python
next_token_logits_3 = next_token_logits / 5
probabilities = softmax(next_token_logits_3)
```

A temperature of 0.1 produces a very sharp distribution—almost greedy—while a temperature of 5 produces a very flat one where even low-probability tokens have a reasonable chance of being selected.

Once we have a probability distribution, we need a way to sample from it that is not just "take the maximum." The probability distribution used for sampling the next token is the multinomial probability distribution [c3036].

Sampling from a distribution means drawing values where the outcome is not predetermined; for a multinomial distribution, samples are drawn according to the probability scores of mutually exclusive outcomes [c3037]. The multinomial distribution is used for $k$ mutually exclusive outcomes, each with corresponding probabilities, where $n$ independent trials are conducted to predict outcomes [c3038].

The multinomial function samples the next token proportional to its probability score [c3049]. 

 

```python
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

# Source: [c3040]
```python
torch.multinomial(weights, 5) # ERROR without replacement
# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

```python

# Returns: tensor([2, 1, 1, 1])
```

The first example draws 2 samples without replacement from a weight vector where index 1 has weight 10 and index 2 has weight 3—so index 1 is much more likely to appear first. The second example shows that you cannot draw more samples than there are elements without enabling replacement. The third enables replacement, allowing the same index to be drawn multiple times.

#### Top-K Sampling

Even with temperature scaling and multinomial sampling, there is a risk that very low-probability tokens occasionally get selected, producing nonsensical output. Top-K sampling addresses this by restricting the sampled tokens to the top K most likely tokens and excluding all other tokens [c760, c762].

Top-k sampling is a method that selects only the top k tokens with the highest logits for the next token prediction [c769]. Top-k sampling restricts sampled tokens to the top k most likely tokens, preventing random or low-probability tokens from becoming the next token [c802].

The mechanism is straightforward: find the K highest logits, then replace every other logit with negative infinity. After softmax, negative-infinity logits become zero probability, so they can never be sampled.

PyTorch provides `torch.topk` for finding the top values:

# Source: [c771]
```python
torch.topk(logits, k=3) returns the top 3 values and their indices
```

And `torch.where` for replacing the rest:

# Source: [c773]
```python
torch.where(condition, logits, -inf) replaces logits that don't meet the condition with negative infinity
```

#### Putting It All Together: The Decoding Workflow

The complete decoding workflow combines both strategies [c804]:

1. Apply top-k sampling: keep only the top-k logits.
3. Replace all non-top-k logits with negative infinity.
4. Apply temperature scaling: divide the remaining logits by the temperature $T$.
5. Apply softmax to convert logits to probabilities.
6. Sample from the multinomial distribution to predict the next token.

[FIGURE: Flowchart showing the six-step decoding pipeline: GPT logits → top-k mask → replace with -inf → divide by T → softmax → multinomial sample → next token]

Text generation strategies including temperature scaling and top-k sampling can be integrated together to reduce overfitting in text generation [c724]. Top-k sampling and temperature scaling are decoding strategies that help avoid overfitting [c798]. 

This is also why ChatGPT produces new output each time: it uses decoding strategies like top-k sampling and temperature scaling rather than memorizing user input [c799].

Here is an example call to the `generate` function using both strategies:

# Source: [c3556]
```python
generate(model, input_token_ids=['every', 'effort', 'moves', 'you'], max_new_tokens=25, temperature=1.5, top_k=50)
```

Top-k sampling was introduced as a second decoding strategy to reduce loss and overfitting [c3457].

---

 Loading and saving model weights is especially important when dealing with large models like the large language model built in this series [c719]. Saving and loading model weights helps save memory and time [c720]. Building the foundation for weight saving and loading is also necessary for working with pre-trained weights from OpenAI [c729].

This lecture focuses on learning how to load and save PyTorch model weights [c718].

#### The State Dictionary

The two main PyTorch functions for saving and loading model parameters are `torch.save()` and `model.load_state_dict()` [c730].

`model.state_dict()` is a dictionary mapping each layer of a PyTorch model to its parameter tensors [c731]. Learnable parameters of any `torch.nn.Module` are contained in `model.parameters()` [c733].

[FIGURE: Diagram showing a GPT model with its layers (embedding, transformer blocks, output head) and arrows pointing to key-value pairs in the state_dict]

#### The Optimizer State Dictionary

When resuming training—not just inference—you also need to save the optimizer state. The optimizer state dictionary stores both hyperparameters (such as learning rate and weight decay) and historical data used by the optimizer (such as past gradient values and squared gradient values) [c742].

`optimizer.state_dict()` returns the state of the optimizer as a dictionary containing two entries: `'state'` (a dict holding current optimization state per parameter) and `'param_groups'` (a list containing all parameter groups) [c746].

Saving and loading model and optimizer states is especially important for training large language models to avoid losing progress and having to restart from scratch [c749].

#### Loading a Checkpoint

To resume from a saved checkpoint, you reconstruct both the model and the optimizer, then load their respective state dictionaries. Here is the complete pattern:

# Source: [c747]
```python
checkpoint = torch.load('model_and_optimizer.pth')
model = GPTModel()
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()
```

The key insight here is that `GPTModel()` is instantiated with its architecture (so PyTorch knows the shape of every tensor), and then the saved weights are injected via `load_state_dict`. The optimizer is similarly reconstructed and then populated with its historical state, so momentum and adaptive learning rate estimates are preserved exactly where training left off.

---

### Loading Pre-Trained GPT-2 Weights

We now have all the pieces: a working GPT model class, decoding strategies, and the ability to load weights from disk. The next step is to load OpenAI's publicly released GPT-2 weights into our custom model class and verify that the result generates coherent text [c3453, c726].

The lecture series is building a GPT model from scratch using a custom-defined GPT model class [c3452]. The GPT-2 weights will be loaded into this GPT model class and used for text generation [c3468].

#### GPT-2 Model Sizes

GPT-2 model sizes available include 124M, 355M, 774M, and 1.5B parameters [c3461]. The four variants differ in embedding dimension, number of layers, and number of attention heads:

# Source: [c3533]
```python
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

[FIGURE: Table comparing the four GPT-2 model sizes with their parameter counts, embedding dimensions, layer counts, and head counts]

 N_heads is the number of attention heads present in each transformer block [c3494]. N_layers is the number of transformer blocks in the model [c3495].

#### Downloading the GPT-2 Files

OpenAI originally saved GPT-2 weights using TensorFlow, while the lecture series code uses PyTorch [c3469]. 

# Source: [c3462]
```python
def download_and_load_gpt2(model_size, models_dir):
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

To invoke it:

# Source: [c3479]
```python

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

#### What the Downloaded Files Contain

Each downloaded file serves a specific purpose:

- **checkpoint**: Contains the path where all current parameters and weights of the GPT-2 model are stored [c3481].
- **model.ckpt files**: Store all the weights of the GPT-2 model [c3482].
- **encoder.json**: A vocabulary mapping tokens to their corresponding token IDs [c3483].
- **vocab.bpe**: Contains a list of byte-pair encoded token merges, ordered by merge frequency with the highest probability merges at the top [c3485]. Byte-pair encoding is a sub-word tokenization scheme that merges the most frequently occurring pairs of tokens into single tokens [c3486].
- **hparams.json**: Contains all the hyperparameter settings and configuration values for the GPT-2 model [c3488].

[FIGURE: Directory tree showing the downloaded GPT-2 files and a brief annotation of each file's role]

#### The `settings` and `params` Dictionaries

After calling `download_and_load_gpt2`, you receive two dictionaries. The `settings` dictionary contains the same hyperparameters as those in hparams.json: vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks [c3503].

The `params` dictionary for GPT-2 has five keys [c3505]:

| Key | Contents |
|-----|----------|
| `WTE` | Token embedding parameters [c3506] |
| `WPE` | Positional embedding parameters [c3507] |
| `blocks` | All trainable weights within the transformer blocks [c3511] |
| `G` | Final layer normalization scale [c3505] |
| `B` | Final layer normalization shift [c3505] |

WTE (token embeddings) stores parameters for converting input token IDs into token embeddings [c3506]. 

One important structural difference: in GPT-2, the query, key, and value weight matrices in the attention layer are fused into a single large matrix called `C_ATTN` [c3513]. Our custom model stores them separately, so we need to split this matrix when loading. 

[FIGURE: Diagram showing how the fused C_ATTN matrix in GPT-2 maps to separate Q, K, V matrices in the custom model]

Because OpenAI saved the weights in TensorFlow format, we need a helper function to read them and organize them into a Python dictionary that mirrors our model's structure:

# Source: [c3521]
```python
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
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

This function iterates over every variable in the TensorFlow checkpoint, strips the `model/` prefix from the variable name, and uses the remaining path components to build a nested Python dictionary. Variables whose names start with `h` (e.g., `h0`, `h1`, ...) are routed into the corresponding entry of `params["blocks"]`.

#### Assigning Weights to the Custom Model

Once the `params` dictionary is populated, we assign its values to the corresponding tensors in our PyTorch model. A shape-checking helper ensures we do not silently load mismatched weights:

# Source: [c3544]
```python
def assign(left, right):
 if left.shape != right.shape:
 raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
 return right
```

The GPT model class initializes token embeddings, positional embeddings, transformer blocks, layer normalization, and an output head layer [c3540]. Each of these components receives its weights from the corresponding key in `params`.

The output projection layer weights and biases from GPT-2 are assigned to the attention object's output projection [c3548]. Feed forward neural network weights and biases for both the fully connected layer and projection layer are assigned from downloaded GPT-2 values [c3550].

After all assignments are complete, we can call `download_and_load_gpt2` one more time to confirm the full pipeline:

# Source: [c3524]
```python
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

#### Verifying the Loaded Weights

The proof is in the output. Without GPT-2 weights, the model generates incoherent output [c3561]. With pre-trained GPT-2 weights loaded, inference execution is fast because the model does not require training [c3562]. Output from the pre-trained GPT-2 model is more coherent than output from the untrained model [c3564].

Model weights are loaded correctly when the model produces coherent text [c3571].

Here is an example generation call that exercises both decoding strategies with the loaded model:

# Source: [c3556]
```python
generate(model, input_token_ids=['every', 'effort', 'moves', 'you'], max_new_tokens=25, temperature=1.5, top_k=50)
```

With `temperature=1.5` and `top_k=50`, the model samples from the top 50 most likely tokens at each step, with a relatively high temperature that encourages diversity. The result should be grammatically coherent English that continues the prompt naturally.

---

### Why Build a Custom Model Class?

One might ask: if OpenAI already provides GPT-2, why go through the effort of building a custom model class and loading weights into it? The custom-built GPT model allows exploration of hyperparameter tuning and architectural modifications that are not possible with ChatGPT [c3567]. By owning the model code, you can change the number of layers, swap out the attention mechanism, experiment with different normalization schemes, or fine-tune on domain-specific data—none of which is possible through an API.

---

### Summary

This chapter covered three major topics:

1. **Decoding strategies.** Greedy decoding is deterministic and prone to repetition. Temperature scaling divides logits by a temperature parameter before softmax, sharpening or flattening the probability distribution [c753, c3058]. Multinomial sampling draws the next token proportionally to its probability rather than always taking the maximum [c754, c3049]. Top-k sampling restricts the candidate set to the K most likely tokens, preventing low-probability tokens from ever being selected [c760, c802]. The full decoding pipeline applies these in sequence: top-k masking → temperature scaling → softmax → multinomial sample [c804].

2. **Model persistence.** `model.state_dict()` captures all learnable parameters as a dictionary [c731]. `optimizer.state_dict()` captures both hyperparameters and historical gradient information [c742, c746]. Saving and reloading both is essential for resuming long training runs without losing progress [c749].

3. **Loading pre-trained GPT-2 weights.** OpenAI's GPT-2 comes in four sizes (124M, 355M, 774M, 1558M) [c3461] and was originally saved in TensorFlow format [c3469]. A download-and-parse pipeline converts these weights into a nested Python dictionary, which is then assigned to the corresponding tensors in our custom PyTorch model using a shape-checking `assign` helper [c3544]. The result is a model that generates coherent text immediately, without any additional training [c3562, c3564].
