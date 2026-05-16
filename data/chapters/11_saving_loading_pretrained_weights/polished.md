## Saving, Loading, and Using Pre-Trained GPT-2 Weights

Building a language model from scratch is only half the battle—knowing how to preserve, restore, and reuse its learned parameters is what makes that work practical. At this point in the book, you have built a GPT model from scratch and trained it on a small dataset [c3452]. The model works, but it has a problem: it was trained on a single small book for 10 epochs [c3455], and as a result it overfit—memorizing the training text and reproducing it in predictions rather than generalizing [c723]. Before addressing that, we need to understand the mechanics of saving and loading model state in PyTorch, because those same mechanics underpin everything that follows.

This chapter covers two related topics. First, we examine PyTorch's checkpointing system: how to save and restore model parameters and optimizer state so that training can be interrupted and resumed without losing progress. Second, we walk through loading OpenAI's publicly released GPT-2 weights into our custom model architecture and using them for text generation.

---

### PyTorch Model Checkpointing

#### Why Bother Saving Weights?

Saving and loading model weights helps save memory and time [c720]. This matters even for small experiments, but it becomes critical when dealing with large models like the large language model built in this series [c719]. The next step in this series is to load pre-trained weights from OpenAI into the model instead of training from scratch [c726], so building the foundation for weight saving and loading is a prerequisite [c729].

#### The Model State Dictionary

The two main PyTorch functions for saving and loading model parameters are `torch.save()` and `model.load_state_dict()` [c730]. To understand how they work, you first need to understand what they operate on.

`model.state_dict()` is a dictionary mapping each layer of a PyTorch model to its parameter tensors [c731]. Learnable parameters of any `torch.nn.Module` are contained in `model.parameters()` [c733].

To save a model's parameters, you pass the state dictionary to `torch.save()`. To restore them, you call `model.load_state_dict()` with the dictionary you previously saved. After loading, the model should be put in evaluation mode using `model.eval()` [c737].

#### The Optimizer State Dictionary

The optimizer also carries state that matters for training continuity. The optimizer state dictionary stores both hyperparameters (such as learning rate and weight decay) and historical data used by the optimizer (such as past gradient values and squared gradient values) [c742]. More specifically, `optimizer.state_dict()` returns the state of the optimizer as a dictionary containing two entries: `'state'` (a dict holding current optimization state per parameter) and `'param_groups'` (a list containing all parameter groups) [c746]. It is therefore recommended to save the optimizer state alongside model parameters when checkpointing training [c741].

When loading a saved checkpoint, the optimizer state dictionary restores both the optimizer parameters and the history of gradients and squared gradients [c748]. Without this, resuming training would effectively reset the optimizer's momentum, which can destabilize training or slow convergence.

#### Saving and Loading a Full Checkpoint

The `.pth` file extension is a convention for PyTorch model files, though technically any file extension can be used [c736].

To restore from a checkpoint, you reconstruct the model and optimizer objects, then load their respective state dictionaries from the saved file. The following code demonstrates the full restoration sequence:

```python
# Source: [c747]
checkpoint = torch.load('model_and_optimizer.pth')
model = GPTModel()
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()
```

If you were loading for inference only, you would call `model.eval()` instead [c737]. Saving and loading model and optimizer states is especially important for training large language models to avoid losing progress and having to restart from scratch [c749].

---

### Loading OpenAI's GPT-2 Weights

#### The Big Picture

The goal of this section is to integrate the custom GPT model class with the publicly released GPT-2 weights from OpenAI [c3453], loading those weights into the model and using it for text generation [c3468].

There is one immediate complication: OpenAI originally saved GPT-2 weights using TensorFlow, while the code in this series uses PyTorch [c3469]. This means we cannot simply call `torch.load()` on the downloaded files—a conversion step is required, and TensorFlow must be installed to perform it [c3470].

Previously, a small GPT-2 model was trained for educational purposes using a limited dataset consisting of a single book [c3465]. The pre-trained GPT-2 weights fix this: with them loaded, the model produces coherent text [c3564], and inference is fast because no training is required [c3562].

#### The Seven Downloaded Files

The GPT-2 weights consist of seven files with a total size of approximately 500 megabytes [c3475, c3527], and downloading them typically takes 5 to 10 minutes at speeds of 2 to 5 MB per second [c3528]. Available model sizes include 124M, 355M, 774M, and 1.5B parameters [c3461], and the weights can also be found on platforms such as Kaggle [c3460].

Each of the seven files serves a distinct purpose:

- **`checkpoint`**: Contains the path where all current parameters and weights of the GPT-2 model are stored [c3481].
- **`model.ckpt.data-00000-of-00001`**, **`model.ckpt.index`**, **`model.ckpt.meta`**: These three files together store all the weights of the GPT-2 model [c3482].
- **`encoder.json`**: A vocabulary mapping tokens to their corresponding token IDs [c3483].
- **`vocab.bpe`**: Contains a list of byte-pair encoded token merges, ordered by merge frequency with the highest-probability merges at the top [c3485]. Byte-pair encoding is a sub-word tokenization scheme that merges the most frequently occurring pairs of tokens into single tokens [c3486].
- **`hparams.json`** (also referred to as `params.json` in the download list): Contains all the hyperparameter settings and configuration values for the GPT-2 model [c3488].

Pre-processing steps are required after downloading these files before they can be integrated with the GPT architecture [c3476].

#### The Download Helper Module

The `GPT_download3.py` file contains three functions: `download_and_load_gpt2` (the main function), `download_file` (a helper), and `load_gpt2_params_from_tf_checkpoint` (a second helper) [c3477]. The main function performs two sequential steps: downloading the seven files to the local machine, then loading the parameters into a dictionary called `params` with a specific format [c3478].

```python
# Source: [c3462]
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

The function validates the requested model size against the four allowed sizes, constructs the download URLs, and iterates through all seven filenames. To invoke it:

```python
# Source: [c3479]
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

This call returns two dictionaries, `settings` and `params`, each examined in detail below.

#### The Settings Dictionary

The `settings` dictionary contains the same hyperparameters as those in `hparams.json`: vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks [c3503]. The values shown are for the smallest GPT-2 model (124 million parameters) [c3496]:

- **Embedding dimension**: The size of the vector space into which each token ID is converted; for GPT-2 small, each token is represented as a 768-dimensional vector [c3492].
- **N_heads**: The number of attention heads present in each transformer block [c3494].
- **N_layers**: The number of transformer blocks in the model [c3495].

[FIGURE: Table showing GPT-2 small hyperparameters: vocab_size, context_length, emb_dim=768, n_heads=12, n_layers=12]

#### The Params Dictionary

The `params` dictionary for GPT-2 has five keys: `WTE` (token embeddings), `WPE` (positional embeddings), `blocks` (transformer block parameters), final normalization scale (`G`), and final normalization shift (`B`) [c3505]. Each key corresponds to a distinct part of the model architecture:

- **`WTE`**: Stores parameters for converting input token IDs into token embeddings [c3506].
- **`WPE`**: Stores parameters for adding positional information to token embeddings [c3507].
- **`blocks`**: Contains all trainable weights and parameters within the transformer blocks, including attention layers, feed-forward networks, output projections, and layer normalization parameters [c3511].
- **`G` and `B`**: The final layer normalization scale and shift parameters [c3505]. Layer normalization in GPT-2 includes trainable scale (`G`) and shift (`B`) parameters applied after mean subtraction and variance normalization [c3517].

[FIGURE: Diagram of the params dictionary tree: WTE, WPE, blocks (list of per-layer dicts), G, B]

One architectural detail worth noting: in GPT-2, the query, key, and value weight matrices in the attention layer are fused into a single large matrix called `C_ATTN` [c3513]. When mapping these weights into our custom model, we need to split `C_ATTN` back into the three separate Q, K, V matrices.

#### Loading Parameters from the TensorFlow Checkpoint

The `tensorflow.train.latest_checkpoint()` function locates the model checkpoint directory and identifies the checkpoint path named `model.CKPT` [c3502]. The `load_gpt2_params_from_tf_checkpoint()` function then takes that checkpoint path and a settings dictionary as inputs, returning a `params` dictionary containing the loaded model parameters [c3504].

Internally, `load_gpt2_params_from_tf_ckpt` loads parameter values from the TensorFlow checkpoint and converts them into the `params` dictionary structure with the five keys described above [c3522]:

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

Variables whose names begin with `h` are routed into the appropriate per-layer block dictionary; all others go into the top-level `params` dictionary. The function ultimately returns both the `settings` dictionary (containing vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks) and the `params` dictionary [c3523].

---

### Mapping GPT-2 Weights into the Custom GPT Model

#### Setting Up the Model Configuration

The goal now is to replace the random weight initializations in the GPT model with the downloaded GPT-2 parameters from the `params` dictionary [c3541]. The GPT model class initializes token embeddings, positional embeddings, transformer blocks, layer normalization, and an output head layer [c3540], so we need to assign downloaded weights to each of these components in turn.

We begin by setting up the model configuration:

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

#### A Shape-Checking Assignment Helper

Before assigning any weights, it is worth introducing a small defensive helper:

```python
# Source: [c3544]
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return right
```

A tiny mistake in the weight loading process would cause the model to fail [c3572], so catching shape mismatches early is valuable.

#### Assigning Weights Layer by Layer

With the helper in place, we can assign weights to each component of the model, following the structure of the `params` dictionary:

- **Attention block weights**: The query, key, and value weight matrices are updated by assigning `Q_W`, `K_W`, and `V_W` from the downloaded GPT-2 parameters [c3546].
- **Output projection**: The output projection layer weights and biases from GPT-2 are assigned to the attention object's output projection [c3548].
- **Feed-forward network**: Weights and biases for both the fully connected layer and the projection layer are assigned from the downloaded GPT-2 values [c3550].
- **Final layer normalization**: The final layer normalization scale and shift values are assigned from the downloaded GPT-2 parameters [c3552].

[FIGURE: Diagram showing the mapping from params dictionary keys (WTE, WPE, blocks[i], G, B) to GPTModel attributes (tok_emb, pos_emb, trf_blocks[i], final_norm)]

#### Weight Tying

One important detail in the assignment process is weight tying: GPT-2 reuses the token embedding weights for the output head layer instead of defining separate weights [c3553]. This reduces the total parameter count from 164 million to 124 million [c3554]. Intuitively, the embedding matrix encodes a relationship between tokens and a vector space, and the output projection decodes that same relationship in reverse—sharing the weights enforces consistency between the two operations while significantly reducing the parameter count.

In practice, this means we do not assign separate weights to the output head; we simply point it at the same tensor as the token embedding layer.

#### Downloading and Loading in Practice

```python
# Source: [c3524]
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

The weight assignment functions then iterate through the model's layers and call `assign()` to copy each tensor from `params` into the corresponding attribute of the `GPTModel` instance. Model weights are loaded correctly when the model produces coherent text [c3571].

---

### Text Generation with Pre-Trained Weights

#### Running Inference

With pre-trained GPT-2 weights loaded, inference is fast because the model does not require any training [c3562]. Temperature scaling and top-k sampling can be combined to improve generation quality and reduce repetition [c724]; top-k sampling was introduced as a decoding strategy specifically to reduce loss and overfitting [c3457].

```python
# Source: [c3556]
generate(model, input_token_ids=['every', 'effort', 'moves', 'you'], max_new_tokens=25, temperature=1.5, top_k=50)
```

Top-K sampling with K=50 restricts token selection to the 50 most likely candidates at each step [c3558].

#### Comparing Trained vs. Untrained Output

Without GPT-2 weights, the model generates incoherent output [c3561]. With the pre-trained weights loaded, the output is markedly more coherent [c3564]—a direct consequence of the knowledge encoded in GPT-2's weights rather than anything learned during our limited training run.

#### Fine-Tuning Possibilities

Because we are working with a custom-built GPT model, we retain full control over hyperparameter tuning and architectural modifications that would not be possible with a closed system like ChatGPT [c3567]. For reference, when fine-tuning from the pre-trained weights, the Adam optimizer is used with a learning rate of 5e-4 and weight decay of 0.1 [c3569].

---

### Summary

This chapter covered two complementary skills. The first is PyTorch checkpointing: using `model.state_dict()` and `optimizer.state_dict()` to save and restore the full training state, enabling long training runs to be interrupted and resumed without losing progress [c720, c749]. The second is loading externally trained weights—specifically OpenAI's GPT-2—into a custom PyTorch model architecture.

The key technical challenges along the way were:

1. **Format mismatch**: GPT-2 weights are stored in TensorFlow format; our model uses PyTorch [c3469]. The `load_gpt2_params_from_tf_ckpt` function bridges this gap [c3522].
2. **Fused attention matrices**: GPT-2 stores Q, K, and V as a single `C_ATTN` matrix [c3513], which must be split before assignment.
3. **Weight tying**: The token embedding and output head share weights [c3553], reducing the parameter count from 164M to 124M [c3554].
4. **Shape verification**: The `assign()` helper catches mismatches early [c3544], because even a tiny mistake in the weight loading process would cause the model to fail [c3572].

The payoff is immediate: a model that generates coherent text [c3564] without any training, simply by inheriting the knowledge encoded in GPT-2's weights.