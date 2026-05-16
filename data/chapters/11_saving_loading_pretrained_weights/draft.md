## Saving, Loading, and Using Pre-Trained GPT-2 Weights

At this point in the book, you have built a GPT model from scratch and trained it on a small dataset [c3452]. The model works, but it has a problem: it was trained on a single small book for 10 epochs [c3455], and as a result it overfit—memorizing the training text and reproducing it in predictions rather than generalizing [c723]. The solution is to stop training from scratch and instead load the weights that OpenAI has already released publicly for GPT-2. Before we get there, though, we need to understand the mechanics of saving and loading model state in PyTorch, because those same mechanics underpin everything that follows.

This chapter covers two related topics. First, we look at PyTorch's checkpointing system: how to save and restore model parameters and optimizer state so that training can be interrupted and resumed without losing progress. Second, we walk through the full pipeline for downloading OpenAI's GPT-2 weights, parsing the seven files they ship, mapping those weights into our custom `GPTModel` class, and finally running inference to confirm that the loaded model produces coherent text.

---

### PyTorch Model Checkpointing

#### Why Bother Saving Weights?

Saving and loading model weights helps save memory and time [c720]. This matters even for small experiments, but it becomes critical when dealing with large models like the large language model built in this series [c719]. If training is interrupted—by a crash, a timeout on a cloud instance, or simply because you want to pause and resume—you need to be able to restore the exact state of both the model and the optimizer. The next step in this series is to load pre-trained weights from OpenAI into the model instead of training from scratch [c726], so building the foundation for weight saving and loading is a prerequisite [c729].

#### The Model State Dictionary

The two main PyTorch functions for saving and loading model parameters are `torch.save()` and `model.load_state_dict()` [c730]. To understand how they work, you first need to understand what they operate on.

`model.state_dict()` is a dictionary mapping each layer of a PyTorch model to its parameter tensors [c731]. Learnable parameters of any `torch.nn.Module` are contained in `model.parameters()` [c733]. The state dictionary is essentially a flat snapshot of every weight matrix and bias vector in the model, keyed by the layer name.

To save a model's parameters, you pass the state dictionary to `torch.save()`. To restore them, you call `model.load_state_dict()` with the dictionary you previously saved. After loading, the model should be put in evaluation mode using `model.eval()` [c737]. This is important because some layers (like dropout and batch normalization) behave differently during training versus inference.

#### The Optimizer State Dictionary

Saving the model parameters alone is not always sufficient. The optimizer also carries state that matters for training continuity. The optimizer state dictionary stores both hyperparameters (such as learning rate and weight decay) and historical data used by the optimizer (such as past gradient values and squared gradient values) [c742].

More specifically, `optimizer.state_dict()` returns the state of the optimizer as a dictionary containing two entries: `'state'` (a dict holding current optimization state per parameter) and `'param_groups'` (a list containing all parameter groups) [c746]. For an optimizer like AdamW, the historical gradient and squared-gradient values are what allow the adaptive learning rate to work correctly when training resumes. It is recommended to save the optimizer state in addition to model parameters when checkpointing training [c741].

When loading a saved checkpoint, the optimizer state dictionary restores both the optimizer parameters and the history of gradients and squared gradients [c748]. Without this, resuming training would effectively reset the optimizer's momentum, which can destabilize training or slow convergence.

#### Saving and Loading a Full Checkpoint

The standard pattern is to bundle the model state dictionary and the optimizer state dictionary into a single dictionary and save it as one file. The `.pth` file extension is a convention for PyTorch model files, though technically any file extension can be used [c736].

To restore from such a checkpoint, you reconstruct the model and optimizer objects, then load their respective state dictionaries from the saved file. The following code demonstrates the full restoration sequence:

# Source: [c747]
```python
checkpoint = torch.load('model_and_optimizer.pth')
model = GPTModel()
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()
```

Notice that `model.train()` is called at the end here because the intent is to resume training. If you were loading for inference only, you would call `model.eval()` instead [c737].

Saving and loading model and optimizer states is especially important for training large language models to avoid losing progress and having to restart from scratch [c749]. With the checkpointing mechanics established, we can now turn to the more interesting problem: loading weights that someone else has already trained.

---

### Loading OpenAI's GPT-2 Weights

#### The Big Picture

The lecture series is building a GPT model from scratch using a custom-defined GPT model class [c3452], and the goal of this section is to integrate that custom class with the publicly released GPT-2 OpenAI weights [c3453]. The GPT-2 weights will be loaded into the GPT model class and used for text generation [c3468].

There is one immediate complication: OpenAI originally saved GPT-2 weights using TensorFlow, while the lecture series code uses PyTorch [c3469]. This means we cannot simply call `torch.load()` on the downloaded files. We need a conversion step. TensorFlow must be installed to load the GPT-2 weights, which were originally saved in TensorFlow format [c3470].

Previously, a small GPT-2 model was trained for educational purposes using a limited dataset consisting of a single book [c3465]. The output was incoherent because the model had not seen enough data. The pre-trained GPT-2 weights fix this: with them loaded, the model produces coherent text [c3564], and inference execution is fast because the model does not require training [c3562].

#### The Seven Downloaded Files

The GPT-2 weights consist of seven files with a total size of approximately 500 megabytes [c3475, c3527]. Downloading them typically takes 5 to 10 minutes with download speeds of 2 to 5 MB per second [c3528]. GPT-2 model sizes available include 124M, 355M, 774M, and 1.5B parameters [c3461], and the weights are also available on platforms such as Kaggle [c3460].

The GPT-2 model download process involves downloading seven files from an API to the local machine [c3500]. Let's look at what each file contains, because understanding the file structure is essential for parsing them correctly.

- **`checkpoint`**: Contains the path where all current parameters and weights of the GPT-2 model are stored [c3481].
- **`model.ckpt.data-00000-of-00001`**, **`model.ckpt.index`**, **`model.ckpt.meta`**: These three files together store all the weights of the GPT-2 model [c3482].
- **`encoder.json`**: A vocabulary mapping tokens to their corresponding token IDs [c3483].
- **`vocab.bpe`**: Contains a list of byte-pair encoded token merges, ordered by merge frequency with the highest probability merges at the top [c3485]. Byte-pair encoding is a sub-word tokenization scheme that merges the most frequently occurring pairs of tokens into single tokens [c3486].
- **`hparams.json`** (also referred to as `params.json` in the download list): Contains all the hyperparameter settings and configuration values for the GPT-2 model [c3488].

Pre-processing steps are required after downloading GPT-2 weights before they can be integrated with the GPT architecture [c3476].

#### The Download Helper Module

The `GPT_download3.py` file contains three functions: `download_and_load_gpt2` (the main function), `download_file` (a helper function), and `load_gpt2_params_from_tf_checkpoint` (a helper function) [c3477].

The `download_and_load_gpt2` function performs two sequential steps: first downloading the seven files to the local computer, then loading the parameters from the downloaded files into a dictionary called `params` with a specific format [c3478]. Here is the implementation:

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

The function validates the requested model size against the four allowed sizes, constructs the download URLs, and iterates through all seven filenames. To invoke it:

# Source: [c3479]
```python
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

This call returns two dictionaries: `settings` and `params`. We will examine both in detail.

#### The Settings Dictionary

The `settings` dictionary contains the same hyperparameters as those in `hparams.json`: vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks [c3503]. The hyperparameter values shown are for the smallest GPT-2 model (124 million parameters) [c3496].

To understand what these values mean:

- **Embedding dimension**: The size of the vector space into which each token ID is converted. For GPT-2 small, each token is represented as a 768-dimensional vector [c3492].
- **N_heads**: The number of attention heads present in each transformer block [c3494].
- **N_layers**: The number of transformer blocks in the model [c3495].

[FIGURE: Table showing GPT-2 small hyperparameters: vocab_size, context_length, emb_dim=768, n_heads=12, n_layers=12]

#### The Params Dictionary

The `params` dictionary for GPT-2 has five keys: `WTE` (token embeddings), `WPE` (positional embeddings), `blocks` (transformer block parameters), final normalization scale (`G`), and final normalization shift (`B`) [c3505].

Each key corresponds to a distinct part of the model architecture:

- **`WTE`** (token embeddings): Stores parameters for converting input token IDs into token embeddings [c3506].
- **`WPE`** (positional embeddings): Stores parameters for adding positional information to token embeddings [c3507].
- **`blocks`**: Contains all trainable weights and parameters within the transformer blocks, including attention layers, feed-forward networks, output projections, and layer normalization parameters [c3511].
- **`G` and `B`**: The final layer normalization scale and shift parameters [c3505]. Layer normalization in GPT-2 includes trainable scale (`G`) and shift (`B`) parameters applied after mean subtraction and variance normalization [c3517].

[FIGURE: Diagram of the params dictionary tree: WTE, WPE, blocks (list of per-layer dicts), G, B]

One architectural detail worth noting: in GPT-2, the query, key, and value weight matrices in the attention layer are fused into a single large matrix called `C_ATTN` [c3513]. This is a common implementation optimization—instead of three separate matrix multiplications, a single larger multiplication is performed and the result is split. When we map these weights into our custom model, we need to split `C_ATTN` back into the three separate Q, K, V matrices.

#### Loading Parameters from the TensorFlow Checkpoint

The `tensorflow.train.latest_checkpoint()` function locates the model checkpoint directory and identifies the model checkpoint path named `model.CKPT` [c3502]. The `load_gpt2_params_from_tf_checkpoint()` function takes a TensorFlow checkpoint path and a settings dictionary as inputs and returns a `params` dictionary containing the loaded model parameters [c3504].

The `load_gpt2_params_from_tf_ckpt` function loads parameter values from a TensorFlow checkpoint and converts them into the `params` dictionary structure with the five keys [c3522]. Here is the full implementation:

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

Intuitively, this function walks the TensorFlow variable name hierarchy—which uses `/`-separated path strings like `model/h0/attn/c_attn/w`—and reconstructs a nested Python dictionary that mirrors that hierarchy. Variables whose names start with `h` (for "hidden layer") are routed into the appropriate index of the `blocks` list. All other variables go into the top-level `params` dictionary.

The function returns two values: the `settings` dictionary (containing vocabulary size, context length, embedding dimension, number of attention heads, and number of transformer blocks) and the `params` dictionary [c3523].

---

### Mapping GPT-2 Weights into the Custom GPT Model

#### Setting Up the Model Configuration

We now have the `params` dictionary populated with GPT-2 weights. The next task is to replace the random weight initializations in the GPT model with downloaded GPT-2 parameters from the `params` dictionary [c3541].

The GPT model class initializes token embeddings, positional embeddings, transformer blocks, layer normalization, and an output head layer [c3540]. We need to assign the downloaded weights to each of these components in turn.

First, we set up the model configuration. The four GPT-2 variants have different architectural dimensions:

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

This pattern of copying a base configuration and updating it with model-specific values makes it easy to switch between GPT-2 variants without rewriting the model class.

#### A Shape-Checking Assignment Helper

Before assigning any weights, it is useful to have a helper function that verifies shape compatibility. A mismatch in tensor shapes is a common source of silent bugs—the assignment might succeed but produce garbage outputs.

# Source: [c3544]
```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return right
```

The consequence of using this helper throughout the weight-loading process is that any shape mismatch raises an error immediately, rather than propagating silently. A tiny mistake in the weight loading process would cause the model to fail [c3572], so this defensive check is valuable.

#### Assigning Weights Layer by Layer

With the helper in place, we can assign weights to each component of the model. The process follows the structure of the `params` dictionary.

**Attention block weights**: Attention block query, key, and value weight matrices are updated by assigning `Q_W`, `K_W`, and `V_W` from downloaded GPT-2 parameters [c3546]. Recall that GPT-2 stores these as the fused `C_ATTN` matrix, so the assignment code must split that matrix along the appropriate dimension before assigning to the three separate weight tensors in our custom model.

**Output projection**: The output projection layer weights and biases from GPT-2 are assigned to the attention object's output projection [c3548].

**Feed-forward network**: Feed forward neural network weights and biases for both the fully connected layer and projection layer are assigned from downloaded GPT-2 values [c3550].

**Final layer normalization**: The final layer normalization scale and shift values are assigned from GPT-2 downloaded parameters [c3552].

[FIGURE: Diagram showing the mapping from params dictionary keys (WTE, WPE, blocks[i], G, B) to GPTModel attributes (tok_emb, pos_emb, trf_blocks[i], final_norm)]

#### Weight Tying

There is one more architectural detail that requires special handling: weight tying. GPT-2 uses weight tying, where token embedding weights are reused for the output head layer instead of defining separate weights [c3553]. This means the matrix that maps token IDs to embedding vectors is the same matrix (transposed) that maps the final hidden states back to vocabulary logits.

Weight tying reduces the total parameters from 164 million to 124 million in the GPT-2 architecture [c3554]. Intuitively, this makes sense: the embedding matrix encodes a relationship between tokens and a vector space, and the output projection needs to decode that same relationship in reverse. Sharing the weights enforces consistency between the two operations and reduces the parameter count significantly.

When loading weights, this means we do not assign separate weights to the output head—we simply point it at the same tensor as the token embedding layer.

#### Downloading and Loading in Practice

To put it all together, here is the sequence that downloads the weights and loads them:

# Source: [c3524]
```python
from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

After this call, `settings` contains the hyperparameters and `params` contains the weight tensors. The weight assignment functions described above then iterate through the model's layers and call `assign()` to copy each tensor from `params` into the corresponding attribute of the `GPTModel` instance.

Model weights are loaded correctly when the model produces coherent text [c3571]. This is the practical test: if the output is coherent, the mapping was correct; if it is garbage, something went wrong in the assignment.

---

### Text Generation with Pre-Trained Weights

#### Running Inference

With pre-trained GPT-2 weights loaded, inference execution is fast because the model does not require training [c3562]. We can immediately call the `generate` function that was built in earlier chapters.

The text generation strategies including temperature scaling and top-k sampling can be integrated together to reduce overfitting in text generation [c724]. Top-k sampling was introduced as a second decoding strategy to reduce loss and overfitting [c3457]. Here is an example generation call:

# Source: [c3556]
```python
generate(model, input_token_ids=['every', 'effort', 'moves', 'you'], max_new_tokens=25, temperature=1.5, top_k=50)
```

Top-K sampling with K=50 restricts token selection to the 50 most likely candidates during generation [c3558]. The temperature of 1.5 controls the sharpness of the probability distribution over those candidates—higher values produce more varied output, lower values produce more conservative output.

#### Comparing Trained vs. Untrained Output

The difference in output quality is stark. Without GPT-2 weights, the model generates incoherent output [c3561]. With pre-trained GPT-2 weights loaded, output from the pre-trained GPT-2 model is more coherent than output from the untrained model [c3564].

This comparison makes intuitive sense. The untrained model has random weights, so its output is essentially random token sequences. The pre-trained model has weights that encode statistical patterns from a large corpus, so it produces grammatically plausible and semantically coherent continuations.

#### Fine-Tuning Possibilities

Loading pre-trained weights is not the end of the story—it is the beginning of a new one. The custom-built GPT model allows exploration of hyperparameter tuning and architectural modifications that are not possible with ChatGPT [c3567]. Because we own the model class and have direct access to every weight tensor, we can modify the architecture, freeze specific layers, or fine-tune on a domain-specific dataset.

For reference, the Adam optimizer is used with a learning rate of 5e-4 and weight decay of 0.1 [c3569] when fine-tuning from the pre-trained weights.

---

### Summary

This chapter covered two complementary skills. The first is PyTorch checkpointing: using `model.state_dict()` and `optimizer.state_dict()` to save and restore the full training state, so that long training runs can be interrupted and resumed without losing progress [c720, c749]. The second is loading externally trained weights: downloading OpenAI's seven GPT-2 files, parsing the TensorFlow checkpoint format into a Python dictionary, and mapping each weight tensor into the correct attribute of our custom `GPTModel` class.

The key technical challenges along the way were:

1. **Format mismatch**: GPT-2 weights are in TensorFlow format; our model is in PyTorch [c3469]. The `load_gpt2_params_from_tf_ckpt` function bridges this gap [c3522].
2. **Fused attention matrices**: GPT-2 stores Q, K, V as a single `C_ATTN` matrix [c3513], which must be split before assignment.
3. **Weight tying**: The token embedding and output head share weights [c3553], reducing the parameter count from 164M to 124M [c3554].
4. **Shape verification**: The `assign()` helper catches mismatches early [c3544], because a tiny mistake in the weight loading process would cause the model to fail [c3572].

The payoff is immediate and tangible: a model that generates coherent text [c3564] without any training, simply by inheriting the knowledge encoded in GPT-2's weights.