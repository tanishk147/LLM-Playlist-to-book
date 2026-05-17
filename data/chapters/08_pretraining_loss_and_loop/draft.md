## Pre-Training: Loss Functions, Training Loop, and Evaluation

With a working GPT model architecture in hand, the next challenge is teaching it to actually learn. Pre-training is the process by which a language model adjusts its millions of parameters so that its predictions grow progressively more accurate. This chapter covers everything you need to take a randomly initialized model and drive it toward useful behavior: the loss functions that measure how wrong the model is, the training loop that iterates over data and applies gradient updates, and the evaluation strategy that tells you whether the model is genuinely learning or merely memorizing.

We will calculate training loss and validation loss on an actual dataset [c3737], observe how those numbers evolve over time, and learn to recognize the warning signs of overfitting. By the end of the chapter you will have a complete, working pre-training loop of roughly 15 to 20 lines of Python [c1667].

---

### The Language Modeling Objective

Before writing a single line of training code, it is worth being precise about what we are optimizing and why.

#### Autoregressive Prediction

Large language models are autoregressive models where input-output pairs are constructed from the text itself without pre-labeling [c3753]. This is a crucial observation: unlike image classification, where a human must annotate each example, a language model generates its own supervision signal from raw text. Every document in the training corpus implicitly contains thousands of (input, next-token) pairs.

Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754]. Given a sequence of tokens, the model receives a window of up to that many tokens as input and must predict what comes next. Targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. In other words, if the input is `[t₁, t₂, t₃]`, the target is `[t₂, t₃, t₄]`.

More formally, $X$ represents the input tensor and $Y$ represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767], and the target output is the expected output sequence that corresponds to a given input sequence, which the LLM should learn to predict during training [c3770].

#### From Raw Text to Batches

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. Two parameters govern this process:

- **`max_length`** specifies the context size — the number of tokens to consider at once [c3822].
- **`stride`** specifies how many steps to advance before creating the next input-output pair [c3823].

Stride is the step size used to move through the dataset when constructing consecutive input-output pairs [c3758]. A stride equal to `max_length` produces non-overlapping windows; a smaller stride produces heavily overlapping windows that expose the model to more training signal per token.

The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. Two additional parameters are worth noting:

- **`shuffle`** shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828].
- **`drop_last`** drops the last batch if its size is smaller than the specified batch size [c3829].

This section applies loss computation to the entire dataset, split into training and validation portions [c3730].

---

### A Taxonomy of Loss Functions

Loss functions are the mathematical bridge between model outputs and the training signal. It is useful to survey the landscape before focusing on the one that matters most for language modeling.

#### Regression Losses

For tasks where the output is a continuous value, several loss functions are commonly used. Let $y_i$ denote the true value and $f(x_i)$ the model's prediction for the $i$-th example.

**Mean Bias Error** simply averages the signed errors [c474]:

$$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

Because positive and negative errors cancel, MBE is rarely used as a training objective but can diagnose systematic bias.

**Mean Absolute Error** removes the cancellation problem by taking absolute values [c475]:

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - f(x_i)|$$

**Mean Squared Error** penalizes large errors more heavily by squaring the residuals [c476]:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

**Root Mean Squared Error** brings the loss back to the same units as the target [c477]:

$$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2}$$

**Huber Loss** is a hybrid that behaves like MSE for small errors and like MAE for large ones, making it robust to outliers [c478]:

$$\mathcal{L}_{Huber} = \begin{cases} \frac{(y_i - f(x_i))^2}{2} & \text{if } |y_i - f(x_i)| \leq \delta \\ \delta|y_i - f(x_i)| - \frac{\delta}{2} & \text{otherwise} \end{cases}$$

**Log-Cosh Loss** is another smooth approximation to MAE [c479]:

$$\mathcal{L}_{LogCosh} = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(f(x_i) - y_i))$$

#### Classification Losses

When the output is a discrete category, a different family of losses applies.

**Binary Cross-Entropy** is used for two-class problems [c480]:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

**Hinge Loss** is the classic support-vector-machine objective [c481]:

$$\mathcal{L}_{Hinge} = \max(0, 1 - (f(x) \cdot y))$$

**Kullback-Leibler Divergence** measures how one probability distribution differs from another [c483]:

$$\mathcal{L}_{KL} = -\sum_{i=1}^{N} y_i \cdot \log\left(\frac{y_i}{f(x_i)}\right)$$

#### Cross-Entropy Loss: The Language Modeling Standard

For multi-class classification — which is exactly what next-token prediction is — the standard choice is **Cross-Entropy Loss** [c482]:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{i,c} \log(f(x_i)_c)$$

Here $M$ is the number of classes (the vocabulary size), $y_{i,c}$ is 1 if class $c$ is the correct label for example $i$ and 0 otherwise, and $f(x_i)_c$ is the predicted probability for class $c$.

Cross-entropy loss measures the difference between two probability distributions, specifically between predicted logits and target indices in language model training [c550]. The goal of LLM training is to minimize the cross-entropy loss function so that predicted probabilities for target tokens approach one [c1609].

---

### Cross-Entropy in Depth

Because cross-entropy is the workhorse of LLM training, it deserves a careful walkthrough.

#### From Logits to Probabilities

The transformer block is the main engine of the GPT architecture that transforms input embedding vectors into context vectors [c3775]. Context vectors are richer representations than input embedding vectors because they encode both semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776].

After the final transformer block, a linear projection maps each context vector to a vector of raw scores over the vocabulary. Logits are the raw output values produced by the GPT model architecture before normalization into probabilities [c3779]. A logit tensor is then converted into a softmax tensor, which is a probability tensor [c3785].

#### Negative Log-Likelihood

Cross-entropy loss is computed by taking the logarithm of the target probabilities, summing them, taking the mean, and then negating the result [c3790]. This is equivalent to **negative log-likelihood (NLL)**, which is the negative of the logarithm of the target probabilities [c3791].

Intuitively, the logarithm of a probability is always non-positive (since probabilities lie in $(0, 1]$). When the model assigns high probability to the correct token, $\log(p)$ is close to zero; when it assigns low probability, $\log(p)$ is a large negative number. Negating and averaging gives a loss that is small when the model is confident and correct, and large when it is wrong.

More precisely, cross entropy loss is computed as: negative mean of log probabilities at target indices, or $-\frac{1}{N} \sum \log(p_i)$ where $p_i$ are probabilities at target indices [c552].

The goal of training an LLM is to minimize the negative log-likelihood loss to approach zero [c3792]. Cross-entropy loss (negative log likelihood) is employed to measure the difference between predicted probabilities and target token indices [c1608].

#### Computing Loss in PyTorch

PyTorch's `torch.nn.functional.cross_entropy` handles the softmax and log internally, which is both numerically stable and convenient. The function signature is straightforward:

# Source: [c581]
```python
torch.nn.functional.cross_entropy(logit_tensor, target_tensor)
```

It expects logits (not probabilities) as the first argument and integer class indices as the second. Internally it applies log-softmax and then computes the negative log-likelihood.

When working with batched sequences, the logits tensor has shape `(batch_size, sequence_length, vocab_size)` and the targets tensor has shape `(batch_size, sequence_length)`. PyTorch's cross-entropy function expects a 2-D logits tensor, so we must flatten the first two dimensions before calling it.

# Source: [c569]
```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)  # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The same operation appears in a more compact form when working with a full training batch:

# Source: [c555]
```python
torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

And in the context of a complete forward pass over a batch:

# Source: [c3851]
```python
nn.functional.cross_entropy(flattened_logits, flattened_target_batch)
```

To find the predicted token at each position — useful for quick accuracy checks — we take the argmax over the vocabulary dimension. `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension (columns) [c547].

---

### Perplexity: An Interpretable Loss Metric

Raw cross-entropy loss values are hard to interpret intuitively. A loss of 3.5 is better than 4.2, but what does that mean in practice? **Perplexity** provides a more interpretable scale.

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. Perplexity is a metric for measuring loss that is calculated as $e$ raised to the loss value [c582]:

$$\text{Perplexity} = \exp(\text{loss})$$

[c574]

To see this concretely: a perplexity of 1 means the model is perfectly certain about every next token. A perplexity of $V$ (the vocabulary size) means the model is no better than random guessing over the entire vocabulary. A well-trained language model on English text typically achieves perplexity in the tens to low hundreds, depending on the domain and model size.

The conversion is trivial in code — simply call `torch.exp()` on the scalar loss value — but the resulting number is far easier to communicate to collaborators and to compare across papers.

---

### Utility Functions: Text ↔ Token IDs

Before building the training loop, it is helpful to have two small utility functions that convert between raw text and token IDs.

The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538]. Tiktoken provides a byte pair encoder that operates at character and sub-word level [c3817]. The reverse operation, `token_ids_to_text`, converts token IDs back into text, serving as the reverse of `text_to_token_ids` [c539].

These two functions will be used throughout training and evaluation to generate sample text from the model and to inspect what the model is actually predicting.

---

### Model Parameter Accounting

Before training, it is instructive to count how many parameters the model has and where they live. This helps set expectations about memory requirements and training time.

Two of the largest parameter groups are the embedding tables:

- **Token embedding parameters** = vocabulary size × embedding dimension [c1630]
- **Positional embedding parameters** = context size × embedding dimension [c1631]

At the output end of the network:

- **Final layer parameters** = embedding dimension × vocabulary size [c1644]

Embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

[GAP: Explicit total parameter count for the 124M GPT-2 model and a full breakdown by layer type (attention, feed-forward, layer norm, etc.) is not provided in the supplied claims.]

---

### The Pre-Training Algorithm

With the loss function defined, we can now describe the pre-training algorithm at a high level.

LLM pre-training involves minimizing the loss function to make model outputs as close as possible to target value tensors [c1611]. The pre-training algorithm consists of three steps per batch [c1622]:

1. **Forward pass**: compute the loss between model predictions and targets.
2. **Backward pass**: compute gradients of the loss with respect to all model parameters.
3. **Parameter update**: adjust parameters in the direction that reduces the loss.

Back propagation is used to calculate loss gradients during LLM pre-training [c1612]. The backward pass is the most important step in the pre-training loop because it calculates the gradients of the loss [c1616].

The parameter update follows the gradient descent rule [c1618]:

$$p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$$

The pre-training workflow is fully differentiable, allowing back propagation to compute partial derivatives of the loss with respect to all model parameters [c1626]. This is what makes the whole system work: every operation from the embedding lookup through the transformer blocks to the final linear projection is differentiable, so gradients flow all the way back to the first layer.

#### Epochs and Batches

One epoch is going through the entire training set once [c1613]. The training set is divided into batches, and the pre-training loop processes one batch per iteration [c1614]. Cross entropy loss is calculated for the entire batch during each training iteration [c1615].

The pre-training loop processes all batches in the training set, then repeats for multiple training epochs [c1621]. Structurally, the pretraining loop has two nested loops: an outer loop iterating over epochs and an inner loop iterating over batches in the training dataset [c1657].

[FIGURE: Nested loop diagram showing outer epoch loop containing inner batch loop, with forward pass → loss → backward pass → optimizer step labeled inside the inner loop]

---

### Implementing the Training Loop

The training pipeline for LLMs consists of steps starting with text generation and text evaluation [c490]. In 'target' refers to the true values that the model should predict [c502].

The core of the backward pass is a single method call:

# Source: [c1624]
```python
loss.backward()
```

This call triggers PyTorch's autograd engine to walk backward through the computation graph and accumulate gradients in each parameter's `.grad` attribute. After that, the optimizer reads those gradients and updates the parameters.

A training loop has been implemented where the loss function is minimized and the large language model learns [c1717]. The pretraining loop is approximately 15 to 20 lines of code in Python, with the `loss.backward()` method and `optimizer.step()` being the core gradient update mechanism [c1667].

An evaluation step is performed during training to monitor training loss and validation loss as the model trains [c1659]. An LLM training function will be defined to implement backpropagation and minimize training and validation loss [c3869].

[GAP: The complete training loop code (including optimizer initialization, the epoch/batch loops, the optimizer.zero_grad() call, loss.backward(), optimizer.step(), and the periodic evaluation call) is not provided verbatim in the supplied claims.]

---

### Evaluating the Model During Training

Training loss alone is not enough to understand whether a model is learning well. We also need to track **validation loss** — the loss on data the model has never seen during training.

#### The evaluate_model Function

The `evaluate_model` function calculates loss for both the training loader and validation loader over the entire dataset [c1660]. During model evaluation, the model is set to evaluation mode with gradient tracking disabled and dropout disabled to compute loss without gradient updates [c1665].

Setting the model to evaluation mode (`model.eval()`) is important for two reasons. First, it disables dropout, which is a stochastic regularization technique that randomly zeros out activations during training. During evaluation we want deterministic outputs. Second, wrapping the evaluation in `torch.no_grad()` tells PyTorch not to build a computation graph, which saves memory and speeds up the forward pass.

The function execution produces both training loss and validation loss values as output [c3863]. Obtaining training and validation loss enables backpropagation in the next step [c3868].

[FIGURE: Diagram showing evaluate_model function receiving training_loader and validation_loader, computing average loss over all batches for each, and returning two scalar values]

#### Recognizing Overfitting

Divergence between training and validation loss, with validation loss much larger than training loss, indicates the model is overfitting to the training data [c1707]. This is the classic overfitting signature: the model has memorized the training set rather than learning generalizable patterns.

When training on a small dataset like *The Verdict* (a short story used as a demonstration corpus), overfitting is expected and even instructive — it confirms that the training loop is working correctly, since a model that can overfit a small dataset is at least capable of learning. The remedy for real pre-training is simply more data.

[FIGURE: Line plot with two curves (training loss and validation loss) over training steps, showing validation loss diverging upward from training loss after a certain point, labeled as overfitting]

---

### Temperature Scaling and Text Generation During Training

Periodically generating sample text during training is a practical way to get a qualitative sense of progress. Even before the loss numbers tell a clear story, you can often see the model's outputs shift from random noise to recognizable English.

Temperature scaling is a decoding strategy used to control randomness in model predictions [c1719]. At high temperatures the probability distribution over the vocabulary is flattened, producing more varied (and often incoherent) outputs. At low temperatures the distribution is sharpened, producing more predictable (and often repetitive) outputs.

[GAP: The exact formula for temperature scaling applied to logits (dividing logits by temperature before softmax) is not provided in the supplied claims.]

The `text_to_token_ids` and `token_ids_to_text` utility functions described earlier are used here: encode a prompt, run the model's generation loop, then decode the resulting token IDs back to a string for inspection.

---

### Putting It All Together: The Training Pipeline

Let us now describe the complete training pipeline end to end.

The lecture covers LLM training and validation loss computation [c3728]. The developed code is generalizable and can be applied to custom datasets for pre-training [c3870].

Here is the conceptual flow:

1. **Tokenize the corpus.** Use tiktoken's byte-pair encoder [c3817] to convert raw text into integer token IDs.

2. **Build the DataLoader.** Wrap the token IDs in a dataset class that produces `(input, target)` pairs using a sliding window of size `max_length` and step size `stride` [c3821, c3822, c3823]. Split the data into training and validation portions [c3730].

3. **Initialize the model.** Instantiate the GPT model with the desired hyperparameters (embedding dimension, number of attention heads, context size, etc.).

4. **Initialize the optimizer.** [GAP: The specific optimizer (AdamW) and its hyperparameters (learning rate, weight decay, betas) are referenced in the chapter summary but not provided in the supplied claims.]

5. **Run the training loop.** For each epoch [c1613], iterate over all batches [c1614]:
   - Compute logits via the forward pass.
   - Flatten logits and targets, then compute cross-entropy loss [c569].
   - Call `loss.backward()` [c1624] to compute gradients.
   - Call `optimizer.step()` to update parameters using $p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$ [c1618].

6. **Evaluate periodically.** Call `evaluate_model` to compute training and validation loss over the full dataset [c1660]. Log both values so you can plot them later.

7. **Generate sample text.** Encode a fixed prompt, run greedy or temperature-scaled decoding, and print the result to get a qualitative read on model quality.

[FIGURE: End-to-end pipeline flowchart: Raw Text → Tokenizer → DataLoader → Training Loop (Forward → Loss → Backward → Update) → Evaluate → Generate Sample]

---

### Worked Example: Loss Computation Step by Step

To make the mechanics concrete, let us trace through a single forward pass and loss computation for a small batch.

Suppose we have a batch of 2 sequences, each of length 3 tokens, and a vocabulary of 50,257 tokens (the GPT-2 vocabulary size). After the forward pass, the model produces a logits tensor of shape `(2, 3, 50257)`.

The targets tensor has shape `(2, 3)` — one integer per position, indicating the correct next token.

We flatten both tensors before passing them to PyTorch's cross-entropy function:

# Source: [c569]
```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)  # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The result is a scalar — the average cross-entropy loss over all 6 token predictions in the batch. To convert to perplexity:

$$\text{Perplexity} = \exp(\text{loss})$$

[c574]

If the model were randomly guessing uniformly over 50,257 tokens, the expected loss would be $\log(50257) \approx 10.8$, giving a perplexity of 50,257. As training progresses, both numbers should fall substantially.

---

### What Comes Next

The next lecture will cover LLM pre-training in full [c3873]. The code developed here is generalizable and can be applied to custom datasets for pre-training [c3870]. Once training and validation loss are in hand, backpropagation can proceed in the next step [c3868].

The training loop described in this chapter is the foundation for everything that follows. Fine-tuning, instruction tuning, and reinforcement learning from human feedback all build on the same basic machinery: a differentiable loss function, a backward pass, and a parameter update rule. Understanding this loop deeply — not just as a recipe but as a mathematical process — is what separates practitioners who can debug and improve models from those who can only run existing scripts.

---

### Summary

This chapter covered the full arc from raw text to a working training loop:

- **Loss functions**: We surveyed the landscape of regression and classification losses [c474, c475, c476, c477, c478, c479, c480, c481, c482, c483], then focused on cross-entropy as the standard for language modeling [c550, c552].

- **Cross-entropy mechanics**: Logits are produced by the model [c3779], converted to probabilities via softmax [c3785], and the negative log-likelihood of the target tokens is minimized [c3790, c3791, c3792].

- **PyTorch implementation**: Flatten the logits and targets, then call `torch.nn.functional.cross_entropy` [c569, c555, c3851].

- **Perplexity**: An interpretable metric computed as $\exp(\text{loss})$ [c574, c582].

- **Training loop structure**: Two nested loops (epochs and batches) [c1657], with forward pass, `loss.backward()` [c1624], and optimizer step [c1618] inside the inner loop.

- **Evaluation**: The `evaluate_model` function computes loss on both training and validation sets [c1660], with the model in eval mode and gradients disabled [c1665].

- **Overfitting**: Divergence between training and validation loss is the diagnostic signal [c1707].