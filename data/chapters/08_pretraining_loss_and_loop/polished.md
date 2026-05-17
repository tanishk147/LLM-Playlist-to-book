# Pre-Training: Loss Functions, Training Loop, and Evaluation

Knowing whether a language model is actually learning requires more than watching it generate text — it requires measuring loss precisely and watching that number fall. This chapter builds that measurement infrastructure from the ground up: we will calculate training loss and validation loss on an actual dataset [c3737], observe how those numbers evolve over time, and learn to recognize the warning signs of overfitting. By the end you will have a complete, working pre-training loop of roughly 15 to 20 lines of Python [c1667].

---

## The Language Modeling Objective

### Autoregressive Prediction

Large language models are autoregressive models where input-output pairs are constructed from the text itself without pre-labeling [c3753]. Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754]. Given a sequence of tokens, the model receives a window of up to that many tokens as input and must predict what comes next. Targets are the input token sequence shifted by one position, creating input-output pairs for training [c542].

More formally, $X$ represents the input tensor and $Y$ represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767], and the target output is the expected output sequence that corresponds to a given input sequence, which the LLM should learn to predict during training [c3770].

### From Raw Text to Batches

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. Two parameters govern this process:

- **`max_length`** specifies the context size — the number of tokens to consider at once [c3822].
- **`stride`** specifies how many steps to advance before creating the next input-output pair [c3823].

Stride is the step size used to move through the dataset when constructing consecutive input-output pairs [c3758]. The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. Two additional parameters control how batches are assembled:

- **`shuffle`** shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828].
- **`drop_last`** drops the last batch if its size is smaller than the specified batch size [c3829].

Loss computation is then applied to the entire dataset, split into training and validation portions [c3730].

---

## A Taxonomy of Loss Functions

### Regression Losses

Let $y_i$ denote the true value and $f(x_i)$ the model's prediction for the $i$-th example.

**Mean Bias Error** simply averages the signed errors [c474]:

$$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

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

### Classification Losses

**Binary Cross-Entropy** is used for two-class problems [c480]:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

**Hinge Loss** is the classic support-vector-machine objective [c481]:

$$\mathcal{L}_{Hinge} = \max(0, 1 - (f(x) \cdot y))$$

**Kullback-Leibler Divergence** measures how one probability distribution differs from another [c483]:

$$\mathcal{L}_{KL} = -\sum_{i=1}^{N} y_i \cdot \log\left(\frac{y_i}{f(x_i)}\right)$$

### Cross-Entropy Loss: The Language Modeling Standard

For multi-class classification — which is exactly what next-token prediction is — the standard choice is **Cross-Entropy Loss** [c482]:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{i,c} \log(f(x_i)_c)$$

Cross-entropy loss measures the difference between two probability distributions, specifically between predicted logits and target indices in language model training [c550]. The goal of LLM training is to minimize this loss so that predicted probabilities for target tokens approach one [c1609].

---

## Cross-Entropy in Depth

### From Logits to Probabilities

The transformer block is the main engine of the GPT architecture, transforming input embedding vectors into context vectors [c3775]. Context vectors are richer representations than input embedding vectors because they encode both the semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776].

After the final transformer block, a linear projection maps each context vector to a vector of raw scores over the vocabulary. These raw scores are called logits — the raw output values produced by the GPT model architecture before normalization into probabilities [c3779]. A logit tensor is then converted into a softmax tensor, yielding a probability tensor [c3785].

### Negative Log-Likelihood

Cross-entropy loss is computed by taking the logarithm of the target probabilities, summing them, taking the mean, and then negating the result [c3790]. This is equivalent to **negative log-likelihood (NLL)** — the negative of the logarithm of the target probabilities [c3791]. More precisely, it equals the negative mean of log probabilities at the target indices:

$$-\frac{1}{N} \sum \log(p_i)$$

where $p_i$ are the probabilities at the target indices [c552].

The goal of training an LLM is to minimize this negative log-likelihood toward zero [c3792]. Cross-entropy loss (negative log-likelihood) is therefore the natural measure of the difference between predicted probabilities and target token indices [c1608].

### Computing Loss in PyTorch

PyTorch's `torch.nn.functional.cross_entropy` expects logits (not probabilities) as the first argument and integer class indices as the second [c581]:

```python
torch.nn.functional.cross_entropy(logit_tensor, target_tensor)
```

When working with a batch of sequences, both tensors must be flattened before the call. For example, given logits of shape `(2, 3, 50257)` and targets of shape `(2, 3)` [c569]:

```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)       # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The same pattern appears in a more compact form when working with a full training batch [c555]:

```python
torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

And in the context of a complete forward pass over a batch [c3851]:

```python
nn.functional.cross_entropy(flattened_logits, flattened_target_batch)
```

To find the predicted token at each position — useful for quick accuracy checks — take the argmax over the vocabulary dimension. `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension (columns) [c547].

---

## Perplexity: An Interpretable Loss Metric

Raw cross-entropy values are hard to interpret intuitively. Perplexity addresses this by measuring how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. It is calculated as $e$ raised to the loss value [c582]:

$$\text{Perplexity} = \exp(\text{loss})$$

[c574]

A well-trained language model on English text typically achieves perplexity in the tens to low hundreds, depending on the domain and model size.

---

## Utility Functions: Text ↔ Token IDs

Two helper functions bridge raw text and the integer representations the model operates on. The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538]; tiktoken provides a byte-pair encoder that operates at the character and sub-word level [c3817]. The reverse operation, `token_ids_to_text`, converts token IDs back into text [c539].

---

## Model Parameter Accounting

Understanding how many parameters a model contains helps set expectations about memory requirements and training time. The main contributors are:

- **Token embedding parameters** = vocabulary size × embedding dimension [c1630]
- **Positional embedding parameters** = context size × embedding dimension [c1631]
- **Final layer parameters** = embedding dimension × vocabulary size [c1644]

Embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

---

## The Pre-Training Algorithm

With the loss function defined, we can describe the pre-training algorithm at a high level.

LLM pre-training involves minimizing the loss function to make model outputs as close as possible to target value tensors [c1611]. Each iteration over a batch consists of three steps [c1622]:

1. **Forward pass** — compute logits and then cross-entropy loss for the entire batch [c1615].
2. **Backward pass** — call `loss.backward()` to compute gradients [c1624]. This is the most important step because it calculates the gradients of the loss with respect to every parameter [c1616]. Back propagation is used to calculate these loss gradients [c1612].
3. **Parameter update** — apply the gradient descent rule [c1618]:

$$p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$$

The pre-training workflow is fully differentiable, allowing back propagation to compute partial derivatives of the loss with respect to all model parameters [c1626].

### Epochs and Batches

One epoch is a single pass through the entire training set [c1613]. The training set is divided into batches, and the pre-training loop processes one batch per iteration [c1614]. After exhausting all batches, the loop repeats for multiple epochs [c1621]. Structurally, the pretraining loop has two nested loops: an outer loop iterating over epochs and an inner loop iterating over batches in the training dataset [c1657].

[FIGURE: Nested loop diagram showing outer epoch loop containing inner batch loop, with forward pass → loss → backward pass → optimizer step labeled inside the inner loop]

---

## Implementing the Training Loop

The training pipeline for LLMs begins with text generation and text evaluation [c490]. In this context, "target" refers to the true values that the model should predict [c502]. The core of the inner loop is a single call [c1624]:

```python
loss.backward()
```

This triggers PyTorch's autograd engine to walk backward through the computation graph and accumulate gradients in each parameter's `.grad` attribute. The optimizer then reads those gradients and updates the parameters.

The resulting training loop is approximately 15 to 20 lines of Python, with `loss.backward()` and `optimizer.step()` as the core gradient update mechanism [c1667]. A training loop has been implemented where the loss function is minimized and the large language model learns [c1717].

An evaluation step is performed during training to monitor training loss and validation loss as the model trains [c1659], and an LLM training function will be defined to implement backpropagation and minimize both losses [c3869].

---

## Evaluating the Model During Training

### The `evaluate_model` Function

The `evaluate_model` function calculates loss for both the training loader and validation loader over the entire dataset [c1660]. During evaluation, the model is set to evaluation mode with gradient tracking disabled and dropout disabled, so loss is computed without any gradient updates [c1665]. This matters for two reasons: evaluation mode disables stochastic components such as dropout, giving deterministic outputs; and wrapping the evaluation in `torch.no_grad()` tells PyTorch not to build a computation graph, saving memory and speeding up the forward pass.

The function returns both training loss and validation loss as scalar outputs [c3863]. Having these two values in hand enables backpropagation in the next step [c3868].

[FIGURE: Diagram showing evaluate_model function receiving training_loader and validation_loader, computing average loss over all batches for each, and returning two scalar values]

### Recognizing Overfitting

Divergence between training and validation loss — with validation loss growing substantially larger than training loss — indicates that the model is overfitting to the training data [c1707].

[FIGURE: Line plot with two curves (training loss and validation loss) over training steps, showing validation loss diverging upward from training loss after a certain point, labeled as overfitting]

---

## Temperature Scaling and Text Generation During Training

Even before the loss numbers tell a clear story, you can often see the model's outputs shift from random noise to recognizable English. Temperature scaling is a decoding strategy used to control the randomness of those outputs [c1719].

---

## Putting It All Together: The Training Pipeline

The code developed in this chapter is generalizable and can be applied to custom datasets for pre-training [c3870]. The complete pipeline proceeds as follows [c3728]:

1. **Tokenize the corpus.** Use tiktoken's byte-pair encoder [c3817] to convert raw text into integer token IDs.

2. **Build the DataLoader.** Wrap the token IDs in a dataset class that produces `(input, target)` pairs using a sliding window of size `max_length` and step size `stride` [c3821, c3822, c3823]. Split the data into training and validation portions [c3730].

3. **Run the training loop.** For each epoch [c1613], iterate over all batches [c1614]:
   - Compute logits via the forward pass.
   - Flatten logits and targets, then compute cross-entropy loss [c569].
   - Call `loss.backward()` [c1624] to compute gradients.
   - Call `optimizer.step()` to update parameters using $p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$ [c1618].

4. **Evaluate periodically.** Call `evaluate_model` to compute training and validation loss over the full dataset [c1660]. Log both values so you can plot them later.

[FIGURE: End-to-end pipeline flowchart: Raw Text → Tokenizer → DataLoader → Training Loop (Forward → Loss → Backward → Update) → Evaluate → Generate Sample]

---

## Worked Example: Loss Computation Step by Step

To make the mechanics concrete, consider tracing through a single forward pass and loss computation for a small batch. After the forward pass produces logits of shape `(2, 3, 50257)` and targets of shape `(2, 3)`, both tensors are flattened before being passed to PyTorch's cross-entropy function [c569]:

```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)       # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

To convert the resulting loss to perplexity [c574]:

$$\text{Perplexity} = \exp(\text{loss})$$

As training progresses, both numbers should fall substantially.

---

## What Comes Next

The next lecture will cover LLM pre-training in full [c3873]. The code developed here is generalizable and can be applied to custom datasets [c3870], and once training and validation loss are in hand, backpropagation can proceed in the next step [c3868]. Fine-tuning, instruction tuning, and reinforcement learning from human feedback all build on the same basic machinery: a differentiable loss function, a backward pass, and a parameter update rule.

---

## Summary

This chapter covered the full arc from raw text to a working training loop:

- **Loss functions**: We surveyed regression and classification losses [c474, c475, c476, c477, c478, c479, c480, c481, c482, c483], then focused on cross-entropy as the standard for language modeling [c550, c552].
- **Cross-entropy mechanics**: Logits are produced by the model [c3779], converted to probabilities via softmax [c3785], and the negative log-likelihood of the target tokens is minimized [c3790, c3791, c3792].
- **PyTorch implementation**: Flatten the logits and targets, then call `torch.nn.functional.cross_entropy` [c569, c555, c3851].
- **Perplexity**: An interpretable metric computed as $\exp(\text{loss})$ [c574, c582].
- **Training loop structure**: Two nested loops (epochs and batches) [c1657], with forward pass, `loss.backward()` [c1624], and optimizer step [c1618] inside the inner loop.
- **Evaluation**: The `evaluate_model` function computes loss on both training and validation sets [c1660], with the model in eval mode and gradients disabled [c1665].
- **Overfitting**: Divergence between training and validation loss is the key diagnostic signal [c1707].