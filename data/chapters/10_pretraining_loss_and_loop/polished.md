## Pre-Training: Loss Functions, Training Loop, and Evaluation

This chapter builds each of those three pieces from scratch. We start by understanding exactly what cross-entropy loss means in the context of next-token prediction, then implement helper functions that compute loss over batches and entire data loaders, and finally assemble the full pre-training loop—complete with backpropagation, parameter updates, and periodic evaluation.

---

### The Loss Landscape: Why Cross-Entropy?

#### A Taxonomy of Loss Functions

For regression—predicting a continuous value—common choices include:

**Mean Bias Error (MBE)** simply averages the signed errors [c474]:

$$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

**Mean Absolute Error (MAE)** removes the sign by taking absolute values [c475]:

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - f(x_i)|$$

**Mean Squared Error (MSE)** squares the residuals, penalizing large errors more heavily [c476]:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

**Root Mean Squared Error (RMSE)** restores the original units by taking the square root [c477]:

$$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2}$$

**Huber Loss** blends MSE and MAE: it behaves quadratically for small errors and linearly for large ones, controlled by a threshold $\delta$ [c478]:

$$\mathcal{L}_{Huber} = \begin{cases} \frac{(y_i - f(x_i))^2}{2} & \text{if } |y_i - f(x_i)| \leq \delta \\ \delta|y_i - f(x_i)| - \frac{\delta}{2} & \text{otherwise} \end{cases}$$

**Log-Cosh Loss** is a smooth approximation to MAE [c479]:

$$\mathcal{L}_{LogCosh} = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(f(x_i) - y_i))$$

For classification tasks, the model no longer predicts a scalar; it predicts a probability distribution over discrete classes. The standard choices are [c484]:

**Binary Cross-Entropy (BCE)** for two-class problems [c480]:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

**Hinge Loss**, used in support-vector-style classifiers [c481]:

$$\mathcal{L}_{Hinge} = \max(0, 1 - (f(x) \cdot y))$$

**Cross-Entropy Loss** for multi-class problems [c482]:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{i,c} \log(f(x_i)_c)$$

**Kullback-Leibler Divergence** measures how one probability distribution diverges from another [c483]:

$$\mathcal{L}_{KL} = -\sum_{i=1}^{N} y_i \cdot \log\left(\frac{y_i}{f(x_i)}\right)$$

#### Why Language Modeling Uses Cross-Entropy

Cross-entropy loss measures the difference between two probability distributions—specifically, between the predicted logits and the target token indices [c550]. The goal of LLM training is to minimize this loss so that the predicted probabilities for target tokens approach one [c1609].

Concretely, cross-entropy loss is computed by taking the logarithm of the probability the model assigns to the correct token at each position, averaging those log-probabilities, and negating the result [c3790]:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_i)$$

where $p_i$ are the probabilities at the target indices [c552]. This quantity is also called **negative log likelihood (NLL)**—the negative of the logarithm of the target probabilities [c3791]. As the model becomes less certain or more wrong, the loss grows; the goal of training is to drive it toward zero [c3792].

---

### From Logits to Loss: The Forward Pass in Detail

#### The Model's Output: Logits

Large language models are autoregressive models in which input-output pairs are constructed from the text itself without pre-labeling [c3753]. The transformer block is the main engine of the GPT architecture, transforming input embedding vectors into context vectors [c3775]—richer representations that encode both the semantic meaning of a token and its relationship to every other token in the sequence [c3776]. After the transformer blocks, the model produces **logits**: the raw output values before normalization into probabilities [c3779]. A logit tensor is then converted into a softmax tensor, yielding a probability tensor [c3785].

#### Constructing Input-Output Pairs

Before computing any loss, we need properly structured training data. Targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. More precisely, $X$ represents the input tensor and $Y$ represents the target tensor (the actual desired output values) [c3763]. Input-target pairs are created by pairing each input with its corresponding target output [c3767], where the target is the expected output sequence the LLM should learn to predict [c3770]. In the context of language model training, "target" refers to the true values that the model should predict [c502].

#### The Dataloader

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. Two key parameters govern this process:

- **`max_length`**: the context size—the number of tokens to consider at once [c3822]. Context size is the maximum number of tokens an LLM can see before predicting the next token [c3754].
- **`stride`**: how many steps to advance before creating the next input-output pair [c3823]. Stride is the step size used to move through the dataset when constructing consecutive pairs [c3758].

The PyTorch `DataLoader` makes batch processing convenient [c3826], and two additional parameters control its behavior:

- **`shuffle`**: shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828].
- **`drop_last`**: drops the last batch if its size is smaller than the specified batch size [c3829].

Tiktoken provides a byte pair encoder that operates at the character and sub-word level [c3817], making it the natural tokenizer for this pipeline. The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538], and `token_ids_to_text` performs the reverse conversion [c539].

---

### Computing Cross-Entropy Loss in PyTorch

#### The Flat-Tensor Pattern

PyTorch's `cross_entropy` function expects a 2-D logit tensor and a 1-D target tensor, so we must flatten both before passing them to the loss function. For a batch of shape `(2, 3, 50257)` [c569]:

```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)       # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The same pattern applies in any loss-computation context [c555, c581]:

```python
torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

#### Selecting the Predicted Token

During evaluation, we often want to know which token the model actually predicts. `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension [c547], effectively picking the highest-logit token at each position.

#### The `calculate_loss_batch` Function

The `calculate_loss_batch` function computes cross-entropy loss for a single input-target batch pair by flattening both the logits and target tensors before passing them to `nn.functional.cross_entropy` [c3852]:

```python
nn.functional.cross_entropy(flattened_logits, flattened_target_batch)
```

Cross-entropy loss is used to measure the discrepancy between target tokens and output tokens [c580].

---

### Perplexity: An Interpretable Loss Metric

Raw cross-entropy values can be difficult to interpret in isolation. Perplexity offers a more intuitive alternative: it measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. It is calculated as $e$ raised to the loss value [c582]:

$$\text{Perplexity} = \exp(\text{loss})$$ [c574]

As training progresses and the loss decreases, perplexity falls correspondingly.

---

### The Pre-Training Loop

With loss computation in hand, we can now build the full training loop. The training pipeline for LLMs consists of multiple steps, starting with text generation and text evaluation [c490], and this chapter focuses on training and validation loss computation [c3728], building on the cross-entropy loss calculation covered previously [c3734].

#### High-Level Structure

The pre-training loop has two nested loops: an outer loop iterating over epochs and an inner loop iterating over batches in the training dataset [c1657]. One epoch is a single pass through the entire training set [c1613]. The training set is divided into batches, and the loop processes one batch per iteration [c1614]; after all batches are exhausted, the outer loop repeats for the next epoch [c1621].

[FIGURE: Nested loop diagram showing outer epoch loop containing inner batch loop, with forward pass → loss → backward pass → optimizer step labeled inside the inner loop]

#### The Core Update Rule

At each iteration, parameters are updated according to [c1618]:

$$p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$$

LLM pre-training involves minimizing the loss function to make model outputs as close as possible to the target value tensors [c1611], and back propagation is used to calculate the required loss gradients [c1612]. The algorithm consists of three steps [c1622]:

1. **Forward pass** — run the model and compute cross-entropy loss.
2. **Backward pass** — compute gradients of the loss with respect to all parameters.
3. **Parameter update** — apply the optimizer step using those gradients.

The backward pass is the most important step because it calculates the gradients of the loss [c1616]:

```python
loss.backward()
```

The pre-training workflow is fully differentiable, allowing back propagation to compute partial derivatives of the loss with respect to all model parameters [c1626]. In practice, the entire loop is approximately 15 to 20 lines of Python, with `loss.backward()` and `optimizer.step()` forming the core gradient update mechanism [c1667].

#### Iterating Over Batches

The pretraining loop iterates through batches from the training data loader, dividing each batch into an input batch and a target batch [c1652]. Cross-entropy loss is calculated between these two components for the entire batch at each iteration [c1615, c1653].

#### Evaluation During Training

An evaluation step is performed periodically to monitor both training loss and validation loss as the model trains [c1659]. Computing these losses on an actual dataset [c3737] also enables backpropagation in the subsequent step [c3868].

[FIGURE: Training curve plot showing training loss and validation loss decreasing over epochs, with evaluation checkpoints marked at regular intervals]

#### Model Parameters: A Quick Accounting

The key parameter counts for the major components are:

- **Token embedding parameters** = vocabulary size × embedding dimension [c1630]
- **Positional embedding parameters** = context size × embedding dimension [c1631]
- **Final layer parameters** = embedding dimension × vocabulary size [c1644]

Embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

---

### Putting It All Together: Training on "The Verdict"

With all the pieces defined, we can now describe a full training run on a real text corpus.

#### Data Preparation

The developed code is generalizable and can be applied to custom datasets for pre-training [c3870]. With a small corpus like "The Verdict," setting the stride equal to the context size avoids redundant samples; with larger corpora, overlapping strides can increase the effective dataset size.

#### The Training Function

An LLM training function is defined to implement backpropagation and minimize training and validation loss [c3869]. It wraps the nested epoch-batch loop described above, calling `calculate_loss_batch` on each mini-batch, invoking `loss.backward()`, and then stepping the optimizer. A training loop has been implemented where the loss function is minimized and the large language model learns [c1717]. The next lecture will cover LLM pre-training in full detail [c3873].

#### Decoding Predictions During Training

Temperature scaling is a decoding strategy used to control randomness in model predictions [c1719].

[FIGURE: Diagram showing logits → temperature scaling → softmax → token sampling pipeline]

#### Observing Overfitting

[FIGURE: Overfitting curve showing training loss continuing to decrease while validation loss flattens or rises after a certain epoch]

A well-regularized model trained on a large, diverse corpus will show both curves tracking closely together.

---

### Summary

This chapter developed the full machinery needed to pre-train a GPT-style language model from scratch:

1. **Loss functions**: We surveyed the family of regression and classification losses and established why cross-entropy—equivalently, negative log likelihood—is the right choice for next-token prediction [c550, c1608]. The formula is $-\frac{1}{N}\sum_i \log(p_i)$ [c552], and the training goal is to drive this value toward zero [c3792].

2. **Perplexity**: Defined as $\exp(\text{loss})$ [c574], perplexity provides a more interpretable metric that reflects how well the model's predicted distribution matches the true word distribution [c572].

3. **Batch loss computation**: The `calculate_loss_batch` function flattens logits from `(batch, seq, vocab)` to `(batch×seq, vocab)` and targets from `(batch, seq)` to `(batch×seq,)` before calling `nn.functional.cross_entropy` [c3852, c569].

4. **The training loop**: The loop is nested—outer over epochs [c1613], inner over batches [c1614]—and each iteration performs a forward pass, `loss.backward()` [c1624], and an optimizer step [c1622]. The entire workflow is fully differentiable [c1626].

5. **Evaluation**: Training and validation loss are monitored throughout training [c1659], and perplexity provides a human-readable summary of model quality at each checkpoint [c582].