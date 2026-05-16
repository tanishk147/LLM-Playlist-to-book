## Pre-Training: Loss Functions, the Training Loop, and Evaluation

By this point in the book, you have a working GPT model that can accept token sequences and produce logits. The architecture is in place. What remains is the machinery that actually *teaches* the model to produce useful logits—the loss function, the training loop, and the evaluation metrics that tell you whether training is going in the right direction. This chapter covers all three, then closes with a look at decoding strategies that make the model's output more interesting at inference time.

---

### The Big Picture: What Pre-Training Is Doing

Before diving into code, it is worth being precise about what pre-training means for a language model.

Large language models are autoregressive models where input-output pairs are constructed from the text itself without any pre-labeling [c3753]. There is no human annotator deciding what the correct answer is; the training signal comes entirely from the structure of the text. The model sees a sequence of tokens and must predict the next one. If it predicts well, the loss is low; if it predicts poorly, the loss is high, and the parameters are adjusted accordingly.

The key structural idea is simple: targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. If your input is `["The", "cat", "sat"]`, your target is `["cat", "sat", "on"]`. Every token in the input simultaneously acts as a context element for predicting the next token and as a target for the token that came before it. This is what makes language modeling so data-efficient—a single sentence of length $N$ yields $N-1$ training examples.

Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754]. The dataloader that feeds training uses two parameters to control how it sweeps through the corpus: `max_length`, which specifies the context size (number of tokens to consider at once) [c3822], and `stride`, which specifies how many steps to advance before creating the next input-output pair [c3823]. A stride equal to the context size gives non-overlapping windows; a stride of 1 gives maximally overlapping windows and more training examples from the same text.

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. Two additional parameters worth knowing: the `shuffle` parameter shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828], and the `drop_last` parameter drops the last batch if its size is smaller than the specified batch size [c3829], which prevents shape mismatches during training.

One epoch is going through the entire training set once [c1613]. In practice, LLMs are trained for many epochs on large corpora, but even a single epoch over a large enough dataset can produce a capable model.

---

### Loss Functions: A Taxonomy

Before focusing on the specific loss used for language models, it is useful to survey the landscape of loss functions. Different tasks call for different choices, and understanding the alternatives sharpens your intuition for why cross-entropy is the right tool here.

#### Regression Losses

Regression losses measure the discrepancy between a continuous prediction $f(x_i)$ and a continuous target $y_i$.

**Mean Bias Error** simply averages the signed errors [c474]:

$$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

Because positive and negative errors cancel, MBE is rarely used as a training objective—it is more useful as a diagnostic.

**Mean Absolute Error** takes the absolute value before averaging, so errors cannot cancel [c475]:

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - f(x_i)|$$

MAE is robust to outliers but has a non-differentiable point at zero, which can complicate gradient-based optimization.

**Mean Squared Error** squares the errors, penalizing large deviations more heavily [c476]:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

MSE is the workhorse of regression. Its gradient is smooth everywhere, but it is sensitive to outliers because squaring amplifies large errors.

**Root Mean Squared Error** brings the loss back to the same units as the target [c477]:

$$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2}$$

**Huber Loss** is a hybrid that behaves like MSE for small errors and like MAE for large errors, giving you the best of both worlds [c478]:

$$\mathcal{L}_{Huber} = \begin{cases} \frac{(y_i - f(x_i))^2}{2} & \text{if } |y_i - f(x_i)| \leq \delta \\ \delta|y_i - f(x_i)| - \frac{\delta}{2} & \text{otherwise} \end{cases}$$

The hyperparameter $\delta$ controls the transition point between the two regimes.

**Log-Cosh Loss** is another smooth approximation to MAE [c479]:

$$\mathcal{L}_{LogCosh} = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(f(x_i) - y_i))$$

#### Classification Losses

Classification losses operate on probability distributions rather than continuous values.

**Binary Cross-Entropy** is used when there are exactly two classes [c480]:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

**Hinge Loss** is the loss behind support vector machines [c481]:

$$\mathcal{L}_{Hinge} = \max(0, 1 - (f(x) \cdot y))$$

**Cross-Entropy Loss** generalizes binary cross-entropy to $M$ classes [c482]:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{i,c} \log(f(x_i)_c)$$

**Kullback-Leibler Divergence** measures how one probability distribution differs from another [c483]:

$$\mathcal{L}_{KL} = -\sum_{i=1}^{N} y_i \cdot \log\left(\frac{y_i}{f(x_i)}\right)$$

---

### Cross-Entropy Loss for Language Models

Language modeling is a classification problem: at each position, the model must classify the next token as one of the $V$ tokens in the vocabulary. Cross-entropy loss is therefore the natural choice.

Cross entropy loss measures the difference between two probability distributions, specifically between predicted logits and target indices in language model training [c550]. Intuitively, the model outputs a probability distribution over the vocabulary, and the loss measures how much probability mass the model assigned to the *correct* token. If the model is confident and correct, the loss is near zero; if the model is confident and wrong, the loss is large.

#### From Logits to Loss: The Mechanics

Recall the forward pass of the GPT model. The transformer block is the main engine of the GPT architecture that transforms input embedding vectors into context vectors [c3775]. Context vectors are richer representations than input embedding vectors because they encode both semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776]. After the transformer blocks, a final linear projection produces logits—the raw output values produced by the GPT model architecture before normalization into probabilities [c3779].

A logit tensor is converted into a softmax tensor, which is a probability tensor [c3785]. Once you have probabilities, cross-entropy loss is computed by taking the logarithm of the target probabilities, summing them, taking the mean, and then negating the result [c3790]. This is equivalent to negative log likelihood (NLL)—the negative of the logarithm of the target probabilities [c3791].

More precisely, cross entropy loss is computed as: negative mean of log probabilities at target indices, or $-\frac{1}{N} \sum \log(p_i)$ where $p_i$ are probabilities at target indices [c552].

To see why this formula makes sense: if the model assigns probability 1.0 to the correct token, $\log(1.0) = 0$, so the loss is 0. If the model assigns probability 0.01 to the correct token, $\log(0.01) \approx -4.6$, so the loss is 4.6. The negative sign turns the negative log-probability into a positive loss value that we want to minimize.

#### Implementing Cross-Entropy in PyTorch

PyTorch provides `torch.nn.functional.cross_entropy`, which accepts raw logits (not softmax probabilities) and handles the softmax and log internally for numerical stability. The function signature is straightforward:

# Source: [c581]
```python
torch.nn.functional.cross_entropy(logit_tensor, target_tensor)
```

The function expects logits of shape `(batch_size, num_classes)` and targets of shape `(batch_size,)` containing integer class indices. For language modeling, the "classes" are vocabulary tokens, so `num_classes` equals the vocabulary size.

#### Flattening for Batched Sequences

In practice, the model produces logits of shape `(batch_size, sequence_length, vocab_size)`. For example, with a batch of 2 sequences each of length 3 and a vocabulary of 50,257 tokens (the GPT-2 vocabulary size), the logits tensor has shape `(2, 3, 50257)`. PyTorch's cross-entropy function expects a 2D logits tensor, so you need to flatten the first two dimensions:

# Source: [c569]
```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)  # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The `-1` in `view(-1, 50257)` tells PyTorch to infer the first dimension automatically. After flattening, you have 6 independent prediction problems (one per token position per sequence in the batch), each asking: "given the context up to this point, which of the 50,257 tokens comes next?"

The same operation can be written more concisely:

# Source: [c555]
```python
torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

And in a slightly different notation that emphasizes the dataset-level view:

# Source: [c3851]
```python
nn.functional.cross_entropy(flattened_logits, flattened_target_batch)
```

---

### Perplexity: A More Interpretable Metric

Raw cross-entropy loss values are hard to interpret in isolation. A loss of 3.5 is better than a loss of 4.0, but what does that mean in practice? Perplexity provides a more intuitive framing.

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. Concretely, perplexity is a metric for measuring loss that is calculated as $e$ raised to the loss value [c582]:

$$\text{Perplexity} = \exp(\text{loss})$$

[c574]

Intuitively, perplexity can be thought of as the effective vocabulary size the model is "confused" between at each step. A perplexity of 1 means the model is perfectly certain (it always assigns probability 1 to the correct token). A perplexity of 50,257 means the model is completely random—it treats every token in the vocabulary as equally likely. A well-trained GPT-2 model achieves perplexity in the tens to low hundreds on standard benchmarks.

The relationship $\text{Perplexity} = \exp(\text{loss})$ means that reducing loss by 1 nat (one unit of natural-log loss) cuts perplexity by a factor of $e \approx 2.718$. This is why even small improvements in loss translate to meaningful improvements in generation quality.

---

### The Training Loop

With the loss function defined, we can now build the training loop. The pre-training algorithm consists of three steps: finding the loss, performing the backward pass to get loss gradients, and updating parameters based on those gradients [c1622].

#### The Forward Pass

The forward pass runs the model on a batch of inputs to produce logits, then computes the loss against the targets. In terms of the model's internal computation:

1. Token IDs are looked up in the embedding table. Token embedding parameters scale as vocabulary size × embedding dimension [c1630].
2. Positional embeddings are added. Positional embedding parameters scale as context size × embedding dimension [c1631].
3. The combined embeddings pass through the transformer blocks, which produce context vectors [c3775, c3776].
4. A final linear layer projects context vectors to logits. Final layer parameters scale as embedding dimension × vocabulary size [c1644].

The embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

#### The Backward Pass

The backward pass is the most important step in the pre-training loop because it calculates the gradients of the loss [c1616]. In PyTorch, this is a single call:

# Source: [c1624]
```python
loss.backward()
```

This call traverses the computation graph in reverse, computing the gradient of the loss with respect to every parameter in the model. These gradients tell the optimizer which direction to move each parameter to reduce the loss.

#### Parameter Updates

Once gradients are computed, the optimizer updates the parameters. The fundamental update rule is [c1618]:

$$p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$$

This is gradient descent: move each parameter in the direction that reduces the loss, scaled by the learning rate (step size). In practice, LLM pre-training uses the AdamW optimizer rather than vanilla gradient descent. AdamW adapts the learning rate for each parameter individually and includes weight decay regularization, which helps prevent overfitting.

[GAP: Explicit code for the full training loop including optimizer initialization, zero_grad, loss.backward, and optimizer.step is not present in the supplied claims.]

#### Utility Functions for Text and Tokens

During training and evaluation, it is useful to convert between text and token IDs. The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538]. The `token_ids_to_text` function converts token IDs back into text, serving as the reverse of `text_to_token_ids` [c539]. Tiktoken provides a byte pair encoder that operates at character and sub-word level [c3817].

In the context of language model training, "target" refers to the true values that the model should predict [c502]. The `X` tensor represents the input and `Y` represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767]. Target output is the expected output sequence that corresponds to a given input sequence, which the LLM should learn to predict during training [c3770].

---

### Evaluating the Model During Training

Training a model without monitoring its progress is like driving without looking at the road. You need to track both training loss and validation loss to understand whether the model is learning and whether it is generalizing.

#### Training Loss vs. Validation Loss

Training loss measures how well the model fits the data it is currently being trained on. Validation loss measures how well the model generalizes to data it has never seen. The gap between these two numbers is the primary diagnostic for overfitting.

When training on a small dataset, overfitting is a real concern. The model may memorize the training examples rather than learning generalizable patterns, leading to low training loss but high validation loss. This is especially common when the model has many parameters relative to the amount of training data.

[GAP: Explicit code for the evaluate_model function, the train_model_simple function, and the loss-plotting routine is not present in the supplied claims.]

#### Interpreting Learning Curves

A healthy learning curve shows both training and validation loss decreasing together over time. If training loss decreases but validation loss plateaus or increases, the model is overfitting. If both losses plateau at a high value, the model may be underfitting—perhaps the learning rate is too low, or the model is too small for the task.

[FIGURE: Two-panel plot showing training loss and validation loss curves over epochs; left panel shows healthy convergence (both curves decreasing), right panel shows overfitting (training loss decreasing, validation loss increasing)]

---

### Generating Text During Training

It is useful to periodically generate text samples during training to get a qualitative sense of how the model is progressing. Even before the loss has converged, you can often see the model transitioning from random-looking output to grammatically plausible text to semantically coherent text.

The simplest generation strategy is greedy decoding: greedy decoding is the process of always selecting the token with the highest probability as the next token using `torch.argmax` [c3045]. In PyTorch, `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension (columns) [c547].

Greedy decoding is deterministic—given the same input, it always produces the same output. This is useful for reproducibility but can lead to repetitive or degenerate text. We will address this with better decoding strategies later in the chapter.

---

### Decoding Strategies: Beyond Greedy

LLM decoding strategies are techniques used to control randomness and improve quality when generating the next token in sequence [c750]. The naive decoding strategy selects the generated token corresponding to the largest probability score among all tokens in the vocabulary [c751]. While simple, this greedy approach has well-known failure modes: it tends to produce repetitive text and can get stuck in loops.

Two complementary strategies address these problems: temperature scaling and top-k sampling. Top-k sampling and temperature scaling are decoding strategies that help avoid overfitting [c798].

#### Temperature Scaling

Temperature scaling is a technique used in Large Language Models to control randomness and diversity in generated text [c3025]. More precisely, temperature scaling is dividing the logits by a number greater than zero to control the probability distribution and token selection process [c3058].

Temperature scaling is a decoding strategy where logits are divided by a temperature value before applying softmax to convert them into probabilities [c753]. Temperature is a parameter that controls the entropy of the probability distribution over tokens in language model sampling [c3074]. Temperature scaling adjusts the sharpness of probability distributions to either sharpen or flatten them, helping prevent overfitting through multinomial sampling [c801].

To see how this works in practice, consider two extremes:

**Low temperature (e.g., 0.1):** Dividing logits by a small number makes them larger in magnitude, which sharpens the softmax distribution. The highest-probability token gets even more probability mass, and the distribution approaches a one-hot vector. The model becomes more deterministic and "confident."

# Source: [c3061]
```python
next_token_logits_2 = next_token_logits / 0.1
probabilities = softmax(next_token_logits_2)
```

**High temperature (e.g., 5):** Dividing logits by a large number makes them smaller in magnitude, which flattens the softmax distribution. Probability mass spreads more evenly across tokens, and the model becomes more random and "creative."

# Source: [c3063]
```python
next_token_logits_3 = next_token_logits / 5
probabilities = softmax(next_token_logits_3)
```

Temperature scaling is a decoding strategy used to control randomness in model predictions [c1719]. Temperature scaling is a technique for controlling randomness when sampling the next token from a probability distribution [c3032].

#### Multinomial Sampling

Once you have a probability distribution (whether temperature-scaled or not), you need a way to sample from it. The probability distribution used for sampling the next token is the multinomial probability distribution [c3036].

Sampling from a distribution means drawing values where the outcome is not predetermined; for a multinomial distribution, samples are drawn according to the probability scores of mutually exclusive outcomes [c3037]. The multinomial distribution is used for $k$ mutually exclusive outcomes, each with corresponding probabilities, where $n$ independent trials are conducted to predict outcomes [c3038].

Multinomial distribution sampling selects the next token probabilistically proportional to its probability score rather than deterministically choosing the maximum probability token [c754]. The multinomial function samples the next token proportional to its probability score [c3049].

PyTorch's `torch.multinomial` implements this directly. Here is a simple example with four tokens having weights `[0, 10, 3, 0]`:

# Source: [c3039]
```python
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

Token 0 and token 3 have zero weight, so they are never sampled. Token 1 has weight 10 and token 2 has weight 3, so token 1 is sampled roughly 77% of the time and token 2 roughly 23% of the time.

By default, `torch.multinomial` samples without replacement. If you ask for more samples than there are non-zero-weight tokens, you get an error:

# Source: [c3040]
```python
torch.multinomial(weights, 5)  # ERROR without replacement
# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

For token generation, you typically want sampling with replacement (each draw is independent):

# Source: [c3041]
```python
torch.multinomial(weights, 4, replacement=True)
# Returns: tensor([2, 1, 1, 1])
```

#### Top-K Sampling

Even with temperature scaling, multinomial sampling can occasionally select very low-probability tokens, producing nonsensical output. Top-k sampling addresses this by restricting the candidate set.

Top-K sampling is a decoding strategy that restricts sampled tokens to the top K most likely tokens and excludes all other tokens [c760]. Top-k sampling restricts the sample tokens to the top k most likely tokens and excludes all other tokens [c762]. More precisely, top-k sampling is a method that selects only the top k tokens with the highest logits for the next token prediction [c769]. Top-k sampling restricts sampled tokens to the top k most likely tokens, preventing random or low-probability tokens from becoming the next token [c802].

The implementation uses two PyTorch operations. First, find the top-k logits:

# Source: [c771]
```python
torch.topk(logits, k=3) returns the top 3 values and their indices
```

Then, set all other logits to negative infinity so that softmax assigns them zero probability:

# Source: [c773]
```python
torch.where(condition, logits, -inf) replaces logits that don't meet the condition with negative infinity
```

After masking, apply softmax to the remaining logits to get a valid probability distribution, then sample from it using `torch.multinomial`.

[FIGURE: Diagram showing top-k filtering: a bar chart of token probabilities, with the top-3 bars highlighted and all others set to zero, followed by renormalization]

#### Combining Temperature and Top-K

In practice, temperature scaling and top-k sampling are used together. The typical pipeline is:

1. Compute logits from the model.
2. Divide logits by the temperature parameter.
3. Keep only the top-k logits; set the rest to $-\infty$.
4. Apply softmax to get probabilities.
5. Sample from the resulting distribution using `torch.multinomial`.

This combination gives you control over both the sharpness of the distribution (temperature) and the size of the candidate set (top-k). A common default is temperature around 1.0 and $k$ around 50, but these are hyperparameters that should be tuned for your specific application.

[GAP: Explicit code for the combined generate function incorporating temperature and top-k is not present in the supplied claims.]

---

### Putting It All Together

Let us trace through a complete training iteration to consolidate everything covered in this chapter.

**Step 1: Load a batch.** The dataloader produces a batch of input tensors `X` and target tensors `Y`. Both have shape `(batch_size, context_length)`. The targets are the inputs shifted by one position [c542].

**Step 2: Forward pass.** The model processes `X` through its embedding layers, transformer blocks, and final linear layer to produce logits of shape `(batch_size, context_length, vocab_size)`.

**Step 3: Compute loss.** Flatten the logits and targets, then compute cross-entropy loss:

# Source: [c569]
```python
logits_flat = logits.view(-1, 50257)  # Flatten first two dimensions from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)  # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

**Step 4: Backward pass.** Compute gradients:

# Source: [c1624]
```python
loss.backward()
```

**Step 5: Update parameters.** The optimizer applies the update rule $p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$ [c1618] to every parameter in the model.

**Step 6: Evaluate.** Periodically compute validation loss and perplexity ($\exp(\text{loss})$ [c574]) to track generalization.

**Step 7: Generate.** Periodically generate text samples using temperature scaling and top-k sampling to get a qualitative sense of model quality.

---

### Summary

This chapter has covered the core machinery of LLM pre-training:

- **Loss functions.** We surveyed the landscape of regression and classification losses, then focused on cross-entropy loss, which is the appropriate choice for language modeling because next-token prediction is a classification problem over the vocabulary [c550].

- **Cross-entropy mechanics.** The loss is computed as the negative mean log-probability at target indices [c552]. In PyTorch, `torch.nn.functional.cross_entropy` handles the softmax and log internally for numerical stability [c581].

- **Perplexity.** A more interpretable metric defined as $\exp(\text{loss})$ [c574], perplexity measures how well the model's predicted distribution matches the actual distribution of words [c572].

- **The training loop.** Pre-training consists of a forward pass to compute loss, a backward pass to compute gradients [c1616], and a parameter update step [c1618, c1622].

- **Evaluation.** Tracking both training and validation loss reveals whether the model is learning and whether it is overfitting.

- **Decoding strategies.** Temperature scaling [c753] and top-k sampling [c760] are complementary techniques that improve generation quality by controlling the randomness and candidate set of the next-token distribution.

With these tools in hand, you have everything you need to train a GPT model from scratch on real text data. The next chapter will look at how to scale this process and what happens when you train on much larger datasets.