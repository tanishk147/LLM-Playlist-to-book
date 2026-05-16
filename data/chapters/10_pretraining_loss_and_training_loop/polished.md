## Pre-Training: Loss Functions, the Training Loop, and Evaluation

Training a large language model from scratch requires three things working in concert: a loss function that tells the model how wrong it is, a training loop that iterates toward a better answer, and evaluation machinery that tells you whether progress is real. This chapter covers all three, then closes with a look at decoding strategies that make the model's output more interesting at inference time.

---

### The Big Picture: What Pre-Training Is Doing

Large language models are autoregressive models where input-output pairs are constructed from the text itself without any pre-labeling [c3753]. The model sees a sequence of tokens and must predict the next one.

The key structural idea is simple: targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. Every token in the input simultaneously acts as a context element for predicting the next token and as a target for the token that came before it.

Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754]. The dataloader that feeds training uses two parameters to control how it sweeps through the corpus: `max_length`, which specifies the context size (number of tokens to consider at once) [c3822], and `stride`, which specifies how many steps to advance before creating the next input-output pair [c3823].

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. Two additional parameters worth knowing: `shuffle` shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828], and `drop_last` drops the last batch if its size is smaller than the specified batch size [c3829], preventing shape mismatches during training.

One epoch is a single pass through the entire training set [c1613].

---

### Loss Functions: A Taxonomy

#### Regression Losses

Regression losses measure the discrepancy between a continuous prediction $f(x_i)$ and a continuous target $y_i$.

**Mean Bias Error** averages the signed errors [c474]:

$$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

**Mean Absolute Error** takes the absolute value before averaging, so positive and negative errors cannot cancel [c475]:

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - f(x_i)|$$

**Mean Squared Error** squares the errors, penalizing large deviations more heavily [c476]:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

**Root Mean Squared Error** brings the loss back to the same units as the target [c477]:

$$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2}$$

**Huber Loss** is a hybrid that behaves like MSE for small errors and like MAE for large ones [c478]:

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

Cross-entropy loss measures the difference between two probability distributions—specifically between predicted logits and target indices in language model training [c550]. Intuitively, the model outputs a probability distribution over the vocabulary, and the loss measures how much probability mass the model assigned to the *correct* token.

#### From Logits to Loss: The Mechanics

The transformer block is the main engine of the GPT architecture, transforming input embedding vectors into context vectors [c3775]. Context vectors are richer representations than input embeddings because they encode both the semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776]. After the transformer blocks, a final linear projection produces logits—the raw output values of the GPT model before normalization into probabilities [c3779].

A logit tensor is converted into a softmax tensor, yielding a probability tensor [c3785]. Cross-entropy loss is then computed by taking the logarithm of the target probabilities, summing them, taking the mean, and negating the result [c3790]. This is equivalent to negative log likelihood (NLL)—the negative of the logarithm of the target probabilities [c3791]. More precisely, cross-entropy loss equals the negative mean of log probabilities at target indices: $-\frac{1}{N} \sum \log(p_i)$, where $p_i$ are the probabilities at target indices [c552]. The negative sign ensures the result is a positive value that we want to minimize.

#### Implementing Cross-Entropy in PyTorch

PyTorch provides `torch.nn.functional.cross_entropy`, which accepts raw logits (not softmax probabilities) and handles the softmax and logarithm internally for numerical stability [c581]:

```python
torch.nn.functional.cross_entropy(logit_tensor, target_tensor)
```

The function expects logits of shape `(batch_size, num_classes)` and targets of shape `(batch_size,)` containing integer class indices.

#### Flattening for Batched Sequences

PyTorch's cross-entropy function expects a 2D logits tensor, so when working with batched sequences you need to flatten the first two dimensions [c569]:

```python
logits_flat = logits.view(-1, 50257)  # Flatten from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)       # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

The same operation can be written more concisely [c555]:

```python
torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

And in a slightly different notation that emphasizes the dataset-level view [c3851]:

```python
nn.functional.cross_entropy(flattened_logits, flattened_target_batch)
```

---

### Perplexity: A More Interpretable Metric

Cross-entropy loss is mathematically convenient, but its scale is not always easy to interpret. Perplexity provides a more intuitive framing.

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. Concretely, it is calculated as $e$ raised to the loss value [c582]:

$$\text{Perplexity} = \exp(\text{loss})$$ [c574]

A perplexity of 1 would mean the model is perfectly certain about every next token; higher values indicate greater uncertainty.

---

### The Training Loop

With the loss function defined, we can now build the training loop. The pre-training algorithm consists of three steps: computing the loss, performing the backward pass to obtain loss gradients, and updating parameters based on those gradients [c1622].

#### The Forward Pass

The model's internal computation proceeds as follows:

1. Token embedding parameters scale as vocabulary size × embedding dimension [c1630].
2. Positional embedding parameters scale as context size × embedding dimension [c1631].
3. The combined embeddings pass through the transformer blocks, which produce context vectors [c3775, c3776].
4. A final linear layer projects context vectors to logits, with parameters scaling as embedding dimension × vocabulary size [c1644].

The embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

#### The Backward Pass

The backward pass is the most important step in the pre-training loop because it calculates the gradients of the loss [c1616]:

```python
loss.backward()
```

This call traverses the computation graph in reverse, computing the gradient of the loss with respect to every parameter in the model. These gradients tell the optimizer which direction to move each parameter to reduce the loss.

#### Parameter Updates

The fundamental update rule is [c1618]:

$$p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$$

In practice, optimizers such as AdamW are used because AdamW adapts the learning rate for each parameter individually and includes weight decay regularization, which helps prevent overfitting.

#### Utility Functions for Text and Tokens

Two helper functions bridge raw text and the token IDs the model operates on. The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538], and `token_ids_to_text` converts token IDs back into text, serving as its inverse [c539]. Tiktoken provides a byte pair encoder that operates at the character and sub-word level [c3817].

In the context of language model training, "target" refers to the true values that the model should predict [c502]. The `X` tensor represents the input and `Y` represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767], where the target output is the expected sequence the LLM should learn to predict [c3770].

---

### Evaluating the Model During Training

#### Training Loss vs. Validation Loss

Tracking both training loss and validation loss during training is essential for diagnosing whether the model is learning generalizable patterns. The model may memorize training examples rather than learning to generalize, leading to low training loss but high validation loss—a condition known as overfitting.

#### Interpreting Learning Curves

A healthy learning curve shows both training and validation loss decreasing together over time. A growing gap between the two signals overfitting and calls for intervention, such as regularization or early stopping.

---

### Generating Text During Training

Monitoring generated text alongside loss curves gives a qualitative sense of how the model is improving. The simplest generation strategy is greedy decoding: always selecting the token with the highest probability as the next token using `torch.argmax` [c3045]. In PyTorch, `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension [c547].

Greedy decoding is fast and deterministic, but it tends to produce repetitive, low-diversity output. The decoding strategies discussed in the next section address this limitation.

---

### Decoding Strategies: Beyond Greedy

LLM decoding strategies are techniques used to control randomness and improve quality when generating the next token in sequence [c750]. The naive strategy simply selects the token with the largest probability score among all tokens in the vocabulary [c751], which is exactly greedy decoding. Two complementary strategies do better: temperature scaling and top-k sampling, both of which help avoid overfitting [c798].

#### Temperature Scaling

Temperature scaling is a technique used in large language models to control randomness and diversity in generated text [c3025]. More precisely, it involves dividing the logits by a number greater than zero to control the probability distribution and token selection process [c3058]. Logits are divided by the temperature value before applying softmax to convert them into probabilities [c753]. Temperature is a parameter that controls the entropy of the probability distribution over tokens [c3074], adjusting the sharpness of that distribution to either concentrate or spread probability mass, which helps prevent overfitting through multinomial sampling [c801].

To see how this works in practice, consider two extremes:

**Low temperature (e.g., 0.1):** The highest-probability token receives even more probability mass, and the distribution approaches a one-hot vector. The model becomes more deterministic and "confident." [c3061]

```python
next_token_logits_2 = next_token_logits / 0.1
probabilities = softmax(next_token_logits_2)
```

**High temperature (e.g., 5):** Probability mass spreads more evenly across tokens, and the model becomes more random and "creative." [c3063]

```python
next_token_logits_3 = next_token_logits / 5
probabilities = softmax(next_token_logits_3)
```

In short, temperature scaling is a decoding strategy for controlling randomness in model predictions [c1719, c3032].

#### Multinomial Sampling

Rather than always picking the single most likely token, multinomial sampling draws the next token proportionally to its probability score. The probability distribution used for this purpose is the multinomial probability distribution [c3036].

Sampling from a distribution means drawing values where the outcome is not predetermined; for a multinomial distribution, samples are drawn according to the probability scores of mutually exclusive outcomes [c3037]. The multinomial distribution is used for $k$ mutually exclusive outcomes, each with corresponding probabilities, where $n$ independent trials are conducted to predict outcomes [c3038]. This approach selects the next token probabilistically proportional to its probability score rather than deterministically choosing the maximum [c754, c3049].

PyTorch's `torch.multinomial` implements this directly [c3039]:

```python
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

By default, `torch.multinomial` samples without replacement, so requesting more samples than there are non-zero weights raises an error [c3040]:

```python
torch.multinomial(weights, 5)  # ERROR without replacement
# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

Passing `replacement=True` removes this restriction [c3041]:

```python
torch.multinomial(weights, 4, replacement=True)
# Returns: tensor([2, 1, 1, 1])
```

#### Top-K Sampling

Even with temperature scaling, multinomial sampling can occasionally select very low-probability tokens, producing nonsensical output. Top-k sampling addresses this by restricting the candidate set.

Top-k sampling is a decoding strategy that restricts sampled tokens to the top $k$ most likely tokens and excludes all others [c760, c762], selecting only the $k$ tokens with the highest logits for the next token prediction [c769] and thereby preventing random or low-probability tokens from being chosen [c802].

The implementation has two steps. First, find the top-$k$ logits [c771]:

```python
torch.topk(logits, k=3)  # returns the top 3 values and their indices
```

Then set all other logits to negative infinity so that softmax assigns them zero probability [c773]:

```python
torch.where(condition, logits, -inf)  # replaces non-top-k logits with -inf
```

After masking, apply softmax to the remaining logits to obtain a valid probability distribution, then sample from it using `torch.multinomial`.

[FIGURE: Diagram showing top-k filtering: a bar chart of token probabilities, with the top-3 bars highlighted and all others set to zero, followed by renormalization]

#### Combining Temperature and Top-K

Temperature scaling and top-k sampling are most effective when used together. The typical pipeline is:

1. Divide logits by the temperature value.
2. Retain only the top-$k$ logits, setting the rest to $-\infty$.
3. Apply softmax to obtain a probability distribution.
4. Sample from the resulting distribution using `torch.multinomial`.

This combination gives you independent control over both the sharpness of the distribution (temperature) and the size of the candidate set (top-k).

---

### Putting It All Together

To consolidate everything covered in this chapter, here is a complete training iteration traced step by step.

**Step 1: Load a batch.** The dataloader yields an input tensor `X` and a target tensor `Y`. The targets are the inputs shifted by one position [c542].

**Step 2: Forward pass.** The model transforms token embeddings through the transformer blocks into context vectors, then projects them to logits [c3775, c3776, c3779].

**Step 3: Compute loss.** Flatten logits and targets, then apply cross-entropy [c569]:

```python
logits_flat = logits.view(-1, 50257)  # Flatten from (2, 3, 50257) to (6, 50257)
targets_flat = targets.view(-1)       # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

**Step 4: Backward pass.** Compute gradients [c1624]:

```python
loss.backward()
```

**Step 5: Update parameters.** The optimizer applies the update rule $p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$ [c1618] to every parameter in the model.

**Step 6: Evaluate.** Periodically compute validation loss and perplexity ($\exp(\text{loss})$ [c574]) to track generalization.

---

### Summary

- **Loss functions.** We surveyed the landscape of regression and classification losses, then focused on cross-entropy loss, which is the appropriate choice for language modeling because next-token prediction is a classification problem over the vocabulary [c550].
- **Cross-entropy mechanics.** The loss is computed as the negative mean log-probability at target indices [c552]. PyTorch's `torch.nn.functional.cross_entropy` handles the softmax and logarithm internally for numerical stability [c581].
- **Perplexity.** Defined as $\exp(\text{loss})$ [c574], perplexity measures how well the model's predicted distribution matches the actual distribution of words [c572], providing a more interpretable complement to raw loss.
- **The training loop.** Pre-training consists of a forward pass to compute loss, a backward pass to compute gradients [c1616], and a parameter update step [c1618, c1622].
- **Decoding strategies.** Temperature scaling [c753] and top-k sampling [c760] are complementary techniques that improve generation quality by controlling the randomness and candidate set of the next-token distribution.

The next chapter will examine how to scale this process and what happens when training on much larger datasets.