## Pre-Training: Loss Functions, the Training Loop, and Evaluation

 This chapter covers all three, then closes with a look at decoding strategies that make the model's output more interesting at inference time.

---

### The Big Picture: What Pre-Training Is Doing

Large language models are autoregressive models where input-output pairs are constructed from the text itself without any pre-labeling [c3753]. The model sees a sequence of tokens and must predict the next one. 

The key structural idea is simple: targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. Every token in the input simultaneously acts as a context element for predicting the next token and as a target for the token that came before it. 

Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754]. The dataloader that feeds training uses two parameters to control how it sweeps through the corpus: `max_length`, which specifies the context size (number of tokens to consider at once) [c3822], and `stride`, which specifies how many steps to advance before creating the next input-output pair [c3823]. 

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. Two additional parameters worth knowing: the `shuffle` parameter shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828], and the `drop_last` parameter drops the last batch if its size is smaller than the specified batch size [c3829], which prevents shape mismatches during training.

One epoch is going through the entire training set once [c1613]. 

---

### Loss Functions: A Taxonomy

 

#### Regression Losses

Regression losses measure the discrepancy between a continuous prediction $f(x_i)$ and a continuous target $y_i$.

**Mean Bias Error** simply averages the signed errors [c474]:

**Mean Absolute Error** takes the absolute value before averaging, so errors cannot cancel [c475]:

**Mean Squared Error** squares the errors, penalizing large deviations more heavily [c476]:

 

**Root Mean Squared Error** brings the loss back to the same units as the target [c477]:

**Huber Loss** is a hybrid that behaves like MSE for small errors and like MAE for large errors, giving you the best of both worlds [c478]:

The hyperparameter $\delta$ controls the transition point between the two regimes.

**Log-Cosh Loss** is another smooth approximation to MAE [c479]:

#### Classification Losses

Classification losses operate on probability distributions rather than continuous values.

**Binary Cross-Entropy** is used when there are exactly two classes [c480]:

**Hinge Loss** is the loss behind support vector machines [c481]:

**Cross-Entropy Loss** generalizes binary cross-entropy to $M$ classes [c482]:

**Kullback-Leibler Divergence** measures how one probability distribution differs from another [c483]:

---

### Cross-Entropy Loss for Language Models

 

Cross entropy loss measures the difference between two probability distributions, specifically between predicted logits and target indices in language model training [c550]. Intuitively, the model outputs a probability distribution over the vocabulary, and the loss measures how much probability mass the model assigned to the *correct* token. 

#### From Logits to Loss: The Mechanics

 The transformer block is the main engine of the GPT architecture that transforms input embedding vectors into context vectors [c3775]. Context vectors are richer representations than input embedding vectors because they encode both semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776]. After the transformer blocks, a final linear projection produces logits—the raw output values produced by the GPT model architecture before normalization into probabilities [c3779].

A logit tensor is converted into a softmax tensor, which is a probability tensor [c3785]. Once you have probabilities, cross-entropy loss is computed by taking the logarithm of the target probabilities, summing them, taking the mean, and then negating the result [c3790]. This is equivalent to negative log likelihood (NLL)—the negative of the logarithm of the target probabilities [c3791].

More precisely, cross entropy loss is computed as: negative mean of log probabilities at target indices, or $-\frac{1}{N} \sum \log(p_i)$ where $p_i$ are probabilities at target indices [c552].

 The negative sign turns the negative log-probability into a positive loss value that we want to minimize.

#### Implementing Cross-Entropy in PyTorch

PyTorch provides `torch.nn.functional.cross_entropy`, which accepts raw logits (not softmax probabilities) and handles the softmax and log internally for numerical stability. 

# Source: [c581]
```python

```

The function expects logits of shape `(batch_size, num_classes)` and targets of shape `(batch_size,)` containing integer class indices. 

#### Flattening for Batched Sequences

 PyTorch's cross-entropy function expects a 2D logits tensor, so you need to flatten the first two dimensions:

# Source: [c569]
```python

targets_flat = targets.view(-1) # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

 

The same operation can be written more concisely:

# Source: [c555]
```python

```

And in a slightly different notation that emphasizes the dataset-level view:

# Source: [c3851]
```python

```

---

### Perplexity: A More Interpretable Metric

 Perplexity provides a more intuitive framing.

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. Concretely, perplexity is a metric for measuring loss that is calculated as $e$ raised to the loss value [c582]:

[c574]

 

 

---

### The Training Loop

With the loss function defined, we can now build the training loop. The pre-training algorithm consists of three steps: finding the loss, performing the backward pass to get loss gradients, and updating parameters based on those gradients [c1622].

#### The Forward Pass

 In terms of the model's internal computation:

 Token embedding parameters scale as vocabulary size × embedding dimension [c1630].
2. Positional embedding parameters scale as context size × embedding dimension [c1631].
3. The combined embeddings pass through the transformer blocks, which produce context vectors [c3775, c3776].
4. A final linear layer projects context vectors to logits. Final layer parameters scale as embedding dimension × vocabulary size [c1644].

The embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

#### The Backward Pass

The backward pass is the most important step in the pre-training loop because it calculates the gradients of the loss [c1616]. 

# Source: [c1624]
```python

```

This call traverses the computation graph in reverse, computing the gradient of the loss with respect to every parameter in the model. These gradients tell the optimizer which direction to move each parameter to reduce the loss.

#### Parameter Updates

 The fundamental update rule is [c1618]:

 AdamW adapts the learning rate for each parameter individually and includes weight decay regularization, which helps prevent overfitting.

#### Utility Functions for Text and Tokens

 The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538]. The `token_ids_to_text` function converts token IDs back into text, serving as the reverse of `text_to_token_ids` [c539]. Tiktoken provides a byte pair encoder that operates at character and sub-word level [c3817].

In the context of language model training, "target" refers to the true values that the model should predict [c502]. The `X` tensor represents the input and `Y` represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767]. Target output is the expected output sequence that corresponds to a given input sequence, which the LLM should learn to predict during training [c3770].

---

### Evaluating the Model During Training

 

#### Training Loss vs. Validation Loss

 

 The model may memorize the training examples rather than learning generalizable patterns, leading to low training loss but high validation loss. 

#### Interpreting Learning Curves

A healthy learning curve shows both training and validation loss decreasing together over time. 

---

### Generating Text During Training

 

The simplest generation strategy is greedy decoding: greedy decoding is the process of always selecting the token with the highest probability as the next token using `torch.argmax` [c3045]. In PyTorch, `torch.argmax(input, dim=-1)` returns the indices of the maximum value along the last dimension (columns) [c547].

 We will address this with better decoding strategies later in the chapter.

---

### Decoding Strategies: Beyond Greedy

LLM decoding strategies are techniques used to control randomness and improve quality when generating the next token in sequence [c750]. The naive decoding strategy selects the generated token corresponding to the largest probability score among all tokens in the vocabulary [c751]. 

Two complementary strategies address these problems: temperature scaling and top-k sampling. Top-k sampling and temperature scaling are decoding strategies that help avoid overfitting [c798].

#### Temperature Scaling

Temperature scaling is a technique used in Large Language Models to control randomness and diversity in generated text [c3025]. More precisely, temperature scaling is dividing the logits by a number greater than zero to control the probability distribution and token selection process [c3058].

Temperature scaling is a decoding strategy where logits are divided by a temperature value before applying softmax to convert them into probabilities [c753]. Temperature is a parameter that controls the entropy of the probability distribution over tokens in language model sampling [c3074]. Temperature scaling adjusts the sharpness of probability distributions to either sharpen or flatten them, helping prevent overfitting through multinomial sampling [c801].

To see how this works in practice, consider two extremes:

 The highest-probability token gets even more probability mass, and the distribution approaches a one-hot vector. The model becomes more deterministic and "confident."

# Source: [c3061]
```python

probabilities = softmax(next_token_logits_2)
```

 Probability mass spreads more evenly across tokens, and the model becomes more random and "creative."

# Source: [c3063]
```python

probabilities = softmax(next_token_logits_3)
```

Temperature scaling is a decoding strategy used to control randomness in model predictions [c1719]. Temperature scaling is a technique for controlling randomness when sampling the next token from a probability distribution [c3032].

#### Multinomial Sampling

 The probability distribution used for sampling the next token is the multinomial probability distribution [c3036].

Sampling from a distribution means drawing values where the outcome is not predetermined; for a multinomial distribution, samples are drawn according to the probability scores of mutually exclusive outcomes [c3037]. The multinomial distribution is used for $k$ mutually exclusive outcomes, each with corresponding probabilities, where $n$ independent trials are conducted to predict outcomes [c3038].

Multinomial distribution sampling selects the next token probabilistically proportional to its probability score rather than deterministically choosing the maximum probability token [c754]. The multinomial function samples the next token proportional to its probability score [c3049].

PyTorch's `torch.multinomial` implements this directly. 

# Source: [c3039]
```python

torch.multinomial(weights, 2)
# Returns: tensor([1, 2])
```

 

By default, `torch.multinomial` samples without replacement. 

# Source: [c3040]
```python

# RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

# Source: [c3041]
```python

# Returns: tensor([2, 1, 1, 1])
```

#### Top-K Sampling

Even with temperature scaling, multinomial sampling can occasionally select very low-probability tokens, producing nonsensical output. Top-k sampling addresses this by restricting the candidate set.

Top-K sampling is a decoding strategy that restricts sampled tokens to the top K most likely tokens and excludes all other tokens [c760]. Top-k sampling restricts the sample tokens to the top k most likely tokens and excludes all other tokens [c762]. More precisely, top-k sampling is a method that selects only the top k tokens with the highest logits for the next token prediction [c769]. Top-k sampling restricts sampled tokens to the top k most likely tokens, preventing random or low-probability tokens from becoming the next token [c802].

 First, find the top-k logits:

# Source: [c771]
```python

```

Then, set all other logits to negative infinity so that softmax assigns them zero probability:

# Source: [c773]
```python

```

After masking, apply softmax to the remaining logits to get a valid probability distribution, then sample from it using `torch.multinomial`.

[FIGURE: Diagram showing top-k filtering: a bar chart of token probabilities, with the top-3 bars highlighted and all others set to zero, followed by renormalization]

#### Combining Temperature and Top-K

 The typical pipeline is:

 Sample from the resulting distribution using `torch.multinomial`.

This combination gives you control over both the sharpness of the distribution (temperature) and the size of the candidate set (top-k). 

---

### Putting It All Together

Let us trace through a complete training iteration to consolidate everything covered in this chapter.

 The targets are the inputs shifted by one position [c542].

# Source: [c569]
```python

targets_flat = targets.view(-1) # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

# Source: [c1624]
```python

```

**Step 5: Update parameters.** The optimizer applies the update rule $p_{\text{new}} = p_{\text{old}} - \text{step\_size} \times \text{loss\_gradient}$ [c1618] to every parameter in the model.

**Step 6: Evaluate.** Periodically compute validation loss and perplexity ($\exp(\text{loss})$ [c574]) to track generalization.

---

### Summary

- **Loss functions.** We surveyed the landscape of regression and classification losses, then focused on cross-entropy loss, which is the appropriate choice for language modeling because next-token prediction is a classification problem over the vocabulary [c550].

- **Cross-entropy mechanics.** The loss is computed as the negative mean log-probability at target indices [c552]. In PyTorch, `torch.nn.functional.cross_entropy` handles the softmax and log internally for numerical stability [c581].

- **Perplexity.** A more interpretable metric defined as $\exp(\text{loss})$ [c574], perplexity measures how well the model's predicted distribution matches the actual distribution of words [c572].

- **The training loop.** Pre-training consists of a forward pass to compute loss, a backward pass to compute gradients [c1616], and a parameter update step [c1618, c1622].

- **Decoding strategies.** Temperature scaling [c753] and top-k sampling [c760] are complementary techniques that improve generation quality by controlling the randomness and candidate set of the next-token distribution.

 The next chapter will look at how to scale this process and what happens when you train on much larger datasets.
