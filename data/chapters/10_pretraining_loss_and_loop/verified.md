## Pre-Training: Loss Functions, Training Loop, and Evaluation

 This chapter builds each of those three pieces from scratch. We start by understanding exactly what cross-entropy loss means in the context of next-token prediction, then implement helper functions that compute loss over batches and entire data loaders, and finally assemble the full pre-training loop—complete with backpropagation, parameter updates, and periodic evaluation. 

---

### The Loss Landscape: Why Cross-Entropy?

#### A Taxonomy of Loss Functions

For regression—predicting a continuous value—common choices include:

**Mean Bias Error (MBE)** simply averages the signed errors [c474]:

**Mean Absolute Error (MAE)** removes the sign by taking absolute values [c475]:

**Mean Squared Error (MSE)** squares the residuals, penalizing large errors more heavily [c476]:

**Root Mean Squared Error (RMSE)** restores the original units by taking the square root [c477]:

**Huber Loss** blends MSE and MAE: it behaves quadratically for small errors and linearly for large ones, controlled by a threshold $\delta$ [c478]:

**Log-Cosh Loss** is a smooth approximation to MAE [c479]:

For classification tasks, the picture changes. The model no longer predicts a scalar; it predicts a probability distribution over discrete classes. Here the standard choices are [c484]:

**Binary Cross-Entropy (BCE)** for two-class problems [c480]:

**Hinge Loss**, used in support-vector-style classifiers [c481]:

**Cross-Entropy Loss** for multi-class problems [c482]:

**Kullback-Leibler Divergence** measures how one probability distribution diverges from another [c483]:

#### Why Language Modeling Uses Cross-Entropy

 Cross-entropy loss measures the difference between two probability distributions—specifically, between the predicted logits and the target token indices [c550]. The goal of LLM training is to minimize this cross-entropy loss so that the predicted probabilities for target tokens approach one [c1609].

Intuitively, cross-entropy loss is computed by taking the logarithm of the probability the model assigns to the correct token, summing those log-probabilities across all positions, taking the mean, and then negating the result [c3790]. In equation form [c552]:

 This quantity is also called **negative log likelihood (NLL)**—the negative of the logarithm of the target probabilities [c3791]. Cross-entropy loss (negative log likelihood) is the standard measure of the difference between predicted probabilities and target token indices in LLM training [c1608].

 As the model becomes less certain or more wrong, the loss grows. The goal of training an LLM is to drive this negative log likelihood loss toward zero [c3792].

---

### From Logits to Loss: The Forward Pass in Detail

#### The Model's Output: Logits

Large language models are autoregressive models where input-output pairs are constructed from the text itself without pre-labeling [c3753]. The transformer block is the main engine of the GPT architecture, transforming input embedding vectors into context vectors [c3775]. Context vectors are richer representations than raw embeddings because they encode both the semantic meaning of a token and information about how that token relates to other tokens in the sequence [c3776].

After the transformer blocks, the model produces **logits**—the raw output values before normalization into probabilities [c3779]. A logit tensor is then converted into a softmax tensor, which is a probability tensor [c3785].

#### Constructing Input-Output Pairs

Before computing any loss, we need properly structured training data. Targets are the input token sequence shifted by one position, creating input-output pairs for training [c542]. More precisely, $X$ represents the input tensor and $Y$ represents the target tensor (the actual desired output values) in LLM input-output pair construction [c3763]. Input-target pairs are created from training data by pairing inputs with their corresponding target outputs [c3767], and the target output is the expected output sequence that the LLM should learn to predict during training [c3770].

In the context of language model training, "target" refers to the true values that the model should predict [c502].

#### The Dataloader

A dataloader loops over the entire dataset and creates input-output pairs based on a specified context size and stride [c3821]. Two key parameters govern this process:

- **`max_length`**: specifies the context size—the number of tokens to consider at once [c3822]. Context size is the maximum number of tokens an LLM can see before it predicts the next token [c3754].
- **`stride`**: specifies how many steps to advance before creating the next input-output pair [c3823]. Stride is the step size used to move through the dataset when constructing consecutive input-output pairs [c3758].

The PyTorch `DataLoader` is useful for processing data in batches and makes batch processing more convenient [c3826]. 

- **`shuffle`**: shuffles the dataset order when batches are created, which is sometimes useful for generalization [c3828].
- **`drop_last`**: drops the last batch if its size is smaller than the specified batch size [c3829].

Tiktoken provides a byte pair encoder that operates at character and sub-word level [c3817], making it the natural tokenizer for this pipeline. The `text_to_token_ids` function converts text input into token IDs using the tiktoken tokenizer [c538], and the `token_ids_to_text` function converts token IDs back into text, serving as the reverse of `text_to_token_ids` [c539].

---

### Computing Cross-Entropy Loss in PyTorch

#### The Flat-Tensor Pattern

 We must flatten both before passing them to the loss function.

# Source: [c569]
```python

targets_flat = targets.view(-1) # Flatten from (2, 3) to (6,)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

# Source: [c555]
```python

```

And in the context of a loss-computation function:

# Source: [c581]
```python

```

#### Selecting the Predicted Token

During evaluation (as opposed to training), we often want to know which token the model actually predicts. `torch.argmax` with `dim=-1` returns the indices of the maximum value along the last dimension (columns) [c547], effectively picking the highest-logit token at each position.

#### The `calculate_loss_batch` Function

The `calculate_loss_batch` function computes cross-entropy loss for a single input-target batch pair by flattening both the logits and target tensors before passing them to `nn.functional.cross_entropy` [c3852]:

# Source: [c3851]
```python

```

Cross-entropy loss is used to measure the loss between target tokens and output tokens [c580].

---

### Perplexity: An Interpretable Loss Metric

 

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of words in the dataset [c572]. It is calculated as $e$ raised to the loss value [c582]:

[c574]

 As training progresses and the loss decreases, perplexity falls correspondingly.

---

### The Pre-Training Loop

With loss computation in hand, we can now build the full training loop. The training pipeline for LLMs consists of multiple steps, starting with text generation and text evaluation [c490]. This lecture covers LLM training and validation loss computation [c3728], building on the cross-entropy loss calculation covered previously [c3734].

#### High-Level Structure

The pre-training loop has two nested loops: an outer loop iterating over epochs and an inner loop iterating over batches in the training dataset [c1657]. One epoch is going through the entire training set once [c1613]. The training set is divided into batches, and the pre-training loop processes one batch per iteration [c1614]. After processing all batches, the loop repeats for multiple training epochs [c1621].

[FIGURE: Nested loop diagram showing outer epoch loop containing inner batch loop, with forward pass → loss → backward pass → optimizer step labeled inside the inner loop]

#### The Core Update Rule

 The parameter update rule is [c1618]:

LLM pre-training involves minimizing the loss function to make model outputs as close as possible to target value tensors [c1611]. Back propagation is used to calculate loss gradients during LLM pre-training [c1612].

The pre-training algorithm consists of three steps [c1622]:
1. **Find the loss** — run the forward pass and compute cross-entropy.
2. **Backward pass** — compute gradients of the loss with respect to all parameters.
3. **Update parameters** — apply the optimizer step using those gradients.

The backward pass is the most important step in the pre-training loop because it calculates the gradients of the loss [c1616]. 

# Source: [c1624]
```python

```

The pre-training workflow is fully differentiable, allowing back propagation to compute partial derivatives of the loss with respect to all model parameters [c1626]. The pretraining loop is approximately 15 to 20 lines of Python code, with `loss.backward()` and `optimizer.step()` being the core gradient update mechanism [c1667].

#### Iterating Over Batches

The pretraining loop iterates through batches from a training data loader, dividing each batch into an input batch and a target batch [c1652]. The loss is calculated between the input batch and target batch using categorical cross entropy [c1653]. Cross-entropy loss is calculated for the entire batch during each training iteration [c1615].

#### Evaluation During Training

An evaluation step is performed during training to monitor training loss and validation loss as the model trains [c1659]. This lecture calculates training loss and validation loss on an actual dataset [c3737]. Obtaining training and validation loss enables backpropagation in the next step [c3868].

[FIGURE: Training curve plot showing training loss and validation loss decreasing over epochs, with evaluation checkpoints marked at regular intervals]

#### Model Parameters: A Quick Accounting

 The key parameter counts are:

- **Token embedding parameters** = vocabulary size × embedding dimension [c1630]
- **Positional embedding parameters** = context size × embedding dimension [c1631]
- **Final layer parameters** = embedding dimension × vocabulary size [c1644]

Embedding dimension is the size of the vector space into which token embeddings are projected to capture semantic meaning [c495]. The number of attention heads specifies how many self-attention mechanism blocks exist within one transformer block [c496].

---

### Putting It All Together: Training on "The Verdict"

With all the pieces defined, we can now describe the full training run on a real text corpus.

#### Data Preparation

The developed code is generalizable and can be applied to custom datasets for pre-training [c3870]. 

 With a small corpus like "The Verdict," a stride equal to the context size avoids redundant samples; with larger corpora, overlapping strides can increase the effective dataset size.

#### The Training Function

An LLM training function will be defined to implement backpropagation and minimize training and validation loss [c3869]. The function wraps the nested epoch-batch loop described above, calling `calculate_loss_batch` on each mini-batch, invoking `loss.backward()`, and then stepping the optimizer.

A training loop has been implemented where the loss function is minimized and the large language model learns [c1717]. The next lecture will cover LLM pre-training in full detail [c3873].

#### Decoding Predictions During Training

 Temperature scaling is a decoding strategy used to control randomness in model predictions [c1719]. 

[FIGURE: Diagram showing logits → temperature scaling → softmax → token sampling pipeline]

#### Observing Overfitting

 

[FIGURE: Overfitting curve showing training loss continuing to decrease while validation loss flattens or rises after a certain epoch]

 A well-regularized model trained on a large, diverse corpus will show both curves tracking closely together. 

---

### Summary

1. **Loss functions**: We surveyed the full family of regression and classification losses and established why cross-entropy—equivalently, negative log likelihood—is the right choice for next-token prediction [c550, c1608]. The formula is $-\frac{1}{N}\sum_i \log(p_i)$ [c552], and the training goal is to drive this toward zero [c3792].

2. **Perplexity**: We defined perplexity as $\exp(\text{loss})$ [c574], a more interpretable metric that reads as the effective branching factor of the model's predictions [c572].

3. **Batch loss computation**: The `calculate_loss_batch` function flattens logits from `(batch, seq, vocab)` to `(batch×seq, vocab)` and targets from `(batch, seq)` to `(batch×seq,)` before calling `nn.functional.cross_entropy` [c3852, c569].

4. **The training loop**: The loop is nested—outer over epochs [c1613], inner over batches [c1614]—and each iteration performs a forward pass, `loss.backward()` [c1624], and an optimizer step [c1622]. The entire workflow is fully differentiable [c1626].

5. **Evaluation**: Training and validation loss are monitored throughout training [c1659], and perplexity provides a human-readable summary of model quality [c582].
