## Building the Transformer Block: Layer Norm, GELU, Feed-Forward, and Shortcut Connections

The transformer block is the fundamental building block of GPT and other large language model architectures [c2051]. Before we can assemble a complete transformer block, we need to understand and implement each of its sub-components individually. This chapter works through those components one by one—layer normalization, the GELU activation function, the feed-forward network, and shortcut connections—and then assembles them into a working transformer block.

The five components of a transformer block are: masked multi-head attention, layer normalization, dropout, a feed-forward neural network, and the GELU activation function [c2058]. We have already covered multi-head attention in detail. Here we focus on the remaining pieces, which together form the normalization and transformation machinery that makes deep transformer networks trainable [c277, c278].

---

### Layer Normalization

#### Why Normalize at All?

Training deep neural networks is hard. One reason is a phenomenon called **internal covariate shift**: the input distribution to each layer changes across training iterations, making weight updates difficult and delaying convergence [c1038]. When the distribution of activations shifts unpredictably from one batch to the next, the layers downstream must constantly adapt to a moving target. Normalization is a technique that stabilizes these distributions.

Layer normalization addresses this by normalizing the output of every layer based on the mean and standard deviation of that layer's own activations [c1099]. The key idea is that we compute statistics *per sample* rather than across the batch.

#### Layer Norm vs. Batch Norm

It is worth pausing to distinguish layer normalization from batch normalization, because the two are often confused.

- **Batch normalization** performs normalization for an entire batch [c1100]. It computes the mean and variance across the batch dimension, which means the normalization statistics depend on what other samples happen to be in the same mini-batch.
- **Layer normalization** normalizes along the feature dimension (columns) [c1097]. It computes statistics independently for each sample, so the normalization of one sample is completely unaffected by other samples in the batch.

For language models, layer normalization is strongly preferred. Sequence lengths vary, batch sizes can be small, and at inference time you often process a single sequence—all situations where batch statistics are unreliable.

#### The Math

Layer normalization normalizes each sample in a batch independently by subtracting the mean and dividing by the square root of variance computed across the features for that sample [c1058].

Concretely, given a sample with feature values $X_1, X_2, X_3, X_4$, the mean is [c1044]:

$$\mu = \frac{X_1 + X_2 + X_3 + X_4}{4} = 2.15$$

The variance is [c1045]:

$$\sigma^2 = \frac{1}{4}\left[(X_1 - \mu)^2 + (X_2 - \mu)^2 + (X_3 - \mu)^2 + (X_4 - \mu)^2\right]$$

And the normalized value for each feature is [c1046]:

$$\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2}}$$

In formula form, layer normalization is [c1059]:

$$y_{\text{normalized}} = \frac{y - \mu}{\sqrt{\text{variance}}}$$

applied independently to each sample in the batch.

After normalization, the result has mean 0 and variance 1 [c2064]. This is the raw normalization step.

#### Scale and Shift Parameters

Raw normalization to zero mean and unit variance is useful, but it is too rigid. The network should be able to learn that some layers benefit from a different scale or offset. This is why layer normalization includes **scale** and **shift** parameters.

Scale and shift are trainable parameters in layer normalization that have the same dimension as the input and are applied to the normalized output [c1078]. They allow the model to undo the normalization if that turns out to be optimal for a given layer, while still benefiting from the numerical stability that normalization provides during the early stages of training.

#### A Note on Bessel's Correction

When computing variance in PyTorch, you will encounter the `unbiased` flag. Bessel's correction is the use of $n - 1$ instead of $n$ in the formula for the sample variance and sample standard deviation, where $n$ is the number of observations in a sample [c1083]. In layer normalization we typically use the biased estimator (dividing by $n$, i.e., `unbiased=False`) because we are normalizing the entire feature vector of a sample, not estimating a population statistic from a subset.

#### Computing Layer Norm Step by Step in PyTorch

Let us walk through the computation manually before looking at the full class. First, create a small batch and pass it through a linear layer:

# Source: [c1057]
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```

Now compute the mean across the feature dimension:

# Source: [c1060]
```python
mean = output.mean(dim=-1, keepdim=True)
```

Then compute the variance:

# Source: [c1062]
```python
variance = output.var(dim=-1, keepdim=True)
```

And finally normalize:

# Source: [c1063]
```python
normalized_output = (output - mean) / torch.sqrt(variance)
```

The `dim=-1` argument is critical: it tells PyTorch to compute statistics along the last dimension (the feature dimension), which is exactly what layer normalization requires [c1097].

#### The LayerNorm Class

Layer normalization is a class that takes the output of a layer and applies normalization to it [c1066]. The forward method takes input with a certain number of rows and embedding dimension columns [c1073]. Here is the complete implementation:

# Source: [c2107]
```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

A few things to notice:

- `self.eps = 1e-5` is a small epsilon value added to prevent division by zero [c2104]. Without it, a sample with zero variance would cause a numerical explosion.
- `self.scale` is initialized to ones and `self.shift` to zeros, so at initialization the layer normalization is an identity transformation. The model learns to deviate from this as training proceeds.
- `unbiased=False` uses the biased variance estimator (dividing by $n$) rather than Bessel's correction.
- The final line `self.scale * norm_x + self.shift` applies the trainable scale and shift element-wise.

---

### The GELU Activation Function

#### The Problem with ReLU

The most familiar activation function is ReLU (Rectified Linear Unit). ReLU returns the input value for $x > 0$ and returns zero for $x < 0$ [c2002]. It is simple, fast, and works well in many settings. However, it has a well-known failure mode.

The **dead neuron problem** occurs when a neuron's output becomes negative, ReLU sets it to zero, and the neuron stops contributing to the learning process [c2005]. Because the gradient of ReLU is exactly zero for all negative inputs, a neuron that consistently receives negative pre-activations will never receive a gradient signal and will never update its weights [c2004]. In a deep network, this can silently kill large fractions of the model's capacity.

#### Introducing GELU

GELU (Gaussian Error Linear Unit) is a smooth activation function that is differentiable at $x = 0$, unlike ReLU [c2075]. The exact definition is [c2006]:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution.

Intuitively, GELU multiplies the input by the probability that a standard normal random variable is less than or equal to $x$. For large positive $x$, $\Phi(x) \approx 1$ and GELU behaves like the identity. For large negative $x$, $\Phi(x) \approx 0$ and GELU suppresses the signal. But crucially, near zero the transition is smooth and differentiable, which means gradients can flow even for slightly negative inputs—avoiding the hard cutoff that kills neurons in ReLU.

#### The Approximation Used in GPT-2

Computing $\Phi(x)$ exactly requires the error function, which is expensive. In practice, GPT-2 uses a polynomial approximation [c2009]:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

This approximation is the one used in GPT-2 [c2108] and is what we implement here.

#### Implementing GELU

# Source: [c2109]
```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
```

[GAP: The closing parenthesis of the `return` statement and the class appear to be missing from the source claim; the code as given in c2109 is syntactically incomplete.]

The structure mirrors the approximation formula exactly. `torch.sqrt(torch.tensor(2.0 / torch.pi))` computes $\sqrt{2/\pi}$, and `torch.pow(x, 3)` computes $x^3$.

[FIGURE: Side-by-side plot of ReLU and GELU activation functions over the range x ∈ [-3, 3], showing ReLU's hard zero cutoff vs. GELU's smooth curve that allows small negative outputs near zero]

---

### The Feed-Forward Network

#### Purpose and Position

The goal of implementing the feed-forward neural network sub-module is to create a component that is part of the LLM transformer block [c1999]. Within the transformer block, the feed-forward neural network processes each element of the input sequence separately, without considering relationships with other elements [c2092]. This is in contrast to the self-attention block, which analyzes relationships between input elements and assigns attention scores based on how one input element relates to others [c2091].

The two components are therefore complementary: attention mixes information across positions, while the feed-forward network transforms each position independently.

#### The Expansion-Contraction Pattern

The feed-forward neural network in a transformer block has an expansion layer followed by a compression layer, preserving input and output dimensions [c2070]. Specifically, the FeedForward neural network consists of three layers [c2110]:

1. An **expansion layer** that increases the dimension to 4 times the embedding dimension
2. A **GELU activation function**
3. A **contraction layer** that reduces back to the original embedding dimension

The expansion layer in the feed-forward network has a hidden layer with neurons four times larger than the embedding dimension [c2072]. For GPT-2's embedding dimension of 768, this means the feed-forward network projects from 768 dimensions to 3072, applies GELU, and then projects back from 3072 to 768 [c2031].

Why expand and then contract? Intuitively, the expansion into a higher-dimensional space gives the network more room to represent complex, non-linear transformations of each token's representation. The GELU activation introduces non-linearity in that expanded space, and the contraction brings the result back to the dimension expected by the rest of the block.

The feedforward module uses GELU activation function between the expansion and contraction layers [c2038]. The feed-forward network processes input through expansion, GELU activation, and contraction, with output dimensions matching the input dimensions [c2120].

#### Implementation

`nn.Sequential` is a PyTorch module that chains neural network layers together by forwarding outputs sequentially to inputs of subsequent modules [c2033]. This makes it a natural fit for the feed-forward network, where data flows linearly through expansion → activation → contraction.

The FF object is an instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation function, initializing weights randomly [c2119].

[GAP: The complete FeedForward class code block is not present in the supplied claims. Claims c2110 and c2031 describe the architecture but do not supply a verbatim code listing for the full class.]

The architecture described by the claims is:

```
FeedForward:
    Linear(emb_dim → 4 * emb_dim)
    GELU()
    Linear(4 * emb_dim → emb_dim)
```

This matches the description in [c2110] and the concrete dimensions in [c2031].

---

### Shortcut Connections and the Vanishing Gradient Problem

#### The Vanishing Gradient Problem

To understand why shortcut connections matter, we first need to understand the problem they solve.

The **vanishing gradient problem** occurs when gradients become extremely small or approach zero during backpropagation through deep layers, preventing weight updates and causing training stagnancy [c224]. During backpropagation, gradients are computed by repeatedly applying the chain rule as we move backward through the network. In a deep network, this means multiplying many partial derivatives together. If each of those partial derivatives is a small number (less than 1), the product shrinks exponentially with depth. By the time the gradient reaches the early layers, it is effectively zero, and those layers stop learning.

Shortcut connections help transformers solve the vanishing gradient problem [c275].

#### What Are Shortcut Connections?

Shortcut connections are also known as skip connections or residual connections [c211, c217, c241]. They are achieved by adding the output of one layer to the output of a later layer [c228]. In a transformer block diagram, shortcut connections are represented by plus symbols with associated arrows that bypass the linear flow of data [c216, c2050].

More precisely, shortcut connections add the output of one layer to the output of the previous layer, creating an alternative path for gradient flow [c2079]. A neural network with shortcut connections allows the output of any layer to be added to the input of the previous layer [c247]. Shortcut connections can be added between any layers of the transformer block [c274].

[FIGURE: Diagram of a residual block showing input y_l flowing both through a transformation f(y_l) and directly via a skip arrow, with the two paths summing to produce y_{l+1} = f(y_l) + y_l]

#### The Residual Block Formula

In a residual block with shortcut connections, the output of layer $l+1$ is [c234]:

$$y_{l+1} = f(y_l) + y_l$$

where $f(y_l)$ is the output of the neural network transformation and $y_l$ is the output of the previous layer. The addition of $y_l$ directly is the skip connection.

In code, this is expressed as [c255]:

# Source: [c255]
```python
x = x + layer_output
```

This single line is the entire implementation of a shortcut connection. The input `x` is added to whatever the layer produced.

#### Why Shortcut Connections Preserve Gradients: The Math

To see why shortcut connections help with vanishing gradients, let us work through the calculus.

The partial derivative of loss with respect to $y_l$ can be expressed using the chain rule as [c236]:

$$\frac{\partial L}{\partial y_l} = \frac{\partial L}{\partial y_{l+1}} \times \frac{\partial y_{l+1}}{\partial y_l}$$

Now, when $y_{l+1} = f(y_l) + y_l$, the partial derivative $\frac{\partial y_{l+1}}{\partial y_l}$ becomes [c237]:

$$\frac{\partial y_{l+1}}{\partial y_l} = \frac{\partial f(y_l)}{\partial y_l} + 1$$

The consequence is crucial: even if $\frac{\partial f(y_l)}{\partial y_l}$ is very small (the vanishing gradient scenario), the total derivative is $\frac{\partial f(y_l)}{\partial y_l} + 1$, which is at least 1. The gradient can never vanish to zero through this path because the shortcut connection contributes a constant additive term of 1 to the derivative.

With shortcut connections, gradient magnitude remains stable across layers, preventing the vanishing gradient problem [c2082].

#### Demonstrating the Effect

To see the vanishing gradient problem in action and verify that shortcut connections fix it, we can build a simple deep network and compare gradients with and without skip connections.

A deep neural network class takes `layer_sizes` as an argument, which specifies the number of neurons in each layer. For example, `[3, 3, 3, 3, 1]` means 5 layers with 3 neurons each and a final layer with 1 neuron [c248].

The loss function is defined as the squared difference between the model output and the ground truth target [c260]:

$$\text{loss} = (Y - \text{target})^2$$

In code:

# Source: [c262]
```python
loss = (model(x) - target) ** 2
loss.backward()
```

After calling `loss.backward()`, we can inspect the `.grad` attribute of each layer's weights to see how large the gradients are at each depth. Without shortcut connections, you will observe that gradients shrink dramatically in the early layers. With shortcut connections, the gradients remain at a similar magnitude throughout the network.

[FIGURE: Bar chart comparing gradient magnitudes at each layer depth for a deep network with and without shortcut connections, showing exponential decay without shortcuts vs. stable magnitudes with shortcuts]

---

### Assembling the Transformer Block

#### Components Review

We now have all the pieces we need. The GPT architecture has four key components: layer normalization, GELU activation, feed-forward neural network, and shortcut connections [c2148]. The full transformer block stacking order is [c2083]:

1. Layer normalization
2. Multi-head attention
3. Dropout
4. Shortcut connection (add input to output of attention)
5. Layer normalization
6. Feed-forward neural network with GELU
7. Dropout
8. Shortcut connection (add input to output of feed-forward)

[FIGURE: Vertical flow diagram of the complete transformer block showing the two sub-paths (attention path and feed-forward path), each with pre-layer-norm, the operation, dropout, and a residual addition]

#### Pre-Layer Norm vs. Post-Layer Norm

The original transformer model applied layer normalization *after* the self-attention and feed-forward neural network, a pattern called **post-layer norm** [c2130]. Modern transformer implementations use **pre-layer norm** instead [c2132].

Layer normalization is applied *before* the multi-head attention mechanism and *before* the feed-forward neural network, a pattern called pre-layer norm [c2129]. This is the design used in GPT-2 and most modern LLMs.

Why does this matter? Pre-layer norm stabilizes the residual stream. Because the shortcut connection adds the un-normalized input directly to the output, the residual stream accumulates values across many layers. Normalizing *before* each operation keeps the inputs to each sub-layer well-conditioned, which improves training stability.

Layer normalization is implemented twice within the transformer block: before the multi-head attention and before the feed-forward neural network [c2067].

#### The Transformer Block Components in Detail

Let us look at each named component of the transformer block class:

- **ATT**: The ATT object is an instance of the multi-head attention class that takes embedding vectors and converts them into context vectors, with input and output dimensions equal to the embedding dimension [c2113]. Multi-head attention takes an input matrix X and multiplies it with trainable queries, keys, and values matrices to produce context vectors [c2059].

- **FF**: The FF object is an instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation function, initializing weights randomly [c2119].

- **norm_one** and **norm_two**: `norm_one` is the first layer normalization layer applied before the multi-head attention, and `norm_two` is the second layer normalization layer applied before the feed-forward neural network [c2121].

- **drop_shortcut**: The `drop_shortcut` object is a dropout layer from PyTorch (`torch.nn.Dropout`) [c2123]. Dropout randomly turns off some layer outputs during training to improve generalization and prevent overfitting [c2068].

In a transformer block, each element of an input sequence is represented by a fixed-size vector equal to the embedding dimension [c2084]. This means the block's input and output shapes are identical: `(batch_size, sequence_length, embedding_dimension)`. The shortcut connections rely on this shape consistency—you can only add two tensors if they have the same shape.

#### The Transformer Block Class

A transformer block class in PyTorch includes a multi-head attention mechanism and a feed-forward neural network [c2128]. The transformer block implements shortcut connections (residual connections) that add the input of the block to the output of each component [c2133].

A dropout layer and residual shortcut connection are applied after the feed-forward network to prevent vanishing gradients [c2141].

[GAP: The complete TransformerBlock class code listing is not present in the supplied claims. The architecture is described across multiple claims (c2083, c2121, c2123, c2128, c2133) but no single verbatim code block for the full class is available.]

Based on the architectural description, the forward pass of the transformer block follows this logic:

```
# First sub-block: attention
shortcut = x
x = norm_one(x)
x = ATT(x)
x = drop_shortcut(x)
x = x + shortcut          # residual connection

# Second sub-block: feed-forward
shortcut = x
x = norm_two(x)
x = FF(x)
x = drop_shortcut(x)
x = x + shortcut          # residual connection
```

This structure directly implements the stacking order described in [c2083], with pre-layer norm [c2129] and shortcut connections after each sub-block [c2133].

---

### Putting It All Together: The GPT Architecture

The GPT architecture consists of four main components: the GPT backbone, layer normalization, GELU activation function, and feed-forward neural network [c2049]. The transformer block is the fundamental unit that is stacked repeatedly to form the full model.

[FIGURE: High-level diagram of the GPT architecture showing token embeddings feeding into a stack of N transformer blocks, each containing the pre-norm attention and pre-norm feed-forward sub-blocks, followed by a final layer norm and output projection]

Each transformer block takes a sequence of embedding vectors, applies masked self-attention to let each position attend to previous positions, applies a position-wise feed-forward transformation, and returns a sequence of the same shape. Stacking many such blocks allows the model to build increasingly abstract representations of the input sequence.

The design choices we have made in this chapter—pre-layer norm, GELU activation, the 4× expansion factor in the feed-forward network, and shortcut connections throughout—are not arbitrary. They reflect lessons learned from training large models and are the specific choices made in GPT-2.

---

### Summary

In this chapter we built every sub-component of the transformer block from scratch:

1. **Layer normalization** normalizes each sample independently along the feature dimension [c1058], using trainable scale and shift parameters [c1078] and a small epsilon for numerical stability [c2104]. It addresses internal covariate shift [c1038] and is applied twice per transformer block [c2067].

2. **GELU activation** provides a smooth, differentiable alternative to ReLU [c2075] that avoids the dead neuron problem [c2005]. It is approximated by the formula $0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$ [c2009, c2108].

3. **The feed-forward network** expands the embedding dimension by a factor of 4, applies GELU, and contracts back to the original dimension [c2110], processing each sequence position independently [c2092].

4. **Shortcut connections** add the input of each sub-block to its output [c228], providing an additive term of 1 in the gradient computation [c237] that prevents gradients from vanishing in deep networks [c224, c275].

5. **The transformer block** assembles these components in pre-layer-norm order [c2129, c2132]: normalize → attend → dropout → add → normalize → feed-forward → dropout → add [c2083].

With the transformer block complete, we have the core machinery needed to build a full GPT model. The next step is to stack multiple transformer blocks, add token and positional embeddings at the input, and add a final linear projection at the output to produce logits over the vocabulary.