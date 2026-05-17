## Building the Transformer Block: Layer Norm, GELU, Feed-Forward, and Shortcut Connections

At the heart of every GPT-style model lies the transformer block—a carefully orchestrated sequence of normalization, attention, and transformation operations that, when stacked, enables language models to build rich, abstract representations of text. The transformer block is the fundamental building block of GPT and other large language model architectures [c2051]. Before assembling a complete transformer block, we need to understand and implement each of its sub-components individually.

The five components of a transformer block are: masked multi-head attention, layer normalization, dropout, a feed-forward neural network, and the GELU activation function [c2058]. This chapter focuses on the normalization and transformation machinery—the pieces that make deep transformer networks trainable—building on earlier coverage of layer normalization, GELU activation, feed-forward networks, and shortcut connections [c277, c278].

---

### Layer Normalization

#### Why Normalize at All?

One reason normalization is necessary is a phenomenon called **internal covariate shift**: the input distribution to each layer changes across training iterations, making weight updates difficult and delaying convergence [c1038]. When the distribution of activations shifts unpredictably from one batch to the next, downstream layers must constantly adapt to a moving target. Layer normalization addresses this by normalizing the output of every layer based on the mean and standard deviation of that layer's own activations [c1099].

#### Layer Norm vs. Batch Norm

- **Batch normalization** performs normalization for an entire batch [c1100], computing the mean and variance across the batch dimension. This means the normalization statistics depend on which other samples happen to be in the same mini-batch.
- **Layer normalization** normalizes along the feature dimension (columns) instead [c1097], making each sample's statistics independent of the rest of the batch.

#### The Math

Layer normalization normalizes each sample in a batch independently by subtracting the mean and dividing by the square root of the variance computed across the features for that sample [c1058].

Concretely, given a sample with feature values $X_1, X_2, X_3, X_4$, the mean is [c1044]:

$$\mu = (X_1 + X_2 + X_3 + X_4) / 4 = 2.15$$

The variance is [c1045]:

$$\sigma^2 = \frac{1}{4}\left[(X_1 - \mu)^2 + (X_2 - \mu)^2 + (X_3 - \mu)^2 + (X_4 - \mu)^2\right]$$

And the normalized value for each feature is [c1046]:

$$\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2}}$$

In formula form, layer normalization applies [c1059]:

$$y_{\text{normalized}} = \frac{y - \mu}{\sqrt{\text{variance}}}$$

independently to each sample in the batch. After normalization, the result has mean 0 and variance 1 [c2064].

#### Scale and Shift Parameters

Raw normalization to zero mean and unit variance may not always be optimal—some layers may benefit from a different scale or offset. Scale and shift are trainable parameters in layer normalization that have the same dimension as the input and are applied to the normalized output [c1078]. They allow the model to undo the normalization if that turns out to be optimal for a given layer, while still benefiting from the numerical stability that normalization provides during early training.

#### A Note on Bessel's Correction

When computing variance in PyTorch, you will encounter the `unbiased` flag. Bessel's correction is the use of $n - 1$ instead of $n$ in the formula for sample variance and sample standard deviation, where $n$ is the number of observations in a sample [c1083]. In layer normalization we typically use the biased estimator (dividing by $n$), so `unbiased=False` is the correct setting.

#### Computing Layer Norm Step by Step in PyTorch

Let us walk through the computation manually before looking at the full class. First, create a small batch and pass it through a linear layer:

```python
# Source: [c1057]
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```

Now compute the mean across the feature dimension:

```python
# Source: [c1060]
mean = output.mean(dim=-1, keepdim=True)
```

Then compute the variance:

```python
# Source: [c1062]
variance = output.var(dim=-1, keepdim=True)
```

And finally normalize:

```python
# Source: [c1063]
normalized_output = (output - mean) / torch.sqrt(variance)
```

The `dim=-1` argument is critical: it tells PyTorch to compute statistics along the last dimension (the feature dimension), which is exactly what layer normalization requires [c1097].

#### The LayerNorm Class

Layer normalization is implemented as a class that takes the output of a layer and applies normalization to it [c1066]. Its forward method takes input with a certain number of rows and embedding-dimension columns [c1073].

```python
# Source: [c2107]
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

A few things to notice: `self.eps = 1e-5` is a small epsilon value added to prevent division by zero [c2104], and `scale` and `shift` are initialized to ones and zeros respectively so that the layer initially performs pure normalization before learning any rescaling [c1078].

---

### The GELU Activation Function

#### The Problem with ReLU

ReLU returns the input value for $x > 0$ and returns zero for $x < 0$ [c2002]. This hard cutoff creates the **dead neuron problem**: when a neuron's output becomes negative, ReLU sets it to zero and the neuron stops contributing to the learning process [c2005]. Because the gradient of ReLU is exactly zero for all negative inputs, a neuron that consistently receives negative pre-activations will never receive a gradient signal and will never update its weights [c2004]. In a deep network, this can silently eliminate large fractions of the model's capacity.

#### Introducing GELU

GELU (Gaussian Error Linear Unit) is a smooth activation function that is differentiable at $x = 0$, unlike ReLU [c2075]. The exact definition is [c2006]:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution.

#### The Approximation Used in GPT-2

In practice, GPT-2 uses a polynomial approximation [c2009]:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

This is the approximation used in GPT-2 [c2108] and is what we implement here.

#### Implementing GELU

```python
# Source: [c2109]
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

The structure mirrors the approximation formula exactly.

[FIGURE: Side-by-side plot of ReLU and GELU activation functions over the range x ∈ [-3, 3], showing ReLU's hard zero cutoff vs. GELU's smooth curve that allows small negative outputs near zero]

---

### The Feed-Forward Network

#### Purpose and Position

The goal of the feed-forward sub-module is to serve as a component of the LLM transformer block [c1999]. Within that block, the feed-forward network processes each element of the input sequence separately, without considering relationships with other elements [c2092]. This contrasts with the self-attention block, which analyzes relationships between input elements and assigns attention scores based on how one element relates to others [c2091].

#### The Expansion–Contraction Pattern

The feed-forward neural network in a transformer block has an expansion layer followed by a compression layer, preserving input and output dimensions [c2070]. Specifically, it consists of three stages [c2110]:

1. An **expansion layer** that increases the dimension to 4 times the embedding dimension.
2. A **GELU activation** applied between the two linear layers.
3. A **contraction layer** that reduces back to the original embedding dimension.

The expansion layer's hidden dimension is four times the embedding dimension [c2072]. For GPT-2's embedding dimension of 768, this means the network projects from 768 to 3072, applies GELU, and then projects back from 3072 to 768 [c2031]. Intuitively, expanding into a higher-dimensional space gives the network more room to represent complex, non-linear transformations of each token's representation before compressing back down.

The feedforward module uses GELU between the expansion and contraction layers [c2038], and its output dimensions match its input dimensions [c2120].

#### Implementation

`nn.Sequential` is a PyTorch module that chains neural network layers together by forwarding outputs sequentially to inputs of subsequent modules [c2033], making it a natural fit for the linear flow of expansion → activation → contraction. The FF object is an instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation, initializing weights randomly [c2119].

The architecture is:

```python
# Sources: [c2110], [c2031]
nn.Sequential(
    nn.Linear(emb_dim, 4 * emb_dim),
    GELU(),
    nn.Linear(4 * emb_dim, emb_dim)
)
```

---

### Shortcut Connections and the Vanishing Gradient Problem

#### The Vanishing Gradient Problem

To understand why shortcut connections matter, we first need to understand the problem they solve. The **vanishing gradient problem** occurs when gradients become extremely small or approach zero during backpropagation through deep layers, preventing weight updates and causing training stagnancy [c224]. In a deep network, gradients are computed by multiplying many partial derivatives together; if each factor is less than one, the product shrinks exponentially with depth. Shortcut connections help transformers solve this problem [c275].

#### What Are Shortcut Connections?

Shortcut connections are also known as skip connections or residual connections [c211, c217, c241]. They are achieved by adding the output of one layer to the output of a later layer [c228]. In a transformer block diagram, they are represented by plus symbols with associated arrows that bypass the linear flow of data [c216, c2050]. More precisely, shortcut connections add the output of one layer to the output of the previous layer, creating an alternative path for gradient flow [c2079], and they can be added between any layers of the transformer block [c274].

[FIGURE: Diagram of a residual block showing input y_l flowing both through a transformation f(y_l) and directly via a skip arrow, with the two paths summing to produce y_{l+1} = f(y_l) + y_l]

#### The Residual Block Formula

In a residual block with shortcut connections, the output of layer $l+1$ is [c234]:

$$y_{l+1} = f(y_l) + y_l$$

In code, this is expressed as [c255]:

```python
x = x + layer_output
```

#### Why Shortcut Connections Preserve Gradients: The Math

The partial derivative of loss with respect to $y_l$ can be expressed using the chain rule as [c236]:

$$\frac{\partial L}{\partial y_l} = \frac{\partial L}{\partial y_{l+1}} \cdot \frac{\partial y_{l+1}}{\partial y_l}$$

When $y_{l+1} = f(y_l) + y_l$, the second factor becomes [c237]:

$$\frac{\partial y_{l+1}}{\partial y_l} = \frac{\partial f(y_l)}{\partial y_l} + 1$$

The additive constant of 1 ensures that even if $\frac{\partial f(y_l)}{\partial y_l}$ is very small, the overall gradient is never smaller than $\frac{\partial L}{\partial y_{l+1}}$ itself. As a result, gradient magnitude remains stable across layers, preventing the vanishing gradient problem [c2082].

#### Demonstrating the Effect

To see the vanishing gradient problem in action and verify that shortcut connections fix it, we can build a simple deep network and compare gradients with and without skip connections. A deep neural network class takes `layer_sizes` as an argument specifying the number of neurons in each layer—for example, `[3, 3, 3, 3, 1]` means 5 layers with 3 neurons each and a final layer with 1 neuron [c248].

The loss function is defined as the squared difference between the model output and the ground truth target [c260]:

$$\text{loss} = (Y - \text{target})^2$$

In code:

```python
# Source: [c262]
loss = (model(x) - target) ** 2
loss.backward()
```

Without shortcut connections, gradients shrink dramatically in the early layers. With shortcut connections, they remain at a similar magnitude throughout the network.

---

### Assembling the Transformer Block

#### Components Review

The GPT architecture has four key components: layer normalization, GELU activation, a feed-forward neural network, and shortcut connections [c2148]. The full transformer block stacking order is [c2083]:

$$\text{layer normalization} \to \text{multi-head attention} \to \text{dropout} \to \text{shortcut connection} \to \text{layer normalization} \to \text{feed-forward (with GELU)} \to \text{dropout} \to \text{shortcut connection}$$

#### Pre-Layer Norm vs. Post-Layer Norm

The original transformer model applied layer normalization *after* the self-attention and feed-forward sub-layers, a pattern called **post-layer norm** [c2130]. Modern transformer implementations use **pre-layer norm** instead [c2132], applying normalization *before* the multi-head attention mechanism and *before* the feed-forward neural network [c2129].

Why does this matter? Because the shortcut connection adds the un-normalized input directly to the output, the residual stream accumulates values across many layers. Normalizing *before* each operation keeps the inputs to each sub-layer well-conditioned, improving training stability. Layer normalization is therefore implemented twice within the transformer block: once before multi-head attention and once before the feed-forward network [c2067].

#### The Transformer Block Components in Detail

Each named component of the transformer block class plays a specific role:

- **ATT**: An instance of the multi-head attention class that takes embedding vectors and converts them into context vectors, with input and output dimensions equal to the embedding dimension [c2113]. It multiplies an input matrix X with trainable query, key, and value matrices to produce context vectors [c2059].

- **FF**: An instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation, initializing weights randomly [c2119].

- **norm_one** and **norm_two**: `norm_one` is the first layer normalization layer, applied before multi-head attention; `norm_two` is the second, applied before the feed-forward network [c2121].

- **drop_shortcut**: A dropout layer (`torch.nn.Dropout`) [c2123] that randomly turns off some layer outputs during training to improve generalization and prevent overfitting [c2068].

In a transformer block, each element of an input sequence is represented by a fixed-size vector equal to the embedding dimension [c2084].

#### The Transformer Block Class

A transformer block class in PyTorch includes a multi-head attention mechanism and a feed-forward neural network [c2128], and implements shortcut connections that add the input of the block to the output of each component [c2133]. A dropout layer and residual shortcut connection are applied after the feed-forward network to prevent vanishing gradients [c2141].

Based on the architectural description, the forward pass follows this logic:

```python
# Sources: [c2083], [c2129], [c2133]
# First sub-block: multi-head attention
shortcut = x
x = norm_one(x)
x = ATT(x)
x = drop_shortcut(x)
x = x + shortcut  # residual connection

# Second sub-block: feed-forward
shortcut = x
x = norm_two(x)
x = FF(x)
x = drop_shortcut(x)
x = x + shortcut  # residual connection
```

This structure directly implements the stacking order described in [c2083], with pre-layer norm [c2129] and shortcut connections after each sub-block [c2133].

---

### Putting It All Together: The GPT Architecture

The GPT architecture consists of four main components: the GPT backbone, layer normalization, GELU activation function, and feed-forward neural network [c2049]. Each transformer block processes the full sequence and passes its output to the next block; stacking many such blocks allows the model to build increasingly abstract representations of the input.

---

### Summary

This chapter built every sub-component of the transformer block from scratch:

1. **Layer normalization** normalizes each sample independently along the feature dimension [c1058], using trainable scale and shift parameters [c1078] and a small epsilon for numerical stability [c2104]. It addresses internal covariate shift [c1038] and is applied twice per transformer block [c2067].

2. **GELU activation** provides a smooth, differentiable alternative to ReLU [c2075] that avoids the dead neuron problem [c2005]. It is approximated by the formula $0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$ [c2009, c2108].

3. **The feed-forward network** expands the embedding dimension by a factor of 4, applies GELU, and contracts back to the original dimension [c2110], processing each sequence position independently [c2092].

4. **Shortcut connections** add the input of each sub-block to its output [c228], contributing an additive term of 1 to the gradient computation [c237] that prevents gradients from vanishing in deep networks [c224, c275].

5. **The transformer block** assembles these components in pre-layer-norm order [c2129, c2132]—normalize → attend → dropout → add → normalize → feed-forward → dropout → add [c2083]—and can be stacked repeatedly to form a complete GPT model.