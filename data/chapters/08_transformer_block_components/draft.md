## Building the Transformer Block: LayerNorm, GELU, Feed-Forward, and Shortcut Connections

The transformer block is the engine of every modern large language model. Before we can assemble it, we need to understand and implement each of its sub-components individually. This chapter works through four of those components—layer normalization, the GELU activation function, the feed-forward network, and shortcut (residual) connections—and then snaps them together into a complete, working transformer block.

By the end of this chapter you will have PyTorch implementations of every piece and a clear mental model of *why* each design decision was made. Let's start from the bottom up.

---

### Layer Normalization

#### The Problem: Internal Covariate Shift

Training a deep neural network is an iterative process: you push data forward, compute a loss, and push gradients backward. At each step, the weights of every layer change. But when the weights of layer $k$ change, the distribution of activations that layer $k+1$ receives also changes—even if layer $k+1$'s own weights haven't moved yet. This phenomenon is called **internal covariate shift** [c1038].

Formally, internal covariate shift is the problem where the input distribution to each layer changes across training iterations, making weight updates difficult and delaying convergence [c1038]. Intuitively, each layer is trying to hit a moving target: the statistics of its inputs keep shifting, so the layer must constantly re-adapt rather than making steady progress toward a good solution.

Layer normalization is the standard remedy in transformer architectures. The core idea is simple: before passing activations to the next layer, normalize them so they have a consistent mean and variance.

#### The Math

Layer normalization normalizes each sample in a batch independently by subtracting the mean and dividing by the square root of variance computed across the features for that sample [c1058]. This is the key distinction from batch normalization, which performs normalization for an entire batch [c1100]. Layer normalization normalizes along the feature dimension (columns) [c1097], while batch normalization normalizes along the batch dimension (rows).

To make this concrete, suppose a single sample has four feature values $X_1, X_2, X_3, X_4$. The mean is computed as [c1044]:

$$\mu = \frac{X_1 + X_2 + X_3 + X_4}{4} = 2.15$$

The variance is [c1045]:

$$\sigma^2 = \frac{1}{4}\left[(X_1 - \mu)^2 + (X_2 - \mu)^2 + (X_3 - \mu)^2 + (X_4 - \mu)^2\right]$$

And the normalized value for each feature is [c1046]:

$$\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2}}$$

The result is that layer normalization normalizes the output of every layer based on the mean and standard deviation [c1099], so that the mean becomes 0 and the variance becomes 1 [c2064].

#### A Quick Empirical Check

Before building the full class, it is instructive to see normalization in action on a small example. The following code creates a two-sample batch, passes it through a linear layer and ReLU, and then we can inspect the output statistics:

# Source: [c1057]
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```

With the output tensor in hand, computing the mean and variance manually mirrors the formulas above. The mean across the feature dimension is:

# Source: [c1060]
```python
mean = output.mean(dim=-1, keepdim=True)
```

The variance is:

# Source: [c1062]
```python
variance = output.var(dim=-1, keepdim=True)
```

And the normalized output is:

# Source: [c1063]
```python
normalized_output = (output - mean) / torch.sqrt(variance)
```

Notice that `dim=-1` operates along the last dimension—the feature dimension—which is exactly what layer normalization requires [c1097].

#### Scale and Shift: Trainable Parameters

Raw normalization forces every layer's output to have mean 0 and variance 1. That sounds helpful, but it also removes any learned scaling that the network might have developed. To give the model the ability to undo normalization when that is useful, layer normalization introduces two trainable parameters: **scale** and **shift** [c1078].

Scale and shift have the same dimension as the input and are applied to the normalized output [c1078]. After normalization, the output becomes:

$$y = \text{scale} \cdot \hat{x} + \text{shift}$$

These parameters are learned during training [c2105], so the network can decide for itself how much normalization to apply.

#### A Note on Bessel's Correction

When computing variance in PyTorch, you will notice the flag `unbiased=False` in the implementation below. This relates to **Bessel's correction**, which is the use of $n - 1$ instead of $n$ in the formula for sample variance [c1083]. For layer normalization we normalize over the full set of features for a given sample (not a statistical sample from a larger population), so we use $n$ in the denominator—hence `unbiased=False`.

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

A few things to note:

- `self.eps = 1e-5` is a small epsilon value added inside the square root to prevent division by zero [c2104].
- `self.scale` is initialized to ones and `self.shift` to zeros, so at initialization the layer is an identity transformation. The network learns to deviate from this as needed.
- The layer normalization formula applied here is $y_{\text{normalized}} = (y - \mu) / \sqrt{\sigma^2}$, applied independently to each sample in the batch [c1059].

[FIGURE: Diagram comparing batch normalization (normalizing across the batch dimension for each feature) versus layer normalization (normalizing across the feature dimension for each sample), showing a 2D grid of batch × features with arrows indicating the normalization direction for each method]

---

### GELU: The Activation Function of Choice

#### Why Not ReLU?

The ReLU activation function is simple and effective: it returns the input value for $x > 0$ and returns zero for $x < 0$ [c2002]. For many years it was the default choice. But ReLU has a well-known failure mode called the **dead neuron problem**.

The dead neuron problem occurs when a neuron's output becomes negative, ReLU sets it to zero, and the neuron stops contributing to the learning process [c2005]. Once a neuron is dead—once its pre-activation is consistently negative—the gradient through it is also zero, so no weight update can revive it. In a large model with millions of neurons, a significant fraction can die during training, wasting capacity.

#### GELU: A Smooth Alternative

GELU (Gaussian Error Linear Unit) is a smooth activation function that is differentiable at $x = 0$, unlike ReLU [c2075]. The exact definition is [c2006]:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution [c2006]. Intuitively, GELU weights each input value by the probability that a standard normal random variable is less than or equal to it. For large positive $x$, $\Phi(x) \approx 1$ and GELU behaves like the identity; for large negative $x$, $\Phi(x) \approx 0$ and GELU suppresses the value—but smoothly, not with a hard cutoff.

Because $\Phi(x)$ involves the error function, which has no closed form, a practical approximation is used in GPT-2 [c2108]:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

[c2009]

[FIGURE: Side-by-side plot of ReLU and GELU activation functions over the range x ∈ [-3, 3], highlighting GELU's smooth, non-zero gradient for negative x values and the hard zero of ReLU for x < 0]

#### The GELU Class

The PyTorch implementation wraps this approximation in a module:

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

This matches the approximation formula exactly [c2108]. The consequence of using GELU instead of ReLU is that gradients can flow even for slightly negative inputs, reducing the risk of dead neurons and generally improving training stability in deep transformer models.

---

### The Feed-Forward Network

#### Purpose and Architecture

The goal of implementing a feed-forward neural network sub-module is to create a component that is part of the LLM transformer block [c1999]. Within the transformer block, the feed-forward network processes each element of the sequence separately, without considering relationships with other elements [c2092]. That relationship-aware work is handled by the self-attention mechanism; the feed-forward network's job is to apply a learned non-linear transformation to each position independently.

The feed-forward network in a transformer block has an **expansion layer** followed by a **compression layer**, preserving input and output dimensions [c2070]. Specifically, the architecture consists of three layers [c2110]:

1. An **expansion layer** that increases the dimension to 4 times the embedding dimension.
2. A **GELU activation function**.
3. A **contraction layer** that reduces back to the original embedding dimension.

[FIGURE: Diagram of the feed-forward network showing input of dimension d_model, expansion to 4×d_model through a linear layer, GELU activation, and contraction back to d_model through a second linear layer]

For GPT-2's embedding dimension of 768, this means the first linear layer projects from 768 to 3072 dimensions, and the second projects from 3072 back to 768 [c2031]. The feedforward module uses GELU activation function between the expansion and contraction layers [c2038].

#### Why Expand and Then Contract?

The expansion-contraction architecture preserves input dimensionality while allowing richer exploration of the feature space [c2029]. By temporarily lifting the representation into a higher-dimensional space, the network can learn more complex, non-linear relationships between features. The expansion and contraction allows exploration of a richer parameter space and helps LLMs learn better relationships between parameters [c2073]. After this exploration, the contraction brings the representation back to the original dimension so it can be combined with the residual connection and passed to the next block.

#### Implementation

`nn.Sequential` is a PyTorch module that chains neural network layers together by forwarding outputs sequentially to inputs of subsequent modules [c2033]. The feed-forward network uses this to compose its three layers cleanly:

The feed-forward network processes input through expansion, GELU activation, and contraction, with output dimensions matching the input dimensions [c2120]. The FF object is an instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation function, initializing weights randomly [c2119].

---

### Shortcut Connections and the Vanishing Gradient Problem

#### The Vanishing Gradient Problem

Before we can appreciate shortcut connections, we need to understand the problem they solve. The **vanishing gradient problem** occurs when gradients become extremely small or approach zero during backpropagation through deep layers, preventing weight updates and causing training stagnancy [c224].

To see why this happens, recall that backpropagation computes gradients by repeatedly applying the chain rule. Each time a gradient passes through a layer, it is multiplied by that layer's local derivative. If those derivatives are consistently less than 1—which is common—the product of many such terms shrinks exponentially as it travels backward through a deep network. By the time the gradient reaches the early layers, it is so small that those layers effectively stop learning.

Shortcut connections help transformers solve the vanishing gradient problem [c275].

#### What Are Shortcut Connections?

Shortcut connections are also known as skip connections or residual connections [c211, c217, c241]. They are achieved by adding the output of one layer to the output of a later layer [c228]. In a diagram of the transformer block, shortcut connections are represented by plus symbols with associated arrows that bypass the linear flow of data [c216].

Shortcut connections add the output of one layer to the output of the previous layer, creating an alternative path for gradient flow [c2079]. This alternative path is the key insight: gradients no longer have to travel exclusively through the sequence of transformations—they can take a shortcut directly from later layers back to earlier ones [c273].

Shortcut connections can be added between any layers of the transformer block [c274], and they solve the vanishing gradient problem by providing these alternative paths for gradient flow through the network [c233].

#### The Math Behind Shortcut Connections

Consider a residual block where the output of layer $l+1$ is [c234]:

$$y_{l+1} = f(y_l) + y_l$$

where $f(y_l)$ is the output of the neural network transformation and $y_l$ is the output of the previous layer.

Using the chain rule, the partial derivative of the loss with respect to $y_l$ is [c236]:

$$\frac{\partial L}{\partial y_l} = \frac{\partial L}{\partial y_{l+1}} \cdot \frac{\partial y_{l+1}}{\partial y_l}$$

Now, because $y_{l+1} = f(y_l) + y_l$, we can compute the inner derivative [c237]:

$$\frac{\partial y_{l+1}}{\partial y_l} = \frac{\partial f(y_l)}{\partial y_l} + 1$$

The critical observation is the **+1** term. Even if $\partial f(y_l)/\partial y_l$ becomes very small (the vanishing gradient scenario), the total derivative is at least 1. This means the gradient signal is never completely killed by passing through a residual block—there is always a direct path carrying the full gradient magnitude. With shortcut connections, gradient magnitude remains stable across layers, preventing the vanishing gradient problem [c2082].

[FIGURE: Graph showing gradient magnitude across layers with and without shortcut connections: without shortcuts, gradient magnitude decays exponentially toward zero in early layers; with shortcuts, gradient magnitude remains roughly constant across all layers]

#### Implementing Shortcut Connections

The implementation is elegantly simple. After computing a layer's output, you add the original input back:

# Source: [c255]
```python
x = x + layer_output
```

That single line is the entire residual connection. To verify that shortcut connections actually stabilize gradients, we can set up a small experiment. First, define a loss function as the squared difference between the model output and the ground truth target [c260]:

$$\text{loss} = (Y - \text{target})^2$$

Then compute gradients:

# Source: [c262]
```python
loss = (model(x) - target) ** 2
loss.backward()
```

After calling `loss.backward()`, you can inspect the `.grad` attribute of each layer's weights to confirm that gradient magnitudes are consistent across layers when shortcut connections are present.

A deep neural network class for this experiment takes `layer_sizes` as an argument, which specifies the number of neurons in each layer—for example, `[3, 3, 3, 3, 1]` means 5 layers with 3 neurons each and a final layer with 1 neuron [c248].

---

### Assembling the Transformer Block

Now that we have all the pieces, we can put them together. The transformer block is the fundamental building block of GPT and other LLM architectures [c2051].

#### Components

The five components of a transformer block are: masked multi-head attention, layer normalization, dropout, feed-forward neural network, and GELU activation function [c2058].

Let's look at each component's role:

- **Multi-head attention** takes an input matrix X and multiplies it with trainable queries, keys, and values matrices to produce context vectors [c2059]. The self-attention block analyzes the relationship between input elements and assigns attention scores based on how one input element relates to other input elements [c2091].

- **Layer normalization** is implemented twice within the transformer block: before the multi-head attention and before the feed-forward neural network [c2067]. The first layer normalization (`norm_one`) is applied before the multi-head attention, and the second (`norm_two`) is applied before the feed-forward neural network [c2121].

- **Dropout** randomly turns off some layer outputs during training to improve generalization and prevent overfitting [c2068]. The `drop_shortcut` object is a dropout layer from PyTorch (`torch.nn.Dropout`) [c2123].

- **The feed-forward neural network** processes each element separately without considering relationships with other elements [c2092]. The FF object is an instance of the feed-forward class [c2119].

- **Shortcut connections** are applied after both the attention sub-layer and the feed-forward sub-layer [c2133].

In a transformer block, each element of an input sequence is represented by a fixed-size vector equal to the embedding dimension [c2084]. Transformer block operations such as multihead attention and feedforward layers are designed to preserve the dimensionality of input vectors [c2085], which is what makes the residual addition possible—you can only add tensors of the same shape.

#### Pre-Layer Norm vs. Post-Layer Norm

The original transformer model applied layer normalization *after* the self-attention and feed-forward neural network, a pattern called **post-layer norm** [c2130]. Modern GPT-style models instead apply layer normalization *before* each sub-layer—a pattern called **pre-layer norm**. This is the arrangement we implement here, where `norm_one` precedes attention and `norm_two` precedes the feed-forward network [c2121].

#### The Stacking Order

The transformer block stacking order is [c2083]:

1. Layer normalization
2. Multi-head attention
3. Dropout
4. Shortcut connection (add input to output)
5. Layer normalization
6. Feed-forward neural network with GELU
7. Dropout
8. Shortcut connection (add input to output)

[FIGURE: Detailed diagram of the transformer block showing the two sub-layers (attention and feed-forward), each preceded by layer normalization and followed by dropout and a residual addition, with arrows showing the shortcut paths bypassing each sub-layer]

A transformer block class in PyTorch includes a multi-head attention mechanism and a feed-forward neural network [c2128]. The transformer block implements shortcut connections (residual connections) that add the input of the block to the output of each component [c2133]. A dropout layer and residual shortcut connection are applied after the feed-forward network to prevent vanishing gradients [c2141].

The ATT object is an instance of the multi-head attention class that takes embedding vectors and converts them into context vectors, with input and output dimensions equal to the embedding dimension [c2113].

#### Putting It All Together

The complete transformer block forward pass follows the stacking order above. For the attention sub-layer:

```
shortcut = x
x = norm_one(x)
x = attention(x)
x = drop_shortcut(x)
x = x + shortcut          # residual connection
```

For the feed-forward sub-layer:

```
shortcut = x
x = norm_two(x)
x = feedforward(x)
x = drop_shortcut(x)
x = x + shortcut          # residual connection
```

The pattern is identical for both sub-layers: save the input, normalize, transform, apply dropout, add the saved input back. This clean symmetry is one of the reasons the transformer architecture is so easy to reason about and extend.

---

### The GPT Architecture in Context

The four key components we have built in this chapter—layer normalization, GELU activation, feed-forward neural network, and shortcut connections—are exactly the four key components of the GPT architecture [c2148, c2049]. Together with the multi-head attention mechanism covered in earlier chapters, they form a complete transformer block.

Previous lectures covered layer normalization, GELU activation, and feed-forward neural networks as building blocks for transformers [c277], and shortcut connections are covered in detail as a building block for understanding the transformer block [c278]. Now that all these pieces are in hand, the path to a full GPT model is clear: stack $N$ of these transformer blocks, add a token embedding layer at the input and a linear projection at the output, and you have the complete architecture.

[FIGURE: High-level diagram of the full GPT architecture showing token embeddings feeding into a stack of N transformer blocks, each containing the components built in this chapter, followed by a final layer normalization and linear output projection]

---

### Summary

This chapter built four essential sub-components of the transformer block from scratch:

1. **Layer normalization** addresses internal covariate shift by normalizing each sample's features to zero mean and unit variance, then applying learned scale and shift parameters. It normalizes along the feature dimension, unlike batch normalization which normalizes along the batch dimension.

2. **GELU activation** replaces ReLU with a smooth, differentiable function that avoids the dead neuron problem. Its approximation formula—$0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$—is the version used in GPT-2 [c2108].

3. **The feed-forward network** uses an expansion-contraction architecture: it projects from the embedding dimension to four times that size, applies GELU, and projects back. This allows the model to explore a richer feature space while preserving dimensionality for the residual connection [c2029, c2110].

4. **Shortcut connections** solve the vanishing gradient problem by adding a direct path from a layer's input to its output. The mathematical consequence—a guaranteed +1 term in the gradient—ensures that gradient magnitude remains stable across many layers [c237, c2082].

With these components assembled into a transformer block and stacked repeatedly, we have the structural foundation of a working GPT model. The next step is to wire up the full model: embeddings, positional encodings, the stack of transformer blocks, and the output head.