## Building the Transformer Block: LayerNorm, GELU, Feed-Forward, and Shortcut Connections

The transformer block is the fundamental building block of GPT and other large language model architectures [c2051]. Before we can assemble it, we need to understand and implement each of its sub-components individually. Let's start from the bottom up.

---

### Layer Normalization

#### The Problem: Internal Covariate Shift

At each training step, the weights of every layer change, which means the input distribution seen by each subsequent layer also shifts. This phenomenon is called **internal covariate shift**: the problem where the input distribution to each layer changes across training iterations, making weight updates difficult and delaying convergence [c1038].

#### The Math

Layer normalization normalizes each sample in a batch independently by subtracting the mean and dividing by the square root of variance computed across the features for that sample [c1058]. This is the key distinction from batch normalization, which performs normalization for an entire batch [c1100]. Layer normalization normalizes along the feature dimension (columns) [c1097], while batch normalization normalizes along the batch dimension (rows).

The mean is computed as [c1044]:

$$\mu = (X_1 + X_2 + X_3 + X_4) / 4$$

The variance is [c1045]:

$$\sigma^2 = \frac{1}{4}\left[(X_1 - \mu)^2 + (X_2 - \mu)^2 + (X_3 - \mu)^2 + (X_4 - \mu)^2\right]$$

And the normalized value for each feature is [c1046]:

$$\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2}}$$

The result is that layer normalization normalizes the output of every layer based on the mean and standard deviation [c1099], so that the mean becomes 0 and the variance becomes 1 [c2064].

#### A Quick Empirical Check

The following code creates a two-sample batch, passes it through a linear layer and ReLU, and prints the output so we can inspect its statistics [c1057]:

```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```

With the output tensor in hand, we can compute the mean and variance manually to mirror the formulas above. The mean across the feature dimension is [c1060]:

```python
mean = output.mean(dim=-1, keepdim=True)
```

The variance is [c1062]:

```python
variance = output.var(dim=-1, keepdim=True)
```

And the normalized output is [c1063]:

```python
normalized_output = (output - mean) / torch.sqrt(variance)
```

Notice that `dim=-1` operates along the last dimension—the feature dimension—which is exactly what layer normalization requires [c1097].

#### Scale and Shift: Trainable Parameters

To give the model the ability to undo normalization when useful, layer normalization introduces two trainable parameters: **scale** and **shift** [c1078]. Both have the same dimension as the input and are applied to the normalized output [c1078]. These parameters are learned during training [c2105], so the network can decide for itself how much normalization to apply.

#### A Note on Bessel's Correction

When computing variance in PyTorch, you will notice the flag `unbiased=False` in the implementation below. This relates to **Bessel's correction**, which is the use of $n - 1$ instead of $n$ in the formula for sample variance [c1083]. For layer normalization we normalize over the full set of features for a given sample—not a statistical sample drawn from a larger population—so we use $n$ in the denominator, hence `unbiased=False`.

#### The LayerNorm Class

Layer normalization is a class that takes the output of a layer and applies normalization to it [c1066]. Its forward method takes input with a certain number of rows and embedding-dimension columns [c1073]:

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

A few things to note:

- `self.eps = 1e-5` is a small epsilon value added inside the square root to prevent division by zero [c2104].
- `self.scale` is initialized to ones and `self.shift` to zeros, so at initialization the layer acts as an identity transformation; the network learns to deviate from this as needed.
- The normalization formula $y_{\text{normalized}} = (y - \mu) / \sqrt{\sigma^2}$ is applied independently to each sample in the batch [c1059].

---

### GELU: The Activation Function of Choice

#### Why Not ReLU?

The ReLU activation function is simple and effective: it returns the input value for $x > 0$ and returns zero for $x < 0$ [c2002]. Its main weakness is the **dead neuron problem**, which occurs when a neuron's output becomes negative, ReLU sets it to zero, and the neuron stops contributing to the learning process entirely [c2005]. In a large model with millions of neurons, a significant fraction can die during training, wasting capacity.

#### GELU: A Smooth Alternative

GELU (Gaussian Error Linear Unit) is a smooth activation function that is differentiable at $x = 0$, unlike ReLU [c2075]. Its exact definition is [c2006]:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution [c2006]. Because $\Phi(x)$ involves the error function, which has no closed form, a practical approximation is used in GPT-2 [c2108]:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,\bigl(x + 0.044715\, x^3\bigr)\right)\right)$$

[c2009]

[FIGURE: Side-by-side plot of ReLU and GELU activation functions over the range x ∈ [-3, 3], highlighting GELU's smooth, non-zero gradient for negative x values and the hard zero of ReLU for x < 0]

#### The GELU Class

The PyTorch implementation wraps this approximation in a module [c2109]:

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

This matches the GPT-2 approximation formula exactly [c2108].

---

### The Feed-Forward Network

#### Purpose and Architecture

The feed-forward neural network sub-module is a core component of the LLM transformer block [c1999]. Within the block, it processes each element of the sequence separately, without considering relationships with other elements [c2092].

The network follows an **expansion–contraction** design: an expansion layer followed by a compression layer, with input and output dimensions preserved [c2070]. Specifically, the architecture consists of three layers [c2110]:

1. An **expansion layer** that increases the dimension to 4× the embedding dimension.
2. A **GELU activation function**.
3. A **contraction layer** that reduces back to the original embedding dimension.

For GPT-2's embedding dimension of 768, the first linear layer projects from 768 to 3072 dimensions and the second projects from 3072 back to 768 [c2031]. The GELU activation sits between these two linear layers [c2038].

#### Why Expand and Then Contract?

The expansion-contraction architecture preserves input dimensionality while allowing richer exploration of the feature space [c2029]. By temporarily lifting the representation into a higher-dimensional space, the network can learn more complex, non-linear relationships between features; the expansion and contraction together allow exploration of a richer parameter space and help LLMs learn better relationships between parameters [c2073]. The contraction then brings the representation back to the original dimension so it can be combined with the residual connection and passed to the next block.

#### Implementation

`nn.Sequential` is a PyTorch module that chains neural network layers together by forwarding outputs sequentially to inputs of subsequent modules [c2033]. The feed-forward network processes input through expansion, GELU activation, and contraction, with output dimensions matching the input dimensions [c2120]. The FF object is an instance of the feed-forward class that takes the embedding dimension from configuration and creates layers with GELU activation, initializing weights randomly [c2119].

---

### Shortcut Connections and the Vanishing Gradient Problem

#### The Vanishing Gradient Problem

The **vanishing gradient problem** occurs when gradients become extremely small or approach zero during backpropagation through deep layers, preventing weight updates and causing training stagnancy [c224]. Shortcut connections are the primary mechanism by which transformers solve this problem [c275].

#### What Are Shortcut Connections?

Shortcut connections—also known as skip connections or residual connections [c211, c217, c241]—are achieved by adding the output of one layer to the output of a later layer [c228]. In a diagram of the transformer block, they are represented by plus symbols with associated arrows that bypass the linear flow of data [c216].

By creating an alternative path for gradient flow [c2079], shortcut connections ensure that gradients no longer have to travel exclusively through the sequence of transformations—they can bypass one or more layers entirely [c273]. Shortcut connections can be added between any layers of the transformer block [c274], and they solve the vanishing gradient problem by providing these alternative paths [c233].

#### The Math Behind Shortcut Connections

Consider a residual block where the output of layer $l+1$ is [c234]:

$$y_{l+1} = f(y_l) + y_l$$

Using the chain rule, the partial derivative of the loss with respect to $y_l$ is [c236]:

$$\frac{\partial L}{\partial y_l} = \frac{\partial L}{\partial y_{l+1}} \cdot \frac{\partial y_{l+1}}{\partial y_l}$$

Because $y_{l+1} = f(y_l) + y_l$, the inner derivative becomes [c237]:

$$\frac{\partial y_{l+1}}{\partial y_l} = \frac{\partial f(y_l)}{\partial y_l} + 1$$

The guaranteed $+1$ term means the gradient can never vanish completely through this path. As a result, gradient magnitude remains stable across layers, preventing the vanishing gradient problem [c2082].

#### Implementing Shortcut Connections

After computing a layer's output, the original input is added back [c255]:

```python
x = x + layer_output
```

To verify that shortcut connections actually stabilize gradients, we can set up a small experiment. The loss function is defined as the squared difference between the model output and the ground truth target [c260]:

```python
loss = (model(x) - target) ** 2
loss.backward()
```

[c262]

A deep neural network class for this experiment takes `layer_sizes` as an argument, which specifies the number of neurons in each layer—for example, `[3, 3, 3, 3, 1]` means 5 layers with 3 neurons each and a final layer with 1 neuron [c248].

---

### Assembling the Transformer Block

#### Components

The five components of a transformer block are: masked multi-head attention, layer normalization, dropout, feed-forward neural network, and GELU activation function [c2058]. Here is the role each plays:

- **Multi-head attention** takes an input matrix X and multiplies it with trainable queries, keys, and values matrices to produce context vectors [c2059]. The self-attention mechanism analyzes the relationship between input elements and assigns attention scores based on how each element relates to the others [c2091].

- **Layer normalization** appears twice within the transformer block: before the multi-head attention and before the feed-forward neural network [c2067]. The first instance (`norm_one`) precedes multi-head attention and the second (`norm_two`) precedes the feed-forward network [c2121].

- **Dropout** randomly turns off some layer outputs during training to improve generalization and prevent overfitting [c2068]. The `drop_shortcut` object is a dropout layer from PyTorch (`torch.nn.Dropout`) [c2123].

- **The feed-forward neural network** processes each element separately without considering relationships with other elements [c2092], implemented as the FF object described above [c2119].

- **Shortcut connections** are applied after both the attention sub-layer and the feed-forward sub-layer [c2133].

In a transformer block, each element of an input sequence is represented by a fixed-size vector equal to the embedding dimension [c2084]. Both multi-head attention and the feed-forward layers are designed to preserve this dimensionality [c2085], which is what makes the residual addition possible—you can only add tensors of the same shape.

#### Pre-Layer Norm vs. Post-Layer Norm

The original transformer model applied layer normalization *after* the self-attention and feed-forward sub-layers, a pattern called **post-layer norm** [c2130]. The arrangement implemented here applies `norm_one` before attention and `norm_two` before the feed-forward network [c2121].

#### The Stacking Order

The full transformer block stacking order is [c2083]:

1. Layer normalization → multi-head attention → dropout → shortcut connection
2. Layer normalization → feed-forward neural network (with GELU) → dropout → shortcut connection

A transformer block class in PyTorch includes a multi-head attention mechanism and a feed-forward neural network [c2128], with shortcut connections that add the input of the block to the output of each component [c2133]. A dropout layer and residual shortcut connection are applied after the feed-forward network to prevent vanishing gradients [c2141]. The ATT object is an instance of the multi-head attention class that takes embedding vectors and converts them into context vectors, with input and output dimensions equal to the embedding dimension [c2113].

#### Putting It All Together

The complete transformer block forward pass follows the stacking order above. For the attention sub-layer:

```python
shortcut = x
x = norm_one(x)
x = attention(x)
x = drop_shortcut(x)
x = x + shortcut  # residual connection
```

For the feed-forward sub-layer:

```python
shortcut = x
x = norm_two(x)
x = feedforward(x)
x = drop_shortcut(x)
x = x + shortcut  # residual connection
```

---

### The GPT Architecture in Context

The four sub-components built in this chapter—layer normalization, GELU activation, feed-forward neural network, and shortcut connections—are exactly the four key components of the GPT architecture [c2148, c2049]. Together with the multi-head attention mechanism covered in earlier chapters, they form a complete transformer block. Layer normalization, GELU activation, and feed-forward networks serve as foundational building blocks for transformers [c277], and shortcut connections are essential for making deep transformer stacks trainable [c278].

---

### Summary

This chapter built four essential sub-components of the transformer block from scratch:

- **Layer normalization** normalizes each sample independently along the feature dimension, unlike batch normalization which normalizes along the batch dimension. Trainable scale and shift parameters let the network recover expressive power after normalization [c1078, c1097].

- **GELU activation** is a smooth, differentiable alternative to ReLU that avoids the dead neuron problem. Its approximation formula—$0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$—is the version used in GPT-2 [c2108].

- **The feed-forward network** expands the representation to 4× the embedding dimension, applies GELU, then contracts back to the original dimension. This allows the model to explore a richer feature space while preserving dimensionality for the residual connection [c2029, c2110].

- **Shortcut connections** add the input of a sub-layer directly to its output, creating an alternative gradient path. The guaranteed $+1$ term in the gradient ensures that magnitude remains stable across many layers [c237, c2082].