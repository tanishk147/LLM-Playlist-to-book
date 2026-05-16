# The Transformer Architecture: From Attention Is All You Need to GPT

The story of modern large language models begins with a single paper. In 2017, a research team published "Attention Is All You Need," introducing the transformer — a deep neural network architecture that would reshape the field of machine learning [c1245, c1252]. In the years since, that paper has accumulated more than one hundred thousand citations [c1254], a testament to how thoroughly it changed what was possible. Most modern LLMs rely on the transformer architecture [c1244], and the GPT architecture that underlies ChatGPT originated directly from that work [c1255].

This chapter builds a complete mental model of the transformer: where it came from, how it is structured, how GPT and BERT diverged from the original design, and how the full pipeline from raw token IDs to output logits fits together. By the end, you will have enough context to understand every design decision in the GPT implementation we build throughout the rest of this book.

---

To appreciate why the transformer was such a leap forward, it helps to understand what came before it.

The dominant sequence-modeling tools before 2017 were Recurrent Neural Networks (RNNs) and their more capable cousins, Long Short-Term Memory (LSTM) networks. RNNs maintain a feedback loop to incorporate memory, allowing them to carry information from one step of a sequence to the next [c1313]. LSTM networks extend this idea by maintaining two separate paths — one for short-term memories and one for long-term memories [c1314]. Despite this improvement, both architectures struggled with what researchers call *long-range dependencies*: the need to understand context from distant parts of a sequence, such as an earlier sentence, to accurately predict the next word in a later one [c1280]. The feedback-loop design of RNNs means that information from many steps back tends to fade or become distorted by the time it is needed.

Attention mechanisms changed this. They have become an integral part of sequence modeling, allowing the modeling of dependencies without regard to the distance between input or output sequences [c1277]. The self-attention mechanism specifically allows the model to weigh the importance of different words and tokens relative to each other [c1279] and to capture long-range dependencies by looking far back in a sequence to identify the next word [c1281]. Indeed, without masked multi-head attention, large language models would lose their power and revert to relying on recurrent and long short-term memory networks [c373].

The paper "Attention Is All You Need" is named precisely because of the self-attention mechanism and the intuition behind it [c1284]. The name is a statement of intent: you do not need recurrence at all.

---

## The Original Transformer: Built for Machine Translation

The original transformer was proposed for machine translation tasks, specifically for translating English texts into German and French [c1260, c1319]. The transformer architecture consists of two main blocks: an encoder that converts input text to embedding vectors, and a decoder that generates output text from those embedding vectors and partial output [c1273].

[FIGURE: High-level encoder-decoder transformer diagram showing input text flowing into the encoder stack, producing embedding vectors, which feed into the decoder stack alongside partial output text, producing translated output one word at a time]

### The Encoder

The encoder takes tokenized input text and converts it into vector embeddings [c1267]. Intuitively, its job is to read and understand the source sentence, compressing its meaning into a rich numerical representation that the decoder can consult.

### The Decoder

The decoder generates output text one word at a time, receiving both the encoder's vector embeddings and the partial output text produced so far [c1270]. Previously generated words are made available to the model at each step, so the process is autoregressive: the output at one step becomes part of the input for the next [c1271].

### The Eight-Step Pipeline

At a high level, the transformer converts input text to output text through eight steps: tokenization, encoding, decoding with partial output, and word-by-word generation [c1268, c1320].

[FIGURE: Eight-step pipeline diagram: (1) tokenize source text, (2) embed tokens, (3) add positional encoding, (4) pass through encoder attention layers, (5) produce encoder embeddings, (6) feed encoder embeddings + partial target into decoder, (7) decoder attention layers, (8) generate next output word; repeat steps 6–8 until done]

After the original transformer was developed, researchers discovered that architectures derived from it could handle many tasks beyond machine translation [c1262]. Text completion — the predominant role of GPT — was not even in consideration when the transformer paper was originally proposed [c1261]. The transformer turned out to be a general-purpose sequence-processing engine, and researchers quickly began adapting it.

---

## BERT and GPT: Two Paths from One Architecture

Both BERT and GPT have the word "transformers" in their names because they originated from the transformer architecture [c1303], but they took very different paths from that common ancestor.

### BERT: The Encoder-Only Model

BERT stands for Bidirectional Encoder Representations from Transformers [c1288]. As the name implies, BERT uses only the encoder half of the original transformer and has no decoder [c1295, c1326]. It is a transformer variation that predicts hidden or masked words in a sentence [c1324], attending to the sentence from both left and right directions simultaneously [c1325]. This bidirectional view gives BERT a significant advantage for understanding tasks: it can capture nuances and relationships between words by examining the entire sentence at once [c1299].

One practical consequence is that BERT can differentiate between different meanings of the same word — such as "bank" as a financial institution versus "bank" as a river bank — by examining surrounding words [c1300]. BERT models are therefore commonly used for sentiment analysis, because they can capture the meanings of different words and how they relate to each other [c1327]. GPT can also perform sentiment analysis, though it is a specialty of BERT [c1302].

### GPT: The Decoder-Only Model

GPT stands for Generative Pre-trained Transformers [c1289]. It is a pre-trained foundational model [c1290] and the architecture that underlies ChatGPT [c1255]. Where BERT reads in both directions, GPT processes text strictly from left to right, predicting only the next unknown token at each step [c1296, c1328, c1293].

In ChatGPT, when you write an input, GPT gives attention to every preceding token and then predicts what the next word could be — not just the immediately preceding sentence [c1286]. This is the self-attention mechanism at work: even in a left-to-right model, every token can attend to all previous tokens, not just its immediate neighbor.

[FIGURE: Side-by-side comparison of BERT (bidirectional arrows between all tokens, encoder-only) and GPT (left-to-right arrows, decoder-only, next-word output)]

The key architectural difference is straightforward: BERT uses only the encoder half of the original transformer; GPT uses only the decoder half, with the cross-attention to an encoder removed since there is no encoder to attend to.

---

## Beyond Language: Vision Transformers

The transformer is not limited to text. Transformers can be applied to computer vision tasks such as image classification and image segmentation [c1330, c1305]. Vision Transformers (ViT) apply the transformer architecture specifically to tasks such as image recognition and image classification [c1306], and they achieve comparable or better results than Convolutional Neural Networks (CNNs) while requiring substantially fewer computational resources for pre-training [c1307].

This generality is part of what makes the transformer so important. The same core mechanism — self-attention over a sequence of tokens — works whether those tokens represent words, image patches, or other discrete units.

It is also worth being precise about terminology: "transformers" and "large language models" are not interchangeable [c1316]. Not all large language models are based on the transformer architecture [c1308], and not all transformers are large language models [c1304].

---

## Inside the Transformer Block

We now zoom in from the high-level architecture to the building block that makes it all work. The transformer block is the most important part of the large language model architecture, consisting of multiple components linked together [c366].

[FIGURE: Detailed transformer block diagram showing: layer normalization → masked multi-head attention → residual connection → layer normalization → feed-forward network (with GELU activation) → residual connection → output]

The attention mechanism is considered the heart of the transformer [c464, c362], and the transformer block contains a large number of trainable parameters and weights that are optimized during LLM pre-training [c370]. The feed-forward network within the block uses GELU activation [c369], and its weights are optimized during training to predict output words correctly [c1275]. At its simplest, the transformer can be understood as a neural network where parameters are optimized through training [c1276]; the complexity lies in the specific structure of those parameters and how information flows through them.

### Multi-Head Attention

The self-attention mechanism inside the transformer block is implemented as *multi-head attention*. The number of attention heads refers to the count of separate query, key, and value matrix sets created in this process [c390]. The governing equations are:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

[c1258]

where each individual head is computed as:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

[c1259]

Running multiple attention heads in parallel allows the model to simultaneously attend to different aspects of the input — one head might track syntactic relationships while another tracks semantic similarity. The results are concatenated and projected back to the model dimension via the output weight matrix $W^O$.

After all transformer blocks are applied, each input token has a 768-dimensional output vector representation [c433]. This dimensionality matches the embedding dimension used throughout the model.

---

## The GPT Model Configuration

Before tracing the full pipeline, it is useful to examine the concrete hyperparameters that define the GPT-124M model we will implement. Pre-training is training on a large, diverse dataset [c1246], and the model configuration determines the network's capacity for that training.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,           # Vocabulary size
    "context_length": 1024,        # Context length
    "emb_dim": 768,                # Embedding dimension
    "n_heads": 12,                 # Number of attention heads
    "n_layers": 12,                # Number of layers
    "drop_rate": 0.1,              # Dropout rate
    "qkv_bias": False              # Query-Key-Value bias
}
```

[c438]

Each field controls a distinct aspect of the architecture:

- **`vocab_size: 50257`** — the number of distinct tokens the model knows about; each token ID is an integer in the range $[0, 50256]$.
- **`context_length: 1024`** — the maximum number of tokens the model can process in a single forward pass.
- **`emb_dim: 768`** — the dimensionality of every token's vector representation throughout the model. Each token ID in GPT-2 is converted into a vector of 768 dimensions [c379].
- **`n_heads: 12`** — the number of parallel attention heads in each transformer block [c390].
- **`n_layers: 12`** — the number of transformer blocks stacked on top of each other.
- **`drop_rate: 0.1`** — the dropout probability applied during training for regularization.
- **`qkv_bias: False`** — no bias terms are used in the Query-Key-Value projections [c445].

These parameters were covered in earlier lectures [c402], and we will refer back to them throughout the implementation chapters.

---

## The Full GPT Pipeline: From Token IDs to Logits

With the configuration in hand, we can trace the complete data flow through the GPT model — the bird's-eye view that ties every component together.

[FIGURE: Full GPT pipeline: token IDs → token embedding lookup (768-dim) + positional embedding lookup (768-dim) → sum → dropout → N × transformer blocks → layer norm → linear projection (768 → 50257) → logits matrix]

### Step 1: Token Embeddings

Token IDs are converted to token embeddings before being passed to the GPT model [c411]. The architecture workflow begins with token IDs that are converted to input embedding vectors of 768 dimensions [c446]. In GPT-2, the embedding layer is trained to learn token representations that capture semantic meaning [c388], with the embedding matrix parameters initialized randomly from a Gaussian distribution and refined during backpropagation [c417].

### Step 2: Positional Embeddings

Because the transformer has no built-in notion of order — unlike an RNN, which processes tokens sequentially — position information must be injected explicitly. The positional embedding is implemented as a learned lookup table:

```python
self.pos_emb = nn.Embedding(context_length, embedding_dimension)
```

[c425]

To index into this table, we construct a tensor of position indices:

```python
pos = torch.arange(sequence_length)
```

[c426]

This creates a tensor `[0, 1, 2, ..., sequence_length-1]` that indexes into the positional embedding matrix [c426]. The positional embedding for each position is then added to the corresponding token embedding, giving the model a combined representation that encodes both *what* the token is and *where* it appears in the sequence.

### Step 3: Dropout

After the token and positional embeddings are summed, dropout is applied for regularization. The output then passes through the transformer blocks [c449].

### Step 4: Transformer Blocks

The combined embedding passes through 12 stacked transformer blocks. Each block applies masked multi-head attention followed by a feed-forward network, with layer normalization and residual connections around each sub-layer [c468]. The attention mechanism is the heart of each block [c464], and the forward method of the GPT model class takes an input and outputs the next-word prediction by running it through all transformer blocks in sequence [c403].

### Step 5: Output Head — Projecting to Logits

After all transformer blocks have processed the sequence, a final linear transformation maps from the embedding dimension back to the vocabulary. The output head applies a linear transformation from the embedding dimension (768) to the vocabulary size (50,257), converting token representations to logits [c437].

Logits are the final output matrices from a transformer model that contain probability scores for each token in the vocabulary at each position in the sequence [c432]. The shape of the logits tensor is therefore `[batch_size, sequence_length, 50257]` — for every position in the input, the model produces a score for every possible next token.

[FIGURE: Logits matrix diagram showing shape [batch, seq_len, 50257] with each row representing a probability distribution over the vocabulary for one sequence position]

To convert logits to actual probabilities, a softmax function is applied, and the token with the highest probability (or a sample from the distribution) becomes the next generated token. Generating text from output tokens using the logits matrix is the final step in the pipeline [c470].

---

## Putting It All Together

The full architecture can be summarized as a single sequential pass:

1. **Tokenization**: raw text is split into tokens and mapped to integer IDs from a vocabulary of 50,257 tokens.
2. **Embedding**: each token ID is looked up in a learned embedding table, producing a 768-dimensional vector, to which a learned positional embedding is added to encode sequence position.
3. **Dropout**: applied to the summed embeddings during training.
4. **Transformer blocks** (×12): each block applies masked multi-head attention (12 heads) followed by a GELU-activated feed-forward network, with layer normalization and residual connections throughout. The large number of trainable parameters in these blocks are optimized during pre-training [c370].
5. **Output projection**: a linear layer maps the 768-dimensional representations to 50,257-dimensional logits.
6. **Generation**: logits are converted to probabilities, the next token is selected and appended to the sequence, and the process repeats.

The GPT configuration parameters [c438] control every dimension of this pipeline. Changing `n_layers` adds or removes transformer blocks; changing `emb_dim` widens or narrows every vector in the system; changing `n_heads` controls how many parallel attention patterns each block can learn.

---

## Chapter Summary

This chapter has traced the transformer from its origins as a machine translation system to its role as the foundation of modern LLMs. The key points to carry forward:

- The transformer was introduced in 2017 for machine translation and consists of an encoder and a decoder [c1245, c1273].
- Researchers subsequently discovered that transformer-derived architectures could handle many tasks beyond translation [c1262].
- BERT uses only the encoder and is bidirectional, making it well-suited for understanding tasks such as sentiment analysis [c1295, c1325, c1327].
- GPT uses only the decoder and generates text left-to-right, one token at a time [c1296, c1293].
- The transformer block — the core repeating unit — combines masked multi-head attention, a GELU-activated feed-forward network, layer normalization, and residual connections [c366, c369].
- The full GPT pipeline runs from token IDs through embeddings, through stacked transformer blocks, to a final linear projection that produces logits over the vocabulary [c446, c437, c432].
- The GPT-124M configuration uses a vocabulary of 50,257 tokens, a context length of 1,024, an embedding dimension of 768, 12 attention heads, and 12 transformer layers [c438].

In the chapters that follow, we will implement each of these components from scratch, starting with the attention mechanism — the heart of the transformer [c464].