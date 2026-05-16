## Data Preprocessing: Input-Target Pairs, Token Embeddings, and Positional Embeddings

Before a large language model can learn anything, raw text must be transformed into a form the model can actually consume. This chapter walks through the complete data preprocessing pipeline: creating input-target pairs from raw text, batching those pairs efficiently with a DataLoader, converting token IDs into dense vector embeddings, and finally injecting positional information so the model knows where each token sits in a sequence.

The pipeline has four distinct stages [c3023]:

---

### Creating Input-Target Pairs

#### Why LLMs Need a Special Pairing Strategy

Standard supervised learning tasks — classification, regression — come with explicit labels. Large language models, by contrast, use a specific technique for creating input-target pairs that differs from those conventional approaches [c2710].

Input-target pairs in LLMs are structured so that the input is a sequence of words up to a certain point and the target is the next word that follows [c2711]. More precisely, there are two variables, $x$ and $y$, where $x$ contains input tokens and $y$ contains targets that are the inputs shifted by one position [c2729]. Tokens on the left of the arrow represent the input to the model, and the token on the right represents the target token ID the model is supposed to predict [c2752].

For example, given the sequence `["The", "cat", "sat"]`, the model should predict `"on"`; given `["The", "cat", "sat", "on"]`, it should predict `"the"`. The same corpus of text generates thousands of such pairs automatically — no human labeling required.

Input-target pairs must be created before vector embeddings are fed to the LLM training process [c2709], and a dataset for language modeling must consist of these input-output pairs, not just raw tokens [c2774].

#### Context Size and the Sliding Window

A context size of 4 means the model is trained to look at a sequence of 4 tokens to predict the next one [c2736]. Intuitively, context size represents how many words the model should pay attention to at one time [c2738].

Input-output pairs are constructed based on context size, where each pair contains multiple prediction tasks corresponding to predicting the next word at each position [c2978]:

- Input `[t0]` → target `t1`
- Input `[t0, t1]` → target `t2`
- Input `[t0, t1, t2]` → target `t3`
- Input `[t0, t1, t2, t3]` → target `t4`

Input-output pairs are constructed by taking a sequence of tokens as input and the next token as the target output [c2745]. The following code snippet illustrates this loop:

```python
# Source: [c2747]
context = encoded_data[:i]
desired = encoded_data[i]
```

A data loader fetches input-target pairs using a sliding window approach [c2722], advancing the window by one token position each time to produce the next pair [c2773].

**Stride** controls how far the window moves between consecutive pairs — specifically, how many tokens to skip when sliding the context window [c2777].

[FIGURE: Diagram showing a token sequence with a sliding window of context_size=4 and stride=1, illustrating how the window shifts to produce successive input-target pairs]

#### Loading the Raw Text

Before building pairs, we need encoded tokens. The following code reads a text file and encodes it with a tokenizer:

```python
# Source: [c2724]
raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```

This produces a flat list of integer token IDs ready for windowing.

---

### The Dataset and DataLoader Abstraction

#### Why Use a DataLoader?

A data loader is a PyTorch tool that processes data by creating input-output target pairs using a sliding window approach [c2980], then iterating over those pairs and returning them as PyTorch tensors [c2762]. A tensor can be thought of as a multi-dimensional array [c2764]. Batch size specifies how many input-output pairs are processed before updating model parameters [c2984].

#### Dataset Structure

In a data loader implementation, the input tensor $X$ contains rows representing input contexts, and the target tensor $Y$ contains the corresponding prediction targets — that is, the inputs shifted by one word [c2981]. The `__getitem__` method returns the input tensor row and target tensor row at a given index [c2779].

The `create_data_loader` function initializes a tokenizer, creates a dataset instance, and wraps it in a DataLoader for batch processing [c2783]:

```python
# Source: [c2807]
data_loader = DataLoader(dataset, batch_size=8)
```

The DataLoader extracts input-output pairs in batches, which are then converted into vector embeddings before being fed into the model [c2822].

#### Workers and Parallel Loading

The `num_workers` parameter enables parallel computing by distributing data loading across multiple CPU threads [c2821, c2792, c2986], which can significantly speed up data ingestion for large datasets.

#### Verifying the Output

A DataLoader with batch size of 1 and context size of 4 produces input-output pairs where the output tensor is the input tensor shifted by one position [c2794] — confirming that input-target pairs are sequences where the target is the input shifted by one position [c2814].

[FIGURE: Illustration of a batch of input-target pairs as two tensors X and Y, each of shape (batch_size, context_size), with Y being X shifted right by one token]

---

### Token Embeddings: From IDs to Vectors

#### What Is a Token Embedding?

Token embeddings map words into a vector space where each word is represented as a vector with a fixed number of dimensions [c2528], and they serve as the input to training large language models such as GPT [c2515].

The terminology can be confusing: token embeddings are also called vector embeddings or word embeddings, though "word embeddings" is not entirely accurate [c2509]. The reason is that "token" is a broader term than "word" — a token might be a subword, a punctuation mark, or a special symbol — which is why "token embeddings" is the preferred terminology [c2511].

Token embeddings are created by converting token IDs into embedding vectors using an embedding weight matrix [c2541], making them the third step in the overall LLM workflow: (1) tokenize input text into tokens, (2) convert tokens into token IDs, and (3) convert token IDs into token embeddings [c2514].

#### The Embedding Weight Matrix

An embedding layer weight matrix is a lookup table that stores vector embeddings for each token ID in the vocabulary, with dimensions of (vocabulary_size, embedding_dimension) [c2993]. Constructing one requires two quantities: the vector dimension (the size each token is projected to) and the vocabulary size (the number of unique token IDs) [c3300].

The weight matrix serves a dual role: it is both a matrix of trainable weights and a lookup table for retrieving vector representations by token ID [c2568], mapping vocabulary IDs to dense vectors of a specified embedding dimension [c2575].

[FIGURE: A rectangular matrix of shape (vocab_size, embedding_dim) with rows labeled by token IDs, illustrating how a row lookup retrieves the embedding vector for a given token]

#### Word2Vec: A Historical Perspective

The following code loads a classic Word2Vec model trained on Google News:

```python
# Source: [c2531]
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
word_vectors = model
print(word_vectors['computer'])
```

Modern LLMs do not use pre-trained Word2Vec vectors; instead, they learn their own embeddings from scratch as part of end-to-end training. Creating token embeddings for large vocabularies is computationally expensive, which is one reason why training large language models like GPT takes a huge amount of time [c2526].

#### `torch.nn.Embedding`: The PyTorch Lookup Table

PyTorch provides a dedicated module for this purpose. `torch.nn.Embedding` is a lookup table that stores embeddings of a fixed dictionary and size, used to store word embeddings and retrieve them using indices [c2555]:

```python
# Source: [c2998]
torch.nn.Embedding(num_embeddings, embedding_dim)
# Creates an embedding layer with num_embeddings rows and embedding_dim columns
# Stores embeddings of a fixed dictionary and size
```

```python
# Source: [c2556]
embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
```

`torch.nn.Embedding(num_embeddings, embedding_dim)` creates an embedding layer that functions as a lookup table, retrieving embeddings using token indices [c3303]. To inspect the underlying weight matrix:

```python
# Source: [c2559]
embedding_layer.weight
```

The lookup operation retrieves only the embedding vectors corresponding to the input IDs, without computing embeddings for unused vocabulary IDs [c2579]. Just as convolutional neural networks exploit spatial features of an image before training, token embeddings extract a structured representation of each token before it enters the model [c2585].

#### Building the Token Embedding Layer for GPT

```python
# Source: [c3003]
vocab_size = 50257
embedding_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
# Creates embedding matrix with 50257 rows and 256 columns
```

A simpler illustration of the same call:

```python
# Source: [c2577]
embedding = torch.nn.Embedding(vocab_size, embedding_dim)
# Example: torch.nn.Embedding(4, 5) creates a 4x5 embedding matrix
```

With a batch size of 8 and a context size of 4, the token embedding layer produces an output tensor of shape $8 \times 4 \times 256$ [c3017].

---

### The Problem with Token Embeddings Alone

Token embeddings capture semantic meaning — words with similar meanings end up close together in the vector space. However, token embedding does not take into account the position of words in a sequence, so identical words at different positions receive the same embedding vector [c3006]. In other words, token embeddings alone carry no information about where a word appears in a sentence [c2594], and without that information the model cannot distinguish sentences that differ only in word order.

It is therefore important to inject additional position information into the model alongside the semantic content captured by token embeddings [c3280]. Without token embeddings, the semantic relationship between related words like "dog" and "puppy" or "cat" and "kitten" is lost [c3275] — but without positional embeddings, the *order* of those words is lost.

---

### Positional Embeddings

#### What Is Positional Encoding?

Positional encoding is a technique that injects additional positional information into large language models to complement token embeddings [c3008]. It is the final step in LLM data pre-processing, encoding information about the position of each token in a sequence [c3276]. The terms "positional encoding" and "positional embedding" are used interchangeably to refer to vectors in a higher-dimensional space [c3295].

Both absolute and relative positional embeddings enable large language models to understand the order and relationship between tokens, leading to more accurate and context-aware predictions [c3287].

#### Absolute vs. Relative Positional Embeddings

There are two main approaches to encoding position information: absolute positional embeddings and relative positional embeddings [c3348].

In absolute positional embedding, the same token receives different final embeddings when it appears at different positions in the sequence, because a different positional embedding is added to the same token embedding at each position [c3283]. This approach is well-suited for tasks where the fixed order of tokens is crucial, such as sequence generation [c3011]. GPT models (GPT-3, GPT-4) use absolute positional embeddings that are optimized during training [c3290].

Relative positional encoding takes a different approach, emphasizing the relative distance between tokens rather than their absolute positions in the sequence [c3010]. Intuitively, relative encodings let the model reason about "how far apart" two tokens are, rather than "what slot each token occupies."

[FIGURE: Side-by-side comparison of absolute positional embeddings (each position gets a unique fixed vector) vs. relative positional embeddings (distances between positions are encoded)]

#### The Sinusoidal Formula (Original Transformer)

The original Transformer paper proposed a fixed, formula-based approach to absolute positional encoding using sinusoidal functions. For even dimensions:

$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$ [c3292]

And for odd dimensions:

$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$ [c3293]

GPT-style models do not use this fixed formula; they use learned positional embeddings instead [c3290].

#### Implementing Learned Positional Embeddings

Positional embedding matrix values are initialized randomly, similar to token embedding initialization, and both must be optimized during training [c3021, c3022].

To generate positional embeddings for a sequence of length `max_length`, we first create a sequence of position indices:

```python
# Source: [c3329]
torch.arange(max_length)
# creates a sequence of integers from 0 to max_length-1,
# used to index into the positional embedding lookup table
```

Passing positions 0, 1, 2, and 3 to the positional embedding lookup table generates four 256-dimensional positional embedding vectors [c3331]. Because positions are identical across all sequences in a batch, the same four positional embedding vectors are reused for every input sequence [c3327].

[FIGURE: A positional embedding lookup table of shape (context_size, embedding_dim), showing how position indices 0, 1, 2, 3 map to four distinct embedding vectors]

---

### Combining Token and Positional Embeddings

#### The Addition Operation

Positional embeddings are added to token embeddings using Python broadcasting, where the same positional embedding vectors are added to each row of the token embedding matrix [c3342]. Concretely, broadcasting converts the $4 \times 256$ positional embedding tensor into $8 \times 4 \times 256$ by duplicating values eight times — once for each item in the batch [c3018, c3335]. PyTorch handles this expansion automatically, so no manual tiling is required.

[FIGURE: Diagram showing token embedding tensor (8×4×256) plus positional embedding tensor (4×256) via broadcasting, producing input embedding tensor (8×4×256)]

#### The Complete Pipeline

With all four stages in place [c3023]:

1. **Tokenization**: Raw text → token IDs (covered in earlier chapters)
2. **Token embeddings**: Token IDs → dense vectors via `torch.nn.Embedding(vocab_size, embedding_dim)` [c3003]
3. **Positional embeddings**: Position indices → dense vectors via a second `torch.nn.Embedding(context_size, embedding_dim)` [c3021]
4. **Input embeddings**: Token embeddings + positional embeddings → final input tensor [c3342]

Token embeddings have shape $8 \times 4 \times 256$ and positional embeddings have shape $4 \times 256$ [c3017]. The resulting input embeddings feed directly into the first layer of the transformer. During inference, the model's own predictions loop back as new inputs — making it an autoregressive model, where the output of one iteration becomes the input of the next [c2718].

[FIGURE: End-to-end pipeline diagram: raw text → tokenizer → token IDs → DataLoader (sliding window, batching) → token embedding layer → + positional embedding layer → input embeddings → LLM]

---

### Putting It All Together: A Worked Example

Let us trace a concrete example through the entire pipeline.

**Step 1 — Tokenize and encode.** We read a text file and encode it into a list of integer token IDs [c2724].

**Step 2 — Build input-target pairs.** With a context size of 4 and a stride of 1, the sliding window produces pairs where each input is 4 tokens and each target is those same 4 tokens shifted right by one [c2777, c2729]. The data loader uses this sliding window approach to fetch input-output target pairs [c2767].

**Step 3 — Batch.** A DataLoader with batch size 8 groups 8 such pairs into a single batch [c2807]. The input tensor $X$ has shape $8 \times 4$ and the target tensor $Y$ has shape $8 \times 4$ [c2981].

**Step 4 — Token embeddings.** We pass $X$ through `torch.nn.Embedding(50257, 256)` [c3003]. The lookup retrieves one 256-dimensional vector per token ID, producing an output of shape $8 \times 4 \times 256$ [c3017].

**Step 5 — Positional embeddings.** We create position indices `[0, 1, 2, 3]` with `torch.arange(4)` [c3329] and pass them through a second `torch.nn.Embedding(4, 256)`, producing a tensor of shape $4 \times 256$ [c3331].

**Step 6 — Combine.** Broadcasting expands the positional embedding from $4 \times 256$ to $8 \times 4 \times 256$ [c3018], and the element-wise sum yields the final input embedding tensor of shape $8 \times 4 \times 256$ [c3342].

---

### Summary

- **Input-target pairs** are constructed by pairing each sequence of tokens with the next token as the target. The sliding window with a configurable stride generates these pairs efficiently [c2722, c2777].
- **The DataLoader** wraps a dataset of input-target pairs into batches of PyTorch tensors, with optional parallel loading via multiple worker threads [c2762, c2821].
- **Token embeddings** are dense vector representations stored in a lookup table (`torch.nn.Embedding`). Each token ID maps to a row in the embedding weight matrix [c2993, c3317], and the values are learned during training [c3022].
- **Token embeddings alone are insufficient** because they carry no information about token order [c3006, c2594]. Positional embeddings address this gap.
- **Positional embeddings** come in two flavors — absolute and relative [c3348]. GPT-style models use learned absolute positional embeddings [c3290], initialized randomly and optimized during training [c3021, c3022].
- **Input embeddings** are the element-wise sum of token and positional embeddings, combined via broadcasting [c3342, c3018], and form the final input to the transformer's first layer.