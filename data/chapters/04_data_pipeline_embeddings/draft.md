## Data Preprocessing: Input-Target Pairs, Token Embeddings, and Positional Embeddings

Before a large language model can learn anything, raw text must be transformed into a form the model can actually consume. That transformation is what this chapter is about. We will walk through the complete data preprocessing pipeline: creating input-target pairs from raw text, batching those pairs efficiently with a DataLoader, converting token IDs into dense vector embeddings, and finally injecting positional information so the model knows where each token sits in a sequence. By the end, you will have a clear picture of how a string of text becomes the numerical input that feeds into LLM training.

The pipeline has four distinct stages [c3023]:

1. **Tokenization** — splitting raw text into tokens
2. **Token embeddings** — mapping token IDs to dense vectors
3. **Positional embeddings** — encoding each token's position in the sequence
4. **Input embeddings** — combining token and positional embeddings into a single representation

Chapters 2 and 3 covered tokenization in depth. Here we pick up at stage two and work all the way through stage four, but we begin with a prerequisite that sits between tokenization and embedding: constructing the input-target pairs that define what the model is actually trying to learn.

---

### Creating Input-Target Pairs

#### Why LLMs Need a Special Pairing Strategy

Standard supervised learning tasks — classification, regression — come with explicit labels. Language modeling is different. The labels are already hidden inside the text itself: every token is simultaneously an input (when predicting the next token) and a target (when it is the token being predicted). Large language models use a specific technique for creating input-target pairs that differs from standard machine learning classification and regression tasks [c2710].

The fundamental definition is simple. Input-target pairs in LLMs are structured as: the input is a sequence of words up to a certain point, and the target is the next word that follows [c2711]. More precisely, there are two variables, $x$ and $y$, where $x$ contains input tokens and $y$ contains targets that are the inputs shifted by one position [c2729]. Tokens on the left of the arrow represent the input to the model, and the token on the right represents the target token ID the model is supposed to predict [c2752].

Intuitively, this is the "fill in the next word" game. Given the sequence `["The", "cat", "sat"]`, the model should predict `"on"`. Given `["The", "cat", "sat", "on"]`, it should predict `"the"`. The same corpus of text generates thousands of such pairs automatically — no human labeling required.

Input-target pairs must be created before vector embeddings are fed to the LLM training process [c2709]. A dataset for language modeling must consist of input-output pairs, not just raw tokens [c2774].

#### Context Size and the Sliding Window

The number of tokens the model looks at to make a single prediction is called the **context size**. A context size of 4 means the model is trained to look at a sequence of 4 words or tokens to predict the next word in the sequence [c2736]. Intuitively, context size represents how many words the model should pay attention to at one time to predict the next word [c2738].

Each input-output pair contains multiple prediction tasks, not just one. Input-output pairs for LLM training are constructed based on context size, where each pair contains multiple prediction tasks corresponding to predicting the next word at each position [c2978]. To see this concretely, consider a context window of size 4 over the token sequence `[t0, t1, t2, t3, t4]`:

- Input `[t0]` → target `t1`
- Input `[t0, t1]` → target `t2`
- Input `[t0, t1, t2]` → target `t3`
- Input `[t0, t1, t2, t3]` → target `t4`

Input-output pairs are constructed by taking a sequence of tokens as input and the next token as the target output [c2745]. The following code snippet illustrates this loop:

# Source: [c2747]
```python
for i in range(1, context_size + 1):
    context = encoded_data[:i]
    desired = encoded_data[i]
```

The **sliding window** is the mechanism that generates new pairs by advancing through the token stream. A data loader fetches input-target pairs using a sliding window approach [c2722]. The sliding window approach works by sliding the input window by one token position to create the corresponding output [c2773].

**Stride** controls how far the window moves between consecutive pairs. Stride determines how many tokens to skip when sliding the context window to create the next input-output pair [c2777]. A stride of 1 means maximum overlap between consecutive windows (and therefore the most training pairs); a larger stride reduces overlap and produces fewer, more independent pairs.

[FIGURE: Diagram showing a token sequence with a sliding window of context_size=4 and stride=1, illustrating how the window shifts to produce successive input-target pairs]

#### Loading the Raw Text

Before building pairs, we need encoded tokens. The following code reads a text file and encodes it with a tokenizer:

# Source: [c2724]
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```

This gives us a flat list of integer token IDs. The next step is to organize those IDs into structured input-target pairs.

---

### The Dataset and DataLoader Abstraction

#### Why Use a DataLoader?

Iterating over pairs one at a time is fine for illustration, but real training requires **batches** — groups of pairs processed together before updating model parameters. Batch size specifies how many input-output pairs are processed before updating model parameters [c2984]. A data loader is a PyTorch tool that processes data by creating input-output target pairs using a sliding window approach [c2980].

A data loader is used to iterate over the input dataset and return inputs and targets as PyTorch tensors [c2762]. A tensor can be thought of as a multi-dimensional array [c2764].

#### Dataset Structure

In a data loader implementation, the input tensor $X$ contains rows representing input contexts, and the target tensor $Y$ contains corresponding prediction targets which are the input shifted by one word [c2981]. The `__getitem__` method returns the input tensor row and target tensor row at a given index [c2779].

The `create_data_loader` function initializes a tokenizer, creates a dataset instance, and wraps it in a DataLoader for batch processing [c2783]. Here is a minimal example of wrapping a dataset in a DataLoader:

# Source: [c2807]
```python
data_loader = DataLoader(dataset, batch_size=8)
```

The DataLoader extracts input-output pairs in batches (first batch, second batch, etc.) which are then converted into vector embeddings before being fed into the model [c2822].

#### Workers and Parallel Loading

The number of workers parameter enables parallel computing by distributing data loading across multiple threads on the computer [c2821]. Number of workers is for parallel processing on different threads of the CPU [c2792]. The number of workers parameter in a data loader specifies how many threads are used for parallel processing on the CPU [c2986]. On machines with multiple CPU cores, increasing this value can significantly speed up data loading.

#### Verifying the Output

A DataLoader with batch size of 1 and context size of 4 produces input-output pairs where the output tensor is the input tensor shifted by one position [c2794]. Input-target pairs are sequences where the target is the input shifted by one position, used for training language models [c2814].

[FIGURE: Illustration of a batch of input-target pairs as two tensors X and Y, each of shape (batch_size, context_size), with Y being X shifted right by one token]

---

### Token Embeddings: From IDs to Vectors

#### What Is a Token Embedding?

Once we have batches of integer token IDs, we need to convert them into continuous vectors that the model can process mathematically. Token embeddings map words into a vector space where each word is represented as a vector with a fixed number of dimensions [c2528]. Token embeddings serve as the input to training large language models such as GPT [c2515].

The terminology can be confusing. Token Embeddings are also called Vector Embeddings or Word Embeddings, though Word Embeddings is not entirely accurate terminology [c2509]. The reason is that "token" is a broader term than "word" — a token might be a subword, a punctuation mark, or a special symbol — which is why Token Embeddings is the preferred terminology [c2511].

Token embeddings for large language models are created by converting token IDs into embedding vectors using an embedding weight matrix [c2541]. The large language model workflow has three main steps: (1) tokenizing input text into tokens, (2) converting tokens into token IDs, and (3) converting token IDs into token embeddings, which serve as input to training the model [c2514].

#### The Embedding Weight Matrix

An embedding layer weight matrix is a lookup table that stores vector embeddings for each token ID in the vocabulary, with dimensions of (vocabulary_size, embedding_dimension) [c2993]. A token embedding matrix (or embedding layer) requires two quantities: the vector dimension (the size each token is projected to) and the vocabulary size (the number of unique token IDs) [c3300].

The embedding weight matrix functions both as a matrix of weights for vector embeddings and as a lookup table for finding vector representations by token ID [c2568]. An embedding layer maps vocabulary IDs to dense vectors of a specified embedding dimension [c2575].

[FIGURE: A rectangular matrix of shape (vocab_size, embedding_dim) with rows labeled by token IDs, illustrating how a row lookup retrieves the embedding vector for a given token]

#### Word2Vec: A Historical Perspective

Before the era of end-to-end trained LLMs, word embeddings were often pre-trained separately. The following code loads a classic Word2Vec model trained on Google News:

# Source: [c2531]
```python
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
word_vectors = model
print(word_vectors['computer'])
```

This returns a 300-dimensional vector for the word `'computer'`. Modern LLMs do not use pre-trained Word2Vec vectors; instead, they learn their own embeddings from scratch as part of end-to-end training. Creating token embeddings for large vocabularies is computationally expensive, which is one reason why training large language models like GPT takes a huge amount of time [c2526].

#### `torch.nn.Embedding`: The PyTorch Lookup Table

PyTorch provides a dedicated module for this purpose. `torch.nn.Embedding` is a lookup table that stores embeddings of a fixed dictionary and size, used to store word embeddings and retrieve them using indices [c2555].

# Source: [c2998]
```python
torch.nn.Embedding(num_embeddings, embedding_dim)
# Creates an embedding layer with num_embeddings rows and embedding_dim columns
# Stores embeddings of a fixed dictionary and size
```

# Source: [c2556]
```python
embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
```

`torch.nn.Embedding(num_embeddings, embedding_dim)` creates an embedding layer that stores embeddings of a fixed dictionary and size, functioning as a lookup table that retrieves embeddings using token indices [c3303].

To inspect the underlying weight matrix:

# Source: [c2559]
```python
embedding_layer.weight
```

An embedding layer is a lookup table that maps token IDs to their corresponding vector representations [c3317]. The embedding layer operates as a lookup mechanism that retrieves pre-stored vector representations indexed by token IDs [c3321]. Embedding layers function as lookup tables that map indices to corresponding embedding vectors [c3330].

An embedding layer lookup operation retrieves only the embedding vectors corresponding to the input IDs, without computing embeddings for unused vocabulary IDs [c2579]. This is an important efficiency property: even if the vocabulary has 50,000 entries, a forward pass only touches the rows corresponding to the tokens actually present in the current batch.

In convolutional neural networks, spatial features of an image are exploited before giving it as input for training, similar to how token embeddings work [c2585]. The analogy is instructive: just as a CNN transforms raw pixel values into feature maps, the embedding layer transforms raw token IDs into feature vectors.

#### Building the Token Embedding Layer for GPT

For a GPT-scale model, the embedding layer is substantially larger:

# Source: [c3003]
```python
vocab_size = 50257
embedding_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
# Creates embedding matrix with 50257 rows and 256 columns
```

A simpler form of the same call:

# Source: [c2577]
```python
embedding = torch.nn.Embedding(vocab_size, embedding_dim)
# Example: torch.nn.Embedding(4, 5) creates a 4x5 embedding matrix
```

With a batch size of 8 and a context size of 4, the token embedding layer produces an output tensor of shape $8 \times 4 \times 256$ [c3017]. Each of the 8 sequences in the batch contains 4 tokens, and each token is represented by a 256-dimensional vector.

---

### The Problem with Token Embeddings Alone

Token embeddings capture semantic meaning — words with similar meanings end up close together in the vector space. But they have a critical blind spot.

Token embedding does not take into account the position of words in a sequence, so identical words at different positions receive the same embedding vector [c3006]. Token embeddings alone do not encode positional information about where words appear in a sentence [c2594].

Consider the sentences "The dog bit the man" and "The man bit the dog." The token embeddings for "dog" and "man" are identical in both sentences, even though their roles are completely reversed. Without positional information, the model cannot distinguish these two sentences.

It is important to inject additional position information to the large language model along with capturing semantic meaning through token embeddings [c3280]. Without token embeddings, the semantic relationship between related words like 'dog' and 'puppy' or 'cat' and 'kitten' is lost [c3275] — but without positional embeddings, the *order* of those words is lost. Both are essential.

---

### Positional Embeddings

#### What Is Positional Encoding?

Positional encoding is a technique that injects additional positional information into large language models to complement token embeddings [c3008]. Positional Encoding is the final step in LLM data pre-processing that encodes information about the position of tokens in a sequence [c3276].

The terms positional encoding and positional embedding are used interchangeably to refer to vectors in a higher dimensional space [c3295].

Both absolute and relative positional embeddings enable large language models to understand the order and relationship between tokens, leading to more accurate and context-aware predictions [c3287].

#### Absolute vs. Relative Positional Embeddings

Absolute positional embeddings and relative positional embeddings are two different approaches to encoding position information [c3348].

**Absolute positional embeddings** assign a unique vector to each position index (0, 1, 2, …). In absolute positional embedding, the same token receives different final embeddings when it appears at different positions in the sequence because different positional embeddings are added to the same token embedding [c3283]. Absolute positional encoding is well-suited for tasks where fixed order of tokens is crucial, such as sequence generation [c3011].

GPT models (GPT-3, GPT-4) use absolute positional embeddings that are optimized during the training process [c3290]. This is a key design choice: rather than using a fixed mathematical formula, GPT learns its positional embeddings from data, just like it learns its token embeddings.

**Relative positional embeddings** take a different approach. Relative positional encoding emphasizes the relative distance between tokens rather than their absolute positions in the sequence [c3010]. Intuitively, relative encodings let the model reason about "how far apart" two tokens are, rather than "what slot each token occupies."

[FIGURE: Side-by-side comparison of absolute positional embeddings (each position gets a unique fixed vector) vs. relative positional embeddings (distances between positions are encoded)]

#### The Sinusoidal Formula (Original Transformer)

The original Transformer paper proposed a fixed, formula-based approach to absolute positional encoding. The positional encoding for even dimensions uses a sine function:

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
[c3292]

And for odd dimensions, a cosine function:

$$PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
[c3293]

Here, $pos$ is the position of the token in the sequence, $i$ is the dimension index, and $d_{\text{model}}$ is the embedding dimension. The consequence of this design is that each position gets a unique pattern of sine and cosine values across all dimensions, and the model can potentially generalize to sequence lengths not seen during training.

GPT-style models, however, do not use this fixed formula. They use learned positional embeddings instead [c3290].

#### Implementing Learned Positional Embeddings

The implementation of learned positional embeddings mirrors the token embedding layer exactly — it is simply another `torch.nn.Embedding` table, but indexed by position rather than token ID.

Positional embedding matrix values are initialized randomly, similar to token embedding initialization [c3021]. Both token embedding and positional embedding values must be optimized during the training process of large language models [c3022].

To generate positional embeddings for a sequence of length `max_length`, we first create a sequence of position indices:

# Source: [c3329]
```python
torch.arange(max_length)  # creates a sequence of integers from 0 to max_length-1, used to index into the positional embedding lookup table
```

Passing positions 0, 1, 2, and 3 to the positional embedding lookup table generates four 256-dimensional positional embedding vectors [c3331]. The same four positional embedding vectors are reused across all input sequences in a batch because positions are identical across sequences [c3327].

[FIGURE: A positional embedding lookup table of shape (context_size, embedding_dim), showing how position indices 0, 1, 2, 3 map to four distinct embedding vectors]

---

### Combining Token and Positional Embeddings

#### The Addition Operation

With both token embeddings (shape $8 \times 4 \times 256$) and positional embeddings (shape $4 \times 256$) in hand, the final step is to add them together. Positional embeddings are added to token embeddings using Python broadcasting, where the same positional embedding vectors are added to each row of the token embedding matrix [c3342].

When adding token embeddings and positional embeddings in Python, broadcasting converts the $4 \times 256$ positional embedding tensor into $8 \times 4 \times 256$ by duplicating values eight times [c3018]. The same positional embedding values are added to each row of the token embeddings across all 8 batches during the broadcasting operation [c3335].

Intuitively, broadcasting means we do not need to manually tile the positional embedding matrix across the batch dimension — PyTorch handles this automatically. The result is an **input embedding** tensor of shape $8 \times 4 \times 256$, where each vector encodes both the identity of the token and its position in the sequence.

[FIGURE: Diagram showing token embedding tensor (8×4×256) plus positional embedding tensor (4×256) via broadcasting, producing input embedding tensor (8×4×256)]

#### The Complete Pipeline

We now have all four stages of the data preprocessing pipeline in place [c3023]:

1. **Tokenization**: Raw text → token IDs (covered in earlier chapters)
2. **Token embeddings**: Token IDs → dense vectors via `torch.nn.Embedding(vocab_size, embedding_dim)` [c3003]
3. **Positional embeddings**: Position indices → dense vectors via a second `torch.nn.Embedding(context_size, embedding_dim)` [c3021]
4. **Input embeddings**: Token embeddings + positional embeddings → final input tensor [c3342]

The LLM data pre-processing pipeline consists of four stages: tokenization, token embeddings, positional embeddings, and input embeddings [c3023]. Token embeddings have shape $8 \times 4 \times 256$ and positional embeddings have shape $4 \times 256$ [c3017].

An autoregressive model is one where the output of one iteration becomes the input of the next iteration [c2718]. The input embeddings produced by this pipeline feed directly into the first layer of the transformer, and during inference, the model's own predictions loop back as new inputs — hence "autoregressive."

[FIGURE: End-to-end pipeline diagram: raw text → tokenizer → token IDs → DataLoader (sliding window, batching) → token embedding layer → + positional embedding layer → input embeddings → LLM]

---

### Putting It All Together: A Worked Example

Let us trace a concrete example through the entire pipeline to make sure all the pieces connect.

**Step 1 — Tokenize and encode.** We read a text file and encode it into a list of integer token IDs [c2724]. Suppose the encoded sequence has 1,000 tokens.

**Step 2 — Build input-target pairs.** With a context size of 4 and a stride of 1, the sliding window produces pairs where each input is 4 tokens and each target is those same 4 tokens shifted right by one [c2777, c2729]. The data loader uses a sliding window approach to fetch input-output target pairs [c2767].

**Step 3 — Batch.** A DataLoader with batch size 8 groups 8 such pairs into a single batch [c2807]. The input tensor $X$ has shape $8 \times 4$ and the target tensor $Y$ has shape $8 \times 4$ [c2981].

**Step 4 — Token embeddings.** We pass $X$ through `torch.nn.Embedding(50257, 256)` [c3003]. The lookup retrieves one 256-dimensional vector per token ID, producing an output of shape $8 \times 4 \times 256$ [c3017].

**Step 5 — Positional embeddings.** We create position indices `[0, 1, 2, 3]` with `torch.arange(4)` [c3329] and pass them through a second `torch.nn.Embedding(4, 256)`. This produces a tensor of shape $4 \times 256$ [c3331].

**Step 6 — Input embeddings.** We add the two tensors. Broadcasting expands the positional embedding from $4 \times 256$ to $8 \times 4 \times 256$ [c3018], and the element-wise sum gives us the final input embedding tensor of shape $8 \times 4 \times 256$ [c3342].

This tensor is what the transformer's attention mechanism will receive. Every vector in it encodes both *what* the token is (semantic content from the token embedding) and *where* it sits in the sequence (positional content from the positional embedding).

---

### Summary

This chapter covered the full data preprocessing pipeline that bridges raw text and the numerical inputs consumed by an LLM:

- **Input-target pairs** are created by treating each token as a prediction target for the preceding context. The sliding window with a configurable stride generates these pairs efficiently [c2722, c2777].
- **The DataLoader** wraps a dataset of input-target pairs into batches of PyTorch tensors, with optional parallel loading via multiple worker threads [c2762, c2821].
- **Token embeddings** are dense vector representations stored in a lookup table (`torch.nn.Embedding`). Each token ID maps to a row in the embedding weight matrix [c2993, c3317]. The values are learned during training [c3022].
- **Token embeddings alone are insufficient** because they carry no information about token order [c3006, c2594]. Positional embeddings address this gap.
- **Positional embeddings** come in two flavors — absolute and relative [c3348]. GPT-style models use learned absolute positional embeddings [c3290], initialized randomly and optimized during training [c3021, c3022].
- **Input embeddings** are the element-wise sum of token and positional embeddings, combined via broadcasting [c3342, c3018]. The result is a tensor of shape $(batch\_size, context\_size, embedding\_dim)$ that serves as the input to the transformer.

With this pipeline in place, we are ready to move on to the transformer architecture itself — the attention mechanism, feed-forward layers, and the full GPT model that will learn to predict the next token from these carefully constructed input embeddings.