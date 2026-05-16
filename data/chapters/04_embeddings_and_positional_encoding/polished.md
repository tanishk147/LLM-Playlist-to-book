## Token Embeddings, Positional Encoding, and Input Representations

To bridge the gap between raw text and mathematical computation, we need *embeddings*—dense vectors that live in a continuous space where geometry encodes meaning.

---

### The Three-Step LLM Input Pipeline

Before diving into the mechanics, it helps to see the big picture. The large language model workflow has three main steps: tokenizing input text into tokens, converting tokens into token IDs, and converting token IDs into token embeddings, which serve as input to training the model [c2514]. Token embeddings are also called vector embeddings or word embeddings, though "word embeddings" is not entirely accurate terminology because a token is a broader concept than a word [c2509, c2511].

[FIGURE: Three-stage pipeline: raw text → tokenizer → token IDs → embedding layer → token embedding vectors]

Without token embeddings, the semantic relationship between related words like "dog" and "puppy" or "cat" and "kitten" would be lost entirely [c3275]. The full data pre-processing pipeline that feeds an LLM consists of four stages: tokenization, token embeddings, positional embeddings, and input embeddings [c3023]. We will work through each of the last three stages in this chapter.

---

### What Is a Token Embedding?

Token embeddings map words into a vector space where each word is represented as a vector with a fixed number of dimensions [c2528]. Intuitively, each dimension is a learned axis in a semantic space. A simple conceptual example: vector embeddings can be constructed by representing each word as a vector where each dimension corresponds to a semantic feature—for example, `has_a_tail`, `is_eatable`, `has_4_legs`, `makes_sound`, `is_a_pet` [c2521].

The key advantage of this representation is that vector embeddings can capture semantic meaning between words, whereas random number assignment and one-hot encoding cannot [c2523]. Dense embeddings place semantically similar words close together in the vector space, a property that neither of those alternatives can provide.

#### Semantic Arithmetic

Word embeddings encode semantic properties such that masculine words (king, man) and feminine words (queen, woman) are distinguishable through vector operations [c2534]. Concretely, vector arithmetic on word embeddings preserves semantic relationships: $\text{king} + \text{woman} - \text{man} \approx \text{queen}$ [c2532, c2587]. Semantically unrelated word pairs such as "paper" and "water" have low similarity scores in Word2Vec embeddings [c2536], and the norm of the vector difference between two word embeddings serves as an indicator of how semantically close or distant the words are [c2538].

```python
# Source: [c2531]
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
word_vectors = model
print(word_vectors['computer'])
```

This prints a 300-dimensional vector for the word "computer." The analogy to convolutional neural networks is instructive: in CNNs, spatial features of an image are exploited before the image is used as training input, similar to how token embeddings work [c2585].

#### How Embeddings Are Learned

Token embeddings are created by training a neural network on textual data to learn vector representations where semantically similar words have similar vectors [c2524, c2992]. Training data provides information about semantic relationships between words, which is used to optimize the embedding weight parameters [c2549]. Importantly, token embedding values are learned and optimized during the LLM training process—they are not fixed at initialization [c3000]. Creating token embeddings for large vocabularies is computationally expensive, which is one reason why training large language models like GPT takes a huge amount of time [c2526].

---

### The Embedding Layer as a Lookup Table

An embedding layer is a lookup table that stores embeddings of a fixed dictionary and size, used to store word embeddings and retrieve them using indices [c2555]. More precisely, it maps vocabulary IDs to dense vectors of a specified embedding dimension [c2575].

The embedding layer weight matrix stores vector embeddings for each token ID in the vocabulary, with dimensions of `(vocabulary_size, embedding_dimension)` [c2993]. The first row of the matrix is the vector for token ID 0, the second row is the vector for token ID 1, and so on [c2563]. In this way, the weight matrix functions both as a matrix of learned weights and as a lookup table for finding vector representations by token ID [c2568].

[FIGURE: Embedding weight matrix with vocab_size rows and embedding_dim columns; arrows from token IDs to their corresponding rows]

A token embedding matrix requires two quantities: the vector dimension (the size each token is projected to) and the vocabulary size (the number of unique token IDs) [c3300].

```python
# Source: [c2998]
torch.nn.Embedding(num_embeddings, embedding_dim)
# Creates an embedding layer with num_embeddings rows and embedding_dim columns
# Stores embeddings of a fixed dictionary and size
```

For a small toy example with six tokens and three-dimensional embeddings:

```python
# Source: [c2556]
embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
```

You can inspect the underlying weight matrix directly:

```python
# Source: [c2559]
embedding_layer.weight
```

The embedding layer retrieves rows from this weight matrix using token IDs, and crucially, it retrieves only the embedding vectors corresponding to the input IDs—without computing embeddings for unused vocabulary entries [c3001, c2579]. You can pass multiple token IDs in a single call to retrieve all their vectors at once [c2571].

#### Mathematical Equivalence to a Linear Layer

There is a useful mathematical insight here: the output of `embedding(token_ids)` is equivalent to $X \cdot W^T$, where $X$ is the one-hot encoded input matrix and $W^T$ is the transposed weight matrix of a linear layer [c2581].

---

### Building the Token Embedding Layer for GPT

Token embeddings for large language models are created by converting token IDs into embedding vectors using an embedding weight matrix [c2541]. For our implementation, input tokens are encoded into a 256-dimensional vector space for demonstration purposes [c3296].

```python
# Source: [c3003]
vocab_size = 50257
embedding_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
# Creates embedding matrix with 50257 rows and 256 columns
```

For reference, the general pattern is:

```python
# Source: [c2577]
embedding = torch.nn.Embedding(vocab_size, embedding_dim)
# Example: torch.nn.Embedding(4, 5) creates a 4x5 embedding matrix
```

Note that the token embedding weight matrix for GPT-2 uses 768-dimensional vectors rather than 256, giving it 50,257 rows by 768 columns [c2590]. In our demonstration setup, each token in the input batch is converted to a single 256-dimensional embedding vector [c3320, c3339].

---

### The Data Loader: Feeding Batches to the Embedding Layer

Before embeddings can be computed, we need a mechanism to produce batches of token IDs. A data loader is a PyTorch tool that processes data by creating input-output target pairs using a sliding window approach [c2980]. Input-output pairs for LLM training are constructed based on context size, where each pair contains multiple prediction tasks corresponding to predicting the next word at each position [c2978]. In a data loader implementation, the input tensor `X` contains rows representing input contexts, and the target tensor `Y` contains the corresponding prediction targets—the input shifted by one word [c2981].

[FIGURE: Sliding window over a token sequence showing how X and Y are constructed by shifting by one position]

Two important hyperparameters govern the data loader:

- **Batch size** specifies how many input-output pairs are processed before updating model parameters [c2984].
- **Number of workers** specifies how many threads are used for parallel processing on the CPU [c2986].

---

### The Problem with Token Embeddings Alone

Here is a critical limitation: token embedding does not take into account the position of words in a sequence, so identical words at different positions receive the same embedding vector [c3006, c3277]. Token embeddings alone carry no information about where a word appears in a sentence [c2594].

It is therefore important to inject additional position information into the model alongside the semantic content captured by token embeddings [c3280]. Positional encoding is the final step in LLM data pre-processing that encodes information about the position of tokens in a sequence [c3276], and it is the technique that complements token embeddings by supplying this missing order information [c3008].

---

### Positional Encoding: Two Philosophies

Absolute positional embeddings and relative positional embeddings are two different approaches to encoding position information, and both enable large language models to understand the order and relationship between tokens, leading to more accurate and context-aware predictions [c3348, c3287].

#### Absolute Positional Encoding

In absolute positional embedding, the same token receives different final embeddings when it appears at different positions in the sequence, because a different positional embedding is added to the same token embedding at each position [c3283]. This approach is well-suited for tasks where the fixed order of tokens is crucial, such as sequence generation [c3011]. GPT models (GPT-3, GPT-4) use absolute positional embeddings that are optimized during the training process [c3290, c3012].

#### Relative Positional Encoding

Relative positional encoding, by contrast, emphasizes the relative distance between tokens rather than their absolute positions in the sequence [c3010]. This makes it suitable for tasks like language modeling or long sequences where the same phrase can appear in different parts of the sequence [c3013], and it allows models to generalize better to sequences of varying lengths [c3285].

#### The Original Sinusoidal Formulation

The terms positional encoding and positional embedding are used interchangeably to refer to vectors in a higher-dimensional space [c3295]. The original sinusoidal formulas are:

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \quad [c3292]$$

$$PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \quad [c3293]$$

These formulas produce a unique, deterministic vector for each position without any learned parameters.

---

### GPT's Learned Positional Embeddings

Unlike the sinusoidal approach, GPT learns its positional embeddings from data. The size of the positional embedding matrix is determined by the context length (number of positions) and the embedding dimension [c3015]. For a context length of four, only four positional embedding vectors are needed—one for each position [c3341, c3328]. The resulting matrix has shape `4 by 256`, where 4 is the sequence length and 256 is the embedding dimension [c3333].

Like token embeddings, positional embedding values are initialized randomly and must be optimized during training [c3021, c3022].

```python
# Source: [c3329]
torch.arange(max_length)  # creates a sequence of integers from 0 to max_length-1
```

Passing positions 0, 1, 2, and 3 to the positional embedding lookup table generates four 256-dimensional positional embedding vectors [c3331]. Because positions are identical across all sequences in a batch, the same four positional embedding vectors are reused for every input sequence [c3327].

[FIGURE: Positional embedding lookup table with context_length rows and embedding_dim columns; positions 0,1,2,3 mapped to their respective vectors]

---

### Combining Token and Positional Embeddings

The final input representation is formed by adding positional embeddings to token embeddings via Python broadcasting, where the same positional embedding vectors are added to each row of the token embedding matrix [c3342].

Token embeddings have shape `8 by 4 by 256` and positional embeddings have shape `4 by 256` [c3017]. Broadcasting automatically expands the `4 by 256` positional embedding tensor to `8 by 4 by 256` by duplicating its values eight times—once for each sequence in the batch [c3018, c3335].

[FIGURE: Broadcasting diagram: (4,256) positional embedding tensor expanded to (8,4,256) and added element-wise to (8,4,256) token embedding tensor]

Intuitively, every sequence in the batch receives the same positional "correction" applied to its token vectors: the first token in every sequence gets the first positional vector added to it, the second token gets the second positional vector, and so on. As a result, the same token receives different final embeddings when it appears at different positions in the sequence [c3283].

---

### The Complete Pre-Processing Pipeline

The LLM data pre-processing pipeline consists of four stages: tokenization, token embeddings, positional embeddings, and input embeddings [c3023]. The `torch.nn.Embedding` layer ties it all together:

```python
# Source: [c3303]
torch.nn.Embedding(num_embeddings, embedding_dim)
# Creates an embedding layer that stores embeddings of a fixed dictionary and size,
# functioning as a lookup table that retrieves embeddings using token indices.
```

The gradient signal from the language modeling objective flows back through the transformer, through the addition operation, and into both embedding matrices simultaneously, allowing both to be optimized end-to-end.

---

### Summary

This chapter covered the three-stage transformation from raw token IDs to the rich vector representations that feed a transformer. The key ideas are:

1. **Token embeddings** convert integer token IDs into dense vectors by looking up rows in a learned weight matrix [c2993, c3317]. The matrix has shape `(vocab_size, embedding_dim)` and is optimized during training [c3000].
2. **Semantic geometry** emerges from training: similar words cluster together, and vector arithmetic encodes relationships like $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ [c2532].
3. **Token embeddings alone are position-blind**: the same token ID always produces the same vector regardless of where it appears in the sequence [c3006, c3277].
4. **Positional embeddings** inject order information by adding a position-specific vector to each token embedding [c3008]. GPT uses learned absolute positional embeddings [c3012, c3290].
5. **The input embedding** is the element-wise sum of the token embedding and the positional embedding, combined via broadcasting across the batch dimension [c3342, c3018].
6. **Both embedding matrices** are initialized randomly and optimized jointly during LLM training [c3021, c3022].

In the next chapter, we will see how these input embeddings flow into the self-attention mechanism, where the model learns to relate tokens to one another across the full context window.