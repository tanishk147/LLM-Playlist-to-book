## Tokenization: From Words to Byte Pair Encoding

Before a large language model can process text, that text must be converted into a form the model can actually work with: numbers. 

[FIGURE: High-level pipeline showing raw text → tokenizer → token IDs → vector embeddings, with arrows between each stage]

---

### The Role of Tokenization in the LLM Pipeline

Data preprocessing is the procedure of processing text data—sentences, paragraphs, documents—before it is fed as input to a large language model [c2895]. Within that preprocessing pipeline, tokenization is the very first step: it converts raw text into tokens [c2898]. Those tokens are then mapped to integer IDs, and those IDs are later converted into vector representations that the model actually trains on [c3257].

Concretely, the tokenization pipeline for LLMs involves three stages: tokenizing input text into words, converting those words into token IDs, and later converting token IDs into vector representations [c3257]. This chapter focuses on the first two stages. The third stage—embedding—is covered in the next chapter.

[FIGURE: Three-stage pipeline: (1) "Hello world" → ["Hello", "world"] tokens, (2) tokens → [4312, 995] IDs, (3) IDs → embedding vectors]

---

### Three Families of Tokenization

There are three main types of tokenization algorithms: word-based tokenizers, sub-word-based tokenizers, and character-based tokenizers [c2419]. Each makes a different trade-off between vocabulary size, handling of rare words, and semantic expressiveness. 

#### Word-Based Tokenization

Word-based tokenization is the most intuitive approach: it converts an entire dataset into chunks of words [c2909], breaking down sentences into individual words as tokens [c2938]. In previous tokenization schemes, every word was treated as a unique token, along with special characters like comma, full stop, and exclamation mark [c2417].

To see this concretely, consider splitting on whitespace:

# Source: [c2914]
```python

text = "Hello world. This is a test."
tokens = re.split(r'\s+', text)
```

This gives you one token per whitespace-separated chunk. 

 Word-based tokenization treats each unique word as a single token [c2466]. A large corpus will contain hundreds of thousands of unique words, producing an enormous vocabulary. Worse, any word not seen during training becomes an **out-of-vocabulary (OOV) word**—a word not present in the vocabulary used by the tokenizer [c2423]. Word-based tokenization requires handling these OOV words with special context tokens [c2939].

#### Character-Based Tokenization

At the opposite extreme, character-based tokenization uses individual characters as tokens instead of words [c2942]. For example, the word "dinosaur" is split into eight separate character tokens (d, i, n, o, s, a, u, r), whereas in word-based tokenization it would be one token [c2435].

The upside is a tiny vocabulary: character-based tokenization requires a vocabulary of approximately 256 characters, which is much smaller than word-based tokenization [c2943]. It also completely solves the out-of-vocabulary problem—every possible string can be represented as a sequence of characters [c2944].

 The model must learn to compose meaning from scratch at every level.

#### Subword-Based Tokenization

Subword-based tokenization sits between the two extremes. It uses subwords as tokens, which are neither full words nor individual characters [c2947]. Sub-word based tokenization combines features of word tokenization and character tokenization by selectively splitting words based on frequency [c2439].

 This retains semantic similarity by sharing root word tokens across related words like "tokens" and "tokenizing" [c2952]. Byte pair encoding (BPE) is the most widely used algorithm for implementing this approach [c2953], and it is the algorithm used by GPT models [c2948].

[FIGURE: Comparison table of three tokenization types showing vocabulary size, OOV handling, and sequence length trade-offs]

---

### Building a Word-Based Tokenizer from Scratch

Even though word-based tokenization is not what GPT uses, building one from scratch is the best way to understand the machinery that all tokenizers share: a vocabulary, an encode method, and a decode method [c2908]. We will use a real text file as our corpus.

#### Loading the Raw Text

# Source: [c2912]
```python

 raw_text = f.read()
```

The variable `raw_text` stores the entire text content read from The Verdict file [c3181].

#### Splitting into Tokens

A naive whitespace split misses punctuation. 

# Source: [c2917]
```python

tokens = [token for token in tokens if token.strip()]
```

This splits on commas, periods, semicolons, colons, question marks, exclamation marks, quotation marks, hyphens, parentheses, and forward slashes, as well as any whitespace [c3196]. The list comprehension filters out empty strings that result from consecutive delimiters [c3193]. An alternative regex pattern that achieves similar splitting is:

# Source: [c3190]
```python

```

followed by:

# Source: [c3192]
```python

```

Using item.strip() filters out whitespace tokens from the tokenization result [c3193]. Applied to the book example, the raw text yields 4,690 tokens after applying the regex-based tokenization scheme [c3199].

#### Building the Vocabulary

A vocabulary is a dictionary where every unique token is mapped to a unique integer called a token ID [c2919]. A token ID is a unique integer assigned to each unique token in the vocabulary [c3201]. To build one, collect all unique tokens, sort them for reproducibility, and enumerate them:

# Source: [c2921]
```python

vocab_dict = {token: idx for idx, token in enumerate(vocab)}
```

You can also capture the vocabulary size at the same time:

# Source: [c3205]
```python

vocab_size = len(vocab)
```

#### Encoding and Decoding

Encoding is the process of converting tokens into token IDs [c3209]. Decoding is the reverse: a decoder maps token IDs back to tokens, needed to convert LLM numerical output back into text [c3210].

A tokenizer class is a Python class that contains an encode method to convert text into token IDs and a decode method to convert token IDs back into text [c2922]. The `__init__` method of the tokenizer class takes a vocabulary parameter, which is a mapping from tokens to token IDs [c3218]. Internally, the class stores two dictionaries: `str2int`, which maps tokens (strings) to token IDs (integers) [c3219], and `int2str`, which is the reverse mapping from token IDs (integers) back to tokens (strings) [c3220].

# Source: [c2923]
```python

 def __init__(self, vocab):
 self.string_to_int = vocab
 self.int_to_string = {idx: token for token, idx in vocab.items()}
 
 def encode(self, text):
 tokens = re.split(r'[,\.;:\?!"\-\(\)/]|\s+', text)
 tokens = [token for token in tokens if token.strip()]
 return [self.string_to_int[token] for token in tokens]
 
 def decode(self, ids):
 text = ' '.join([self.int_to_string[idx] for idx in ids])
 return re.sub(r'\s+([,\.;:\?!"\-\(\)/])', r'\1', text)
```

The tokenizer class uses `re.split` and `item.strip` methods to convert text into tokens [c3227].

Let's trace through what each method does:

- **encode**: The encode method preprocesses text by splitting it on punctuation marks (comma, full stop, colon, semicolon) and removing whitespace to create individual tokens [c3222]. It then converts individual tokens to token IDs using the `str2int` vocabulary dictionary [c3223]. The encode method takes text as input and outputs token IDs [c3216]. In the context of a transformer architecture, the encode method corresponds to the encoder block, converting text to token IDs for training data [c3214].

- **decode**: The decode method converts token IDs back to individual tokens using the `int2str` reverse dictionary [c3224], then joins individual tokens together into a single string [c3225]. The decode method takes token IDs as input and outputs text [c3217]. A post-processing regex removes the spurious spaces that would otherwise appear before punctuation marks (e.g., `"Hello ,"` becomes `"Hello,"`).

The encode method converts sample text into tokens and assigns token IDs based on the vocabulary [c3212]. The decode method takes token IDs as input, converts them into individual tokens, and returns the original sample text [c3213].

[FIGURE: Diagram showing encode path: text → regex split → token list → vocab lookup → ID list; and decode path: ID list → reverse lookup → token list → join → text]

---

### Handling Unknown Words with Special Tokens

The `SimpleTokenizer` above will raise a `KeyError` if it encounters a word not in its vocabulary. 

Special context tokens are reserved tokens used to handle unknown words and mark boundaries between text sources in tokenization [c3234]. An unknown token is a special token added to the vocabulary to represent words that are not present in the vocabulary [c2927]. The unknown token (UNK) is assigned to any word in the input text that is not present in the vocabulary [c3235].

The end-of-text token (EOS) is a special token inserted between unrelated text sources to signal boundaries and prevent the model from mixing separate documents [c3236]. 

Special tokens are added to the vocabulary to handle words not present in the vocabulary and to separate multiple text sources [c3263].

#### Adding Special Tokens to the Vocabulary

# Source: [c3243]
```python

pre_processed.extend(['<|endoftext|>', '<|unk|>'])
vocab = {token: idx for idx, token in enumerate(sorted(pre_processed))}
```

#### Updating the Encoder to Use the Unknown Token

# Source: [c3245]
```python

if item not in self.str2int:
 tokens.append(self.str2int['<|unk|>'])
else:
 tokens.append(self.str2int[item])
```

#### Other Special Tokens

Some researchers use additional special tokens beyond `<|unk|>` and `<|endoftext|>` [c2937]:

- **BOS (Beginning of Sequence)**: marks the start of a text and signifies to the LLM where a piece of context begins [c3248].
- **EOS (End of Sequence)**: positioned at the end of a text and is useful for separating unrelated text [c3249].
- **Padding token**: used to extend shorter texts in a batch to match the length of the longest text, enabling parallel processing of variable-length sequences [c3250].

[FIGURE: Diagram showing two documents separated by <|endoftext|> token, and a batch of sequences padded to equal length with <|pad|> tokens]

---

### The Limits of Word-Based Tokenization

 Consider the words "old", "older", "finest", and "lowest". Word-based tokenization treats each unique word as a single token, resulting in four tokens for this dataset [c2466]. 

Subword tokenization solves this by retaining root words and breaking down rare words into subword units [c2959]. 

---

### Byte Pair Encoding: The Algorithm

Byte pair encoding (BPE) is a subword-based tokenization method [c2903] and the algorithm used by GPT models [c2948]. It is also an alternative tokenization scheme used in large language models, different from simple regex-based splitting [c3197]. BPE reduces vocabulary length compared to word-level tokenization [c2499].

#### Step 1: Initialize with Characters

The first step in BPE tokenization is to split all words into character-level tokens [c2961]. In BPE preprocessing, a special end-of-word token (`</w>`) is appended to each word to mark word boundaries, enabling the algorithm to distinguish between subwords at word endings and the same subwords within words [c2468]. In BPE tokenization, a special end-of-word token is added after each word to indicate word boundaries [c2960].

For example, the word "low" becomes `l o w </w>`, and "lower" becomes `l o w e r </w>`.

#### Step 2: Count and Merge Pairs

In byte pair encoding, the algorithm identifies the most frequently occurring byte pair and replaces it with a new variable that does not occur in the data [c2956]. 

Intuitively, if "l o" appears very frequently, merging it into "lo" reduces the total number of tokens in the corpus. 

#### Step 3: Apply to New Text

Once the merge table is learned, the BPE tokenizer breaks down words not in its predefined vocabulary into smaller sub-word units or individual characters [c2504]. The BPE tokenizer scans text from left to right and breaks it down into characters or sub-words based on what is present in the vocabulary, then assigns each a unique token ID [c2506].

Byte pair encoding for large language models is a tweaked version of the standard BPE algorithm used to convert entire sentences into subwords [c2460].

[FIGURE: Step-by-step BPE merge example: starting corpus with character tokens, showing three rounds of merging the most frequent pair, with merge table on the right]

#### A Worked Example

After adding end-of-word markers and splitting into characters, the initial token inventory is: `o`, `l`, `d`, `</w>`, `e`, `r`, `f`, `i`, `n`, `s`, `t`, `w`. The algorithm counts all adjacent pairs across all occurrences, finds the most frequent, and merges it. BPE applied to vocabulary achieves subword tokenization by retaining root words and breaking down rare words into subword units [c2959].

---

### Using tiktoken for GPT-2 Compatible BPE

Implementing BPE from scratch is relatively complicated, so in practice we use the `tiktoken` Python library instead [c2487]. The tiktoken library provides tokenizers using the same encodings as GPT-2 [c2967].

#### Installation and Basic Usage

# Source: [c2490]
```python

tokenizer = tiktoken.get_encoding('gpt2')
```

The tiktoken BPE tokenizer has an encode method that converts words into token IDs [c2491]. 

For example:

```python

token_ids = tokenizer.encode(text)
print(token_ids)
# e.g. [15496, 11, 466, 345, 588, 8887, 30]
```

#### How the BPE Tokenizer Handles Unknown Words

 The BPE tokenizer breaks down words not in its predefined vocabulary into smaller sub-word units or individual characters [c2504]. 

The BPE tokenizer scans text from left to right and breaks it down into characters or sub-words based on what is present in the vocabulary, then assigns each a unique token ID [c2506]. This means that even a completely novel word like a proper noun or a technical term will be decomposed into recognizable subword pieces rather than mapped to a generic unknown token.

[FIGURE: Diagram showing tiktoken encoding of an unfamiliar word, decomposing it into subword tokens with their IDs]

---

### Putting It All Together: The Full Tokenization Pipeline

Let's consolidate everything into a clear picture of the pipeline.

Tokenization is the process of preparing input text for training LLMs, divided into two steps: splitting text into individual words (tokens) and converting tokens into token IDs [c3255]. The full pipeline looks like this:

The encode method of a tokenizer class converts sample text into tokens and then converts these tokens into token IDs [c3260]. The decode method of a tokenizer class converts token IDs back into tokenized text and then recovers the original sample text [c3261].

[FIGURE: End-to-end flowchart from raw text file through tokenization, vocabulary lookup, and encoding to a final sequence of integer token IDs ready for embedding]

#### Comparing the Two Tokenizers

---

### Summary

This chapter covered the full tokenization pipeline from first principles to production tooling.

- There are three families of tokenization: word-based, character-based, and subword-based [c2419]. Each makes different trade-offs on vocabulary size, OOV handling, and sequence length.
- Word-based tokenization is easy to implement but suffers from large vocabularies and OOV words [c2939]. We built a `SimpleTokenizer` class with `encode` and `decode` methods [c2922] and extended it with special tokens (`<|unk|>` and `<|endoftext|>`) [c3263].
- Character-based tokenization solves OOV with a tiny vocabulary [c2943, c2944] but produces very long sequences with little per-token semantics.
- Subword-based tokenization, and BPE in particular, is the best of both worlds: it retains root words, handles rare words gracefully, and keeps vocabulary sizes manageable [c2499, c2952].
- BPE works by iteratively merging the most frequent adjacent byte pair [c2956], starting from a character-level initialization [c2961] and using end-of-word markers to preserve word boundaries [c2468].
- The `tiktoken` library provides a production-ready BPE tokenizer compatible with GPT-2 [c2967], accessible with two lines of code [c2490].
