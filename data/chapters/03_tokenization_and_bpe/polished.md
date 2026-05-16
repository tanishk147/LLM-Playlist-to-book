# Tokenization: From Words to Byte Pair Encoding

Before a large language model can process text, that text must be converted into a form the model can actually work with: numbers. This chapter traces that conversion from first principles—splitting raw text into tokens—all the way to the production-ready byte pair encoding used by GPT-2.

[FIGURE: High-level pipeline showing raw text → tokenizer → token IDs → vector embeddings, with arrows between each stage]

---

## The Role of Tokenization in the LLM Pipeline

Data preprocessing is the procedure of processing text data—sentences, paragraphs, documents—before it is fed as input to a large language model [c2895]. Within that preprocessing pipeline, tokenization is the very first step: it converts raw text into tokens [c2898]. Those tokens are then mapped to integer IDs, and those IDs are later converted into vector representations that the model actually trains on [c3257].

Concretely, the pipeline involves three stages: tokenizing input text into words, converting those words into token IDs, and converting token IDs into vector representations [c3257]. This chapter focuses on the first two stages; the third—embedding—is covered in the next chapter.

[FIGURE: Three-stage pipeline: (1) "Hello world" → ["Hello", "world"] tokens, (2) tokens → [4312, 995] IDs, (3) IDs → embedding vectors]

---

## Three Families of Tokenization

There are three main types of tokenization algorithms: word-based tokenizers, sub-word-based tokenizers, and character-based tokenizers [c2419]. Each makes a different trade-off between vocabulary size, handling of rare words, and semantic expressiveness.

### Word-Based Tokenization

Word-based tokenization is the most intuitive approach: it converts an entire dataset into chunks of words [c2909], treating each word as an individual token [c2938]. In earlier tokenization schemes, every word was treated as a unique token, along with special characters such as commas, full stops, and exclamation marks [c2417].

To see this concretely, consider splitting on whitespace:

```python
# Source: [c2914]
import re
text = "Hello world. This is a test."
tokens = re.split(r'\s+', text)
```

This yields one token per whitespace-separated chunk. Because word-based tokenization treats each unique word as a single token [c2466], a large corpus will produce an enormous vocabulary. Worse, any word not seen during training becomes an **out-of-vocabulary (OOV) word**—a word absent from the tokenizer's vocabulary [c2423]—and must be handled with special context tokens [c2939].

### Character-Based Tokenization

At the opposite extreme, character-based tokenization uses individual characters as tokens instead of words [c2942]. For example, "dinosaur" is split into eight separate character tokens (d, i, n, o, s, a, u, r), whereas word-based tokenization would treat it as a single token [c2435].

The upside is a tiny vocabulary: character-based tokenization requires only approximately 256 characters, far fewer than word-based tokenization [c2943]. It also eliminates the OOV problem entirely, since every possible string can be represented as a sequence of characters [c2944]. The trade-off is that the model must learn to compose meaning from individual characters at every level, producing very long sequences with little per-token semantics.

### Subword-Based Tokenization

Subword-based tokenization sits between the two extremes, using tokens that are neither full words nor individual characters [c2947]. It combines features of both approaches by selectively splitting words based on frequency [c2439], which retains semantic similarity by sharing root-word tokens across related words like "tokens" and "tokenizing" [c2952]. Byte pair encoding (BPE) is the most widely used algorithm for implementing this approach [c2953] and is the algorithm used by GPT models [c2948].

[FIGURE: Comparison table of three tokenization types showing vocabulary size, OOV handling, and sequence length trade-offs]

---

## Building a Word-Based Tokenizer from Scratch

Although word-based tokenization is not what GPT uses, building one from scratch is the best way to understand the machinery that all tokenizers share: a vocabulary, an encode method, and a decode method [c2908]. We will use a real text file as our corpus.

### Loading the Raw Text

```python
# Source: [c2912]
with open('the-verdict.txt', 'r') as f:
    raw_text = f.read()
```

The variable `raw_text` stores the entire text content read from The Verdict file [c3181].

### Splitting into Tokens

A naive whitespace split misses punctuation. A more robust approach splits on both punctuation and whitespace:

```python
# Source: [c2917]
tokens = re.split(r'[,\.;:\?!"\-\(\)/]|\s+', text)
tokens = [token for token in tokens if token.strip()]
```

This splits on commas, periods, semicolons, colons, question marks, exclamation marks, quotation marks, hyphens, parentheses, and forward slashes, as well as any whitespace [c3196]. The list comprehension filters out empty strings that result from consecutive delimiters [c3193]. An alternative regex pattern that achieves similar splitting is:

```python
# Source: [c3190]
result = re.split(r'[,.;:?_!()"\|\-\s]', text)
```

followed by:

```python
# Source: [c3192]
[item for item in result if item.strip()]
```

Using `item.strip()` filters out whitespace tokens from the result [c3193]. Applied to the book example, the raw text yields 4,690 tokens after the regex-based tokenization scheme is applied [c3199].

### Building the Vocabulary

A vocabulary is a dictionary where every unique token is mapped to a unique integer called a token ID [c2919]. To build one, collect all unique tokens, sort them for reproducibility, and enumerate them:

```python
# Source: [c2921]
vocab = sorted(set(preprocessed))
vocab_dict = {token: idx for idx, token in enumerate(vocab)}
```

You can also capture the vocabulary size at the same time:

```python
# Source: [c3205]
vocab = sorted(set(preprocessed))
vocab_size = len(vocab)
```

### Encoding and Decoding

Encoding is the process of converting tokens into token IDs [c3209]. Decoding is the reverse: mapping token IDs back to tokens, which is needed to convert the model's numerical output back into readable text [c3210].

A tokenizer class is a Python class containing an encode method to convert text into token IDs and a decode method to convert token IDs back into text [c2922]. Its `__init__` method takes a vocabulary parameter—a mapping from tokens to token IDs [c3218]—and stores two dictionaries internally: `str2int`, which maps tokens (strings) to token IDs (integers) [c3219], and `int2str`, the reverse mapping from token IDs back to tokens [c3220].

```python
# Source: [c2923]
class SimpleTokenizer:
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

The tokenizer class uses `re.split` and `item.strip` to convert text into tokens [c3227]. Here is what each method does:

- **encode**: Splits input text on punctuation and whitespace to produce individual tokens [c3222], then converts each token to its ID using the `str2int` dictionary [c3223]. It takes text as input and outputs token IDs [c3216]. In the context of a transformer architecture, the encode method corresponds to the encoder block, converting text to token IDs for training data [c3214].
- **decode**: Converts token IDs back to individual tokens using the `int2str` dictionary [c3224], then joins them into a single string [c3225]. It takes token IDs as input and outputs text [c3217]. A post-processing regex removes spurious spaces before punctuation (e.g., `"Hello ,"` becomes `"Hello,"`).

[FIGURE: Diagram showing encode path: text → regex split → token list → vocab lookup → ID list; and decode path: ID list → reverse lookup → token list → join → text]

---

## Handling Unknown Words with Special Tokens

The `SimpleTokenizer` above will raise a `KeyError` if it encounters a word not in its vocabulary. The standard solution is to introduce **special tokens**—reserved tokens that handle unknown words and mark boundaries between text sources [c3234].

The **unknown token** (`<|unk|>`) is added to the vocabulary to represent any word absent from it [c2927, c3235]. The **end-of-text token** (`<|endoftext|>`) is inserted between unrelated text sources to signal document boundaries and prevent the model from mixing separate documents [c3236]. Together, these special tokens handle words not present in the vocabulary and separate multiple text sources [c3263].

### Adding Special Tokens to the Vocabulary

```python
# Source: [c3243]
# Adding special tokens to vocabulary
pre_processed.extend(['<|endoftext|>', '<|unk|>'])
vocab = {token: idx for idx, token in enumerate(sorted(pre_processed))}
```

### Updating the Encoder to Use the Unknown Token

```python
# Source: [c3245]
# In SimpleTokenizer v2 encode method: if a word is not in vocabulary, replace it with unknown token
if item not in self.str2int:
    tokens.append(self.str2int['<|unk|>'])
else:
    tokens.append(self.str2int[item])
```

### Other Special Tokens

Some researchers use additional special tokens beyond `<|unk|>` and `<|endoftext|>` [c2937]:

- **BOS (Beginning of Sequence)**: marks the start of a text and signals to the LLM where a piece of context begins [c3248].
- **EOS (End of Sequence)**: positioned at the end of a text, useful for separating unrelated passages [c3249].
- **Padding token**: extends shorter texts in a batch to match the length of the longest text, enabling parallel processing of variable-length sequences [c3250].

[FIGURE: Diagram showing two documents separated by <|endoftext|> token, and a batch of sequences padded to equal length with <|pad|> tokens]

---

## The Limits of Word-Based Tokenization

Consider the words "old", "older", "finest", and "lowest". Word-based tokenization treats each as a distinct token, yielding four tokens for this small dataset [c2466]—even though "old" and "older" clearly share a root. Subword tokenization solves this by retaining root words and breaking down rare words into subword units [c2959].

---

## Byte Pair Encoding: The Algorithm

Byte pair encoding (BPE) is a subword-based tokenization method [c2903] used by GPT models [c2948] and distinct from simple regex-based splitting [c3197]. Compared to word-level tokenization, BPE reduces vocabulary size [c2499] while still handling rare and novel words gracefully.

### Step 1: Initialize with Characters

The first step is to split all words into character-level tokens [c2961]. A special end-of-word token (`</w>`) is appended to each word to mark word boundaries, enabling the algorithm to distinguish between subwords at word endings and the same subwords appearing within words [c2468, c2960]. For example, "low" becomes `l o w </w>` and "lower" becomes `l o w e r </w>`.

### Step 2: Count and Merge Pairs

The algorithm then identifies the most frequently occurring adjacent byte pair and replaces it with a new token not already present in the data [c2956]. If "l o" appears very frequently, merging it into "lo" reduces the total number of tokens in the corpus. This process repeats iteratively, building up a merge table of learned subword units.

### Step 3: Apply to New Text

Once the merge table is learned, the BPE tokenizer breaks down words not in its predefined vocabulary into smaller subword units or individual characters [c2504]. It scans text from left to right, decomposing each word based on what is present in the vocabulary and assigning each piece a unique token ID [c2506]. The version of BPE used for large language models is a tweaked variant of the standard algorithm, adapted to convert entire sentences into subwords [c2460].

[FIGURE: Step-by-step BPE merge example: starting corpus with character tokens, showing three rounds of merging the most frequent pair, with merge table on the right]

### A Worked Example

After adding end-of-word markers and splitting into characters, the initial token inventory for "old", "older", "finest", and "lowest" is: `o`, `l`, `d`, `</w>`, `e`, `r`, `f`, `i`, `n`, `s`, `t`, `w`. The algorithm counts all adjacent pairs across all occurrences, finds the most frequent, merges it into a new token, and repeats—ultimately retaining root words and decomposing rare words into subword units [c2959].

---

## Using tiktoken for GPT-2 Compatible BPE

Implementing BPE from scratch is relatively complicated, so in practice we use the `tiktoken` Python library instead [c2487]. The tiktoken library provides tokenizers using the same encodings as GPT-2 [c2967].

### Installation and Basic Usage

```python
# Source: [c2490]
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
```

The tiktoken BPE tokenizer has an encode method that converts text into token IDs [c2491]:

```python
token_ids = tokenizer.encode(text)
print(token_ids)
# e.g. [15496, 11, 466, 345, 588, 8887, 30]
```

### How the BPE Tokenizer Handles Unknown Words

Rather than mapping unfamiliar words to a generic unknown token, the BPE tokenizer breaks them down into smaller subword units or individual characters [c2504]. It scans text from left to right, decomposing each word based on what is present in the vocabulary and assigning each piece a unique token ID [c2506]. This means that even a completely novel word—a proper noun or a technical term—will be represented as a sequence of recognizable subword pieces.

[FIGURE: Diagram showing tiktoken encoding of an unfamiliar word, decomposing it into subword tokens with their IDs]

---

## Putting It All Together: The Full Tokenization Pipeline

Tokenization prepares input text for LLM training in two steps: splitting text into tokens and converting those tokens into token IDs [c3255]. The encode method of a tokenizer class handles the forward direction—converting sample text into tokens and then into token IDs [c3260]—while the decode method handles the reverse, converting token IDs back into text [c3261].

[FIGURE: End-to-end flowchart from raw text file through tokenization, vocabulary lookup, and encoding to a final sequence of integer token IDs ready for embedding]

---

## Summary

This chapter covered the full tokenization pipeline, from first principles to production tooling.

- **Three families of tokenization** exist—word-based, character-based, and subword-based [c2419]—each making different trade-offs on vocabulary size, OOV handling, and sequence length.
- **Word-based tokenization** is easy to implement but suffers from large vocabularies and OOV words [c2939]. We built a `SimpleTokenizer` class with `encode` and `decode` methods [c2922] and extended it with special tokens (`<|unk|>` and `<|endoftext|>`) [c3263].
- **Character-based tokenization** eliminates OOV with a tiny vocabulary [c2943, c2944] but produces long sequences with little per-token semantics.
- **Subword-based tokenization**, and BPE in particular, offers the best of both worlds: it retains root words, handles rare words gracefully, and keeps vocabulary sizes manageable [c2499, c2952].
- **BPE** works by iteratively merging the most frequent adjacent byte pair [c2956], starting from a character-level initialization [c2961] and using end-of-word markers to preserve word boundaries [c2468].
- The **`tiktoken` library** provides a production-ready BPE tokenizer compatible with GPT-2 [c2967], accessible with just two lines of code [c2490].