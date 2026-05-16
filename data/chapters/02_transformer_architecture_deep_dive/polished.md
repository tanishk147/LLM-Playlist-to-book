## The Transformer Architecture: Encoders, Decoders, BERT, and GPT

Modern large language models did not appear out of nowhere. They are the product of a decades-long search for better ways to process sequential data—text, speech, music—and the story of how we got here is worth telling carefully. This chapter traces that history, from the early days of recurrent networks through the attention mechanism and into the transformer architecture that powers models like BERT and GPT today. By the end, you will understand not just *what* these components are, but *why* they were designed the way they were and what problems each design was trying to solve.

---

### The Transformer at a Glance

Before diving into history, it helps to have a rough map of the destination.

The transformer architecture consists of two main blocks [c1273]:

1. An **encoder block** that converts input text into embedding vectors.
2. A **decoder block** that generates output text from those embedding vectors and partial output already produced.

[FIGURE: High-level transformer diagram showing input text flowing into an encoder block that produces embedding vectors, which feed into a decoder block alongside partial output text, producing final output text word by word]

The encoder takes tokenized input text and converts it into vector embeddings [c1267], while the decoder generates output text one word at a time, receiving both those embeddings and the partial output produced so far [c1270]. Because each newly generated word is fed back into the decoder, previously generated words are always available to the model when predicting the next one [c1271]. Think of the encoder as a reading-comprehension module and the decoder as a writing module: the encoder reads and understands; the decoder writes, one token at a time, consulting the encoder's understanding as it goes.

The transformer uses feed-forward layers with weights and parameters that are optimized during training to predict output words correctly [c1275]. The full simplified pipeline can be broken down into 8 steps covering tokenization, encoding, decoding with partial output, and word-by-word generation [c1320].

One terminological note before proceeding: the terms "transformers" and "large language models" should not be used interchangeably—they are different concepts [c1316]. Not all large language models are based on transformer architecture [c1308]. The transformer is one architecture; LLMs are a broader category, many of which happen to use transformers.

---

### Why Attention Is the Engine

Transformers are conceptualized as the secret sauce behind large language models, with the attention mechanism serving as the engine that drives the transformer [c1479]. That engine is a key factor in the strong performance of models like ChatGPT [c1480].

Attention mechanisms have become an integral part of sequence modeling, allowing dependencies to be modeled without regard to the distance between positions in the input or output [c1277]. That phrase—*without regard to distance*—is the crucial one. To understand why it matters, we first need to understand the problem attention was invented to solve.

---

### The Problem: Long-Range Dependencies

Consider a sentence like: *"The trophy didn't fit in the suitcase because it was too big."* What does "it" refer to—the trophy or the suitcase? A human reader resolves this instantly by connecting "it" to "trophy" across several intervening words. Now imagine that ambiguity stretched across multiple sentences or paragraphs.

Long-range dependencies refer to the need to understand context from distant parts of a sequence—such as earlier sentences—to accurately predict the next word in a later sentence [c1280]. More concretely, long-term dependencies arise in complex sentence structures where multiple clauses or phrases are connected, making it difficult for language models to identify relationships between distant words [c1481].

The self-attention mechanism addresses this directly, allowing the model to look far back in a sequence to identify the next word [c1281]. In ChatGPT, for instance, GPT gives attention to every sentence in the input and then predicts the next word—rather than considering only the immediately preceding sentence [c1286]. Beyond resolving references, self-attention also allows the model to weigh the importance of different words and tokens relative to each other [c1279], and attention blocks in multi-head attention ensure that long-range dependencies are captured throughout the network [c1283]. The paper that introduced the transformer was named "Attention Is All You Need" precisely because of the self-attention mechanism and the intuition it embodied [c1284].

---

### Predecessors: RNNs and LSTMs

To appreciate what transformers solved, we need to understand what came before them.

#### Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining a hidden state that captures information about previous inputs [c1535]—the key innovation that enables them to process sequential data at all [c1538]. By incorporating a feedback loop, RNNs effectively give the network a form of memory [c1313], and this made them a significant advancement for generative models, enabling the generation of sequences of text, music, and more [c1537].

[FIGURE: RNN unrolled diagram showing input tokens x1, x2, x3 feeding into hidden states h1, h2, h3 sequentially, with each hidden state depending on the previous one]

For language translation, RNNs employ an encoder-decoder architecture in which both the encoder and decoder are recurrent networks that maintain and update hidden states at each time step [c1500, c1510]. The encoder processes the entire input sequence and produces a **context vector**—the final hidden state—which is then passed to the decoder [c1499, c1504]. That context vector conveys the encoded meaning of the input text [c1505], and the decoder uses it to generate the translated sentence one word at a time [c1506].

Critically, the decoder has access only to this final hidden state, not to any of the intermediate hidden states produced along the way [c1511, c1513, c1514]. In other words, the decoder never directly sees the individual input words—only the encoder's compressed summary of them [c1521].

[FIGURE: RNN encoder-decoder diagram showing encoder processing input tokens into a single context vector, which is passed to the decoder that generates output tokens one at a time]

#### The Vanishing Gradient and Loss of Context

This design creates a fundamental bottleneck: compressing an entire input sentence—or paragraph, or document—into a single fixed-size vector means that the longer the input, the more information gets squeezed into that one vector, and the more gets lost.

A major shortcoming of RNNs is precisely that they must remember the entire encoded input in a single hidden state before passing it to the decoder [c1522]. Because the decoder has no access to earlier inputs, context loss becomes severe with longer sequences [c1556]. This **loss of context**—the decoder's inability to capture longer dependencies because it relies on only one final hidden state—was one of the biggest issues that made RNNs less effective compared to attention-based models like GPT [c1517, c1519].

#### Long Short-Term Memory Networks

Long Short-Term Memory (LSTM) networks were developed to address some of these limitations. By maintaining two separate paths—one for short-term memories and one for long-term memories—LSTMs mitigate the vanishing gradient problem that plagued standard RNNs [c1314, c1541].

Even so, both RNNs and LSTMs struggled with longer context [c1542]. The architectural improvements of LSTMs did not eliminate the fundamental bottleneck of compressing everything into a fixed context vector.

[FIGURE: LSTM cell diagram showing two separate pathways—a cell state (long-term memory) and a hidden state (short-term memory)—with input, forget, and output gates]

---

### The Bahdanau Attention Mechanism

The breakthrough came in 2014, three years before the transformer itself. The Bahdanau attention mechanism was introduced in the paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio [c1523, c1524]. The Transformer architecture is built on this earlier attention mechanism [c1546].

The core idea is elegant: instead of forcing the decoder to work from a single context vector, give it access to *all* of the encoder's hidden states. The Bahdanau mechanism allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525, c1544]. In the attention-augmented architecture, all encoder hidden states—not just the final one—are passed to the decoder at every decoding step [c1530].

[FIGURE: Bahdanau attention diagram showing all encoder hidden states h1...hN being passed to the decoder, with attention weights alpha1...alphaN computed at each decoding step to form a weighted context vector]

This solves the long-range dependency problem: the decoder can access all input tokens even in long sentences and selectively focus on the most relevant ones [c1527, c1526]. The key property that makes this possible is **dynamic focus**—at every decoding step, the decoder chooses which inputs to attend to and how much weight to give each one [c1533, c1557].

It is worth distinguishing this from what came next. Traditional attention operates between two different sequences—an input and an output—determining which parts of the output are most related to which parts of the input [c1558]. This is powerful, but it still requires a separate input and output sequence to compare.

Intuitively, this is like a human translator who, when writing each word of a translation, can glance back at any word in the original sentence rather than relying solely on a mental summary of the whole thing.

---

### From Attention to Self-Attention

The next conceptual leap was to apply attention *within a single sequence*. This is self-attention.

Self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. The term "self" captures this precisely: rather than relating an input sequence to an output sequence, the mechanism computes attention weights by relating different positions within a single input sequence to each other [c1553, c1559].

Two forms of self-attention are worth distinguishing:

- **Simplified self-attention** is the purest and most basic form of the technique [c1487].
- **Self-attention with trainable weights** introduces learnable parameters that form the basis of the actual mechanism used in LLMs [c1488].

In practice, **multi-head attention**—which runs several self-attention operations in parallel and combines their outputs—is the main attention mechanism used in GPT and modern LLMs [c1486].

The Transformer architecture, introduced in 2017, was built with self-attention at its core, directly inspired by the Bahdanau attention mechanism [c1534, c1545].

[FIGURE: Self-attention diagram showing a single input sequence where each token attends to every other token in the same sequence, with attention weights shown as a matrix]

---

### The Transformer Architecture in Detail

With self-attention as its foundation, the transformer dispensed with recurrence entirely. There are no hidden states passed from one time step to the next; instead, the entire input sequence is processed in parallel, with self-attention allowing every position to communicate directly with every other position.

The encoder-decoder architecture consists of two sub-modules: an encoder that processes the input sequence and a decoder that generates the output sequence [c1497]. The encoder converts tokenized input text into vector embeddings [c1267]; the decoder generates output text one word at a time, receiving both those embeddings and the partial output produced so far [c1270]. The full pipeline—tokenization, encoding, decoding with partial output, and word-by-word generation—unfolds across 8 steps [c1320, c1268].

The original transformer was developed for machine translation, specifically translating English text into German and French [c1319, c1269]. Both BERT and GPT have the word "transformers" in their names because they originated from this architecture [c1303].

---

### BERT: The Encoder-Only Transformer

BERT stands for Bidirectional Encoder Representations from Transformers [c1288]. As the name suggests, BERT uses only the encoder half of the transformer architecture and has no decoder [c1295, c1326].

Rather than predicting the next word in a sequence, BERT is trained to predict hidden or masked words within a sentence [c1324]—a task called masked language modeling. To do this effectively, BERT pays attention to the sentence from both left and right directions simultaneously, making it truly bidirectional [c1325, c1297]. This allows BERT to capture nuances and relationships between words by considering the full sentence context from both sides [c1299].

[FIGURE: BERT bidirectional attention diagram showing a masked token in the center of a sentence with attention arrows flowing both left and right from all other tokens toward the masked position]

This bidirectional nature makes BERT particularly well-suited for tasks that require understanding the full context of a sentence before making a prediction—classification, named entity recognition, question answering, and especially sentiment analysis [c1327]. GPT can also perform sentiment analysis, but it is a specialty of BERT [c1302].

---

### GPT: The Decoder-Only Transformer

GPT stands for Generative Pre-trained Transformers [c1289] and is a pre-trained, foundational model [c1290].

Where BERT reads a sentence in both directions, GPT performs strictly left-to-right analysis: it receives incomplete text and predicts the next word one word at a time, attending only to tokens that have already been generated [c1328, c1296, c1293].

[FIGURE: GPT left-to-right generation diagram showing a sequence of tokens where each new token is predicted based only on the tokens to its left, with a causal mask blocking rightward attention]

Multi-head attention is the main attention mechanism driving GPT and modern LLMs [c1486]. The contrast with BERT is stark and instructive:

| Property | BERT | GPT |
|---|---|---|
| Architecture | Encoder only | Decoder only |
| Directionality | Bidirectional | Left-to-right |
| Training objective | Predict masked words | Predict next word |
| Primary use | Classification, analysis | Text generation |

BERT's bidirectionality makes it powerful for understanding tasks; GPT's autoregressive, left-to-right design makes it powerful for generation. Both inherit from the transformer, but they use different halves of it.

---

### Beyond Language: Vision Transformers

The transformer architecture has proven general enough to extend well beyond text. Vision Transformers (ViT) apply the transformer to computer vision tasks such as image recognition and image classification [c1306]. This demonstrates that the core ideas of the transformer—self-attention and parallel processing of sequences—are not inherently linguistic; they apply wherever structured sequences of data need to be modeled.

---

### The Historical Arc

It is worth pausing to appreciate how far the field traveled in a short time.

1. **RNNs** introduced the hidden state, enabling memory of previous inputs and making sequential generation possible [c1535, c1538, c1537].
2. **LSTMs** addressed the vanishing gradient problem with separate long-term and short-term memory routes [c1541], but still struggled with longer context [c1542].
3. **Bahdanau attention (2014)** broke the single-context-vector bottleneck by giving the decoder access to all encoder hidden states [c1523, c1525], introducing the concept of dynamic focus [c1533].
4. **The Transformer (2017)** placed self-attention at its core—inspired by Bahdanau attention [c1534]—and eliminated recurrence entirely, as described in "Attention Is All You Need" [c1284].
5. **BERT and GPT** emerged as specialized variants: one encoder-only and bidirectional, one decoder-only and left-to-right, each optimized for a different family of tasks [c1295, c1326, c1296, c1328].

[FIGURE: Timeline diagram showing progression from RNNs → LSTMs → Bahdanau Attention (2014) → Transformer (2017) → BERT and GPT, with brief annotations at each step]

---

### Why the Transformer Won

The transformer's dominance over its predecessors comes down to a few concrete advantages.

**Parallelism.** RNNs process sequences step by step, with each hidden state depending on the previous one. Transformers process the entire sequence at once using self-attention, making training dramatically faster on modern hardware.

**Long-range dependencies.** Self-attention allows every token to directly attend to every other token in the sequence [c1548], eliminating the information bottleneck that plagued RNNs and LSTMs [c1522, c1542]. Dependencies can be modeled without regard to the distance between positions [c1277].

**Scalability.** The transformer's architecture scales well with data and compute—a large part of why models like GPT have grown to billions of parameters while remaining trainable.

**Flexibility.** By using only the encoder (BERT), only the decoder (GPT), or both (the original transformer), practitioners can tailor the architecture to their task, while both variants trace their lineage directly to the original design [c1303].

---

### Summary

This chapter traced the evolution from recurrent networks to the transformer architecture. RNNs introduced the hidden-state mechanism that enabled sequential memory [c1535], LSTMs extended this with a dual-memory design to combat vanishing gradients [c1314], and the Bahdanau attention mechanism of 2014 first gave decoders access to all encoder states rather than just the final one [c1523, c1525]. The 2017 transformer [c1545] took self-attention—the idea that each position in a sequence can attend to all other positions within that same sequence [c1548]—and made it the architecture's core [c1534]. From that foundation, BERT emerged as an encoder-only, bidirectional model suited for understanding tasks [c1295, c1325], while GPT emerged as a decoder-only, left-to-right model suited for generation [c1328, c1296].

In the next chapter, we will go deeper into the mechanics of self-attention itself—how attention weights are computed, what queries, keys, and values are, and how multi-head attention extends the basic mechanism into the form used in production LLMs.