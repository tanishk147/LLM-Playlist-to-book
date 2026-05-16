## The Transformer Architecture: Encoders, Decoders, BERT, and GPT

Modern large language models did not appear out of nowhere. They are the product of a decades-long search for better ways to process sequential data—text, speech, music—and the story of how we got here is worth telling carefully. This chapter traces that history, from the early days of recurrent networks through the attention mechanism and into the transformer architecture that powers models like BERT and GPT today. By the end, you will understand not just *what* these components are, but *why* they were designed the way they were and what problems each design was trying to solve.

---

### The Transformer at a Glance

Before diving into history, it helps to have a rough map of the destination. 

The transformer architecture consists of two main blocks [c1273]:

1. An **encoder block** that converts input text into embedding vectors.
2. A **decoder block** that generates output text from those embedding vectors and partial output already produced.

[FIGURE: High-level transformer diagram showing input text flowing into an encoder block that produces embedding vectors, which feed into a decoder block alongside partial output text, producing final output text word by word]

The encoder is the component that takes tokenized input text and converts it into vector embeddings [c1267]. The decoder is the component that generates output text one word at a time, receiving both the vector embeddings from the encoder and the partial output text generated so far [c1270]. The transformer generates output one word at a time, making previously generated words available to the model when predicting the next word [c1271].

Intuitively, you can think of the encoder as a reading comprehension module and the decoder as a writing module. The encoder reads and understands; the decoder writes, one token at a time, consulting the encoder's understanding as it goes.

The transformer uses feed-forward layers with weights and parameters that are optimized during training to predict output words correctly [c1275]. The full simplified transformer architecture can be broken down into 8 steps: tokenization, encoding, decoding with partial output, and word-by-word generation [c1320].

It is worth noting upfront that the terms "transformers" and "large language models" should not be used interchangeably—they are different concepts [c1316]. Not all large language models are based on transformer architecture [c1308]. The transformer is one architecture; LLMs are a broader category of models, many of which happen to use transformers.

---

### Why Attention Is the Engine

Transformers are conceptualized as the secret sauce behind large language models, with the attention mechanism serving as the engine that drives the transformer [c1479]. The attention mechanism is a key factor in the strong performance of large language models like ChatGPT [c1480].

Attention mechanisms have become an integral part of sequence modeling, allowing modeling of dependencies without regard to the distance between input or output sequences [c1277]. This last point—*without regard to distance*—is the crucial one. To understand why it matters, we need to understand the problem that attention was invented to solve.

---

### The Problem: Long-Range Dependencies

Consider a sentence like: *"The trophy didn't fit in the suitcase because it was too big."* What does "it" refer to—the trophy or the suitcase? A human reader resolves this instantly by connecting "it" to "trophy" across several intervening words. Now imagine this ambiguity stretched across multiple sentences or paragraphs.

Long-range dependencies refer to the need to understand context from distant parts of a sequence—such as earlier sentences—to accurately predict the next word in a later sentence [c1280]. Long-term dependencies in sentences are complex sentence structures where multiple clauses or phrases are connected, making it difficult for language models to identify relationships between distant words [c1481].

The self-attention mechanism allows the model to capture long-range dependencies so that it can look far behind to sentences closer to the current one to identify the next word [c1281]. In ChatGPT, when you write an input, GPT gives attention to every sentence and then predicts what the next word could be, rather than just looking at the sentence immediately before the current one [c1286].

The self-attention mechanism also allows the model to weigh the importance of different words and tokens relative to each other [c1279]. Attention blocks in multi-head attention ensure that long-range dependencies in sentences are captured [c1283]. The paper that introduced the transformer was named "Attention Is All You Need" precisely because of the self-attention mechanism and the intuition behind attention that it introduced [c1284].

---

### Predecessors: RNNs and LSTMs

To appreciate what transformers solved, you need to understand what came before them.

#### Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining a hidden state that captures information about previous inputs [c1535]. The hidden state is the key innovation of RNNs that enables them to process sequential data [c1538]. In an RNN, the hidden state captures memory of previous inputs in the sequence [c1502].

RNNs were a significant advancement for generative models, allowing for the generation of sequences of text, music, and more [c1537]. Recurrent Neural Networks maintain a feedback loop to incorporate memory [c1313].

[FIGURE: RNN unrolled diagram showing input tokens x1, x2, x3 feeding into hidden states h1, h2, h3 sequentially, with each hidden state depending on the previous one]

For language translation, RNNs employ an encoder-decoder architecture [c1500]. In sequence-to-sequence models, the encoder and decoder are both recurrent neural networks that each maintain and update hidden states at each time step [c1510].

In the encoder-decoder architecture, the encoder processes the entire input sequence and generates a context vector, which is then passed to the decoder to generate the output sequence [c1499]. The final hidden state from the encoder is called the **context vector** [c1504]. The context vector is passed from the encoder to the decoder to convey the encoded meaning of the input text [c1505]. The decoder uses this final hidden state to generate the translated sentence one word at a time [c1506].

The decoder has access only to the final hidden state from the encoder, not to previous intermediate hidden states [c1511]. The encoder processes the entire input text into one final hidden state which serves as the memory cell for the decoder [c1513]. A recurrent neural network cannot directly access earlier hidden states during decoding; it only accesses the final hidden state from the encoder [c1514].

In standard RNN encoder-decoder architecture, the decoder only receives the final hidden state and does not have access to all prior input words [c1521].

[FIGURE: RNN encoder-decoder diagram showing encoder processing input tokens into a single context vector, which is passed to the decoder that generates output tokens one at a time]

#### The Vanishing Gradient and Loss of Context

Here is the fundamental problem: compressing an entire input sentence—or paragraph, or document—into a single fixed-size vector is a bottleneck. The longer the input, the more information gets squeezed into that one vector, and the more information gets lost.

A major shortcoming of RNNs is that they must remember the entire encoded input in a single hidden state before passing it to the decoder [c1522]. In recurrent neural networks, the decoder does not have access to the previous inputs, which leads to context loss when dealing with long sentences [c1556].

Loss of context is the problem that occurs when an RNN decoder struggles to capture longer dependencies and contextual information because it relies on only one final hidden state [c1517]. Loss of context was one of the biggest issues that made RNNs less effective compared to models based on attention mechanisms like GPT [c1519].

#### Long Short-Term Memory Networks

Long Short-Term Memory (LSTM) networks were developed to address some of these limitations. LSTMs incorporate two separate paths: one for short-term memories and one for long-term memories [c1314]. LSTMs solve the vanishing gradient problem by maintaining both a long-term memory route and a short-term memory route [c1541].

However, both RNNs and LSTMs had problems with respect to longer context [c1542]. Even with the architectural improvements of LSTMs, the fundamental bottleneck of compressing everything into a fixed context vector remained.

[FIGURE: LSTM cell diagram showing two separate pathways—a cell state (long-term memory) and a hidden state (short-term memory)—with input, forget, and output gates]

---

### The Bahdanau Attention Mechanism

The breakthrough came in 2014. The Bahdanau attention mechanism was developed in the paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio [c1523].

Attention was introduced in 2014, three years before the 2017 "Attention Is All You Need" paper that introduced the Transformer architecture [c1524]. The Transformer architecture is built on the attention mechanism proposed in 2014 [c1546].

The core idea of Bahdanau attention is elegant: instead of forcing the decoder to work from a single context vector, give it access to *all* of the encoder's hidden states. The Bahdanau attention mechanism allows the decoder to selectively access different parts of the input sequence at each decoding step, rather than relying only on the final hidden state [c1525]. The Bahdanau attention mechanism allows the decoder to have access to each input state during decoding and selectively decide which inputs to give more attention to [c1544].

In the attention-augmented encoder-decoder architecture, all encoder hidden states (not just the final one) are passed to the decoder at every decoding step [c1530].

[FIGURE: Bahdanau attention diagram showing all encoder hidden states h1...hN being passed to the decoder, with attention weights alpha1...alphaN computed at each decoding step to form a weighted context vector]

The attention mechanism solves the problem of long-range dependencies by allowing the decoder to access all input tokens even in long sentences and selectively focus on relevant tokens [c1527]. In the attention mechanism, the decoder has access to all input tokens and can decide how much attention to pay to each token when generating an output word [c1526].

Dynamic focus is the key property of the attention mechanism that allows the decoder to selectively choose which inputs to focus on and how much attention to give to each input at every decoding step [c1533].

In attention mechanisms, when decoding a particular part, the decoder has access to all of the input tokens and decides how much attention to give to each input [c1557]. Traditional attention looks at one input sequence and one output sequence, determining which parts of the output sequence are more related to which parts of the input sequence [c1558].

Intuitively, this is like a human translator who, when writing each word of a translation, can glance back at any word in the original sentence rather than relying solely on a mental summary of the whole thing.

---

### From Attention to Self-Attention

The Bahdanau mechanism applied attention between two different sequences—an input sequence and an output sequence. The next conceptual leap was to apply attention *within a single sequence*. This is self-attention.

Self-attention is a mechanism that allows each position of an input sequence to attend to all positions in the same sequence [c1548]. In self-attention, the term "self" refers to the attention mechanism's ability to compute attention weights by relating different positions in a single input sequence [c1553].

Self-attention looks at one sequence and examines how different parts of that same sequence are related with respect to each other [c1559]. This is in contrast to traditional attention, which looks at one input sequence and one output sequence [c1558].

There are two forms of self-attention worth distinguishing:

- **Simplified self-attention** is the purest and most basic form of the attention technique [c1487].
- **Self-attention with trainable weights** introduces trainable weights which form the basis of the actual mechanism used in LLMs [c1488].

Multi-head attention is the main attention mechanism used in GPT, generative pre-trained transformers, and modern LLMs [c1486].

The Transformer architecture, introduced in 2017, was built with the self-attention mechanism at its core, which was inspired by the Bahdanau attention mechanism [c1534]. The Transformer architecture was introduced in 2017 [c1545].

[FIGURE: Self-attention diagram showing a single input sequence where each token attends to every other token in the same sequence, with attention weights shown as a matrix]

---

### The Transformer Architecture in Detail

With self-attention as its foundation, the transformer dispensed with recurrence entirely. There are no hidden states passed from one time step to the next. Instead, the entire input sequence is processed in parallel, with self-attention allowing every position to communicate with every other position directly.

The encoder-decoder architecture consists of two sub-modules: an encoder that processes the input sequence and a decoder that generates the output sequence [c1497].

The encoder takes tokenized input text and converts it into vector embeddings [c1267]. The decoder generates output text one word at a time, receiving both vector embeddings from the encoder and partial output text [c1270].

The full pipeline, in simplified form, consists of 8 steps covering tokenization, encoding, decoding with partial output, and word-by-word generation [c1320, c1268].

Both BERT and GPT have the word "transformers" in their names because they originated from the transformer architecture [c1303].

---

### BERT: The Encoder-Only Transformer

BERT stands for Bidirectional Encoder Representations from Transformers [c1288]. BERT uses an encoder architecture [c1295]. BERT has only an encoder architecture and does not have a decoder [c1326].

BERT (Bidirectional Encoder Representations from Transformers) is a transformer variation that predicts hidden or masked words in a sentence [c1324]. Rather than predicting the next word in a sequence, BERT is trained to fill in blanks—a task called masked language modeling.

BERT pays attention to a sentence from both left and right directions, making it bidirectional [c1325]. BERT is bidirectional, attending to different parts of the sentence from both left and right directions [c1297]. BERT can capture nuances and relationships between words by looking at the entire sentence from both directions [c1299].

[FIGURE: BERT bidirectional attention diagram showing a masked token in the center of a sentence with attention arrows flowing both left and right from all other tokens toward the masked position]

BERT models are commonly used for sentiment analysis because they can capture the meanings of different words and how they relate to each other [c1327]. GPT can complete missing text and perform sentiment analysis, though sentiment analysis is a specialty of BERT [c1302].

The bidirectional nature of BERT makes it particularly well-suited for tasks where you need to understand the full context of a sentence before making a prediction—classification, named entity recognition, question answering, and sentiment analysis are natural fits.

---

### GPT: The Decoder-Only Transformer

GPT stands for Generative Pre-trained Transformers [c1289]. GPT is a pre-trained or foundational model [c1290].

GPT (Generative Pre-trained Transformer) performs left-to-right analysis, receiving incomplete text and predicting the next word one word at a time [c1328]. GPT processes text from left to right, predicting only rightmost unknown information [c1296]. GPT generates one word at a time [c1293].

[FIGURE: GPT left-to-right generation diagram showing a sequence of tokens where each new token is predicted based only on the tokens to its left, with a causal mask blocking rightward attention]

Multi-head attention is the main attention mechanism used in GPT and modern LLMs [c1486].

The contrast with BERT is stark and instructive:

| Property | BERT | GPT |
|---|---|---|
| Architecture | Encoder only | Decoder only |
| Directionality | Bidirectional | Left-to-right |
| Training objective | Predict masked words | Predict next word |
| Primary use | Classification, analysis | Text generation |

BERT's bidirectionality makes it powerful for understanding tasks. GPT's left-to-right, autoregressive design makes it powerful for generation tasks. Both inherit from the transformer, but they use different halves of it.

---

### Beyond Language: Vision Transformers

The transformer architecture has proven general enough to extend well beyond text. Vision Transformers (ViT) are transformer models applied to computer vision tasks such as image recognition and image classification [c1306]. This demonstrates that the core ideas of the transformer—self-attention, parallel processing of sequences—are not inherently linguistic. They apply wherever you have structured sequences of data.

---

### The Historical Arc

It is worth pausing to appreciate how far the field traveled in a short time.

1. **RNNs** introduced the idea of maintaining a hidden state across a sequence, enabling memory of previous inputs [c1535, c1538]. They were a significant advancement for generative models [c1537].

2. **LSTMs** addressed the vanishing gradient problem by maintaining separate long-term and short-term memory routes [c1541], but still struggled with longer context [c1542].

3. **Bahdanau attention (2014)** broke the bottleneck of the single context vector by giving the decoder access to all encoder hidden states [c1523, c1525]. This introduced the concept of dynamic focus [c1533].

4. **The Transformer (2017)** took self-attention as its core mechanism, inspired by Bahdanau attention [c1534], and eliminated recurrence entirely. It was introduced in the paper "Attention Is All You Need" [c1284].

5. **BERT and GPT** emerged as specialized variants—one encoder-only and bidirectional, one decoder-only and left-to-right—each optimized for different families of tasks [c1295, c1326, c1296, c1328].

[FIGURE: Timeline diagram showing progression from RNNs → LSTMs → Bahdanau Attention (2014) → Transformer (2017) → BERT and GPT, with brief annotations at each step]

---

### Putting It Together: Why the Transformer Won

The transformer's dominance comes down to a few concrete advantages over its predecessors.

**Parallelism.** RNNs process sequences step by step; each hidden state depends on the previous one. Transformers process the entire sequence at once using self-attention, which makes training dramatically faster on modern hardware.

**Long-range dependencies.** Self-attention allows every token to directly attend to every other token in the sequence [c1548]. There is no information bottleneck of the kind that plagued RNNs and LSTMs [c1522, c1542]. Attention mechanisms allow modeling of dependencies without regard to the distance between input or output sequences [c1277].

**Scalability.** The transformer's architecture scales well with data and compute. This is a large part of why models like GPT have grown to billions of parameters while remaining trainable.

**Flexibility.** By using only the encoder (BERT), only the decoder (GPT), or both (original transformer), practitioners can tailor the architecture to their task. Both BERT and GPT trace their lineage directly to the original transformer [c1303].

---

### Summary

This chapter has traced the evolution from recurrent networks to the transformer architecture. We started with RNNs and their hidden-state mechanism [c1535], moved through LSTMs and their dual-memory design [c1314], and arrived at the Bahdanau attention mechanism of 2014 [c1523] that first gave decoders access to all encoder states rather than just the final one [c1525].

The 2017 transformer [c1545] took self-attention—the idea that each position in a sequence can attend to all other positions in the same sequence [c1548]—and made it the core of a new architecture [c1534]. From that foundation, BERT emerged as an encoder-only, bidirectional model suited for understanding tasks [c1295, c1325], while GPT emerged as a decoder-only, left-to-right model suited for generation [c1328, c1296].

In the next chapter, we will go deeper into the mechanics of self-attention itself—how attention weights are computed, what queries, keys, and values are, and how multi-head attention extends the basic mechanism into the form used in production LLMs.
