## Pre-Training vs. Fine-Tuning: Concepts and Motivation

 Understanding what each stage does—and why both exist—is foundational to everything that follows in this book.

This chapter unpacks that lifecycle in depth. 

---

### The Two-Stage Lifecycle

At the highest level, creating an LLM consists of two stages: pre-training and fine-tuning [c1, c9]. 

[FIGURE: A horizontal pipeline diagram showing Stage 1 (Pre-training on raw text corpus → Base/Foundational Model) feeding into Stage 2 (Fine-tuning on labeled data → Task-Specific Model)]

 Throughout this chapter, a *token* is approximately equal to one word [c13]. LLM stands for Large Language Model [c4]. And when we say *transformer*, we mean the neural network architecture that underlies models like GPT and BERT—though it is worth noting upfront that not all transformers are LLMs, and not all LLMs are transformers [c1304, c1308]. We will return to that distinction later in the chapter.

---

### Pre-Training: Building the Foundation

#### What Pre-Training Is

Pre-training means training an LLM on a huge amount of data so that it can perform a wide range of tasks [c23]. More precisely, it is the first training stage of an LLM, and the model it produces is called a *base model* or *foundational model* [c38]. Pre-training is also described simply as training on a large, diverse dataset [c1246].

 Pre-training requires a huge amount of data—billions or even trillions of words—and significant computational power, making it inaccessible to most individuals [c32]. LLMs are typically trained on billions of documents [c3165]. 

#### The Training Objective: Next-Word Prediction

The core task that drives pre-training is deceptively simple. *Word completion* is the task where an LLM is given a set of words and must predict the next word [c18]. The model sees raw text—regular text without any labeling information [c37]—and learns by trying to predict what comes next, over and over, across billions of examples.

 

#### Emergent Capabilities

LLMs trained only for next-word prediction can perform translation, multiple-choice questions, text summarization, sentiment analysis, linguistic acceptability, and question answering—without specific training on any of these tasks [c21]. 

 In traditional NLP, separate models must be trained for different tasks like translation, question answering, and emotion detection, whereas pre-trained LLMs can perform all these tasks with a single model [c22]. 

Why does next-word prediction produce such broad capability? 

LLMs interact effectively with users because they are trained on a huge and diverse set of data [c11]. 

#### What Pre-Training Data Looks Like

Two commonly cited data sources illustrate the kind of material used in pre-training:

- **Common Crawl** is a huge and open repository of all the data on the internet [c16].
- **WebText2** is a corpus consisting of Reddit submissions, blog posts, Stack Overflow articles, and code [c17].

Together, these sources expose the model to an enormous variety of writing styles, topics, languages, and domains. 

#### The Output: A Foundational Model

 GPT-4, for example, is a pre-trained model capable of text completion and other tasks such as sentiment analysis and question answering [c39]. GPT stands for Generative Pre-trained Transformers [c1289], and it is explicitly a pre-trained or foundational model [c1290].

GPT generates one word at a time [c1293] and processes text from left to right, predicting only the rightmost unknown information [c1296]. 

[FIGURE: Diagram contrasting GPT (left-to-right, decoder-only, generates next token) with BERT (bidirectional, encoder-only, predicts masked tokens)]

For contrast, consider BERT—Bidirectional Encoder Representations from Transformers [c1324]. BERT randomly masks some words during training and attempts to predict those masked words [c1292]. Rather than predicting the next word, BERT predicts hidden or masked words in a given sentence [c1291]. BERT is bidirectional, attending to different parts of the sentence from both left and right directions [c1297], and it uses only an encoder architecture without a decoder [c1326].

Because BERT can capture nuances and relationships between words by looking at the entire sentence from both directions [c1299], it can differentiate between different meanings of the same word—for example, "bank" as a financial institution versus "bank" as a river bank—by examining surrounding words [c1300]. This makes BERT particularly well suited to classification tasks: BERT is commonly used for sentiment analysis [c1301, c1327].

Both BERT and GPT have the word "transformers" in their names because they originated from the transformer architecture [c1303]. The original transformer architecture was developed for machine translation tasks, specifically translating English text into German and French [c1319]. The simplified transformer architecture consists of eight steps: tokenization, encoding, decoding with partial output, and word-by-word generation [c1320]. BERT uses an encoder architecture [c1295], while GPT uses a decoder-style design.

 

---

### The Limits of Pre-Training Alone

 Pre-trained models like GPT-4 may not have access to company-specific data and therefore produce generic rather than domain-specific responses [c26].

Think about what this means in practice. 

 Fine-tuning is needed when building applications specific to a particular task or domain, rather than for general-purpose use [c25]. General users and students can use pre-trained models like GPT-4 directly without fine-tuning, but companies and startups deploying LLM applications in production need fine-tuning [c36].

 

---

### Fine-Tuning: Specializing the Foundation

#### What Fine-Tuning Is

 Where pre-training operates on raw, unlabeled text at massive scale, fine-tuning typically requires a labeled dataset [c43]. 

The three main steps for building an LLM are: (1) training on a large corpus of raw text data, (2) pre-training to create a foundational model, and (3) fine-tuning on labeled data for specific tasks [c47]. Stage 3 of building an LLM involves fine-tuning on smaller, specific datasets to build applications such as classifiers or personal assistants [c3161].

 The generalist already knows how to read, write, reason, and communicate. The specialized training teaches them the specific knowledge and conventions of your domain.

#### Types of Fine-Tuning

**Instruction fine-tuning** involves providing instruction-answer pairs as labeled data to teach the model specific tasks like language translation [c41]. The training examples look like: "Translate the following sentence into French: [sentence]" → "[translation]". The model learns to follow instructions by seeing many such pairs.

**Classification fine-tuning** adapts the model to produce categorical outputs—for example, labeling a piece of text as positive or negative sentiment, or routing a customer query to the correct department. Fine-tuned LLMs can be used for specific tasks such as classification, summarization, translation, and building custom chatbots [c45].

Large companies building custom chatbots perform fine-tuning on foundational models rather than using foundational models alone [c46]. 

[FIGURE: Branching diagram showing a pre-trained base model being fine-tuned into multiple specialized models: a classifier, a summarizer, a translation model, and a custom chatbot]

---

### Real-World Case Studies

 The following three case studies illustrate different domains and different reasons for choosing fine-tuning over a general-purpose model.

#### SK Telecom: Customer Service in Korean

SK Telecom fine-tuned a model to improve customer service interactions for telecom-related conversations in Korean, resulting in a 35% increase in conversation summarization quality and a 33% increase in intent recognition accuracy [c27].

This case study highlights two important points. Fine-tuning produced a qualitatively better product for this specific use case.

#### Harvey: Legal AI

Harvey is an AI legal tool that was fine-tuned on legal case history data to assist attorneys and lawyers, addressing the limitation that foundational models lacked extensive knowledge of legal cases [c28].

 Legal reasoning depends on specific precedents, statutes, and jurisdictional rules. Harvey's fine-tuning on legal case data directly addresses this gap—the model learns the specific knowledge that attorneys need, rather than relying on whatever legal content happened to appear in the pre-training corpus.

This also illustrates a risk management dimension. 

#### JPMorgan Chase: Proprietary Financial Data

JPMorgan Chase developed a fine-tuned AI-powered LLM suite specifically for their employees using their proprietary data, rather than relying solely on GPT-4 [c30].

This case study adds a dimension that the previous two do not: data privacy and competitive advantage. By fine-tuning their own model on their own data, JPMorgan Chase keeps that data in-house while still benefiting from the language capabilities of a large pre-trained model.

This pattern—fine-tune a foundational model on proprietary data, deploy internally—is increasingly common among large enterprises. The foundational model provides the expensive, compute-intensive language understanding; the fine-tuning provides the domain-specific knowledge that makes the model useful for the organization's specific needs.

---

### Connecting the Stages: A Unified View

Building a large language model from scratch typically proceeds in three stages: Stage 1 (understanding LLM basics), Stage 2 (pre-training), and Stage 3 (fine-tuning) [c3158]. The three main steps are: training on a large corpus of raw text data, pre-training to create a foundational model, and fine-tuning on labeled data for specific tasks [c47].

[FIGURE: Three-stage pipeline: Stage 1 (raw text corpus) → Stage 2 (pre-training → base model) → Stage 3 (fine-tuning on labeled data → specialized model)]

 It requires billions or trillions of tokens and massive computational resources [c32]. Fine-tuning, by contrast, operates on smaller, specific datasets [c3161]. 

 You do not need to train a model from scratch to build a useful product. You need to understand what the foundational model already knows, identify the gap between that and what your application requires, and close that gap through fine-tuning.

---

### When to Pre-Train vs. When to Fine-Tune

Given everything above, a practical question emerges: when should you do each?

 Pre-training requires a huge amount of data and significant computational power, making it inaccessible to most individuals [c32]. 

 

 Legal case history, proprietary financial data, and specialized technical documentation are all examples [c28, c30].

 Instruction fine-tuning can teach the model to follow specific formats [c41].

 SK Telecom's Korean telecom use case is an example [c27].

 Fine-tuning on proprietary data, deployed internally, addresses this [c30].

If none of these factors apply—if you are a student, a researcher, or a developer building a general-purpose tool—then a pre-trained model used directly may be entirely sufficient [c36].

---

### The Broader Landscape: Transformers, LLMs, and Their Relationship

The terms "transformers" and "large language models" should not be used interchangeably, as they are different concepts [c1316]. Not all transformers are large language models [c1304], and not all large language models are based on transformer architecture [c1308].

Transformers can be used for computer vision tasks in addition to language tasks [c1305]. Vision Transformers (ViT) are transformer models applied to computer vision tasks such as image recognition and image classification [c1306], and they achieve comparable or better results than Convolutional Neural Networks (CNNs) while requiring substantially fewer computational resources for pre-training [c1307]. Transformers can also be used for image segmentation [c1330].

 Recurrent Neural Networks (RNNs) maintain a feedback loop to incorporate memory [c1313], and Long Short-Term Memory (LSTM) networks incorporate two separate paths: one for short-term memories and one for long-term memories [c1314]. Both RNNs and LSTMs could perform sequence modeling and text completion tasks [c1312]. 

 

---

### Summary

 

- **Pre-training** trains a model on a massive corpus of raw, unlabeled text using next-word prediction (or masked-word prediction, in the case of BERT) [c23, c43]. The result is a foundational model with broad, general-purpose capabilities [c38].

- **The emergent capabilities** of pre-trained models are remarkable: a model trained only on next-word prediction can perform translation, summarization, sentiment analysis, and question answering without task-specific training [c21].

- **The limitation** of pre-trained models is that they lack domain-specific knowledge and may produce generic responses when applied to specialized tasks [c26].

- **Fine-tuning** adapts a foundational model to a specific task or domain using labeled data [c43, c47]. It is needed for production applications in specialized domains [c25].

- **Real-world examples**—SK Telecom [c27], Harvey [c28], and JPMorgan Chase [c30]—demonstrate that fine-tuning produces measurable improvements in domain-specific performance.

- **Instruction fine-tuning** and **classification fine-tuning** are the two main approaches, suited to different output types [c41, c45].

- **Transformers and LLMs are related but distinct concepts** [c1316]. Not every transformer is an LLM, and not every LLM is a transformer [c1304, c1308].

In the next chapter, we will move from concepts to mechanics, beginning with the data preparation pipeline: how raw text is tokenized, how tokens are assigned IDs, and how those IDs are converted into the vector embeddings that an LLM actually processes [c3162, c3166, c3173].
