## Large Language Models: Foundations and Landscape

If you have spent any time with ChatGPT, Gemini, or similar tools, you have already interacted with a Large Language Model. But knowing how to *use* one and knowing how to *build* one are very different things. This chapter lays the conceptual groundwork for everything that follows: what LLMs actually are, where they sit in the broader landscape of artificial intelligence, how they are trained, and why they represent such a significant departure from the NLP tools that came before them. By the end, you will have a clear mental map of the territory—one that will make every technical detail in subsequent chapters feel purposeful rather than arbitrary.

---

### What Is a Large Language Model?

Let us start with the simplest possible definition. A Large Language Model—LLM for short [c4]—is a neural network designed to understand, generate, and respond to human-like text [c279]. That single sentence contains three important ideas: *neural network*, *massive scale*, and *text*. We will unpack each in turn.

At its core, a neural network consists of input data feeding into layers of neurons stacked together, with an output layer [c284]. You can think of each layer as a transformation: raw numbers go in, slightly more useful numbers come out, and after enough layers the network has learned to map inputs to outputs in ways that are surprisingly powerful. LLMs are deep neural networks—meaning they have many such layers—trained on massive amounts of data, and they are specifically designed to understand, generate, and respond to human-like text [c287].

The word "large" in the name is not decorative. Model size refers to the number of parameters in a model [c290], and modern LLMs have parameter counts in the billions or even trillions. Those parameters are the learned weights that encode everything the model knows about language, facts, reasoning patterns, and more.

[FIGURE: A simple diagram showing input text flowing into stacked neural network layers and producing output text, with "parameters" labeled on the connections between neurons]

---

### Where Do LLMs Fit? The AI Terminology Hierarchy

The field of AI is littered with overlapping buzzwords—AI, ML, deep learning, generative AI—and it is easy to lose track of how they relate. Here is the precise hierarchy.

Artificial Intelligence is the broadest umbrella term, encompassing any machine that behaves remotely like a human or exhibits some form of intelligence [c331]. Inside that umbrella sits Machine Learning, a subset of AI where machines learn and adapt based on how users interact with them [c332]. Inside ML sits Deep Learning, which specifically involves neural networks—whereas ML also encompasses other approaches like decision trees [c333].

LLMs are a subset of deep learning that deal exclusively with text and do not involve images [c338]. That last qualifier is important: the moment you add images, audio, or video to the mix, you have crossed into a different (though related) territory.

[FIGURE: Concentric circles showing AI as the outermost ring, then ML, then DL, then LLMs at the center, with Generative AI overlapping DL and LLMs]

That related territory is Generative AI—a mixture of large language models and deep learning that deals with multiple modalities including text, images, sound, and video [c339]. Generative AI uses deep neural networks to create new content such as text, images, and various forms of media [c340]. So while every LLM is a piece of deep learning, not every piece of generative AI is an LLM. The hierarchical relationship can be summarized as: AI (broadest) contains ML, which contains DL, which contains LLMs (most specific) [c342].

---

### How LLMs Differ from Earlier NLP Models

To appreciate what makes LLMs remarkable, it helps to understand what came before them. Earlier NLP models were designed for very specific tasks [c315]—a translation model translated, a sentiment analysis model classified sentiment, and that was essentially the end of the story [c316]. Each model was a specialist, and specialists are brittle: a translation model cannot answer a question, and a sentiment classifier cannot summarize a document.

LLMs break this mold entirely. Modern LLMs have more diverse and endless applications compared to earlier NLP models [c323]. A single pretrained LLM can translate, summarize, answer questions, write code, and do dozens of other things—often without any task-specific retraining. This task-agnostic quality is one of the defining characteristics of the LLM era, and understanding *why* it arises requires understanding how LLMs are trained.

---

### The Two-Stage Development Pipeline

Creating an LLM involves two stages: pre-training and fine-tuning [c1, c9]. This pipeline is fundamental enough that the entire book is organized around it, so it is worth spending time here to understand what each stage means and why both are necessary.

[FIGURE: A horizontal pipeline diagram with two boxes labeled "Stage 1: Pre-Training" and "Stage 2: Fine-Tuning" connected by an arrow, with "Foundation Model" as the output of Stage 1 and "Task-Specific Model" as the output of Stage 2]

#### Stage 1: Pre-Training

Pre-training means training an LLM on a huge amount of data so that it can perform a wide range of tasks [c23]. The key word is *huge*. We are not talking about thousands or even millions of examples—we are talking about hundreds of billions of tokens of text scraped from the internet and other sources.

For the purposes of building intuition, one token is approximately equal to one word [c13]. So "hundreds of billions of tokens" translates roughly to hundreds of billions of words—an amount of text no human could read in many lifetimes.

Where does all this text come from? Two prominent sources illustrate the scale:

- **Common Crawl** is a huge and open repository of all the data on the internet [c16].
- **WebText2** is a corpus consisting of Reddit submissions, blog posts, Stack Overflow articles, and code [c17].

The training objective during pre-training is deceptively simple: next-word prediction. Given a sequence of words, predict the next word [c3680]. This is also called word completion—the task where an LLM is given a set of words and must predict the next word [c18]. LLM pretraining involves next-word prediction on large text datasets [c1593].

What makes this objective so powerful is that it requires no human-labeled data. The text itself provides the supervision: every word in a document is simultaneously a training input (given the words before it) and a training label (the correct next word). This approach is called generative pre-training—an unsupervised learning approach where a language model is trained on unlabeled text to predict the next word, using the text itself as training data without requiring external labels [c3358]. The raw text used for training is regular text without any labeling information [c37], which means the training data can be collected at internet scale without expensive human annotation.

In the GPT family of models, this self-supervised structure is made explicit: self-supervised learning in GPT involves dividing a sentence into training and testing portions, where the next word is the known true label [c3414]. GPT models are autoregressive models trained on next-word prediction tasks [c3408], meaning each prediction is conditioned on all previous words in the sequence.

The result of pre-training is a **foundation model** (also called a base model or pre-trained model) [c3397]. Pre-training is the process of training a large language model on an underlying dataset to build a foundational model on unlabeled data [c3686]. This foundation model has absorbed an enormous amount of world knowledge and linguistic structure, but it has not yet been specialized for any particular application.

#### Stage 2: Fine-Tuning

Once a foundation model exists, it can be adapted for specific tasks through fine-tuning. A pretrained LLM can be finetuned using smaller labeled datasets for specific tasks like classification, summarization, or question-answering [c1593]. Fine-tuning uses labeled data—human-annotated examples of the task—to adjust the model's weights so that it performs well on that specific task.

The contrast with pre-training is stark: pre-training uses massive unlabeled datasets; fine-tuning uses small labeled datasets. Pre-training builds general capability; fine-tuning sharpens it for a purpose.

It is worth noting what fine-tuning for classification does to a model's outputs. A classification-finetuned model is restricted to predicting only the classes it has encountered during training; for example, a model trained to classify emails as 'spam' or 'not spam' cannot output any other response [c3695]. This is a deliberate constraint—you want a spam classifier to say "spam" or "not spam," not to compose a poem.

All modern large language models are trained in these two main steps: pre-training on unlabeled data to create a foundational model, followed by fine-tuning on labeled task-specific data [c3699].

---

### The Three-Stage Development Workflow

When building an LLM from scratch, the two-stage pipeline expands into a three-stage development workflow [c1773]:

1. **Stage 1 — Foundation and Architecture**: This covers data preparation, sampling, and the architectural components of the model [c1774]. Before you can train anything, you need to understand how to represent text as numbers, how to feed data into the network, and what the network's internal structure looks like.

2. **Stage 2 — Pre-Training**: After the architecture is in place, the model is trained on large unlabeled datasets [c3685]. At the end of this stage, pre-trained weights (such as those released by OpenAI) can be loaded into the model [c3692]. Functions to save and load model weights are implemented to avoid retraining from scratch every time [c3690].

3. **Stage 3 — Fine-Tuning**: The pre-trained model is adapted for specific applications [c3694].

This book follows that three-stage structure. The goal throughout is to understand the nuts and bolts of how large language models are built, rather than simply running pre-built applications [c1768].

---

### The Transformer Architecture: A Brief Introduction

The architectural backbone of virtually every modern LLM is the **transformer**—a deep neural network architecture introduced in 2017 [c1252]. Understanding the transformer is essential to understanding LLMs, and later chapters will build it piece by piece. For now, it is enough to know that the transformer's power comes from a mechanism called **attention**.

The attention mechanism is the core component of the transformer architecture that makes it powerful [c3710]. Intuitively, attention allows the model to decide which parts of the input are most relevant when producing each part of the output. More precisely, the attention mechanism gives LLMs selective access to the entire input sequence when generating output one word at a time [c3711].

The self-attention mechanism captures long-range dependencies in a sentence, allowing better prediction of the next word [c3355]. This is a crucial capability: in a long sentence, the word that best predicts the next word might be ten or twenty words back. Earlier architectures struggled with this; transformers handle it naturally.

Several components work together to make attention function correctly:

- **Attention scores** are a key component of the attention mechanism [c3667].
- **Positional encoding** provides information about the order in which words appear in a sentence to the model [c3677], since the transformer architecture itself does not inherently process words in sequence [c3668].
- **Vector embeddings** represent words as dense numerical vectors [c3669].
- **Tokens** are the basic units of a sentence that the model operates on [c3672].

The transformer also uses specialized variants of attention for different purposes. **Multi-head attention** applies the attention mechanism across multiple representation subspaces simultaneously [c3683], allowing the model to attend to different aspects of the input at once. **Masked multi-head attention** is a variant used in transformer decoders to prevent the model from attending to future positions [c3684]—a necessary constraint when training on next-word prediction, since the model should not be able to "cheat" by looking ahead.

The GPT approach specifically combines the transformer architecture with unsupervised pre-training, where labels are derived from the sentences themselves without pre-labeling data [c3361].

[FIGURE: A simplified transformer block diagram showing input embeddings with positional encoding feeding into multi-head attention, then a feed-forward layer, with residual connections, producing output representations]

---

### Zero-Shot and Few-Shot Learning

One of the most striking capabilities of large pre-trained language models is their ability to perform tasks they were never explicitly trained on. This phenomenon has two related names depending on how much task-specific information the model is given at inference time.

**Zero-shot learning** is the ability to generalize to completely unseen tasks without any prior specific examples [c3372]. In zero-shot learning, the model predicts the answer given only a description of the task with no supporting examples [c3374]. For instance, in zero-shot translation, the model is given only the task description "translate English to French" and the input word "breakfast" without any translation examples [c3385]. The model is completing tasks without any examples at all [c3387].

**Few-shot learning** is learning from a minimum number of examples which the user provides as input [c3373]. In few-shot learning, the model is provided with a few supplementary examples to perform a task [c3440]. The model sees a few examples of a task to learn from [c3378], and these examples are provided directly in the input prompt rather than through any weight update [c3388].

Between these two extremes sits **one-shot learning**, where the model sees a single example of the task in addition to the task description [c3376].

[FIGURE: Three side-by-side prompt boxes illustrating zero-shot (task description only), one-shot (task description + one example), and few-shot (task description + multiple examples) prompting]

These capabilities are closely related to what researchers call **emergent behavior**—the ability of a model to perform tasks that it was not explicitly trained to perform [c3422, c3450]. A model trained purely on next-word prediction somehow learns to translate, reason, and answer questions. This emergence is not fully understood theoretically, but it is empirically well-established.

The concept of few-shot learning as a property of large language models was formally introduced with GPT-3, released in 2020, which introduced the concept that language models are few-shot learners [c3437]. Zero-shot versus few-shot learning remains a key concept in understanding what modern LLMs can do [c3353].

---

### The Historical Arc: From Transformers to GPT-4

To understand where we are, it helps to trace the path that got us here.

The transformer architecture, introduced in 2017 [c1252], provided the foundation. The GPT family of models then demonstrated that combining transformers with large-scale unsupervised pre-training could produce models with remarkable generalization ability [c3361]. GPT-3, released in 2020, was a landmark: it showed that scaling up model size and training data produced qualitatively new capabilities, including few-shot learning [c3437].

GPT-4 represents the current state of the art in closed-source models—a closed-source model where the parameters and weights are not publicly known [c3400]. This opacity is characteristic of the closed-source approach: closed-source models, such as those released by OpenAI, do not release the weights or architecture, and not many details are known about the model itself [c1579].

---

### Open-Source vs. Closed-Source Models

The distinction between open-source and closed-source models is practically important for anyone building with or on top of LLMs.

An open-source model means the entire architecture of the model is available to the public for anyone to see [c1578]. This transparency allows researchers and engineers to inspect, modify, and build upon the model. In 2022, most large language models were closed-source [c1580]. That landscape has shifted considerably: by 2024, the performance gap between open-source and closed-source models is slowly decreasing [c1582]. As of August 2024, the performance of open-source language models is comparable to closed-source models on the MMLU benchmark [c3401].

Meta's release of Llama 3 405B is a concrete example of this trend—a model that closes the gap between closed-source and open-weight models [c1576]. All the information needed to build capable language models is now available as open-source [c1584].

This matters for this book in a direct way. Modern large language models such as GPT are very powerful and sophisticated [c1575], but the techniques used to build them are no longer secret. The open-source ecosystem has made it possible to study, replicate, and build upon state-of-the-art architectures.

[FIGURE: A timeline from 2017 to 2024 showing key milestones: Transformer paper (2017), GPT-1, GPT-2, GPT-3 (2020, few-shot learning), GPT-4 (closed-source), Llama 3 405B (open-weight, 2024)]

---

### Why Build from Scratch?

Given that powerful pre-trained models are freely available, a reasonable question is: why bother building one from scratch? Most existing online courses on building large language models focus on LLM application development rather than teaching how to build an LLM from scratch [c1590]. Running a pre-built model is straightforward; understanding what is happening inside it is not.

A common mistake learners make when studying large language models is directly switching to applications and running code without understanding how LLMs actually work [c1565]. This shortcut creates a fragile understanding—you can use the tool, but you cannot diagnose problems, adapt architectures, or reason about limitations.

The goal of this book is different: to teach how to build an entire large language model from scratch rather than focusing on deploying pre-built applications, to ensure deep understanding of LLM fundamentals [c1595]. Every design decision, every hyperparameter, every architectural choice will be motivated from first principles. By the time you finish, you will not just know *that* transformers use multi-head attention—you will know *why*, and you will have implemented it yourself.

---

### Chapter Summary

This chapter has established the conceptual landscape for everything that follows. Let us consolidate the key ideas:

- An LLM is a deep neural network designed to understand, generate, and respond to human-like text [c287], with model size measured in parameters [c290].
- LLMs sit within a hierarchy: AI ⊃ ML ⊃ Deep Learning ⊃ LLMs [c342], with Generative AI overlapping DL and LLMs across multiple modalities [c339].
- Earlier NLP models were task-specific specialists [c315]; LLMs are task-agnostic generalists [c323].
- LLM development follows a two-stage pipeline: pre-training on massive unlabeled text using next-word prediction [c23, c1593], followed by fine-tuning on smaller labeled datasets for specific tasks [c1593].
- The transformer architecture, introduced in 2017 [c1252], is the backbone of modern LLMs, with the attention mechanism as its core innovation [c3710].
- Pre-trained LLMs exhibit zero-shot and few-shot learning—the ability to perform tasks without explicit training examples [c3372, c3373]—a form of emergent behavior [c3422].
- GPT-3 (2020) formally established few-shot learning as a property of large language models [c3437]; GPT-4 represents the closed-source frontier [c3400].
- The open-source model landscape has matured rapidly, with performance now comparable to closed-source models on standard benchmarks [c3401].

In the next chapter, we begin Stage 1 of the development workflow: preparing text data and understanding how raw text is transformed into the numerical representations that a neural network can process.