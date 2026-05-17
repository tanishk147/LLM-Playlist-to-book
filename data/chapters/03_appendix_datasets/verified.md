## Datasets Used in This Series

Every machine learning project lives or dies by its data, and building a language model from scratch is no exception. This chapter catalogs every dataset that appears throughout this series—what each one contains, where it comes from, how large it is, and what role it plays. Having this reference in one place will save you from hunting through later chapters when you need to recall why a particular dataset was chosen or how to obtain it.

We cover four distinct categories: a small literary text used for pre-training demonstrations, a spam-classification corpus used for fine-tuning experiments, a compact instruction-following dataset used for instruction fine-tuning, and finally the massive corpora that were used to train GPT-3 in the real world. The first three are datasets you will actually work with in code; the last is context that helps you understand the scale gap between educational examples and production systems.

---

### The Verdict: Pre-Training Dataset

The primary dataset for pre-training demonstrations throughout this series is *The Verdict*, a short story by Edith Wharton [c3175]. The story was published in 1908 [c3176] and is in the public domain, making it freely available to download [c3744]. It is included directly in the companion repository for this series so you do not need to track it down yourself [c3177].

The choice of such a small text is deliberate. A short book is used for this example specifically to ensure fast execution on laptops [c3746]. You should be able to run every pre-training experiment in this series on consumer hardware in a reasonable amount of time, and *The Verdict* makes that possible.

In concrete terms, the text file contains **20,479 total characters** [c2911, c3815]. When tokenized using byte pair encoding (specifically the `tiktoken` encoder), the story produces **5,145 tokens** [c2725, c3818]. To put that in perspective, 5,145 tokens is an extremely short corpus for training a language model [c3819]. A production model would be trained on hundreds of billions of tokens. However, for the purposes of learning how the data pipeline, the model architecture, and the training loop all fit together, this dataset is entirely suitable [c3819].

[FIGURE: Bar chart comparing The Verdict's 5,145 tokens against GPT-3's 300 billion tokens, with a logarithmic y-axis to make both visible]

---

### SMS Spam Collection: Classification Fine-Tuning Dataset

When the series moves from pre-training to fine-tuning for classification, a different dataset takes center stage: the **SMS Spam Collection**. This corpus is used to demonstrate how a pre-trained language model can be adapted to perform binary text classification—distinguishing spam messages from legitimate ones (commonly called "ham").

The dataset contains a total of **5,572 messages** [c3632]. Of these, 425 spam messages were manually extracted from the Grumbletext website [c3623]. The dataset is distributed as a tab-separated values file named `SMSSpamCollection.tsv` [c3629], which can be loaded directly into a pandas DataFrame for manipulation [c3631].

One practical issue with spam datasets is class imbalance: there are typically far more legitimate messages than spam. To address this, the dataset is balanced before training. After balancing, the dataset contains **747 instances of each class** (ham and spam) [c3635], giving a total of 1,494 examples for the classification experiments.

---

### Instruction Fine-Tuning Dataset

The third dataset appears in the instruction fine-tuning portion of the series, where the goal is to teach a model to follow natural-language instructions rather than simply continue text. The dataset used for these demonstrations consists of **1,100 instruction–input–output pairs** [c588, c603, c691].

Each example in the dataset has three fields: an instruction describing the task, an optional input providing context, and an output giving the expected response. This format is the backbone of instruction fine-tuning and mirrors the structure used by much larger, publicly available datasets.

For comparison, the **Stanford Alpaca** repository contains a dataset of **52,000 instruction–output pairs** [c695]—roughly **50 times larger** than the 1,100-pair dataset used in these examples [c696]. The Alpaca dataset is a well-known reference point in the open-source instruction-tuning community. The smaller 1,100-pair dataset is used here for the same reason *The Verdict* is used for pre-training: it keeps training times manageable on a laptop while still illustrating every step of the fine-tuning pipeline.

[FIGURE: Side-by-side comparison of dataset sizes: 1,100 pairs (this series) vs. 52,000 pairs (Stanford Alpaca), shown as stacked bars]

---

### GPT-3 Pre-Training Corpora: Real-World Scale

To ground your intuition about what production pre-training actually looks like, it is worth examining the data that went into GPT-3. GPT-3 was trained on a total of **300 billion tokens** [c3395]. The training data was assembled from several sources, each weighted differently during training.

#### Common Crawl

The dominant source is **Common Crawl**, a free and open repository of web crawl data that has maintained over **250 million pages** spanning **17 years** since 2007 [c3391]. GPT-3 draws **410 billion tokens** from Common Crawl, which constitutes **60%** of the entire training dataset [c3389, c3421]. The sheer size of Common Crawl makes it the backbone of most large language model training runs.

The second major source is **WebText2**, an enhanced version of the original WebText corpus that covers Reddit submissions from 2005 to 2020 [c3393]. GPT-3 uses **19 billion words** from WebText2, accounting for **22%** of the training data [c3392, c3421].

#### Books and Wikipedia

The remaining **18–19%** of GPT-3's training data comes from books and Wikipedia [c3394]. Within the books component, the Books1 corpus alone contributes **12 billion tokens** at an **8% weight** [c3421].

The table below summarizes the composition of GPT-3's training data:

| Source | Tokens | Weight |
|---|---|---|
| Common Crawl | 410 billion | 60% |
| WebText2 | 19 billion words | 22% |
| Books1 | 12 billion | 8% |
| Books + Wikipedia (remainder) | — | ~10% |
| **Total** | **300 billion tokens** | **100%** |

[c3389, c3392, c3394, c3395, c3421]

Intuitively, the weighting scheme matters as much as the raw token counts. A source can be sampled more or less frequently than its size alone would suggest, allowing the training process to emphasize higher-quality text (such as curated books or filtered web pages) relative to noisier sources.

---

### Downloading and Licensing

*The Verdict* is in the public domain and is included directly in the series repository [c3177]. The SMS Spam Collection is a well-known benchmark dataset available for research use [c3629]. The 1,100-pair instruction dataset is provided alongside the series materials. Common Crawl is a free and open repository [c3391], though working with it at full scale requires significant storage and compute resources that go well beyond a laptop setup.

For the purposes of this series, you will only need to download *The Verdict* and the SMS Spam Collection. Everything else is either provided in the repository or is reference material describing production-scale systems rather than something you will run locally.

---

### Summary

| Dataset | Size | Purpose |
|---|---|---|
| *The Verdict* (Edith Wharton, 1908) | 20,479 chars / 5,145 tokens | Pre-training demonstrations |
| SMS Spam Collection | 5,572 messages (balanced: 747 per class) | Classification fine-tuning |
| Instruction dataset | 1,100 pairs | Instruction fine-tuning |
| Stanford Alpaca | 52,000 pairs | Reference / comparison |
| GPT-3 corpora | 300 billion tokens total | Production-scale context |

[c2911, c3818, c3632, c3635, c588, c695, c3395]

With these datasets in hand—and a clear sense of why each one was chosen—you are ready to move into the chapters that build the actual data-loading and tokenization pipelines that feed them into a model.
