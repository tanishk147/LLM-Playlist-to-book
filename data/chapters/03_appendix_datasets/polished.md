## Datasets Used in This Series

Every machine learning project lives or dies by its data, and building a language model from scratch is no exception. This chapter catalogs every dataset that appears throughout this series—what each one contains, where it comes from, how large it is, and what role it plays. Having this reference in one place will save you from hunting through later chapters when you need to recall why a particular dataset was chosen or how to obtain it.

We cover four distinct categories: a small literary text used for pre-training demonstrations, a spam-classification corpus used for classification fine-tuning, a compact instruction-following dataset used for instruction fine-tuning, and the massive corpora used to train GPT-3 in the real world. The first three are datasets you will actually work with in code; the last provides context for understanding the scale gap between educational examples and production systems.

---

### The Verdict: Pre-Training Dataset

The primary dataset for pre-training demonstrations throughout this series is *The Verdict*, a short story by Edith Wharton [c3175]. Published in 1908 [c3176] and now in the public domain, it is freely available to download [c3744] and is included directly in the companion repository for this series [c3177].

The choice of such a small text is deliberate: a short work is used specifically to ensure fast execution on laptops [c3746], so you can run every pre-training experiment on consumer hardware in a reasonable amount of time.

In concrete terms, the text file contains **20,479 total characters** [c2911, c3815]. When tokenized using byte pair encoding (specifically the `tiktoken` encoder), the story produces **5,145 tokens** [c2725, c3818]. That is an extremely short corpus by the standards of language model training [c3819]—a production model would be trained on hundreds of billions of tokens—but for learning how the data pipeline, model architecture, and training loop all fit together, it is entirely suitable [c3819].

[FIGURE: Bar chart comparing The Verdict's 5,145 tokens against GPT-3's 300 billion tokens, with a logarithmic y-axis to make both visible]

---

### SMS Spam Collection: Classification Fine-Tuning Dataset

When the series moves from pre-training to fine-tuning for classification, a different dataset takes center stage: the **SMS Spam Collection**. This corpus demonstrates how a pre-trained language model can be adapted to perform binary text classification—distinguishing spam messages from legitimate ones (commonly called "ham").

The dataset contains **5,572 messages** in total [c3632], of which 425 spam messages were manually extracted from the Grumbletext website [c3623]. It is distributed as a tab-separated values file named `SMSSpamCollection.tsv` [c3629], which can be loaded directly into a pandas DataFrame for manipulation [c3631].

One practical challenge with spam datasets is class imbalance: legitimate messages typically outnumber spam by a wide margin. To address this, the dataset is balanced before training. After balancing, each class contains **747 instances** [c3635], yielding 1,494 examples in total for the classification experiments.

---

### Instruction Fine-Tuning Dataset

The third dataset appears in the instruction fine-tuning portion of the series, where the goal is to teach a model to follow natural-language instructions rather than simply continue text. The dataset used for these demonstrations consists of **1,100 instruction–input–output pairs** [c588, c603, c691].

Each example has three fields: an instruction describing the task, an optional input providing context, and an output giving the expected response. This format is the backbone of instruction fine-tuning and mirrors the structure used by much larger, publicly available datasets.

For comparison, the **Stanford Alpaca** repository contains **52,000 instruction–output pairs** [c695]—roughly **50 times larger** than the 1,100-pair dataset used here [c696]. The smaller dataset is chosen for the same reason *The Verdict* is used for pre-training: it keeps training times manageable on a laptop while still illustrating every step of the fine-tuning pipeline.

[FIGURE: Side-by-side comparison of dataset sizes: 1,100 pairs (this series) vs. 52,000 pairs (Stanford Alpaca), shown as stacked bars]

---

### GPT-3 Pre-Training Corpora: Real-World Scale

To ground your intuition about what production pre-training actually looks like, it is worth examining the data that went into GPT-3. In total, GPT-3 was trained on **300 billion tokens** [c3395], assembled from several sources and weighted differently during training.

#### Common Crawl

The dominant source is **Common Crawl**, a free and open repository of web crawl data that has maintained over **250 million pages** spanning **17 years** since 2007 [c3391]. GPT-3 draws **410 billion tokens** from Common Crawl, constituting **60%** of the entire training dataset [c3389, c3421]—making it the backbone of the training run.

#### WebText2

The second major source is **WebText2**, an enhanced version of the original WebText corpus covering Reddit submissions from 2005 to 2020 [c3393]. GPT-3 uses **19 billion words** from WebText2, accounting for **22%** of the training data [c3392, c3421].

#### Books and Wikipedia

The remaining **18–19%** of GPT-3's training data comes from books and Wikipedia [c3394]. Within the books component, the Books1 corpus alone contributes **12 billion tokens** at an **8% weight** [c3421].

The table below summarizes the full composition of GPT-3's training data:

| Source | Tokens | Weight |
|---|---|---|
| Common Crawl | 410 billion | 60% |
| WebText2 | 19 billion words | 22% |
| Books1 | 12 billion | 8% |
| Books + Wikipedia (remainder) | — | ~10% |
| **Total** | **300 billion tokens** | **100%** |

[c3389, c3392, c3394, c3395, c3421]

It is worth noting that the weighting scheme matters as much as raw token counts. A source can be sampled more or less frequently than its size alone would suggest, allowing the training process to emphasize higher-quality text—such as curated books or filtered web pages—relative to noisier sources.

---

### Downloading and Licensing

*The Verdict* is in the public domain and is included directly in the series repository [c3177]. The SMS Spam Collection is a well-known benchmark dataset available for research use [c3629], and the 1,100-pair instruction dataset is provided alongside the series materials. Common Crawl is a free and open repository [c3391], though working with it at full scale requires storage and compute resources well beyond a typical laptop setup.

For the purposes of this series, you will only need *The Verdict* and the SMS Spam Collection. Everything else is either provided in the repository or serves as reference material describing production-scale systems rather than something you will run locally.

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

With these datasets cataloged—and a clear sense of why each was chosen—you are ready to move into the chapters that build the data-loading and tokenization pipelines that feed them into a model.