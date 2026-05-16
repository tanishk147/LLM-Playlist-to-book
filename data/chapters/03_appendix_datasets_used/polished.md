## Datasets Used in This Series

Before diving into model architecture and training loops, it helps to know exactly what data we will be working with throughout this book. This chapter catalogs every dataset that appears in the series, explains where each one comes from, and provides the key statistics you will need to follow along. There are four datasets in total: a short story used for pre-training demonstrations, a spam-classification corpus, an instruction fine-tuning set, and a much larger reference dataset from Stanford. Each is introduced at the appropriate point in the book, but having them all in one place makes it easy to return here for a quick reminder.

---

### The Verdict — Pre-training Demonstration Data

The simplest dataset in the series is a short piece of fiction. The text used for pre-training demonstrations is *The Verdict*, a short story by Edith Wharton [c3175]. Published in 1908 [c3176], the story is in the public domain and available for free download; it is also included directly in the companion repository [c3177].

> **A note on publication year:** One source in the playlist gives the publication year as 1906 [c3743] while two others give 1908 [c2723, c3176]. The 1908 figure appears more frequently and is used here. [GAP: definitive publication year needs verification against a bibliographic source.]

The story is deliberately small. The full text runs to roughly 20,000 characters [c3745], and when tokenized with byte-pair encoding (BPE) it produces **5,145 tokens** [c2725] — small enough to fit comfortably in memory and to train on quickly, making it ideal for illustrating concepts without requiring a GPU cluster.

[FIGURE: Bar chart comparing the token count of The Verdict (~5,145 tokens) against a typical large pre-training corpus, to give a sense of scale]

Because the dataset is so compact, every experiment that uses it is meant to demonstrate *mechanics*, not to produce a state-of-the-art language model. Imperfect output from a model trained on this data is expected and intentional.

---

### Real-World Pre-training Corpora — GPT-3 as a Reference Point

To put *The Verdict* in perspective, it is worth examining what a production-scale pre-training dataset actually looks like. GPT-3 was trained on a total of **300 billion tokens** [c3395], assembled from several sources:

| Source | Tokens | Share of training data |
|---|---|---|
| Common Crawl | 410 billion (filtered down) | 60 % [c3389] |
| WebText2 | 19 billion words | 22 % [c3392] |
| Books + Wikipedia | — | 18–19 % [c3394] |

**Common Crawl** is a free, open repository of web-crawl data maintained since 2007, spanning more than 250 million pages across 17 years [c3391]. **WebText2** is an enhanced version of the original WebText corpus, covering Reddit submissions from 2005 to 2020 [c3393].

These figures are included not because you will download Common Crawl for the exercises in this book, but to ground your intuition: the 5,145-token *Verdict* is a toy by comparison, and the techniques you learn on it scale directly to corpora many orders of magnitude larger.

---

### SMS Spam Collection — Classification Fine-tuning Data

The second dataset appears when the book covers fine-tuning a pre-trained model for text classification. The task is binary spam detection, and the data comes from the **SMS Spam Collection**, a well-known benchmark from the UCI Machine Learning Repository.

#### Raw Dataset

The SMS Spam Collection contains **5,572 messages** in total [c3632], of which **425 spam messages** were manually extracted from the Grumbletext website [c3623]. The dataset is downloaded and saved as a tab-separated values file named `SMSSpamCollection.tsv` [c3629], which can then be loaded into a pandas DataFrame for convenient manipulation [c3631].

#### Balancing the Classes

Raw spam datasets are almost always imbalanced — legitimate messages far outnumber spam. To prevent a model from simply learning to predict the majority class, the dataset is balanced. After balancing, each class (ham and spam) contains **747 instances** [c3635], for a total of **1,494 examples** [c2397].

Those examples are then split into training, validation, and test sets and wrapped in data loaders. The resulting loader dimensions are:

| Split | Batches | Samples per batch | Tokens per sample |
|---|---|---|---|
| Training | 130 | 8 | 120 [c2404] |
| Validation | 19 | 8 | 120 [c2406] |
| Test | 38 | 8 | 120 [c2405] |

The test loader contains exactly **38 batches**, corresponding to 20 % of the total dataset [c2401]. Because the test set is twice the size of the validation set (20 % vs. 10 %), the number of test batches is exactly twice the number of validation batches [c2402].

[FIGURE: Pie chart showing the 70/10/20 train/validation/test split of the balanced SMS Spam Collection (1,494 total examples)]

---

### Instruction Fine-tuning Dataset — 1,100 Alpaca-Format Pairs

The third dataset is used when the book covers instruction fine-tuning — the process of teaching a pre-trained model to follow natural-language instructions. It contains **1,100 instruction–response pairs** [c205] formatted in the Alpaca style [c691].

#### Splits

The 1,100 examples are divided using an **85 % / 10 % / 5 %** train / validation / test split:

| Split | Examples |
|---|---|
| Training | 935 [c207] |
| Validation | 55 [c208] |
| Test | 110 [c209] |

After loading, the full dataset of 1,100 entries can be accessed by index [c191].

#### Why Such a Small Dataset?

The 1,100-pair size is a deliberate choice, allowing the fine-tuning pipeline to be demonstrated end-to-end on hardware that most readers have access to. For reference, the **Stanford Alpaca** repository contains **52,000 instruction–output pairs** [c695] — roughly **50 times larger** than the dataset used here [c696]. Using 85 % of 1,100 pairs for training yields approximately 800 training examples [c692] (the exact figure after the split is 935 [c207]; ~800 is a convenient approximation for quick mental arithmetic).

[FIGURE: Side-by-side bar chart comparing the 1,100-pair instruction dataset used in this book against the Stanford Alpaca 52,000-pair dataset]

Once you have worked through the book's examples, swapping in the Stanford Alpaca dataset is a natural next step for improving model quality. The pipeline code does not change — only the data loader needs to point at the larger file.

---

### Summary

The table below provides a quick reference for all datasets covered in this chapter:

| Dataset | Task | Size | Notes |
|---|---|---|---|
| *The Verdict* (Edith Wharton, 1908) | Pre-training demo | ~20,000 chars / 5,145 BPE tokens [c2725, c3745] | Public domain; included in repo [c3177] |
| SMS Spam Collection | Binary classification | 5,572 raw → 1,494 balanced [c3632, c2397] | Saved as `.tsv`; balanced to 747 per class [c3635] |
| Instruction fine-tuning set | Instruction following | 1,100 pairs [c205] | 85/10/5 split → 935/55/110 [c207, c208, c209] |
| Stanford Alpaca | Instruction following (reference) | 52,000 pairs [c695] | 50× larger than the in-book dataset [c696] |

Each dataset is introduced again with full context when it first appears in its relevant chapter. This chapter exists so you can always find the raw numbers in one place, without hunting through later material.