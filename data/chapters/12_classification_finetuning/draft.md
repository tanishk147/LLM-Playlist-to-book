## Classification Fine-Tuning: Building a Spam Detector

Fine-tuning is the process of adapting a pre-trained model to a specific task by training the model on additional data [c2337, c3575]. Rather than training a language model from scratch—which requires enormous compute and data—we start from a foundation model that has already learned rich representations of language, then nudge its parameters toward a narrower objective [c169]. In this chapter, that objective is spam detection: given a text message, predict whether it is spam or not spam.

We will build the entire classification fine-tuning pipeline from scratch [c2894]. The journey follows three main stages: dataset preparation, model initialization and modification, and fine-tuning with evaluation [c3615]. Along the way we will implement accuracy and loss utilities, a training loop, and finally run inference on new text the model has never seen [c3618].

---

### 12.1 What Is Classification Fine-Tuning?

Before writing any code it is worth being precise about what kind of fine-tuning we are doing. There are two broad categories: instruction fine-tuning and classification fine-tuning [c3596, c2826].

**Instruction fine-tuning** trains the model to follow natural-language instructions. An example prompt might be: *"Is the following text spam? Answer with a yes or no."* [c3598]. Because the model must search the entire vocabulary and perform non-specific actions, instruction fine-tuning requires greater computational power [c3605].

**Classification fine-tuning** is different. The model classifies inputs into predefined categories without being given explicit instructions [c3600]. In our spam detector, no instruction is given to the LLM; only the input text is provided, and the model outputs a class label—spam or not spam [c3601]. The trade-off is that classification fine-tuning can only handle a narrow set of prompts because the model is constrained to predefined categories [c3607]. For our purposes, that constraint is exactly what we want.

Classification fine-tuning is also used for tasks like sentiment classification into categories such as angry, sad, or happy [c3602], so the pattern you learn here generalises well beyond spam detection.

[FIGURE: Two-column diagram contrasting instruction fine-tuning (prompt → free-form text output) with classification fine-tuning (text input → fixed class label), with spam detection as the example on the right]

#### 12.1.1 The Role of the Foundation Model

A foundation model is a pre-trained large language model trained on large-scale data that can be fine-tuned for downstream tasks [c169]. During fine-tuning, the weights, biases, and parameters of this pre-trained model are updated through training on additional data [c3586]. LLM fine-tuning specifically involves the additional training of a pre-existing model that has previously acquired patterns and features from an extensive dataset, using a smaller domain-specific dataset [c3587]. The fine-tuning process takes that large pre-trained model, trains it further on a custom dataset, and produces a fine-tuned large language model [c3588].

In our project we will use GPT-2 pre-trained weights and then run the training procedure again on the SMS Spam Collection dataset [c152]. Without fine-tuning, the pre-trained GPT model cannot correctly answer spam classification prompts even when given explicit instructions [c159, c2851]. Fine-tuning is what bridges that gap.

---

### 12.2 The SMS Spam Collection Dataset

The classification fine-tuning project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository [c48]. The project involves predicting whether text messages are spam or not spam [c2824], and the label used to denote non-spam emails in this dataset is **HAM** [c2341].

#### 12.2.1 Class Imbalance and Balancing

The original SMS Spam Collection dataset is imbalanced, with approximately 4,800 no-spam messages and 747 spam messages [c50]. Training on a heavily imbalanced dataset can cause a model to simply predict the majority class for every input and still achieve high nominal accuracy. To avoid this, we balance the dataset so that both classes are equally represented.

#### 12.2.2 Downloading the Dataset

The following code downloads the dataset, unzips it, and renames the file with a `.csv` extension so it can be read by pandas.

# Source: [c3630]
```python
# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()

# Downloading the file
with urllib.request.urlopen(url, context=ssl_context) as response:
    with open(zip_path, "wb") as out_file:
        out_file.write(response.read())

# Unzipping the file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extracted_path)

# Add .csv file extension
original_file_path = Path(extracted_path) / "SMSSpamCollection"
os.rename(original_file_path, data_file_path)
print(f"File downloaded and saved as {data_file_path}")
```

#### 12.2.3 Splitting into Train, Validation, and Test Sets

Once the dataset is balanced, we split it into training, validation, and test partitions. The following utility function performs a random split using a 70 / 20 / 10 ratio.

# Source: [c3640]
```python
def random_split(df, train_frac=0.7, validation_frac=0.2):
    train_df = df.sample(frac=train_frac)
    remaining = df.drop(train_df.index)
    validation_df = remaining.sample(frac=validation_frac/(1-train_frac))
    test_df = remaining.drop(validation_df.index)
    return train_df, validation_df, test_df
```

Calling this function on the balanced dataframe produces the following split sizes:

# Source: [c3644]
```python
train_df, validation_df, test_df = random_split(balanced_df)
print(len(train_df))        # 1045
print(len(validation_df))   # 149
print(len(test_df))         # 300
```

The training set contains 1,045 examples, the validation set 149, and the test set 300 [c3644]. The test set is held out entirely until the final stage of the project, when we evaluate the fine-tuned model on data it has never seen before [c137].

---

### 12.3 Building PyTorch Datasets and Data Loaders

With the data split in hand, we need to wrap it in PyTorch abstractions that the training loop can consume. A **Dataset** stores the samples and their corresponding labels [c2350]. A **DataLoader** wraps an iterable around the Dataset to enable easy access to the samples [c2351]. A PyTorch dataset specifies how data is loaded and processed before instantiating a data loader [c2374].

#### 12.3.1 Tokenization

To feed text into the model we must convert it to token IDs. We use **TikToken**, a tokenizer library developed by OpenAI that converts sentences into token IDs [c2364]. TikToken uses a **byte pair encoder**, a sub-word tokenizer that breaks words into smaller units rather than mapping each word to a single token [c2365]. A **vocabulary** in tokenization is a dictionary mapping tokens to their corresponding token IDs [c2370].

Text messages in the dataset vary in length. To batch them together efficiently, we pad all sequences to the same length. The **end of text token** is a special token used to mark the end of a document or sentence in tokenization [c2363]; it is also a natural choice for the padding token because the model has already seen it during pre-training and treats it as a neutral filler.

#### 12.3.2 The Spam Dataset Class

The spam dataset class performs three steps: identifying the longest sequence in the training dataset, converting each text message into token IDs, and padding all sequences to match the longest sequence length [c2375].

[FIGURE: Diagram showing three text messages of different lengths being tokenized and then right-padded with the end-of-text token to the length of the longest sequence]

Intuitively, padding to the longest sequence in the *training* set (rather than the longest sequence in each batch) ensures that all data loaders produce tensors of the same shape, which simplifies the training loop.

The large language model architecture takes input text and outputs a classification of whether the email is spam or not spam [c2348]. The dataset class is the bridge between raw CSV rows and the tensors that architecture expects.

---

### 12.4 Loading Pre-Trained GPT-2 Weights

Before modifying the architecture we need to load the pre-trained GPT-2 weights. The project uses a GPT-based LLM architecture that will be modified for classification fine-tuning [c2837].

#### 12.4.1 Downloading GPT-2

The following function downloads GPT-2 model files from OpenAI's public blob storage. It accepts a model size string and a local directory path.

# Source: [c2844]
```python
def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    # Downloads GPT-2 model files from OpenAI public blob storage
```

#### 12.4.2 Loading TensorFlow Checkpoints

GPT-2 weights are distributed as TensorFlow checkpoints. The following function loads them into a nested Python dictionary that we can then copy into our PyTorch model.

# Source: [c2847]
```python
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Recursively loads TensorFlow checkpoint variables into nested params dictionary
    # Handles token embeddings, positional embeddings, transformer blocks, and layer norm parameters
```

Once the weights are loaded into our GPT model, we have a fully functional language model. The next step is to modify it for classification.

---

### 12.5 Modifying the GPT Architecture for Classification

A pre-trained GPT model has an output head that maps the hidden state of size 768 to a vocabulary of 50,257 tokens. For spam classification we do not need 50,257 output dimensions—we need exactly two: one for spam and one for not spam.

#### 12.5.1 The Classification Head

A **classification head** is a small linear layer added on top of a pre-trained language model to adapt it for classification tasks by mapping hidden states to class logits [c2857]. To adapt GPT-2 for text classification, we replace the original 768→50257 output layer with a 768→2 linear layer that outputs logits for two classes (spam vs. no spam) [c2854]. More generally, for fine-tuning a language model for classification, we replace the original output layer that maps to vocabulary size with a smaller output layer that maps to the number of classes [c2867].

In code, this replacement looks like:

# Source: [c2873]
```python
model.output_head = Linear(in_features=768, out_features=num_classes)
```

The original output layer was replaced with a classification head that takes input of size 768 and outputs 2 values for spam or no spam classification [c60].

[FIGURE: Side-by-side diagram of the original GPT-2 output head (768→50257) versus the new classification head (768→2), with the rest of the transformer stack unchanged]

#### 12.5.2 Which Token's Output Do We Use?

A GPT model processes a sequence and produces one output vector per input token. For a 4-token input, the model produces 4 output rows. For spam classification, we focus fine-tuning on the last output token rather than all output rows, since the last token contains all necessary information [c2879]. In PyTorch, extracting the last token's output from the sequence dimension is written as:

# Source: [c2880]
```python
outputs[:, -1, :]
```

Intuitively, the last token has attended to every preceding token through the causal self-attention mechanism, so its hidden state is a compressed summary of the entire input. Extracting the last output token from a 4-token sequence yields a vector with two class logits (e.g., -3.5898 and 3.9902 for spam/not-spam classification) [c2882].

---

### 12.6 Freezing Layers and Selective Unfreezing

Loading pre-trained weights gives us a strong starting point, but we do not want to update every parameter during fine-tuning. Updating all parameters risks destroying the general language representations the model learned during pre-training—a phenomenon sometimes called catastrophic forgetting. Parameter efficient fine-tuning updates only a subset of parameters while freezing the rest, reducing the number of trainable parameters and memory requirements [c3610].

#### 12.6.1 What to Freeze and What to Unfreeze

Three components will be fine-tuned: the classification head, the final (12th) transformer block, and the final layer normalization module [c2863]. The three sets of parameters that are fine-tuned are: the final output head, the final transformer block, and the final layer normalization module [c2872]. Currently trained parameters include the input classification head, the 12th transformer block, and the final normalization layer [c135].

The practical workflow is:

1. Freeze all model parameters.
2. Selectively unfreeze the final transformer block.
3. Selectively unfreeze the final layer normalization.
4. The new classification head is randomly initialised and therefore always trainable.

Unfreezing the final transformer block:

# Source: [c2875]
```python
for param in model.transformer.h[-1].parameters():
    param.requires_grad = True
```

Unfreezing the final layer normalization:

# Source: [c2876]
```python
for param in model.ln_f.parameters():
    param.requires_grad = True
```

Before training, the parameters in the new output head, final transformer block, and final layer normalization are still random and have not been trained on the classification dataset [c2878]. The training loop will update exactly these parameters while leaving the rest of the network frozen.

[FIGURE: Vertical stack of GPT-2 layers with the bottom layers (embeddings, transformer blocks 1–11) shaded grey (frozen) and the top layers (transformer block 12, layer norm, classification head) highlighted in colour (trainable)]

#### 12.6.2 Parameter-Efficient Methods: LoRA and QLoRA

For reference, two popular parameter-efficient fine-tuning methods go further than simple layer freezing. **LoRA** is a fine-tuning method where instead of fine-tuning all weights in a weight matrix, two smaller matrices that approximate the larger matrix are fine-tuned [c3611]. **QLoRA** is quantized LoRA, a more memory-efficient iteration of LoRA that quantizes the weights of LoRA adapters to lower precision [c3612]. These methods are worth knowing about, though the chapter's implementation uses selective layer unfreezing rather than LoRA.

---

### 12.7 Evaluation Metrics: Accuracy and Loss

Before writing the training loop we need two evaluation utilities: one for accuracy and one for loss.

#### 12.7.1 Classification Accuracy

Classification accuracy measures the percentage of correct predictions across a dataset [c67]. The formula is:

$$\text{Accuracy} = \frac{\text{correct\_predictions}}{\text{total\_number\_of\_examples}}$$ [c80]

**Predicted labels** are the class predictions generated by applying argmax to model logits [c70]. **Output labels** (also called target labels) represent the true values in the dataset [c69].

The following function iterates over a data loader and accumulates correct predictions:

# Source: [c71]
```python
def calculate_accuracy(loader, num_batches=None):
    if num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))
    correct_predictions = 0
    num_examples = 0
    for batch in loader:
        logits = model(batch)
        predicted_labels = argmax(logits)
        target_labels = batch.labels
        correct_predictions += sum(predicted_labels == target_labels)
        num_examples += len(batch)
    accuracy = correct_predictions / num_examples
    return accuracy
```

The `num_batches` parameter allows us to evaluate on a subset of the data during training, which saves computation time [c112].

#### 12.7.2 Cross-Entropy Loss

The loss function for classification is **categorical cross-entropy**, defined as:

$$\mathcal{L} = -\sum_{i} y_i \log(p_i)$$ [c88]

where $y_i$ is the true value and $p_i$ is the predicted probability [c88]. In PyTorch, this is computed with:

# Source: [c93]
```python
torch.nn.functional.cross_entropy()
```

Cross-entropy loss is the standard choice for multi-class classification because it penalises confident wrong predictions heavily and rewards confident correct predictions.

---

### 12.8 The Fine-Tuning Training Loop

With the dataset, model, and metrics in place, we can write the training loop. The model passed to the training function is a GPT model class with a modified architecture that includes a classification head on top [c118].

#### 12.8.1 Epochs and Batches

**One epoch** is one complete pass through the entire dataset [c96]. Training typically runs for several epochs, updating model parameters after each batch via backpropagation.

#### 12.8.2 Evaluation Frequency

Printing metrics after every single batch would flood the console and slow training. Instead, we use two parameters to control evaluation cadence:

- **Evaluation frequency** specifies how many batches must be processed before printing training and validation loss metrics [c110]. Step 6 of the fine-tuning training process: after every 50 batches (evaluation frequency), print training loss and validation loss using the `evaluate_model` function [c113].
- **Evaluation iteration** specifies the number of batches to use for evaluation; using fewer batches (e.g., 5 or 10) saves computation time during evaluation [c112].

#### 12.8.3 Per-Epoch Accuracy Reporting

Step 7 of the fine-tuning training process: after every epoch, calculate and print training accuracy and validation accuracy [c114]. This gives a coarser but more stable signal than per-batch loss, and it is the metric we ultimately care about for a classification task.

[FIGURE: Timeline diagram showing one training epoch divided into batches, with loss printed every 50 batches and accuracy printed at the end of each epoch]

#### 12.8.4 Training Loop Structure

The training loop follows this sequence for each epoch:

1. Iterate over batches in the training data loader.
2. Compute logits by passing the batch through the model.
3. Extract the last token's output: `outputs[:, -1, :]` [c2880].
4. Compute cross-entropy loss against the target labels using `torch.nn.functional.cross_entropy()` [c93].
5. Backpropagate and update only the unfrozen parameters.
6. Every 50 batches, evaluate and print training loss and validation loss [c113].
7. At the end of each epoch, evaluate and print training accuracy and validation accuracy [c114].

The remaining steps after architecture setup are to implement loss and accuracy evaluation utilities, fine-tune the model, and test it on new data [c2893].

---

### 12.9 Saving and Loading the Fine-Tuned Model

After training, we want to persist the model weights so we do not have to retrain from scratch every time. PyTorch provides `torch.save` for this purpose:

# Source: [c156]
```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)
```

Fine-tuning is the process of changing model parameters and retraining on a specific dataset so the model performs well on that particular task [c160]. Saving the fine-tuned weights captures the result of that process in a file that can be loaded later for inference or further fine-tuning.

---

### 12.10 Evaluating on the Test Set and Running Inference

The final stage of the spam classification project involves testing the fine-tuned model on new data that it has not seen before [c137]. The email classification project involves training a large language model to classify emails as spam or no spam [c3614], and the test set we held out in Section 12.2.3 is the right tool for this evaluation.

#### 12.10.1 Why a Held-Out Test Set Matters

The validation set is used during training to monitor generalisation and tune hyperparameters. The test set is touched only once, at the very end, to give an unbiased estimate of real-world performance. Using the test set during training would leak information and produce an optimistic accuracy estimate.

#### 12.10.2 Inference on New Text

For inference on a single new message, the pipeline is:

1. Tokenize the input text using TikToken [c2364].
2. Pad or truncate to the expected sequence length.
3. Pass the token IDs through the fine-tuned model.
4. Extract the last token's output: `outputs[:, -1, :]` [c2880].
5. Apply argmax to the two logits to obtain the predicted label [c70].

Without fine-tuning, the pre-trained GPT model could not correctly answer spam classification prompts even when given explicit instructions [c159]. After fine-tuning, the model has learned to map the last-token hidden state to one of two classes, and inference is fast and deterministic.

---

### 12.11 Putting It All Together

Let us review the complete pipeline we have built:

| Stage | What we did |
|---|---|
| **Dataset** | Downloaded SMS Spam Collection, balanced classes, split 70/20/10 |
| **Tokenization** | Used TikToken BPE tokenizer, padded to max sequence length |
| **PyTorch wrappers** | Implemented `Dataset` and `DataLoader` |
| **Model loading** | Downloaded GPT-2 weights, loaded TF checkpoint into PyTorch |
| **Architecture mod** | Replaced 768→50257 head with 768→2 classification head |
| **Layer freezing** | Froze all layers; unfroze final transformer block and layer norm |
| **Metrics** | Implemented accuracy and cross-entropy loss utilities |
| **Training loop** | Fine-tuned for multiple epochs with periodic loss/accuracy reporting |
| **Persistence** | Saved fine-tuned weights with `torch.save` |
| **Evaluation** | Evaluated on held-out test set; ran inference on new messages |

Classification fine-tuning is one category of fine-tuning, focused on adapting models for classification tasks [c161]. The pattern established here—load a foundation model, swap the output head, freeze most layers, fine-tune a small subset, evaluate rigorously—applies to any classification problem you might encounter: sentiment analysis, topic labelling, intent detection, and more.

#### 12.11.1 Comparison with Instruction Fine-Tuning

It is worth pausing to appreciate what we *did not* have to do. Instruction fine-tuning requires the model to generate free-form text conditioned on a natural-language instruction [c3598], which demands more compute [c3605] and a much larger labelled dataset. Classification fine-tuning constrains the output space to a fixed set of labels [c3600], which makes training faster, evaluation simpler, and deployment more predictable. The cost is flexibility: the model can only handle a narrow set of prompts [c3607]. For production spam filters, that is a perfectly acceptable trade-off.

For scenarios where you need both efficiency and flexibility, parameter-efficient methods like LoRA [c3611] and QLoRA [c3612] offer a middle ground by fine-tuning only a small number of adapter parameters rather than full layer weights [c3610].

---

### 12.12 Summary

In this chapter we built a complete classification fine-tuning pipeline for spam detection. We started from the SMS Spam Collection dataset [c48], balanced it to address class imbalance [c50], and split it into train, validation, and test sets [c3644]. We wrapped the data in PyTorch `Dataset` and `DataLoader` objects [c2350, c2351], using TikToken for tokenization [c2364] and padding sequences to uniform length [c2375].

On the model side, we loaded pre-trained GPT-2 weights [c2844], replaced the vocabulary-sized output head with a two-class classification head [c2854, c2873], and selectively unfroze the final transformer block [c2875] and final layer normalization [c2876] for fine-tuning [c2863]. We implemented accuracy [c71] and cross-entropy loss [c93] utilities, wrote a training loop with configurable evaluation frequency [c110, c113], and saved the resulting model [c156]. Finally, we evaluated on the held-out test set [c137] and ran inference on new messages [c2880].

The next chapter extends these ideas to instruction fine-tuning, where the model learns to follow natural-language directives rather than predict a fixed class label.