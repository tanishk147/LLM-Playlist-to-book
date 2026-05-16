## Classification Fine-Tuning: Building a Spam Detector

Fine-tuning is the process of adapting a pre-trained model to a specific task by training it on additional data [c2337, c3575]. Rather than training a language model from scratch—which requires enormous compute and data—we start from a foundation model that has already learned rich representations of language, then nudge its parameters toward a narrower objective [c169].

We will build the entire classification fine-tuning pipeline from scratch [c2894]. The journey follows three main stages: dataset preparation, model initialization and modification, and fine-tuning with evaluation [c3615]. Along the way we will implement accuracy and loss utilities, a training loop, and finally run inference on text the model has never seen [c3618].

---

### 12.1 What Is Classification Fine-Tuning?

LLM fine-tuning has two broad categories: instruction fine-tuning and classification fine-tuning [c3596, c2826].

**Instruction fine-tuning** trains the model to follow natural-language instructions. An example prompt might be: *"Is the following text spam? Answer with a yes or no."* [c3598]. Because the model must search the entire vocabulary and perform non-specific actions, instruction fine-tuning requires greater computational power than its classification counterpart [c3605].

**Classification fine-tuning**, by contrast, trains the model to classify inputs into predefined categories without being given explicit instructions [c3600]. In our spam detector, no instruction is given to the LLM; only the input text is provided, and the model outputs a class label—spam or not spam [c3601]. The trade-off is that classification fine-tuning can only handle a narrow set of prompts because the model is constrained to predefined categories [c3607]. Beyond spam detection, the same approach applies to tasks like sentiment classification into categories such as angry, sad, or happy [c3602], so the pattern generalises well.

#### 12.1.1 The Role of the Foundation Model

A foundation model is a pre-trained large language model trained on large-scale data that can be fine-tuned for downstream tasks [c169]. During fine-tuning, the weights, biases, and parameters of this pre-trained model are updated through training on additional data [c3586]. More precisely, LLM fine-tuning involves the additional training of a pre-existing model that has already acquired patterns and features from an extensive dataset, now applied to a smaller domain-specific dataset [c3587]. The result is a fine-tuned large language model tailored to the target task [c3588].

In this project we use GPT-2 pre-trained weights and then run the training procedure again on the SMS Spam Collection dataset [c152]. Without fine-tuning, the pre-trained GPT model cannot correctly answer spam classification prompts even when given explicit instructions [c159, c2851].

---

### 12.2 The SMS Spam Collection Dataset

The classification fine-tuning project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository [c48]. The task is to predict whether a text message is spam or not spam [c2824]; the label used to denote non-spam messages in this dataset is **HAM** [c2341].

#### 12.2.1 Class Imbalance and Balancing

The original dataset is imbalanced, containing approximately 4,800 non-spam messages and only 747 spam messages [c50]. Training on such a skewed distribution can cause a model to simply predict the majority class for every input and still achieve high nominal accuracy, so we balance the dataset before splitting.

#### 12.2.2 Downloading the Dataset

The following code downloads the dataset, unzips it, and renames the file with a `.csv` extension so it can be read by pandas [c3630]:

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

With the balanced dataset in hand, we split it into three subsets [c3640]:

```python
def random_split(df, train_frac=0.7, validation_frac=0.2):
    train_df = df.sample(frac=train_frac)
    remaining = df.drop(train_df.index)
    validation_df = remaining.sample(frac=validation_frac/(1-train_frac))
    test_df = remaining.drop(validation_df.index)
    return train_df, validation_df, test_df
```

Calling this function on the balanced dataset yields [c3644]:

```python
train_df, validation_df, test_df = random_split(balanced_df)
print(len(train_df))        # 1045
print(len(validation_df))   # 149
print(len(test_df))         # 300
```

The training set contains 1,045 examples, the validation set 149, and the test set 300 [c3644]. The test set is held out entirely until the final stage of the project, when we evaluate the fine-tuned model on data it has never seen before [c137].

---

### 12.3 Building PyTorch Datasets and Data Loaders

With the data split in hand, we wrap it in PyTorch abstractions that the training loop can consume. A **Dataset** stores the samples and their corresponding labels [c2350]; a **DataLoader** wraps an iterable around the Dataset to enable easy access to those samples [c2351]. More specifically, a PyTorch dataset specifies how data is loaded and processed before a data loader is instantiated [c2374].

#### 12.3.1 Tokenization

To feed text into the model we must first convert it to token IDs. We use **TikToken**, a tokenizer library developed by OpenAI that converts sentences into token IDs [c2364]. TikToken employs a **byte pair encoder**—a sub-word tokenizer that breaks words into smaller units rather than mapping each word to a single token [c2365]. The resulting **vocabulary** is a dictionary mapping tokens to their corresponding token IDs [c2370].

Because text messages in the dataset vary in length, we pad all sequences to the same length so they can be batched efficiently. The **end-of-text token** is a special token used to mark the end of a document or sentence [c2363]; it also serves as a natural padding token because the model has already encountered it during pre-training and treats it as neutral filler.

#### 12.3.2 The Spam Dataset Class

The spam dataset class performs three steps: identifying the longest sequence in the training dataset, converting each text message into token IDs, and padding all sequences to match that longest length [c2375].

[FIGURE: Diagram showing three text messages of different lengths being tokenized and then right-padded with the end-of-text token to the length of the longest sequence]

Padding to the longest sequence in the *training* set—rather than the longest sequence in each individual batch—ensures that all data loaders produce tensors of the same shape, which simplifies the training loop. The large language model architecture then takes this fixed-length input and outputs a classification of whether the message is spam or not spam [c2348].

---

### 12.4 Loading Pre-Trained GPT-2 Weights

The project uses a GPT-based LLM architecture that will be modified for classification fine-tuning [c2837]. Loading pre-trained weights rather than training from random initialization gives the model a strong linguistic foundation before we specialize it for spam detection.

#### 12.4.1 Downloading GPT-2

The `download_and_load_gpt2` function accepts a model size string and a local directory path, then retrieves the corresponding files from OpenAI's public storage [c2844]:

```python
def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    # Downloads GPT-2 model files from OpenAI public blob storage
```

#### 12.4.2 Loading TensorFlow Checkpoints

OpenAI released GPT-2 weights as TensorFlow checkpoints. The following function loads them into a nested Python dictionary that we can then copy into our PyTorch model [c2847]:

```python
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Recursively loads TensorFlow checkpoint variables into nested params dictionary
    # Handles token embeddings, positional embeddings, transformer blocks, and layer norm parameters
```

---

### 12.5 Modifying the GPT Architecture for Classification

With pre-trained weights loaded, we make two targeted changes to the architecture: replacing the output head and selecting which token's representation to classify.

#### 12.5.1 The Classification Head

A **classification head** is a small linear layer added on top of a pre-trained language model to adapt it for classification tasks by mapping hidden states to class logits [c2857]. To adapt GPT-2 for text classification, we replace the original 768→50257 output layer with a 768→2 linear layer that outputs logits for two classes—spam and not spam [c2854]. More generally, for any classification fine-tuning task, the original output layer that maps to vocabulary size is replaced with a smaller layer that maps to the number of target classes [c2867].

In code, this replacement looks like [c2873]:

```python
model.output_head = Linear(in_features=768, out_features=num_classes)
```

The new classification head takes an input of size 768 and outputs 2 values for spam or not-spam classification [c60].

[FIGURE: Side-by-side diagram of the original GPT-2 output head (768→50257) versus the new classification head (768→2), with the rest of the transformer stack unchanged]

#### 12.5.2 Which Token's Output Do We Use?

Rather than aggregating all output positions, we focus fine-tuning on the last output token, since it contains all the information accumulated over the full sequence [c2879]:

```python
outputs[:, -1, :]
```

Extracting the last output token from a 4-token sequence yields a vector with two class logits—for example, -3.5898 and 3.9902 for spam/not-spam classification [c2882].

---

### 12.6 Freezing Layers and Selective Unfreezing

Loading pre-trained weights gives us a strong starting point, but updating every parameter during fine-tuning risks destroying the general language representations the model learned during pre-training—a phenomenon sometimes called catastrophic forgetting. Parameter-efficient fine-tuning addresses this by updating only a subset of parameters while freezing the rest, reducing both the number of trainable parameters and memory requirements [c3610].

#### 12.6.1 What to Freeze and What to Unfreeze

Three components are fine-tuned: the classification head, the final (12th) transformer block, and the final layer normalization module [c2863, c2872, c135]. All other layers remain frozen throughout training.

Unfreezing the final transformer block [c2875]:

```python
for param in model.transformer.h[-1].parameters():
    param.requires_grad = True
```

Unfreezing the final layer normalization [c2876]:

```python
for param in model.ln_f.parameters():
    param.requires_grad = True
```

Before training begins, the parameters in the new output head, final transformer block, and final layer normalization are still at their initial (random or pre-trained) values and have not been trained on the classification dataset [c2878]. The training loop will update exactly these parameters while leaving the rest of the network frozen.

#### 12.6.2 Parameter-Efficient Methods: LoRA and QLoRA

Two popular parameter-efficient fine-tuning methods go further than simple layer freezing. **LoRA** fine-tunes two smaller matrices that approximate the larger weight matrix instead of updating all weights directly [c3611]. **QLoRA** extends this idea by quantizing the LoRA adapter weights to lower precision, making it even more memory-efficient [c3612]. Both methods are instances of the broader parameter-efficient fine-tuning paradigm [c3608].

---

### 12.7 Evaluation Metrics: Accuracy and Loss

Before writing the training loop we need two evaluation utilities: one for accuracy and one for loss.

#### 12.7.1 Classification Accuracy

Classification accuracy measures the percentage of correct predictions across a dataset [c67], computed as:

$$\text{Accuracy} = \frac{\text{correct\_predictions}}{\text{total\_number\_of\_examples}}$$ [c80]

**Predicted labels** are the class predictions generated by applying argmax to model logits [c70]; **target labels** represent the true values in the dataset [c69].

The following function iterates over a data loader and accumulates correct predictions [c71]:

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

The `num_batches` parameter allows evaluation on a subset of the data during training, saving computation time [c112].

#### 12.7.2 Cross-Entropy Loss

Categorical cross-entropy loss is defined as the negative sum over all class labels of $y_i \cdot \log(p_i)$, where $y_i$ is the true value and $p_i$ is the predicted probability [c88]. In PyTorch, this is computed with `torch.nn.functional.cross_entropy()` [c93].

---

### 12.8 The Fine-Tuning Training Loop

With the dataset, model, and metrics in place, we can write the training loop. The model passed to the training function is a GPT model class with a modified architecture that includes a classification head on top [c118].

#### 12.8.1 Epochs and Batches

**One epoch** is one complete pass through the entire dataset [c96]. Training typically runs for several epochs, updating model parameters after each batch via backpropagation.

#### 12.8.2 Evaluation Frequency

Printing metrics after every single batch would flood the console and slow training. Two parameters control evaluation cadence:

- **Evaluation frequency** specifies how many batches must be processed before printing training and validation loss metrics [c110]. In our setup, training loss and validation loss are printed every 50 batches using the `evaluate_model` function [c113].
- **Evaluation iteration** specifies the number of batches to use for each evaluation pass; using fewer batches (e.g., 5 or 10) saves computation time [c112].

#### 12.8.3 Per-Epoch Accuracy Reporting

At the end of each epoch, training accuracy and validation accuracy are calculated and printed [c114], giving a higher-level view of learning progress than the per-batch loss alone.

[FIGURE: Timeline diagram showing one training epoch divided into batches, with loss printed every 50 batches and accuracy printed at the end of each epoch]

#### 12.8.4 Training Loop Structure

Each epoch proceeds as follows:

1. For each batch, pass the input through the model.
2. Extract the last token's output: `outputs[:, -1, :]` [c2880].
3. Compute cross-entropy loss against the target labels using `torch.nn.functional.cross_entropy()` [c93].
4. Backpropagate and update the unfrozen parameters.
5. Every 50 batches, evaluate and print training loss and validation loss [c113].
6. At the end of the epoch, evaluate and print training accuracy and validation accuracy [c114].

The remaining steps after architecture setup are to implement loss and accuracy evaluation utilities, fine-tune the model, and test it on new data [c2893].

---

### 12.9 Saving and Loading the Fine-Tuned Model

After training, we save the model weights so they can be reloaded for inference or further fine-tuning without repeating the training process. PyTorch provides `torch.save` for this purpose [c156]:

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL,
           _use_new_zipfile_serialization=True)
```

Fine-tuning changes model parameters through retraining on a specific dataset so the model performs well on that particular task [c160]; saving the fine-tuned weights captures the result of that process in a file that can be loaded later.

---

### 12.10 Evaluating on the Test Set and Running Inference

The final stage of the project involves testing the fine-tuned model on data it has never seen before [c137]. The held-out test set from Section 12.2.3 is the right tool for this evaluation: because it was never used during training or hyperparameter tuning, it provides an unbiased estimate of real-world performance [c3614].

#### 12.10.1 Why a Held-Out Test Set Matters

Using the test set during training would leak information into the learning process and produce an optimistically biased accuracy estimate. Keeping it strictly separate ensures that the final reported accuracy reflects how the model will behave on genuinely unseen messages.

#### 12.10.2 Inference on New Text

For inference on a single new message, the pipeline is:

1. Tokenize the input text using TikToken [c2364].
2. Pass the token IDs through the model and extract the last token's output: `outputs[:, -1, :]` [c2880].
3. Apply argmax to the two logits to obtain the predicted label [c70].

This pipeline confirms the value of fine-tuning: without it, the pre-trained GPT model could not correctly answer spam classification prompts even when given explicit instructions [c159].

---

### 12.11 Putting It All Together

Classification fine-tuning is one category of fine-tuning, focused on adapting models for classification tasks [c161]. The pattern established here—load a foundation model, swap the output head, freeze most layers, fine-tune a small subset, and evaluate rigorously—applies to any classification problem you might encounter: sentiment analysis, topic labelling, intent detection, and more.

#### 12.11.1 Comparison with Instruction Fine-Tuning

Instruction fine-tuning requires the model to generate free-form text conditioned on a natural-language instruction [c3598], which demands more compute [c3605] and a much larger labelled dataset. Classification fine-tuning constrains the output space to a fixed set of labels [c3600], making training faster, evaluation simpler, and deployment more predictable. The cost is flexibility: the model can only handle a narrow set of prompts [c3607].

For scenarios where you need both efficiency and flexibility, parameter-efficient methods like LoRA [c3611] and QLoRA [c3612] offer a middle ground by fine-tuning only a small number of adapter parameters rather than full layer weights [c3610].

---

### 12.12 Summary

In this chapter we built a complete classification fine-tuning pipeline for spam detection. Starting from the SMS Spam Collection dataset [c48], we balanced it to address class imbalance [c50] and split it into train, validation, and test sets [c3644]. We wrapped the data in PyTorch `Dataset` and `DataLoader` objects [c2350, c2351], using TikToken for tokenization [c2364] and padding all sequences to a uniform length [c2375].

On the model side, we loaded pre-trained GPT-2 weights [c2844], replaced the vocabulary-sized output head with a two-class classification head [c2854, c2873], and selectively unfroze the final transformer block [c2875] and final layer normalization [c2876] for fine-tuning [c2863]. We implemented accuracy [c71] and cross-entropy loss [c93] utilities, wrote a training loop with configurable evaluation frequency [c110, c113], and saved the resulting model [c156]. Finally, we evaluated on the held-out test set [c137] and ran inference on new messages using the last output token [c2880].

The next chapter extends these ideas to instruction fine-tuning, where the model learns to follow natural-language directives rather than predict a fixed class label.