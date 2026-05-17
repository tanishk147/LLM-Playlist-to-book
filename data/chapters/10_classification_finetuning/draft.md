## Classification Fine-Tuning: Spam Detection from Scratch

Fine-tuning is the process of adapting a pre-trained model to a specific task by training the model on additional data [c2337, c3575]. Rather than training a model from scratch—which requires enormous compute and data—we start from a foundation model that has already learned rich representations of language, then nudge its weights toward a narrower objective [c169]. This chapter walks through every step of that process for a concrete, practical task: teaching a GPT-2 model to distinguish spam text messages from legitimate ones.

The project is deliberately built from scratch [c2894]. By the end of this chapter you will have downloaded and balanced a real dataset, written a custom PyTorch `Dataset` and `DataLoader`, surgically modified the GPT-2 architecture to output class logits instead of vocabulary logits, frozen the right subset of parameters, implemented accuracy and loss utilities, run a full training loop, and tested the resulting classifier on messages it has never seen [c2893].

### 10.1 The Big Picture: Classification Fine-Tuning

Before diving into code, it is worth understanding where classification fine-tuning sits in the broader landscape of fine-tuning approaches.

#### 10.1.1 Two Flavors of Fine-Tuning

LLM fine-tuning has two broad categories: instruction fine-tuning and classification fine-tuning [c3596]. Instruction fine-tuning trains the model to follow natural-language instructions and produce free-form text responses. Classification fine-tuning, by contrast, uses a large language model to classify inputs into predefined categories without explicit instructions [c3600].

The practical difference matters. Instruction fine-tuning requires greater computational power than classification fine-tuning because the model must search the entire corpus based on given instructions and perform non-specific actions [c3605]. For spam detection, we do not need the model to generate a sentence explaining its reasoning—we just need it to output one of two labels.

In classification fine-tuning for spam detection, no instruction is given to the LLM; only the input text is provided and the model classifies it as spam or not spam [c3601]. The model architecture is augmented at the end to be suitable for binary classification with outputs of either spam (1) or no spam (0) [c3649].

[FIGURE: Side-by-side comparison of instruction fine-tuning (input → instruction + text → free-form output) versus classification fine-tuning (input → text only → discrete class label)]

#### 10.1.2 Parameter-Efficient Alternatives

For completeness, it is worth knowing that full fine-tuning is not the only option. Parameter efficient fine-tuning updates only a subset of parameters while freezing the rest, reducing the number of trainable parameters and memory requirements [c3610]. LoRA, for instance, is a fine-tuning method where instead of fine-tuning all weights in a weight matrix, two smaller matrices that approximate the larger matrix are fine-tuned [c3611]. QLoRA takes this further by quantizing the weights of LoRA adapters to lower precision [c3612]. This chapter focuses on selective full fine-tuning—freezing most of the model but fully updating the layers we care about—which strikes a good balance between simplicity and effectiveness.

#### 10.1.3 The Three Stages

The fine-tuning process is divided into three main stages: dataset preparation, model initialization and modification, and fine-tuning with evaluation [c3615].

- **Stage 1 – Dataset preparation:** Download the SMS Spam Collection dataset, balance it, tokenize messages, pad sequences, and wrap everything in PyTorch `Dataset` and `DataLoader` objects.
- **Stage 2 – Model initialization and modification:** Load pre-trained GPT-2 weights, replace the output head with a two-class linear layer, and freeze all but the last transformer block and final layer normalization [c3617].
- **Stage 3 – Fine-tuning and evaluation:** Run the training loop, track loss and accuracy on train and validation splits, and finally test on held-out data and new messages [c3618].

### 10.2 The SMS Spam Collection Dataset

#### 10.2.1 Dataset Overview

The fine-tuning classification project uses the SMS Spam Collection dataset from the UCI machine learning repository [c48]. The dataset consists of text messages labeled as either spam or not spam [c49]. The label for non-spam messages is called HAM [c2341]. The SMS Spam Collection dataset contains 425 spam messages manually extracted from the Grumbletext website, among other sources [c3623].

The original dataset is imbalanced: there are 4,825 ham (non-spam) messages and 747 spam messages [c2828]. This imbalance—roughly 6.5 ham messages for every spam message—would cause a naive classifier to achieve high accuracy simply by predicting "ham" for everything. We will address this by balancing the dataset before training.

#### 10.2.2 Downloading the Dataset

The following snippet downloads the dataset archive, extracts it, and renames the file to give it a `.csv` extension:

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

The unverified SSL context is a pragmatic workaround for environments where certificate verification causes download failures. In production code you would want proper certificate handling.

#### 10.2.3 Balancing the Dataset

Because the original dataset has approximately 4,800 ham messages and 747 spam messages [c50], we need to downsample the majority class. The goal is to create a balanced dataset where both classes have the same number of examples. Intuitively, a balanced dataset prevents the model from learning a trivial shortcut—predicting the majority class every time.

After balancing, we split the data into training, validation, and test sets. The following function performs a random split:

# Source: [c3640]
```python
def random_split(df, train_frac=0.7, validation_frac=0.2):
    train_df = df.sample(frac=train_frac)
    remaining = df.drop(train_df.index)
    validation_df = remaining.sample(frac=validation_frac/(1-train_frac))
    test_df = remaining.drop(validation_df.index)
    return train_df, validation_df, test_df
```

Applying this split to the balanced dataset yields the following counts:

# Source: [c3644]
```python
train_df, validation_df, test_df = random_split(balanced_df)
print(len(train_df))  # 1045
print(len(validation_df))  # 149
print(len(test_df))  # 300
```

The training set contains 1,045 examples, the validation set 149, and the test set 300 [c3644]. The validation set is used during training to monitor generalization; the test set is held out until the very end [c137].

### 10.3 Building the SpamDataset

#### 10.3.1 Dataset vs. DataLoader

Two PyTorch abstractions are central here. A `Dataset` stores the samples and their corresponding labels [c2350]. A `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples [c2351]. A PyTorch dataset specifies how data is loaded and processed before instantiating a data loader [c2374].

#### 10.3.2 Tokenization

The `SpamDataset` class uses TikToken to convert text into token IDs [c2364]. TikToken is a tokenizer library developed by OpenAI that converts sentences into token IDs [c2364]. It uses a byte pair encoder—a sub-word tokenizer that breaks words into smaller units, not mapping each word to a single token [c2365]. The vocabulary is a dictionary mapping tokens to their corresponding token IDs [c2370].

An end-of-text token is a special token used to mark the end of a document or sentence in tokenization [c2363]. This token is appended to each message so the model knows where the input ends.

#### 10.3.3 The SpamDataset Class

The spam dataset class performs three steps: identifying the longest sequence in the training dataset, converting each text message into token IDs, and padding all sequences to match the longest sequence length [c2375]. The class loads data from CSV files, tokenizes text using the GPT-2 tokenizer from TikToken, and pads or truncates sequences to uniform length [c2385].

[FIGURE: Diagram showing three text messages of different lengths being tokenized and then padded with end-of-text tokens to the length of the longest sequence]

Padding is necessary because PyTorch's `DataLoader` requires all sequences in a batch to have the same length. Using the end-of-text token as the padding token is a natural choice—it is already in the vocabulary and the model has seen it during pre-training.

The large language model architecture takes input text and outputs a classification of whether the email is spam or not spam [c2348]. The dataset class is the bridge between raw CSV rows and the tensor inputs that architecture expects.

#### 10.3.4 Creating the DataLoaders

Once the `SpamDataset` is defined for each split, we instantiate three `DataLoader` objects—one each for training, validation, and testing. The `DataLoader` handles shuffling (for the training set), batching, and iteration. This is standard PyTorch practice and keeps the training loop clean.

### 10.4 Modifying the GPT-2 Architecture

#### 10.4.1 Loading Pre-Trained Weights

Fine-tuning involves using pre-trained weights from GPT-2 and then running the training procedure again on a specific dataset [c152]. The project uses a GPT-based LLM architecture that will be modified for classification fine-tuning [c2837].

The following function downloads and loads the GPT-2 model files:

# Source: [c2844]
```python
def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    # Downloads GPT-2 model files from OpenAI public blob storage
```

GPT-2 weights are stored in TensorFlow checkpoint format. The following helper loads them into a nested Python dictionary:

# Source: [c2847]
```python
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Recursively loads TensorFlow checkpoint variables into nested params dictionary
    # Handles token embeddings, positional embeddings, transformer blocks, and layer norm parameters
```

Once loaded, these parameters are copied into our PyTorch GPT model. The consequence is that the model starts fine-tuning from a strong initialization rather than random weights.

#### 10.4.2 Why the Pre-Trained Model Cannot Classify Spam Out of the Box

The pre-trained GPT-2 model without fine-tuning struggles to follow instructions and lacks classification capabilities for tasks like spam detection [c2851]. Without fine-tuning, the pre-trained GPT model could not correctly answer spam classification prompts even when given explicit instructions [c159]. This is expected: GPT-2 was trained to predict the next token, not to output a class label. We need to modify its architecture.

#### 10.4.3 Replacing the Output Head

A classification head is a small linear layer added on top of a pre-trained language model (like GPT) to adapt it for classification tasks by mapping hidden states to class logits [c2857].

The original GPT-2 output layer maps the 768-dimensional hidden state to 50,257 vocabulary logits—one per token in the vocabulary. For binary classification we only need two logits. To adapt GPT-2 for text classification, we replace the original 768→50257 output layer with a 768→2 linear layer that outputs logits for two classes (spam vs. no spam) [c2854].

In code, this replacement looks like:

# Source: [c2873]
```python
model.output_head = Linear(in_features=768, out_features=num_classes)
```

For fine-tuning a language model for classification, we replace the original output layer that maps to vocabulary size with a smaller output layer that maps to the number of classes [c2867]. The original output layer was replaced with a classification head that takes input of size 768 and outputs 2 values (for spam or no spam classification) [c60].

[FIGURE: GPT-2 architecture diagram showing the original 768→50257 output head being replaced by a 768→2 classification head]

#### 10.4.4 Which Tokens to Use

A GPT model processes a sequence of tokens and produces an output vector for each position. For spam classification, we focus fine-tuning on the last output token rather than all output rows, since the last token contains all necessary information [c2879]. Intuitively, by the time the model processes the final token, it has attended to every preceding token in the sequence—so the last hidden state is a compressed summary of the entire input.

In PyTorch, extracting the last token's output from a batch of sequences is:

# Source: [c2880]
```python
outputs[:, -1, :]
```

Extracting the last output token from a 4-token sequence yields a vector with two class logits (e.g., -3.5898 and 3.9902 for spam/not-spam classification) [c2882]. The larger logit wins: in this example the model would predict "not spam."

#### 10.4.5 Deciding Which Parameters to Train

Before training, the parameters in the new output head, final transformer block, and final layer normalization are still random and have not been trained on the classification dataset [c2878]. We need to train them. But we do not want to train the entire model—that would be slow and risk destroying the pre-trained representations.

Three components will be fine-tuned: the classification head, the final (12th) transformer block, and the final layer normalization module [c2863]. Everything else is frozen.

To unfreeze the final transformer block:

# Source: [c2875]
```python
for param in model.transformer.h[-1].parameters():
    param.requires_grad = True
```

To unfreeze the final layer normalization:

# Source: [c2876]
```python
for param in model.ln_f.parameters():
    param.requires_grad = True
```

The model passed to the training function is a GPT model class with a modified architecture that includes a classification head on top [c118]. By freezing the bulk of the network and only training the top layers, we preserve the general language understanding learned during pre-training while adapting the model's output behavior to our specific task.

[FIGURE: GPT-2 layer stack with frozen layers (grey) and trainable layers highlighted: classification head, final transformer block, and final layer norm]

### 10.5 Evaluation Utilities: Accuracy and Loss

Before writing the training loop, we need two measurement tools: an accuracy function and a loss function. These let us monitor whether the model is actually learning.

#### 10.5.1 Classification Accuracy

Classification accuracy measures the percentage of correct predictions across a dataset [c67]. The formula is straightforward:

$$\text{Accuracy} = \frac{\text{correct\_predictions}}{\text{total\_number\_of\_examples}}$$
[c80]

Predicted labels are the class predictions generated by applying argmax to model logits [c70]. Output labels are also called target labels, representing the true values in the dataset [c69]. The argmax operation picks the index of the largest logit—in a two-class setting, index 0 means "ham" and index 1 means "spam."

The following function computes accuracy over a data loader, with an optional `num_batches` argument to limit evaluation to a subset of batches:

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

Evaluation frequency specifies how many batches must be processed before printing training and validation loss metrics [c110]. Evaluation iteration specifies the number of batches to use for evaluation; using fewer batches (e.g., 5 or 10) saves computation time during evaluation [c112]. The `num_batches` parameter in `calculate_accuracy` serves exactly this purpose—during training we can pass a small number to get a quick estimate without waiting for a full pass over the data.

#### 10.5.2 Cross-Entropy Loss

Categorical cross-entropy loss is defined as: negative sum over all class labels of $(y_i \cdot \log(p_i))$, where $y_i$ is the true value and $p_i$ is the predicted probability [c88].

$$\mathcal{L} = -\sum_{i} y_i \log(p_i)$$

Intuitively, this loss penalizes confident wrong predictions heavily (because $\log(p_i)$ is very negative when $p_i$ is close to zero) and rewards confident correct predictions.

In PyTorch, we use the built-in function for this computation:

# Source: [c93]
```python
torch.nn.functional.cross_entropy()
```

`torch.nn.functional.cross_entropy()` is the PyTorch function used to compute categorical cross-entropy loss [c93]. It accepts raw logits (not probabilities), applies softmax internally, and computes the negative log-likelihood. This is both numerically stable and convenient.

### 10.6 The Fine-Tuning Training Loop

#### 10.6.1 Training Loop Structure

With the dataset, model, and evaluation utilities in place, we can write the training loop. One epoch is one complete pass through the entire dataset [c96]. The loop iterates over epochs, and within each epoch iterates over batches from the training `DataLoader`.

The high-level structure is:

1. For each batch, run a forward pass to get logits.
2. Compute cross-entropy loss between logits and target labels.
3. Run a backward pass to compute gradients.
4. Update parameters to minimize loss.
5. At regular intervals (controlled by evaluation frequency), compute and print training and validation loss [c110].
6. After every epoch, calculate and print training accuracy and validation accuracy [c114].

The lecture covers implementing two metrics (accuracy and loss function), a backward pass for training, parameter modification to minimize loss, and testing on unseen data [c66].

#### 10.6.2 Epoch-Level Evaluation

Step 7 of the fine-tuning training process: after every epoch, calculate and print training accuracy and validation accuracy [c114]. This gives us a coarser but complete picture of model performance—we see how the model is doing on the full training set and the full validation set at the end of each epoch, not just on a few sampled batches.

[FIGURE: Training curve plot showing training loss and validation loss decreasing over epochs, with accuracy increasing on both splits]

#### 10.6.3 Saving the Model

After training, we save the model weights so they can be reloaded without retraining. PyTorch's `torch.save` handles this:

# Source: [c156]
```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)
```

Fine-tuning is the process of changing model parameters and retraining on a specific dataset so the model performs well on that particular task [c160]. Saving the fine-tuned weights is the natural endpoint of that process—the saved file captures everything the model learned during fine-tuning.

### 10.7 Evaluating the Fine-Tuned Model

#### 10.7.1 Train, Validation, and Test Accuracy

Once training is complete, we evaluate the model on all three splits using `calculate_accuracy`. The training and validation accuracies were monitored during training; the test accuracy is the number we care about most, because it reflects performance on data the model has never seen [c137].

The email classification project involves training a large language model to classify emails as spam or no spam [c3614]. The final stage of the spam classification project involves testing the fine-tuned model on new data that it has not seen before [c137].

#### 10.7.2 Comparing to the Unmodified Model

It is instructive to run the unmodified pre-trained GPT-2 (before fine-tuning) on a few spam classification prompts. The pre-trained GPT-2 model without fine-tuning struggles to follow instructions and lacks classification capabilities for tasks like spam detection [c2851]. Without fine-tuning, the pre-trained GPT model could not correctly answer spam classification prompts even when given explicit instructions [c159]. This contrast motivates the entire fine-tuning pipeline: the pre-trained model has powerful language representations but no mechanism to produce a class label.

### 10.8 Running Inference on New Messages

The final stage includes fine-tuning the model, evaluating the fine-tuned model, and using the model on new data [c3618]. After training and evaluation, we want to use the model in practice—feeding it a raw text message and getting back a prediction.

The inference procedure mirrors the training forward pass:

1. Tokenize the input message using TikToken's GPT-2 tokenizer.
2. Pad or truncate to the expected sequence length.
3. Run a forward pass through the modified GPT-2 model.
4. Extract the last token's output: `outputs[:, -1, :]` [c2880].
5. Apply argmax to the two logits to get the predicted class.
6. Map the class index back to a human-readable label: 0 → "ham", 1 → "spam".

[FIGURE: Inference pipeline diagram: raw text message → tokenizer → padded token IDs → GPT-2 + classification head → two logits → argmax → "spam" or "ham" label]

The classification fine-tuning project involves predicting whether text messages are spam or not spam [c2824]. With a well-trained model, this pipeline should correctly flag messages like "Congratulations! You've won a free prize, call now!" as spam while passing through legitimate messages.

### 10.9 Putting It All Together

Let us step back and trace the full pipeline from raw data to a working spam classifier.

**Stage 1 – Data preparation:**
- Download the SMS Spam Collection dataset from the UCI repository [c48].
- Observe the class imbalance: 4,825 ham vs. 747 spam [c2828].
- Downsample ham to match the spam count, creating a balanced dataset [c50].
- Split into train (1,045), validation (149), and test (300) sets [c3644].
- Build the `SpamDataset` class: tokenize with TikToken, pad to the longest sequence in the training set [c2375, c2385].
- Wrap each split in a `DataLoader` [c2351].

**Stage 2 – Model initialization and modification:**
- Download GPT-2 weights using `download_and_load_gpt2` [c2844].
- Load TensorFlow checkpoint parameters with `load_gpt2_params_from_tf_ckpt` [c2847].
- Replace the 768→50257 output head with a 768→2 classification head [c2854, c2873].
- Freeze all parameters except the classification head, the final transformer block, and the final layer normalization [c2863, c2875, c2876].

**Stage 3 – Fine-tuning and evaluation:**
- Implement `calculate_accuracy` [c71] and use `torch.nn.functional.cross_entropy` for loss [c93].
- Run the training loop: forward pass, loss computation, backward pass, parameter update [c66].
- Print loss every `eval_freq` batches [c110]; print accuracy every epoch [c114].
- After training, evaluate on the test set [c137].
- Save the fine-tuned weights with `torch.save` [c156].
- Run inference on new messages [c3618].

[FIGURE: End-to-end pipeline flowchart: SMS Spam Collection CSV → balanced dataset → SpamDataset → DataLoader → pre-trained GPT-2 → replace output head → freeze layers → training loop → fine-tuned classifier → inference on new messages]

### 10.10 Summary and What Comes Next

This chapter built a complete classification fine-tuning pipeline from scratch [c2894]. We started from the observation that LLM fine-tuning has two broad categories—instruction fine-tuning and classification fine-tuning [c3596]—and focused on the latter because it is computationally lighter [c3605] and well-suited to the spam detection task [c2824].

The key architectural insight is the classification head: a small linear layer (768→2) that replaces the original vocabulary projection layer [c2857, c2854]. By freezing most of the pre-trained model and only training the classification head, the final transformer block, and the final layer normalization [c2863], we preserve the general language understanding encoded in GPT-2's weights while teaching the model to produce class labels.

The evaluation utilities—accuracy [c80] and cross-entropy loss [c88]—give us concrete numbers to track during training. The `num_batches` parameter in the accuracy function lets us trade off evaluation speed against precision during training [c112], while full-dataset evaluation at the end of each epoch gives us the complete picture [c114].

LLM fine-tuning involves the additional training of a pre-existing model that has previously acquired patterns and features from an extensive dataset using a smaller domain-specific dataset [c3587]. The SMS Spam Collection dataset, with its 1,494 balanced examples after preprocessing, is a perfect illustration of how a small, well-curated dataset can be enough to adapt a powerful pre-trained model to a new task [c154].

The techniques introduced here—selective layer unfreezing, classification head replacement, and the train/val/test evaluation discipline—generalize directly to other classification tasks: sentiment analysis, topic classification, intent detection, and more [c3602]. The next natural step is instruction fine-tuning, where instead of predicting a class label the model learns to follow natural-language instructions and generate free-form responses [c2826].