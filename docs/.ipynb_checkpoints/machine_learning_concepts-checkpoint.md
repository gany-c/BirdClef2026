# Machine Learning Concepts in BirdClef 2026



## What is a multi-class problem?

A multi-class problem is a classification task where each example must be assigned to one of three or more classes. For example, classifying an image as cat, dog, or horse is multi-class classification

## Pandas and iloc

iloc selects rows and columns using integer positions (0-based)

df.iloc[row_index, column_index]

## What is the GLOB package?

glob lets you programmatically list files using wildcard patterns

## What is Librosa?

1. Librosa is a  Python library for audio and music signal processing. 
2. It helps you load, transform, and extract features from sound files so you can use them in machine learning models.
3. It can read audio files (.wav, .ogg, .mp3)
4. Sampling Rate is another input with the file: y, sr = librosa.load("bird.ogg", sr=32000)
	1. higher sampling rate = longer vector
	2. lower sampling rate = shorter vector

## What is a spectrogram?

1. It is like a heat map
2. A spectrogram is:
3. A 2D grid (matrix) with a third value encoded as intensity.
	1. X-axis → time
	2. Y-axis → frequency
	3. Value at (x, y) → intensity (energy of that frequency at that time)
4. Why ML models love spectrograms?
	1. Patterns become spatial 
	2. CNNs can detect: 
		- edges → frequency transitions 
		- blobs → sustained tones 
		- textures → noise vs signal

## What is a MEL Scale and MEL Spectrogram?

1. MEL Scale is a transformation of frequency such that:
	1. Low frequencies → high resolution
	2. High frequencies → compressed
	3. So instead of equal spacing like: 100 Hz, 200 Hz, 300 Hz ...; You get bins that are: dense at low freq, sparse at high freq
2. In a mel spectrogram:
	1. The frequency axis is remapped from Hz → Mel scale
		1. X-axis → time ✅ (unchanged)
		2. Y-axis → mel frequency bins (non-linear)
		3. Values → energy

## PyTorch: What is the power_to_db operation in the MEL Spectrogram?

1. This is the line of code: S_dB = librosa.power_to_db(S, ref=np.max)
	1. S is the mel spectrogram
	2. It converts the spectrogram's 3rd dimension i.e. energy/intensity to decibels.

## Spec Augment

 Spec Augment works by intentionally corrupting parts of the Mel spectrogram during training so the model becomes more robust and generalizes better.
SpecAugment modifies the spectrogram by randomly:
- masking time regions 
- masking frequency regions 
- sometimes warping time 
This forces the model to learn more robust features instead of memorizing exact patterns.

## What is EfficientNet and its variations?

EfficientNet is a family of CNN (Convolutional Neural Network) models developed by Google for image classification and feature extraction.

EfficientNet Variants (B0, B1, B2 … B7)

The versions differ mainly in:
1. model size
2. accuracy
3. computational cost
4. input resolution

EfficientNet-B0 is the base model, smallest and fastest. As the model number becomes higher, the model becomes larger, more accurate and slower.

## What is Torch or PyTorch?

1. e.g. import torch
2. PyTorch is an open source ML Library,
3. It does mathematical heavy lifting as follows:
	1. It manages tensors which are multi-dimensional arrays. These can be processed on GPUs or your local MAC
	2. It provides the tools to build Neural Network, through layers like nn.Linear or nn.Conv2d

## What is a PyTorch Dataset?

A PyTorch Dataset is an abstract class (torch.utils.data.Dataset) that stores your data samples and their corresponding labels. (Input and Output)
When you inherit from torch.utils.data.Dataset, you are signing a contract to implement three specific methods

1. __init__ (The Setup): 
2. __len__ (The Size): You must return len(df). This tells the DataLoader how many "steps" it has in one full cycle (epoch).
3. __getitem__ (The Worker): This is the logic you described. It performs the file loading/processing, and returns the final output.

e.g. We can create a custom sub class of Dataset which does the following:
1. Accepts a dataframe and an index as input.
2. retrieves the dataframe row using the index, gets a filepath from the row - e.g. audio file
3. converts the audio file to numpy array and then a tensor and returns the tensor... Usually, the Dataset returns a tuple(tensor, label)

## What is a PyTorch DataLoader?

While the Dataset describes how to load a single piece of data, the DataLoader uses the dataset to load data in batches. 
It supports batching the data, shuffling the input, multi-processing

## What is timm?

1. Python library that provides ready-to-use deep learning model architectures, especially for computer vision.
2. A collection of prebuilt, pretrained neural networks you can use with a few lines of code.
3. timm is a library built on top of PyTorch, not a separate framework. It provides a large collection of image models, layers, training utilities, and a simple factory API for creating pretrained vision models, while PyTorch supplies the underlying tensor, autograd, and training engine.

## What is torch.nn?

torch.nn is the module you use to build and define neural networks (layers + loss functions) in PyTorch.
Internally: timm model => many torch.nn layers combined.
Analogy:
torch.nn → raw ingredients (flour, sugar, eggs)
timm → ready-made cake mix (just bake it)

## What is nn.module?

In PyTorch, nn.Module is the base class of all  neural networks' building blocks. 
You inherit from it so your class gets all the built-in functionality needed for training, inference, and parameter management.
In the sub-class, it is mandatory to implement the 
1. init and 
2. the forward methods

## init method:

1. It defines the architecture of the neural net
2. Any neural net can be viewed as having 2 components, the back bone or feature extractor and the final classifier layer.
3. In my BirdClef implementation I'm using a model called "efficientnet-b0" as the feature extractor.. it is also pre-trained on the images from ImageNet
4. The classifier is a linear layer based on the number of features and layers.

## forward method:

1. it takes in tensors as  input
2. passes them through the backbone,
3. passes the result through the classifier 
4. returns the result.


## HuggingFace and Timm

When you run timm.create_model(..., pretrained=True), timm acts as a client. It reaches out to the Hugging Face (HF) Hub to check for the latest version of the model's weights and downloads them to your machine.


## What is a loss function?

A loss function is a mathematical way to measure how far a model’s prediction is from the correct answer. 
If the prediction is good, the loss is small; if it is bad, the loss is large.
So the loss function determines how the model weights will be adjusted at the end of each training epoch or iteration.
Loss -> Gradient = loss/Model parameter -> Corrective action by optimizer (See below section on optimizer)
E.g. of loss functions: CrossEntropyLoss, BCEWithLogitsLoss, AsymmetricLossMultiLable(ASL)

## What is CrossEntropyLoss?

It is a Loss function for PyTorch, It is the standard choice for multi-class classification, where each example belongs to exactly one class, such as cat vs dog vs horse.
Different Types of Loss Functions:
- CrossEntropyLoss
	- Multi-class classification function where only one class is correct (e.g., distinguishing between a cat, dog, or horse).
	- Internally applies Softmax and expects raw logits.
	- Commonly used in image classification and NLP classification tasks.
- BCEWithLogitsLoss()
	- Binary or multi-label classification function where multiple independent labels can be correct per sample.
	- Internally applies Sigmoid and expects raw logits, treating each class independently.
	- Commonly used in tasks like BirdCLEF, multi-label tagging, and multi-object detection.
- BCEWithLogitsLoss(reduction='none')
	- Variant of BCEWithLogitsLoss that returns per-sample and per-class losses in a tensor (typically of shape [batch_size, num_classes]) instead of automatically averaging them.
	- Utilized when implementing custom sample weights, focal weighting, class balancing, or custom reductions.
- AsymmetricLossMultiLabel (ASL)
	- Specialized loss function tailored for highly imbalanced, sparse multi-label classification and long-tail datasets (e.g., BirdCLEF).
	- Designed to reduce the performance penalty for easy negatives, focus heavily on hard positives, and improve rare-class recall.
	- Gamma_neg Parameter: Controls how aggressively easy negative samples are down-weighted (typical value is 4). Higher values ignore easy negatives more intensely, reduce false-positive dominance, and improve rare-class learning.
	- Gamma_pos Parameter: Controls the learning pressure on positive samples (typical value is 1). Higher values emphasize hard positive examples; it is usually set smaller than gamma_neg.


## Loss Functions vs Evaluation Metrics


1. Loss Functions are used while training the model i.e. in Back Propogation, Evaluation metrics are used to determine how well the model performs after training
2. Learn using Loss Function, Judged by Evaluation Metrics
3. However, sometimes a loss value can be monitored like a metric (e.g. val_loss in your BirdClef training). Evaluation Metrics can also be used at the end of every training Epoch to prevent overfitting.
	1. AUROC (Area Under ROC Curve) 
		1. Measures how well the model separates positive and negative classes across all thresholds. 
		2. Higher AUROC means better ranking and classification quality. 
	2. mAP (mean Average Precision) 
		1. Measures how well correct labels are ranked with high confidence. 
		2. Commonly used in multi-label tasks like BirdCLEF. 
	3. MultilabelF1Score 
		1. Measures F1 score for multi-label classification problems. 
		2. Usually implemented as a metric class that can compute micro, macro, or weighted F1. 
		3. F1 Score 
			1. Measures the balance between precision and recall. 
			2. Useful when both false positives and false negatives matter. 
	4. Macro F1 
		1. Computes F1 score separately for each class and averages them equally. 
		2. Gives equal importance to rare and common classes.

## What is a gradient?

1. A gradient is the rate of change of the loss with respect to a model parameter, like a weight or bias. 
2. more precisely it is the derivative of the loss with respect to the weight.
3. In plain English, it tells you how the loss would change if you nudged that parameter a tiny bit

## What is an Optimizer?

An optimizer is the part of training that updates a model’s weights and biases so its predictions get better over time. 
It uses the loss function and gradients to decide how to change the model’s parameters in the direction that reduces error.

## How it works

1. The model makes a prediction.
2. The loss function measures the error.
3. The optimizer looks at the gradients of that loss.
4. It updates the parameters a little bit.
5. This repeats for many training steps


## What is the Adam optimizer?

Adam (Adaptive Moment Estimation) is an optimization algorithm from Stochastic Gradient Descent family
It adjusts each weight using adaptive learning rates based on past gradients. 
It has its own algorithm, 
1. Big, consistent gradients → larger updates to the model
2. Noisy gradients → smaller updates to the model

```
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

Here, we are initializing the optimizer with the model’s parameters (weights)

## Activation Functions


Activation functions are needed to introduce non-linearity into neural networks.
Without activation functions, a neural network — no matter how many layers it has — would behave like just a single linear equation.

Real-world patterns are usually non-linear.
Examples of activation functions are Sigmoid, Softmax and Relu
Activation functions are used in two different places for two different purposes:
Hidden-layer activations (ReLU, GELU, etc.)
	- Used between layers during forward propagation 
	- Introduce non-linearity so the model can learn complex patterns 
Output activations inside loss functions (Sigmoid, Softmax)
	- Applied to final logits before computing loss 
	- Convert raw outputs into probabilities 
	- Included inside losses like BCEWithLogitsLoss and CrossEntropyLoss for numerical stability and better gradients




## What is softmax?

Softmax is a function that turns a list of raw scores into a set of probabilities that add up to 1. It is commonly used at the output of a multi-class classifier, so the model can say how likely each class is


## Sigmoid

1. An activation function that converts logits into probabilities between 0 and 1. 
2. Commonly used in binary and multi-label classification tasks.



## Choosing Softmax vs Sigmoid

Choosing Between Sigmoid and Softmax
- Use Sigmoid for multi-label classification where multiple classes can be correct simultaneously. 
- Use Softmax for multi-class classification where exactly one class is correct.
Relation with Loss Functions
- BCEWithLogitsLoss already includes Sigmoid internally.
- CrossEntropyLoss already includes Softmax internally.

## What is a Logit?

A logit is the raw output score produced by a neural network before applying an activation function like sigmoid or softmax.
It is usually:
- any real number 
- unbounded 
- not yet a probability

## Why do we set the number of Epochs to 10?

The number of Epochs is the number of training loops. 
The value of 10 is arbitrary, we have 70 now. You train until the score against the validation set doesn't improve

## PyTorch: What does model.train() do?

- It doesn't actually train, it just sets the NN's mode to training
- It toggles internal flags (like Dropout or Batch Normalization) into "Training Mode."
- The Opposite: Later, when you want to predict (inference), you must call model.eval().
- Since model.train() is just a "toggle switch," it doesn't need input.

## PyTorch: What is tqdm?

1. It wraps the Dataloader with a progress bar.
2. It inherits the batch size (e.g., 32) from your train_loader = DataLoader. Every time the bar moves one tick, one full batch has been processed.


## PyTorch: How does pbar, an instance of tqdm get the labels and inputs?

From the original dataframe and dataset:
Original Dataframe-> Dataset (loads audio) -> DataLoader (batches audio) ->tqdm 


## PyTorch: Why Zero the gradient?

```
"""
Zero the parameter gradients
""""
optimizer.zero_grad()
```
- It makes the gradient based on the current iteration only.
- It removes the effect based on previous iterations
- Don't understand the "why". Have to dig deeper


## PyTorch: What is model(inputs) ?

This is the actual forward propagation, unlike model.train(), Which just sets the mode.

```
outputs = model(inputs)
```


## PyTorch: How is Loss calculation and Back propagation done after forward propagation?


1. loss = criterion(outputs, labels) - Calculates the loss in terms of the outputs vs ground truth
2. loss.backward() - Calculates the gradients across all the layers based on the loss
3. optimizer.step() - Actually implements the change in weights across layers using the gradients and learning rate.


## PyTorch: Final Debugging and Display Steps


1. These have no impact on the actual model training
2. running_loss += loss.item() - 
	1. Running loss is the cummulative loss across all batches in an epoch. 
	2. loss.item() converts a tensor into a scalar value
3. pbar.set_postfix(loss=running_loss/len(train_loader))
	1. Shows the loss by total number of batches
	2. I think the denominator should be number of batches run so far and not total number of batches. (still the final answer will be correct)


## What is CosineAnnealingLR

A learning rate scheduler that gradually decreases the learning rate using a cosine-shaped curve during training.
Purpose:
Large learning rate early → faster learning
Small learning rate later → stable fine-tuning

## Cosine Warm Restarts

A variant where the learning rate periodically resets to a high value and then decays again.
Purpose:
- Helps escape local minima/plateaus 
- Encourages better exploration during training


## What is recall-oriented learning?


Recall-oriented learning means training a model to prioritize finding as many true positives as possible, even if it increases false positives.

In the Context of BirdClef:

Your move toward:
- ASL loss 
- Macro F1 
- lower thresholds 
- secondary labels 
is effectively making your BirdCLEF pipeline more recall-oriented for rare species detection.

## Processors Available on Kaggle - Accelerator Options Comparison

| Option | What it is | Best for |
| :--- | :--- | :--- |
| **None** | CPU only | Small experiments/debugging |
| **GPU T4 ×2** | Two NVIDIA T4 GPUs | Modern deep learning training/inference |
| **GPU P100** | Older NVIDIA Tesla P100 GPU | Older CUDA/PyTorch workloads |
| **TPU v5e-8** | Google TPU accelerator | TensorFlow/JAX large-scale training |

> [!WARNING]
> **GPU P100 Note:** May sometimes cause issues because newer libraries may not fully support older CUDA architectures.


## To be Learnt:  

1. My BirdClef implementation specific 
	1. How exactly are the secondary labels in Data used in training?
	2. What and How of the mix up algorithm - training samples
	3. What and How of smoothing in Inference

