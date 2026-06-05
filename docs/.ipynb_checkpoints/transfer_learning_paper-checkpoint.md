
# Transfer Learning with Pseudo Multi-Label Birdcall Classification for DS@GT BirdCLEF 2024

These are notes from https://arxiv.org/html/2407.06291v1

BirdNET is a popular birdcall classification model that utilizes the spectrogram-CNN approach. It is widely distributed in the field due to its high accuracy and ease of use on mobile devices [5]. 


The Google Bird Vocalization Classifier is another model using EfficientNet-B1, a similar CNN architecture, and is trained on many soundscapes. It was released alongside the BirdCLEF 2023 competition and has more than 10,000 species in its output space [6].

We explore the effectiveness of transfer learning, where bird vocalization predictions are surrogates for true labels. 
We hypothesize that transfer learning is an effective technique for the competition because existing models capture underlying structures amenable to optimization by simple linear classifiers. 


## Training Pipeline in the paper


Step 1: The Inputs (Far Left)

The pipeline starts with two sets of raw audio data:
1. train audio: The crowd-sourced, clean, isolated bird clips.
2. test audio: The messy, 4-minute wild soundscapes (where multiple birds are overlapping).

## Step 2: The Two Core Models (Middle-Left)

The audio is immediately split and sent to two completely different pre-trained models simultaneously:
- Google Bird Vocalization (The Teacher / Surrogate): * It takes the train audio and outputs a soft prediction.
	- A "soft prediction" means instead of saying a strict "Yes, this is a Plover," it outputs raw numbers (logits) representing confidence levels for thousands of species.
- BirdNET (The Student / Feature Extractor):
	- It takes the exact same train audio and shrinks it down into a highly descriptive numeric summary called an embedding. It doesn't look at BirdNET's final predictions; it only captures BirdNET's underlying understanding of the sound.

## Step 3: Pairing Data in the Storage Bank (Center Database)

The output from both models is merged into a single database called train embeddings.
Inside this database, every single 5-second chunk of audio now has a paired record:
- The Features: BirdNET's acoustic fingerprint vectors.
- The Labels: Google's soft predictions used as pseudo-labels (surrogate targets).

## Step 4: Training the Domain Multi-Label Classifier (Middle-Right)

Now, the actual learning happens in a small, lightweight custom model (like a simple linear layer or a small PyTorch network)


## Summary: 


So, basically, this pipeline is just compressing the huge NN into a smaller classifier, essentially looking at a classic Model Compression and Knowledge Distillation technique.


## Follow up Questions:

Notes: Transfer Learning with Pseudo Multi-Label Birdcall Classification for DS@GT BirdCLEF 2024


1. The train data already has labels, what is the need to generate pseudo labels for it?
	1. train.csv unlike the train_soundscapes_labels.csv operates on the whole file instead of 5 second intervals.
	2. You may be able to get interval level labelling done, more accurate train data.
	3. But did your earlier analysis notebook say that identification of species present in train data is inaccurate? Or were the inaccurate species present only in soundscape and not in train?
2. My existing NN model (E-B3) is already understanding the relationship between the given labels and given train audio clips. Why should another pipeline do the same in a convoluted way? 
	1. Diversity, leveraging the power of Google's models
3. Can we modify this approach to label the unlabelled data in soundscapes?
	1. First check how well the Google Models are doing on the labelled soundscapes..
	2. If results are positive, Next I label the unlabelled soundscapes
	3. Use the newly labelled soundscapes as training data to train an EfficientNet B0 or B3 - my 3rd model M3
	4. add M3 to the ensemble


## Pseudo Label Predictions:


In the paper, pseudo label predictions are limited to the competition’s species set. If the species is not present, its prediction is set to zero by assigning negative infinity to the logit output.
In our case we should limit the predictions to the primary and secondary labels for the given row.

Similarly, we should follow what is being done in section 5.3 Pseudo Multi-Label Construction of the paper. Only difference being, we should use Primary and Secondary labels instead of the folder name


## Loss Functions in Training


Once the dataset is built, and we train the new model, choose the appropriate loss function from  section 5.4. Training Losses


## Inference and Embeddings generation


When we do inference, again we'll have to generate the embeddingss
We may have to compile the model to get it to run faster - See section 6.4. runtime

We compile the model using TensorFlow Lite at runtime, optimizing operations for the hardware while allowing fallback to non-lite operations. This compilation process results in an order-of-magnitude performance increase, leaving a substantial margin for additional computation
