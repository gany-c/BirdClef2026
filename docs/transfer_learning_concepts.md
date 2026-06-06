
## ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative one.


Simple terms: it measures separation—how well the model distinguishes between two groups.


## Feature extractor

A model/module that converts raw input (audio, image, text) into a numeric vector representation (embedding).
👉 Think:raw data → compact meaningful vector
Example:
- audio waveform → 512-d vector capturing “bird sound type”
---

## Selective State Space Model (Selective SSM)

A sequence model that processes long sequences efficiently by selectively deciding what information to keep or forget over time.
👉 Think:
- like an RNN, but smarter and more parallelizable
- “attention-like memory with linear cost”
Used for:
- long audio
- long time series
---

## Bidirectional SSM layers

An SSM that processes data in both directions:
- forward (past → future)
- backward (future → past)
👉 Think:Like a bidirectional LSTM, but with SSM mechanics.
Why:
- captures context from both sides of a sequence
---

## Metadata awareness (site, hour)

Model also uses extra structured context features, not just raw data.
Example:
- site = forest / city / wetland
- hour = 6 AM / night
👉 Think:“context knobs” that help prediction
Like:
user location + timestamp improves recommendation accuracy
---

## Cross-attention

A mechanism where one data stream looks at another to gather relevant information.
👉 Think:
- Query = “audio features”
- Key/Value = “metadata or another embedding set”
It learns:
“Which parts of metadata matter for this input?”
---

## Prototypical metric learning

A method where each class is represented by a prototype vector (centroid).
Prediction = closest prototype in embedding space.
👉 Think:Instead of classifier weights:
- class = point in space
- input → find nearest class center
Used for:
- few-shot learning
- similarity-based classification
---

## Gated fusion

A way to combine two signals using a learned weighting gate.
Formula idea:
```
output = gate * A + (1 - gate) * B

```
👉 Think:Model learns:
“how much to trust model A vs model B for each input”
---

## Second-pass SSM

A model that re-processes already encoded features for refinement.
👉 Think:
- Pass 1: extract rough understanding
- Pass 2: refine temporal relationships
Like:
draft → editor pass → final version
---

## Metadata embeddings

Convert categorical metadata into dense vectors.
Example:
- site_id = “forest” → [0.12, -0.8, 0.44...]
👉 Think:Same idea as word embeddings, but for structured fields.
---

## scikit-learn MLP Classifier

A simple feedforward neural network classifier from scikit-learn.
👉 Think:
- input vector → hidden layers → softmax output
- used on embeddings (not raw data)
Why used:
- fast baseline classifier on top of embeddings
---

## PCA-compressed embeddings

Embeddings reduced using Principal Component Analysis (PCA).
👉 Think:
- compress high-dimensional vectors → lower dimension
- keep most “important variance”
Example:
- 1024-d embedding → 128-d compressed vector
Why:
- faster inference
- less noise
- lower overfitting risk
---

🔥 One-line mental model

All of these together typically form a typical ML pipeline as a sequence of transformations,:
raw data → feature extractor → embeddings → context fusion (metadata + attention) → sequence modeling (SSM) → classification head (MLP / metric learning)
---

