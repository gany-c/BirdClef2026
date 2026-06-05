# BirdClef2026
This is my attempt at Bird Clef 2026 https://www.kaggle.com/competitions/birdclef-2026/overview final score 0.85

## Structure of the Input Data

https://www.kaggle.com/competitions/birdclef-2026/data

3 components here:

1. Train.csv and its files: species' sounds are isolated, only 206 labels. Each sound file is much more than 5 seconds long.
2. train_soundscapes_labels.csv: labels a subset of the files in the soundscapes directory. 5 second intervals, mixed labels i.e. not isolated. Full 234 labels included.
3. Remaining files in the soundscapes directory with no labels

## Help from AI, Vibe Coding

Vibe coding using Claude, Gemini and ChatGpt gets you as far as 0.85, beyond that it requires a lot of experimentation and hit or miss experimentation. The basic approach was as follows:

1. Turn the sound files into MEL Spectrograms - Sound to image
2. Turn them spectrograms into .npy files
3. Train a neural net i.e. EfficientNet, 30 epochs or more.
4. Early stopping to prevent overfitting is important. Train-Val split is required here. 


## Ensembling

Ensembling seemed to be a curious case. https://github.com/gany-c/BirdClef2026/blob/main/docs/ensembles.md

The best working ensemble (0.85) was of 2 models an EfficientNet B3 trained with 234 labels (0.84), and a B0 trained with 206 labels. 
https://github.com/gany-c/BirdClef2026/blob/main/inference/ensemble-based-inference.ipynb

An amphibian only B0 was tested based on the above Ensemble Success, it didn't help produce a better one. Training for the same: https://github.com/gany-c/BirdClef2026/blob/main/training/family-specific-b0-training-noteboo.ipynb

## Kaggle's Hardware support

Kaggle provides 20 hours of GPU and TPU free every week. You can purchase more capacity right there. You don't have to set up a training ecosystem in anothere cloud like AWS or GCP.


## Advanced Strategies.

These were pursued in the last 10 days but didn't yield results. Probably a good starting point if there is a BirdClef 2027

1. Transfer Learning from existing models like Google's Perch or Birdnet.
2. Pseudo Labeling the datasets i.e. the soundscape
3. Using features other than sound such as time of day, latitude and longitude etc
4. Downloading sound files freom external sources for the species that were underrepresented or missing in train.csv 


### Transfer Learning

1. Transfer Learning from existing models like Google's Perch or Birdnet.
2. They can't be directly used in the inference, because they need GPUs which inference doesn't allow. Their CPU only execution time exceeds the 90 minute limit
3. So a simple linear model has to be trained using these models. The publicly available models didn't produce embeddings https://github.com/gany-c/BirdClef2026/blob/main/docs/transfer_learning_paper.md

### Pseudo Labeling

1. We used our own successful models, i.e. the ones that scored 0.85 as teachers
2. The sound files mentioned in train.csv are longer than 5 seconds, and provide species specific samples in isolation. Pseudo Labeling could be used to help trim the relevant parts of long sound clips. - The model didn't yield good results. https://github.com/gany-c/BirdClef2026/blob/main/pseudo-labeling/chunking-and-pseudo-labelling.ipynb
3. A good portion of the soundscapes were not labelled. Soundscapes have multiple labels as well. An attempt was made to use the best ensemble as a teacher model and label a portion of the soundscapes. These labelled soundscapes were then used in training another model. It didn't really help https://github.com/gany-c/BirdClef2026/blob/main/pseudo-labeling/pseudo-labeling-soundscapes.ipynb
4. Some of the winning submissions seem to have used pseudo-labelling iteratively https://github.com/gany-c/BirdClef2026/blob/main/docs/topper_strategy_2025

### More features

1. Location is given differently in the soundscapes and the training data. In soundscapes it is given as a sector or region with an alphanumeric code e.g. S22. In train.csv it is mentioned as latitude and longitude.

2. Didn't get the time to experiment with mutliple new features.

### External Data Sources

Sound files for the missing species were not externally available, only document pages e.g. https://www.inaturalist.org/taxa/517063-Pithecopus-azureus