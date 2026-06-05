## Ensemble submission or inference notebook

1. Ensemble submission: https://www.kaggle.com/code/gany24558/ensemble-based-inference?scriptVersionId=321047201

## Ensembles

1. May 31 2026: New Binary Ensemble created, score remains at 0.850 (rare-amphibian-oversample/1, syntheticdata/8/)
	1. Weights tried are 0.8-0.2, 0.85 - 0.15, 0.9 - 0.1
2. May 21 2026: New Binary Ensemble created with score of 0.850 ( composite-primary-labels/1/, syntheticdata/8/ )
3. May 21 2026: New Ensemble of 0.847 score created
4. May 13 2026: Models used in Ensemble (0.839 score) are:
    1. "/kaggle/input/models/gany24558/birdclef-apr-20-2026-gc/pytorch/secondary-labels/4/bird_model_best.pth" (possible base model) or "/kaggle/input/models/gany24558/birdclef-apr-20-2026-gc/pytorch/secondary-labels/1/bird_model_best.pth"
    2. "/kaggle/input/models/gany24558/birdclef-apr-20-2026-gc/pytorch/efficient-b3/2/bird_model_best.pth"
    3.  "/kaggle/input/models/gany24558/birdclef-apr-20-2026-gc/pytorch/syntheticdata/8/bird_model_best.pth"


## Models

1. New base - /kaggle/input/models/gany24558/birdclef-apr-20-2026-gc/pytorch/composite-primary-labels/1/bird_model_best.pth
   
### rare-amphibian-oversample/1/bird_model_best.pth(Base - 0.8 to 0.9)

1. Date: Sun May 31 2026
2. Functionality: Uses B3 model, has all functionality of composite-primary-labels, over samples 3 amphibians that are not in train but present in soundscapes, 50 Epochs
3. Individual Score: 0.843
4. Submission notebook: https://www.kaggle.com/code/gany24558/single-model-inference-notebook?scriptVersionId=323467560


### composite-primary-labels/1/bird_model_best.pth ( Base - 0.8 weight)

1. Date: Thu May 21 2026
2. Functionality: Uses B3 model, Uses all primary labels in a single row in training, 50 Epochs
3. Individual Score: 0.840
4. Submission notebook: https://www.kaggle.com/code/gany24558/single-model-inference-notebook?scriptVersionId=321028679


### secondary-labels/4/bird_model_best.pth (Base -0.7 weight)

1. Date: Wed May 13 2026
2. Functionality: Uses B3 model, Uses Ratings and Secondary labels in training, about 50 Epochs
3. Individual Score: 0.828
4. Submission notebook: https://www.kaggle.com/code/gany24558/notebook-gc-2026?scriptVersionId=318840827

### secondary-labels/1/bird_model_best.pth (Former Base - 0.7 weight)

1. Date: Tue May 12 2026
2. Functionality: Uses B3 model, 234 labels, Uses Secondary labels in training, 30 epochs
3. Individual Score: 0.826
4. Submission notebook: https://www.kaggle.com/code/gany24558/notebook-gc-2026?scriptVersionId=318677894

### efficient-b3/2/bird_model_best.pth (Support model - 0.15 weight)

1. Date: May 4 2026
2. Functionality: Uses B3 model, full 234 set of labels or species. Soundscape data usage comes along with this.
3. Individual Score: 0.817
4. Submission notebook: https://www.kaggle.com/code/gany24558/notebook-gc-2026?scriptVersionId=316534848

### syntheticdata/8/bird_model_best.pth (Support model - 0.15 weight)

1. Date: May 2 2026
2. Functionality: Highest scoring B0 model, uses only 206 labels. Soundscape data is not used.
3. Individual Score: 0.742
4. Submission notebook: https://www.kaggle.com/code/gany24558/notebook-gc-2026?scriptVersionId=315933120
