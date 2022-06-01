target =

help:
	@echo "Type:"
	@echo "  'make' or 'make help' to show this help message"
	@echo "  'make data' to download and extract the dataset"
	@echo "  'make features' to extract features from the dataset"
	@echo "  'make train' to train the model"
	@echo "  'make predict target=(target audio file path)' to run predictions"

data:
	@python src/data/make_dataset.py

features:
	@python src/data/build_features.py

train:
	@python src/models/train_model.py

predict:
	@python src/models/predict_model.py $(target)

.PHONY: help data features train predict