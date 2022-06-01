# Gender and Age Estimation of Voice with Mel Spectrogram and CNN

## Prerequisites

`make` is recommended, but not required.

`python -m pip install -r requirements.txt` will install all necessary packages.

`requirements.txt` lists `tensorflow-gpu` as a requirement; if you desire the use of `tensorflow`, edit out `tensorflow-gpu` from `requirements.txt` and replace with `tensorflow`.

## Running

```
Type:  
 'make' or 'make help' to show this help message  
 'make data' to download and extract the dataset  
 'make features' to extract features from the dataset  
 'make train' to train the model  
 'make predict' to run prediction
```

From `make help`

Alternatively, the python scripts can be executed directly, although it is not recommended to do so.

`make data` == `python src/data/make_dataset.py`

`make features` == `python src/data/build_features.py`

`make train` == `python src/models/train_model.py`

`make predict` == `python src/models/predict_model.py`

## License

This project is licensed under the free and permissive MIT license. Read `LICENSE` for more details.
