# Gym Exercise Classification

This project implements time series classification algorithms in `sktime` and `sklearn` to predict the gym exercises from time series data. The aim is to classify the following exercises:

- Bench press
- Deadlift
- Military press
- Squats
- Sit ups
- Press ups
- Pull ups
- Stationary

The project has two main components:

1. Data preprocessing: The time series data is preprocessed to convert it into a format suitable for time series classification algorithms.
2. Model training and evaluation: The preprocessed data is used to train and evaluate time series classification models in `sktime` and `sklearn`.

## Data

The dataset used in this project consists of time series data of exercises performed by individuals using the accelerometer and gyroscope sensors of a smartphone. Each exercise is labeled with one of the eight possible exercise types mentioned above. The dataset is split into training and test sets, with 50% of the data used for training and 50% for testing.

## Preprocessing

The time series data is preprocessed as follows:

1. The time series is segmented into fixed-length windows of 100 data points, giving you a 10 second recording of each exercise.
2. Each window is normalized using z-score normalization.

## Models

The following models are trained and evaluated in this project:

1. K-Nearest Neighbors with Euclidean distance (kNN-ED)
2. K-Nearest Neighbors with Dynamic Time Warping (kNN-DTW)
3. Random Forest
4. AdaBoost
5. Decision Tree
6. Naive Bayes
7. Multi-layer Perceptron (MLP)
8. Time Series Forest (TSF)
9. Boss Ensemble (BOSS)
10. Rocket

The models are trained using the preprocessed data and evaluated using train time, accuracy score, balanced accuracy, precision, recall, f1 score, auroc and confusion matrix.


## Getting Started

To use this project, you need to have the following libraries installed:

- `numpy`
- `pandas`
- `sklearn`
- `sktime`
- `matplotlib`
- `Flask`

Clone the repository to your local machine and navigate to the project directory. Run the following command to preprocess the data:

python format_data.py

This will preprocess the data and save it in the `data` directory.

To create the datasets, run the following command to preprocess the data:

python create_train_test_splits_datasets.py

To train and evaluate the models, run the following command:

python main.py

This will train the models and save the results in the `results` directory.

## Results

The accuracy scores and confusion matrices for the trained models are saved in the `results` directory. The Rocket model achieves the highest accuracy score of 98.9%.

## Conclusion

This project demonstrates the effectiveness of time series classification algorithms in predicting gym exercises from time series data. The Rocket model outperforms the other models in accuracy, suggesting that it may be the most suitable model for this task.


## Running Flask Server
to run flask server locally flask run --host=0.0.0.0