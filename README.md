# Application of Deep Learning for Stock Market Price Prediction

## Supervisor
- Assoc. Prof. Mehmet Keskinoz

## Group Members
- Abdullah Usame Gunay
- Mehmed Emin Guvenlioglu
- Irem Ozaydin
- Sara Nil Acarsoy

Long-Short Term Memory network was implemented using Python's Keras package to predict the closing price of S&P 500 stock market index. Time series data contains the daily data for S&P 500 Index prices and the other features such as DOW JONES, USD/GBP, and APPLE etc.

## Project Files

- run.py: The main file used for running the code. The values for parameters in config file is read and data is loaded using DataLoader class. Then, the model is built with the hyperparameters in the config file. Training data is split into batches and training data is generated using model.train_generator() instead of Keras' model.train() which may cause memory overflows. Lastly, predictions are made point by point and the results are plotted. 

- data_processor.py: DataLoader class contains the functions for getting train data, getting test data, generating the training batch, proceeding to next window, and normalizing the windows. 

    - get_test_data: Creates x, y test data windows
    - get_train_data: Creates x, y train data windows
    - generate_train_batch: Yields a generator of training data from filename on given list of columns split for train/test
    - _next_window: Generates the next data window from the given index location
    - normalise_windows: Normalise the windows with a base value of zero


- model: Contains Model class which is used to build the model instance.
    - load_model: Loads the model from a previously saved file
    - build_model: Sets the parameters in the config file to the model, builds the model and compiles. Elapsed time is recorded for reporting purposes.
    - train: Fits the model with the given x, y, epoch and batch size. Callback functions are used to save the model after every epoch (ModelCheckpoint) and stop the training when the MSE stops improving (EarlyStopping).  Elapsed time is recorded for reporting purposes.
    - train_generator: Same as the train() function but train() function loads the full dataset into memory, then applies the normalizations to each window in-memory causing a memory overflow. So instead, train_generator() is utilized to allow for dynamic training of the dataset which will minimize the memory utilization dramatically.
    - predict_point_by_point: Makes predictions day by day by predicting one step ahead each time 
    

- utils: Contains the Timer class which calculates elapsed time for each epoch

- config.json: The configuration file which include the parameters and hyperparameters for the data, training, model and LSTM network. 

Parameters for data:

- filename: Filename of the input data
- columns: Number of columns that will be used as features
- sequence_length: The size of the each window
- train_test_split: The ratio of train and test data
- normalise: Normalizing the data if True

Training parameters:

- epochs: Number of times the entire data is passed through the model
- batch_size: Number of training examples utilized in one iteration

Model hyperparameters:

- loss: Loss function used for the performance evaluation
- optimizer: Optimizer used for the model
- type: Layer type
- neurons: Number of neurons in layer
- dropout: Ratio of neurons that is dropped after a layer
- activation: Activation function

The optimal values for the above-mentioned parameters and hyperparameters are;

* "sequence_length": 50
* "train_test_split": 0.60
* "epochs": 15
* "batch_size": 32
* "loss": "mse"
* "optimizer": "nadam"
* "activation": "elu"
* "neurons": 100
* "dropout": 0.2

[An article write-up for this code](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)

## Requirements

Install requirements.txt file to make sure correct versions of libraries are being used.

* Python 3.5.x
* TensorFlow 1.10.0
* Numpy 1.15.0
* Keras 2.2.2
* Matplotlib 2.2.2

## Running the code

* Change the directory to the folder containing requirements.txt with cd command
* Install the required libraries in requirements.txt with pip install -r requirements.txt
* After installing the requirements, run the file named run.py
* Profit!
