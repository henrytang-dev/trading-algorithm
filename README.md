# Stock Price Prediction with LSTM (Work in Progress)

Also view current notebook on [Kaggle](https://www.kaggle.com/code/henrytang05/long-term-stock-prediction)

This repository contains a work-in-progress machine learning project aimed at predicting stock prices using Long Short-Term Memory (LSTM) networks. The project is currently in its early stages and focuses on implementing LSTM-based models for stock price prediction. This README provides an overview of the project's goals, the tools and libraries being utilized, and how to get started with the code.

## Table of Contents

- [Introduction](#introduction)
- [Goals](#goals)
- [Tools and Libraries](#tools-and-libraries)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Predicting stock prices is a complex task that involves various factors and can be influenced by market sentiment, economic indicators, and more. This project aims to develop a machine learning model currently based on LSTM networks to predict stock prices. LSTM networks are well-suited for sequential data like time series, making them a promising choice for this task.

## Goals

The main goals of this project include:
- Building a robust algorithm for day-trading and long-term trading
- Develop an effective trading strategy for generating consistent returns
- Learn more about the stock market, finance, economics, and machine learning
- Investigating the model's performance at different time intervals (e.g., daily, weekly) and incorporating backtesting
- Experimenting with different hyperparameters to optimize the model's performance.

## Tools and Libraries

This project uses the following tools and libraries:
- Python
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- TensorFlow for building and training the LSTM model
- Scikit-Learn's MinMaxScaler for scaling data
- Yahoo Finance for loading OHLC stock data

## Data Preprocessing

1. **Data Collection**: Obtain historical stock price data using Yahoo Finance API and store in Pandas dataframe

2. **Data Preprocessing**: Use Pandas to clean and preprocess the data. Handle missing values, convert dates to the appropriate format, and organize the data into sequences suitable for LSTM training.

3. **Data Scaling**: Use Scikit-Learn's MinMaxScaler to scale the data between 0 and 1. This helps ensure similar and accurate weighting of features

## Model Architecture

The model architecture involves creating an LSTM-based neural network. I experimented with different configurations, including the number of LSTM layers, hidden units, dropout layers, and more.

## Training

1. Train the LSTM model using TensorFlow's Keras API. I have experimented with various hyperparameters and layer structures to optimize model performance.

## Evaluation

1. I evaluated the model's performance using Root Mean Squared Error (RMSE) which was consistently between 1-2

2. Visualized the model's predictions against actual stock prices using Matplotlib and Seaborn

## Future Work

- Implement more advanced features, such as sentiment analysis of news related to stocks
- Experiment with different architectures like Bidirectional LSTMs, stacked LSTMs, and transformers
- Incorporating TensorFlow callback functions for statistical analysis
- Supplementing different model structures with dropout and normalization to prevent overfitting and variance
- Incorporate techniques such as Random Forest for identifying important features for training
- Using indicators such as RSI and MACD to improve accuracy of the model

## License

This project is licensed under the [MIT License](LICENSE).

---

Please note that this project is still a work in progress. Contributions and feedback are welcome. If you have any questions or suggestions, please feel free to open an issue or reach out to the contributor. Happy coding!
