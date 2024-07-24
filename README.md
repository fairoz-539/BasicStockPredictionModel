# 📈 BasicStockPredictionModel

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Table of Contents
- [📊 Overview](#-overview)
- [🌟 Features](#-features)
- [🛠️ Dependencies](#️-dependencies)
- [🛠️ Installation](#️-installation)
- [📁 Dataset](#-dataset)
- [🚀 Usage](#-usage)
- [📈 How It Works](#-how-it-works)
- [📊 Results](#-results)
- [📝 Important Notes](#-important-notes)
- [🔍 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📧 Contact](#-contact)
- [🙏 Acknowledgements](#-acknowledgements)

## 📊 Overview

BasicStockPredictionModel is a Python-based project that leverages the power of machine learning to predict future stock prices. Using Linear Regression from the scikit-learn library, this model has been specifically tested with Google stock data, achieving an impressive prediction accuracy of 96-98%.

## 🌟 Features

- 🤖 Utilizes Linear Regression algorithm for robust stock price prediction
- 🎯 Achieves high accuracy (96-98%) on test data
- 📊 Visualizes predictions using Matplotlib for easy interpretation
- 🔢 Processes and analyzes complex stock data with Pandas and NumPy
- 🔮 Predicts future stock prices based on historical data

## 🛠️ Dependencies

The project relies on the following Python libraries:

- 🧮 math: For mathematical operations
- 🗓️ datetime: For handling date and time operations
- 🐼 pandas: For data manipulation and analysis
- 🔢 numpy: For numerical computing
- 🧠 scikit-learn: For machine learning algorithms and data preprocessing
- 📊 matplotlib: For creating static, animated, and interactive visualizations
- 🥒 pickle: For serializing and deserializing Python object structures

Install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## 🛠️ Installation

1. Clone the repository: git clone [https://github.com/fairoz-539/BasicStockPredictionModel/blob/main/BasicStockPredictionModel.py](https://github.com/fairoz-539/BasicStockPredictionModel/blob/main/BasicStockPredictionModel.py)
2. Then Navigate to the file : 
`cd BasicStockPredictionModel`
3. Create a virtual environment (optional but recommended):
`python -m venv venv`
`source venv/bin/activate`
# On Windows, use 
`venv\Scripts\activate`
3. Install the required dependencies:
   `pip install -r requirements.txt`

## 📁 Dataset

This model is trained and tested on Google stock data (GOOGL.csv). Ensure this file is located in the correct directory before running the script. The dataset should include columns for Date, Open, High, Low, Close, and Volume.

## 🚀 Usage

Follow these steps to use the BasicStockPredictionModel:

1. Clone this repository to your local machine
2. Install the required dependencies as mentioned above
3. Place your GOOGL.csv file in the project directory
4. Run the Python script:
```bash
python stock_prediction_model.py
```

## 📈 How It Works

The BasicStockPredictionModel follows these key steps:

### 1. Data Preprocessing 🧹

- Reads the Google stock data from CSV
- Calculates additional features:
  - High-Low Percentage: `(High - Close) / Close * 100`
  - Percent Change: `(Close - Open) / Open * 100`

### 2. Feature Engineering 🛠️

- Creates a 'Label' column as the target variable
- This represents the future stock price we want to predict

### 3. Data Splitting ✂️

- Separates the data into features (X) and target variable (y)
- Splits the data into training and testing sets (80% train, 20% test)

### 4. Model Training 🏋️‍♀️

- Initializes a Linear Regression model
- Fits the model on the training data

### 5. Prediction 🔮

- Uses the trained model to predict future stock prices
- Applies the model to the most recent data points

### 6. Visualization 📊

- Plots the actual stock prices and predicted values using Matplotlib
- Provides a visual representation of the model's performance

## 📊 Results

The BasicStockPredictionModel achieves a prediction accuracy of 96-98% on the test data. This high accuracy demonstrates the model's effectiveness in capturing the underlying patterns in Google's stock price movements.

## 📝 Important Notes

- 🚨 This model is for educational and research purposes only. It should not be used as the sole basis for real-world trading decisions.
- 💼 Always consult with financial experts and consider multiple factors before making investment decisions.
- 🔄 The stock market is influenced by many complex factors. Past performance does not guarantee future results.
- 🛠️ Regular model updates and retraining may be necessary to maintain accuracy as market conditions change.

## 🔍 Future Improvements

- 📊 Incorporate more advanced features like technical indicators
- 🧠 Experiment with other machine learning algorithms (e.g., Random Forests, Neural Networks)
- 🕰️ Implement time series cross-validation for more robust model evaluation
- 📈 Add functionality to handle multiple stocks and different time frames

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or feedback, please send me an [email](fairosahmed.ai@gmail.com) Let's connect.

## 🙏 Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) for their excellent machine learning library
- [Pandas](https://pandas.pydata.org/) for data manipulation tools
- [Matplotlib](https://matplotlib.org/) for data visualization

## 👨‍💻 Author
[Fairoz Ahmed]
