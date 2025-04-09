# Wine Quality Prediction using k-NN

This repository is part of a group project for the course **Ứng dụng AI trong kinh doanh và quản lý** supervised by TS. Lưu Minh Tuấn. The project implements a wine quality prediction system using the k-Nearest Neighbors (k-NN) classification algorithm, combined with an interactive web application built with Streamlit.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Description](#modules-description)
- [License](#license)

## Overview

The main objectives of this project are to:
- Analyze and visualize the wine quality dataset.
- Build and train a k-NN classifier to predict the quality of red wine based on physicochemical properties.
- Evaluate the model performance using key metrics (accuracy, confusion matrix, classification report).
- Provide an interactive web application where users can explore the data, train the model, and predict the quality of wine samples.
- Support multiple languages (English and Vietnamese) for broader accessibility.

## Project Structure
. ├── app.py # Main application file that handles navigation and module imports ├── README.md # This file: project documentation and instructions ├── Data │ └── winequality-red.csv # Red wine quality dataset ├── lang │ └── translations.json # Language translations for the app (English and Vietnamese) └── modules ├── introduction.py # Module for the Introduction section ├── eda.py # Module for Exploratory Data Analysis (EDA) ├── training.py # Module for preprocessing and model training └── prediction.py # Module for wine quality prediction

## Dataset

The project uses the **winequality-red.csv** dataset provided by the University of California, Berkeley. This dataset contains a variety of physicochemical properties of red wine (such as fixed acidity, volatile acidity, citric acid, residual sugar, etc.) along with a quality rating. The file is stored in the `Data` folder.

## Technologies Used

- **Python**: Programming language.
- **Streamlit**: Framework for building interactive web applications.
- **Pandas & NumPy**: Data manipulation and numerical computations.
- **Matplotlib & Seaborn**: Data visualization libraries.
- **Scikit-learn**: Machine learning library for building and evaluating the k-NN classifier.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. **reate a virtual environment (optional but recommended):**

'''bash
python -m venv venv
source venv/bin/activate  
nstall the dependencies:

Ensure you have a requirements.txt file. A sample content for requirements.txt is shown below:

txt

streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
Then run:

bash
Sao chép
pip install -r requirements.txt
