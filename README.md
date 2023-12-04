<br/>
<p align="center">
  <h3 align="center">Unlock the Power of Prediction: Mastering Classification with Machine Intelligence</h3>

  <p align="center">
    Classify with Confidence: Where Data Meets Decisiveness
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Scenario

An automotive industry analyst is tasked with understanding the factors that contribute to the safety rating of vehicles. The analyst has access to a dataset, `car.data.txt`, containing various characteristics of cars, such as maintenance cost, number of doors, luggage capacity, safety ratings, etc. The dataset is categorical and needs to be converted into a numerical format suitable for machine learning algorithms. The analyst is required to build a predictive model that can accurately classify cars based on their safety ratings using these features.

### Solution

1. **Data Preprocessing**:
   - The analyst begins by reading the dataset from the `car.data.txt` file, parsing each line into a list of categorical features.
   - To handle categorical data, the analyst employs `LabelEncoder` from `sklearn.preprocessing` to convert each categorical feature into a numerical format, which is a requirement for most machine learning algorithms.

2. **Data Splitting**:
   - Once the data is numerically encoded, it is split into features (`X`) and labels (`y`), with `y` being the safety rating of each car.
   - The dataset is further divided into a training set and a testing set, with 20% of the data reserved for testing the model's performance.

3. **Model Selection and Hyperparameter Tuning**:
   - The analyst decides to use a `RandomForestClassifier` due to its robustness and ability to handle non-linear relationships between features.
   - To optimize the model, a grid search (`GridSearchCV`) is conducted over a predefined range of hyperparameters (`n_estimators` and `max_depth`), using cross-validation to ensure that the model does not overfit.

4. **Model Training and Evaluation**:
   - With the best hyperparameters found (`max_depth`: 15, `n_estimators`: 200), the analyst trains the RandomForestClassifier.
   - The model's feature importances are analyzed, allowing the analyst to determine which car characteristics are most significant in predicting safety ratings.
   - The model is then evaluated on the test set, achieving a high accuracy of 99%, indicating excellent predictive performance.

5. **Learning Curve Analysis**:
   - To validate the model's performance further and to understand if more data would improve the model, a learning curve is generated.
   - The learning curve plots the model's training accuracy against the number of training samples, which should ideally show improvement in accuracy as more data is used for training.

6. **Visualization and Interpretation**:
   - The analyst creates a plot of the learning curve using `matplotlib.pyplot` to visually analyze how the training and validation scores change with the increase in training data.
   - The plot helps in assessing if the model would benefit from more data or if it has reached a plateau, indicating that the model has learned as much as it can from the data provided.

The solution presented provides a systematic approach to predictive modeling in a real-world automotive safety rating classification task. The analyst can use the insights from feature importance and learning curve analysis to make informed decisions on vehicle safety improvements and to understand the impact of different car features on safety ratings.

The code snippet is a comprehensive example of a machine learning workflow using Python, specifically focusing on the classification of a dataset using a RandomForestClassifier from the `sklearn` library. Let's break down the code into its key components and functionalities:

1. **Importing Libraries**:
   - `numpy`: Used for numerical operations on arrays.
   - `matplotlib.pyplot`: For plotting graphs, such as learning curves.
   - `sklearn.preprocessing`, `sklearn.ensemble`, `sklearn.model_selection`: Sub-modules of `sklearn` used for data preprocessing, machine learning models, and model evaluation techniques respectively.

2. **Data Reading and Preprocessing**:
   - Reads a dataset from 'car.data.txt'.
   - Splits each line into a list of features.
   - Uses `LabelEncoder` to convert categorical data into numerical format since machine learning models in `sklearn` require numerical input.

3. **Data Splitting**:
   - Splits the processed data into features (`X`) and labels (`y`).
   - Further splits these into training and testing sets using `train_test_split` for model training and evaluation.

4. **Hyperparameter Tuning**:
   - Uses `GridSearchCV` to find the best hyperparameters for the `RandomForestClassifier`.
   - Defines a grid of parameters (`param_grid`) to try out.
   - Fits the grid search to the training data and identifies the best parameters and score.

5. **Building and Evaluating the Classifier**:
   - Retrieves the best estimator from the grid search.
   - Analyzes feature importances to understand which features contribute most to the prediction.
   - Evaluates the model's accuracy on the test set.

6. **Learning Curve Analysis**:
   - Generates a learning curve using `learning_curve`.
   - This curve helps in understanding how the model's performance improves with the increase in the number of training samples.
   - Plots this information using `matplotlib.pyplot` to visualize the learning process.

In summary, the script is a complete example of a machine learning task involving data preprocessing, model selection and tuning, training, evaluation, and analysis of the results. It highlights key steps in predictive modeling, particularly using `RandomForestClassifier` for a classification task.

The output results shown, including the learning curve plot, are the culmination of executing the provided Python script for a machine learning task. Here's a detailed explanation of the output results:

1. **Best Hyperparameters**:
   - The output `Best Parameters: {'max_depth': 15, 'n_estimators': 200}` indicates that the best performing RandomForestClassifier has a maximum depth of 15 and 200 trees (n_estimators). These parameters were selected after a grid search cross-validation process over the specified parameter grid.

2. **Cross-Validation Score**:
   - `Best Score: 0.9710406529587192` represents the best cross-validation accuracy score achieved by the RandomForestClassifier with the best hyperparameters. It means that, on average, the model correctly predicted 97.1% of the outcomes during the cross-validation phase.

3. **Feature Importance**:
   - The feature ranking section lists the features in order of importance as determined by the RandomForestClassifier. For example, `feature 5 (0.27721776218307925)` means that the 6th feature (indexing starts at 0) is the most important, contributing approximately 27.7% to the model's decision-making process. The values in parentheses represent the importance scores assigned to each feature.

4. **Test Set Accuracy**:
   - `Test Accuracy: 0.99` shows that the model achieved 99% accuracy on the unseen test data, indicating high generalization performance.

5. **Learning Curve Plot**:
   - The learning curve plot visualizes the performance of the model as the number of training samples increases. It plots the accuracy on the y-axis against the number of training samples on the x-axis. In this case, the plot should show the accuracy plateauing as the number of training samples increases, which is typical when a model has learned as much as it can from the data provided.
   - However, the plot shown has an issue: the y-axis represents accuracy percentages, which should be between 0 and 100. Yet, the plot shows values over 100, which is not possible for accuracy metrics. This indicates there might be an error in the way the accuracy is calculated or plotted.

6. **Process Exit Code**:
   - `Process finished with exit code 0` indicates that the program ran successfully without any errors.

This summary provides insight into the model's selection, training, and evaluation process. 

## Built With

This project leverages several powerful libraries and tools in Python to perform data preprocessing, machine learning, hyperparameter tuning, and visualization. Below is a detailed "Built With" section suitable for a GitHub repository README:

#### Data Processing and Analysis
- **NumPy** - A fundamental package for scientific computing with Python, used for handling high-level mathematical functions and multi-dimensional arrays. [NumPy](https://numpy.org/)

#### Machine Learning Algorithms
- **Scikit-learn** - An efficient and simple toolkit for predictive data analysis built on NumPy, SciPy, and matplotlib. It's used in this project for various tasks:
  - `preprocessing.LabelEncoder` - A utility class to help convert categorical data into a numerical format that can be interpreted by the machine learning algorithms.
  - `ensemble.RandomForestClassifier` - A meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  - `model_selection.train_test_split` - A function to split arrays or matrices into random train and test subsets.
  - `model_selection.GridSearchCV` - An exhaustive search over specified parameter values for an estimator, used to optimize the hyperparameters of the model.
  - `model_selection.learning_curve` - A tool to generate a simple plot of the test and training learning curve which shows the validation and training score of an estimator for varying numbers of training samples.
  - [Scikit-learn (sklearn)](https://scikit-learn.org/stable/)

#### Visualization
- **Matplotlib** - A comprehensive library for creating static, interactive, and animated visualizations in Python. Used in this project to plot the learning curve of the model performance.
  - `matplotlib.pyplot` - A collection of command style functions that make matplotlib work like MATLAB, used here for plotting the learning curve to assess the model's performance.
  - [Matplotlib](https://matplotlib.org/)

#### Dataset
- The dataset used in this project, referenced as `car.data.txt`, contains categorical data related to various characteristics of cars. It's processed and utilized to train a machine learning model to predict safety ratings.

This "Built With" section provides an overview of the technologies and tools implemented in the project, including their purpose and contribution to the workflow. Each component is linked to its official website, where users can find more information and documentation on how to use them.Here are a few examples.

## Getting Started

This section guides you through the setup and running of the machine learning project which predicts vehicle safety ratings using the RandomForestClassifier.

#### Prerequisites
Before you begin, ensure you have met the following requirements:
- You have installed Python 3.6 or higher.
- You have installed pip, Python’s package installer.

#### Installation
To set up this project, follow these steps:

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-repo/vehicle-safety-rating-predictor.git
   cd vehicle-safety-rating-predictor
   ```

2. **Set up a virtual environment** (optional but recommended)
   ```sh
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install the required packages**
   ```sh
   pip install numpy matplotlib scikit-learn
   ```

#### Data
Make sure you have the `car.data.txt` file placed in the parent directory relative to the project or modify the `input_file` variable in the script to point to the correct location of the data file.

#### Usage
Run the script using Python:
```sh
python vehicle_safety_predictor.py
```

This script will:
- Preprocess the data from `car.data.txt`.
- Split the data into training and testing sets.
- Perform hyperparameter tuning to find the best parameters for the RandomForestClassifier.
- Train the RandomForestClassifier with the best found parameters.
- Display the importance of each feature used in prediction.
- Evaluate the model on the test data and print the accuracy.
- Plot the learning curve to show how the model's performance improves with more data.

#### Output
When you run the script, you will see the following output in the terminal:
- The best hyperparameters for the RandomForestClassifier.
- The best cross-validation score obtained during the grid search.
- A ranking of features based on their importance in the model.
- The accuracy of the model on the test data.
- A plot showing the learning curve of the model.

#### Contributing
If you have suggestions for how this project could be improved, or want to report a bug, please file an issue in this repository.

Follow these steps to contribute to the project:
1. Fork the repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

#### License
This project uses the following license: [LICENSE](LICENSE.md).

This "Getting Started" section provides a comprehensive guide for users to set up and run the project. It includes instructions for installation, usage, contributing to the project, and the licensing information.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/TribeOfJudahLion//issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/TribeOfJudahLion//blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TribeOfJudahLion//blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion/) - **

## Acknowledgements

* []()
* []()
* []()
