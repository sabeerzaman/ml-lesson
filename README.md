# Supervised Learning: Classification and Prediction (aka, Regression)

<!-- TOC depthFrom:2 -->

- [Challenge / Activity](#challenge--activity)
  - [Instructions](#instructions)
    - [Classification problem: Breast Cancer Diagnosis from Diagnostic Images](#classification-problem-breast-cancer-diagnosis-from-diagnostic-images)
    - [Regression problem: Predict Boston housing prices](#regression-problem-predict-boston-housing-prices)
  - [Hints](#hints)
- [Jupyter Notebook setup](#jupyter-notebook-setup)
- [Without Jupyter Notebook](#without-jupyter-notebook)
- [Additional Readings / Resources](#additional-readings--resources)

<!-- /TOC -->

## Challenge / Activity

### Instructions

#### Classification problem: Breast Cancer Diagnosis from Diagnostic Images

Use scikit-learn's "Breast cancer wisconsin (diagnostic) dataset" to create a decision tree classifier to help predict if there is a malignant or benign tumor based on diagnostic images of patients. Use 75% of the dataset to train the classifier, and the remaining 25% to validate its accuracy.

Cover the following points:

- What is the prediction accuracy of your trained model?
- What happens if you re-run the code 2-3 times? Does your prediction accuracy remain the same? Can you explain why?
- What happens to the accuracy if you instead did a 50-50 split between train-test data? What about 25-75 split (25% for training, 75% for testing)? Can you explain what's happening?

**Bonus:**

- Use Graphviz to visualize your decision tree
  - What is the "Gini" score that shows up for each node?
- Train using a different classification model (such as logistic regression) - how does that compare against the Decision Tree results above?

#### Regression problem: Predict Boston housing prices

Use scikit-learn's "Boston house prices dataset" to create a linear regression model to help predict housing prices based on various measures about the neighbourhood. Use 75% of the dataset to train the classifier, and the remaining 25% to validate its accuracy.

Cover the following points:

- What is the prediction accuracy of your trained model?
- What happens if you re-run the code 2-3 times? Does your prediction accuracy remain the same? Can you explain why?
- What happens to the accuracy if you instead did a 50-50 split between train-test data? What about 25-75 split (25% for training, 75% for testing)? Can you explain what's happening?

**Bonus:**

- Train using a different regression model (such as a decision tree) - how does that compare against the Linear Regression results above?

### Hints

Here are the notes to key steps / commands from scikit-learn used in the in-class code walkthrough to guide you (you'll need to tailor them to the specific assignment):

1. Load the data:

    ```python
    from sklearn import
    iris = datasets.load_iris()
    ```

2. Train the model:

    ```python
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    ```

3. Validate the model accuracy:

    ```python
    score = clf.score(X_test, y_test)
    ```

If you're really stuck, you can peek at the Jupyter notebooks in the "Lesson Notebooks" folder in this repository... but challenge yourself first!

## Jupyter Notebook setup

The use of [Jupyter notebooks](https://jupyter.org/) are not required, but it's highly recommended.

1. Recommend the use of `minicconda` to manage Python dependencies - see [installation instructions](https://conda.io/en/latest/miniconda.html)
2. Commands to run after installing miniconda:

  ```sh
  $ conda create -n brainstation-lesson jupyter scikit-learn matplotlib python-graphviz
  $ source activate brainstation-lesson
  $ jupyter notebook
  # Will launch the notebook in a new browser tab
  ```

## Without Jupyter Notebook

If you wish to do the challenges and/or follow the lesson code without using a Jupyter notebook (not recommended), you will still need to setup your Python environment, after which you can run each of the commands in the notebook one at time from the Python shell.

1. Recommend the use of `minicconda` to manage Python dependencies - see [installation instructions](https://conda.io/en/latest/miniconda.html)
2. Commands to run after installing miniconda:

  ```sh
  $ conda create -n brainstation-lesson jupyter scikit-learn matplotlib python-graphviz
  $ source activate brainstation-lesson
  $ python
  # Will open up the Python shell
  ```

## Additional Readings / Resources

- [Machine Learning Mastery: Classification versus Regression in Machine Learning](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [A delightful visual guide to understanding decision trees](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
