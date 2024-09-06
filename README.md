# Titanic: Machine Learning from Disaster

### Repository Link: [Titanic-MachineLearning-from-Disaster](https://github.com/UtkarshRaj130/Titanic-MachineLearning-from-Disaster)

## Objective
This project aims to develop a machine learning model to predict whether a passenger on the Titanic survived the disaster or not, using the dataset provided by the Titanic competition on Kaggle.

## Dataset
- **train.csv**: The training dataset containing the details of passengers and their survival status.
- **test.csv**: The test dataset where we predict the survival status of passengers.
- **gender_submission.csv**: A simple benchmark submission file provided by Kaggle.

## Solution Overview
Our approach utilizes **Neural Networks** to classify passengers as either survivors or non-survivors. We focused heavily on **data preprocessing** and **feature engineering** to maximize model performance.

### Steps Followed:
1. **Data Preprocessing**:
   - Filled missing values for `Age`, `Embarked`, and `Fare` using median and mode values.
   - Created a binary feature `CabinAvailable` to indicate whether a cabin number was provided.
   - Engineered new features like `FamilySize` and `IsAlone` to capture the passenger's social grouping.
   - One-hot encoding was applied to categorical variables (`Sex`, `Pclass`, and `Embarked`) to convert them into a format suitable for machine learning models.

2. **Model Architecture**:
   - A simple neural network was constructed using TensorFlow and Keras with:
     - **Input Layer**: 64 neurons with ReLU activation.
     - **Hidden Layer**: 32 neurons with ReLU activation.
     - **Output Layer**: 1 neuron with sigmoid activation to handle binary classification.

3. **Model Training**:
   - The neural network was compiled with the **Adam optimizer** and **binary cross-entropy** as the loss function.
   - The model was trained for **50 epochs** with a **batch size of 32** and **20% validation split**.

4. **Model Evaluation**:
   - Achieved an accuracy of approximately **79%** on the test set.
   - Evaluated using precision, recall, F1-score, and confusion matrix to ensure balanced performance across both survival classes.

5. **Test Set Predictions**:
   - Applied the same preprocessing steps to the test data and used the trained model to predict survival status.
   - Prepared and submitted the predictions in the required `submission.csv` format.

## Kaggle Results
- The model achieved a **Kaggle score of 0.79665**, placing our solution in **736th rank** in the Titanic competition.
- You can view the competition details on Kaggle [here](https://www.kaggle.com/competitions/titanic/overview).

## Files in This Repository
- **Final_Notebook.ipynb**: Contains the complete code for data preprocessing, feature engineering, model building, training, evaluation, and prediction.
- **Report.pdf**: A detailed report describing our solution approach, methodology, and results.
- **Contribution_File.pdf**: A file describing the contributions of the three team members involved in the project.
- **submission.csv**: The final predictions submitted to the Kaggle competition.

## Highlights of Our Solution:
1. **Effective Feature Engineering**: We introduced features like `FamilySize` and `IsAlone`, which provided important insights into passengers’ survival likelihood.
2. **Neural Network Implementation**: Built a well-structured neural network that captures complex, non-linear relationships between features, yielding competitive results.
3. **Kaggle Ranking**: Our model achieved a respectable **score of 0.79665** and secured a **rank of 736**, demonstrating strong performance relative to global participants.

## Future Improvements
- Further tuning of hyperparameters and experimenting with deeper neural network architectures could push the performance even higher.
- Implementing other models like **Gradient Boosting** or **Stacking** could provide better ensemble-based results.