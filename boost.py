import warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_data(data):
    # Replace '?' with NaN to mark missing values
    data.replace('?', pd.NA, inplace=True)
    
    # Handle missing values
    for column in data.columns:
        if data[column].isna().sum() > 0:  # Check if column has missing values
            if data[column].dtype in ['float64', 'int64']:  # For numerical columns
                median_value = data[column].median()  # Replace with median
                data[column].fillna(median_value, inplace=True)
            else:  # For categorical columns
                mode_value = data[column].mode()[0]  # Replace with mode
                data[column].fillna(mode_value, inplace=True)
                    
    return data

def main():
    # Load the training and test datasets
    train_file_path = './data/train_final.csv'
    test_file_path = './data/test_final.csv'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    # Preprocess the training and test data
    train_data_preprocessed = preprocess_data(train_data)
    test_data_preprocessed = preprocess_data(test_data)

    # One-hot encoding for categorical columns in the training data
    train_data_encoded = pd.get_dummies(train_data_preprocessed, drop_first=True)
    test_data_encoded = pd.get_dummies(test_data_preprocessed.drop(columns=['ID']), drop_first=True)
    
    # Split the encoded data into features (X) and target (y)
    X = train_data_encoded.drop(columns=['income>50K'])
    y = train_data_encoded['income>50K']
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the XGBoost model
    xgb_model = xgb.XGBClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
    'n_estimators': [200, 400, 600],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'max_depth': [3, 5, 7],  # Maximum depth of trees
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.4, 0.6, 0.8],  # Subsample ratio of columns for each tree
    'gamma': [0.1, 0.3, 0.5],  # Minimum loss reduction required to make a further partition
    'reg_alpha': [0.01, 0.1],  # L1 regularization term on weights
    'reg_lambda': [0.5, 1.0],  # L2 regularization term on weights
    }
    
    # 'n estimators': 400
    # 'learning rate': 0.05
    # 'max depth': 5
    # 'subsample': 1.0
    # 'colsample bytree': 0.6
    # 'gamma': 0.1
    # 'reg alpha':0.01
    # 'reg lambda': 1.0
    
    # Define the scoring metric
    scoring = make_scorer(roc_auc_score, needs_proba=True)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring, cv=3, verbose=1, n_jobs=-1)
    
    # Fit the GridSearchCV model
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Evaluate the best model on the validation set
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation AUC (Best Model): {auc_score}")
    
    ################################ Test ################################ 
    # Align the columns of the test set with the training set
    test_data_encoded = test_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict probabilities on the test data
    test_predictions = best_model.predict_proba(test_data_encoded)[:, 1]

    # Create a submission DataFrame
    submission = pd.DataFrame({
        'ID': test_data_preprocessed['ID'],
        'Prediction': test_predictions
    })

    # Save the submission file
    submission_file_path = './data/submission_predictions_boost.csv'
    submission.to_csv(submission_file_path, index=False)

    # Display the file path
    print(f"Submission file saved at: {submission_file_path}")
    
if __name__ == '__main__':
    main()