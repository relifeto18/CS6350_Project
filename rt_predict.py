import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

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
    
    # Train a Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    # rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    ########## Grid Search ##########
    # Create a parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],        # Number of trees in the forest
        'max_depth': [10, 20, 30],              # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
        'max_features': ['sqrt', 'log2']        # Number of features to consider when looking for the best split
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                            scoring='roc_auc',    # Use AUC as the evaluation metric
                            cv=5,                # 5-fold cross-validation
                            verbose=2,           # Print progress for each combination
                            n_jobs=-1)           # Use all available CPU cores for parallel processing

    # Fit GridSearchCV on the training data
    # rf_model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Print the best hyperparameters
    print(f"Best Hyperparameters: {best_params}")
    
    # # Predict probabilities on the validation set
    # y_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]  # Probabilities for class 1 (income > 50K)
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # Evaluate model performance using AUC score
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"AUC score: {auc_score}")
    
    ################################ Test ################################ 
    # Align the columns of the test set with the training set
    test_data_encoded = test_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict probabilities on the test data
    # test_predictions = rf_model.predict_proba(test_data_encoded)[:, 1]
    test_predictions = best_model.predict_proba(test_data_encoded)[:, 1]

    # Create a submission DataFrame
    submission = pd.DataFrame({
        'ID': test_data_preprocessed['ID'],
        'Prediction': test_predictions
    })

    # Save the submission file
    submission_file_path = './data/submission_predictions.csv'
    submission.to_csv(submission_file_path, index=False)

    # Display the file path
    print(f"Submission file saved at: {submission_file_path}")


if __name__ == '__main__':
    main()