import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
    xgb_model = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)

    # Fit the model
    xgb_model.fit(X_train, y_train)

    # Predict probabilities on the validation set
    y_val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]

    # Evaluate AUC
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation AUC: {auc_score}")
    
    ################################ Test ################################ 
    # Align the columns of the test set with the training set
    test_data_encoded = test_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict probabilities on the test data
    # test_predictions = rf_model.predict_proba(test_data_encoded)[:, 1]
    test_predictions = xgb_model.predict_proba(test_data_encoded)[:, 1]

    # Create a submission DataFrame
    submission = pd.DataFrame({
        'ID': test_data_preprocessed['ID'],
        'Prediction': test_predictions
    })

    # Save the submission file
    submission_file_path = './data/submission_predictions_1.csv'
    submission.to_csv(submission_file_path, index=False)

    # Display the file path
    print(f"Submission file saved at: {submission_file_path}")
    
if __name__ == '__main__':
    main()