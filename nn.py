import torch
import warnings
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

    # Align categorical features between training and test sets using one-hot encoding
    full_data = pd.concat([train_data_preprocessed, test_data_preprocessed.drop(columns=['ID'])])
    full_data_encoded = pd.get_dummies(full_data, drop_first=True)

    # Split the full data back into training and test sets
    X_full = full_data_encoded.drop(columns=['income>50K'], errors='ignore')
    X_train = X_full.iloc[:len(train_data_preprocessed)]
    X_test = X_full.iloc[len(train_data_preprocessed):]
    y_train = train_data_preprocessed['income>50K']

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Normalize features for neural network compatibility
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Create DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the neural network
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size):
            super(NeuralNetwork, self).__init__()
            # First hidden layer
            self.fc1 = nn.Linear(input_size, 256)  # Increased neurons for more capacity
            self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
            self.relu1 = nn.LeakyReLU(0.1)  # Leaky ReLU activation
            self.dropout1 = nn.Dropout(0.2)  # Reduced dropout

            # Second hidden layer
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.relu2 = nn.LeakyReLU(0.1)
            self.dropout2 = nn.Dropout(0.2)

            # Third hidden layer
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.relu3 = nn.LeakyReLU(0.1)
            self.dropout3 = nn.Dropout(0.2)

            # Output layer
            self.fc4 = nn.Linear(64, 1)  # Output layer
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = self.relu3(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
            x = self.sigmoid(self.fc4(x))
            return x

    # Initialize the model
    input_size = X_train_tensor.shape[1]
    model = NeuralNetwork(input_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Learning rate decay

    # Training loop with stopping condition
    num_epochs = 50
    best_val_auc = 0
    target_auc = 0.93
    best_model_path = "./data/best_model.pth"
    submission_file_path = './data/submission_predictions_nn.csv'
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                y_true.extend(y_batch.numpy())
                y_pred.extend(outputs.numpy())
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(y_true, y_pred)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        # Save the model if it improves the best AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_val_auc:.4f}")

        # Early stopping condition
        if val_auc >= target_auc:
            print(f"Stopping training early as validation AUC reached {val_auc:.4f}, exceeding the target of {target_auc}.")
            break
                
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze().numpy()

    # Prepare submission file
    submission = pd.DataFrame({
        "ID": test_data["ID"],
        "Prediction": test_predictions
    })
    submission.to_csv(submission_file_path, index=False)
    print(f"Submission file saved at: {submission_file_path}")


# Run the main function
if __name__ == "__main__":
    main()
