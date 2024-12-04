import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

# Preprocessing Function
def preprocess_data(data):
    data.replace('?', pd.NA, inplace=True)
    for column in data.columns:
        if data[column].isna().sum() > 0:
            if data[column].dtype in ['float64', 'int64']:
                data[column].fillna(data[column].median(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)
    return data

# Load Data
train_file_path = './data/train_final.csv'
test_file_path = './data/test_final.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocess the data
train_data_preprocessed = preprocess_data(train_data)
test_data_preprocessed = preprocess_data(test_data)

# One-hot encoding
train_data_encoded = pd.get_dummies(train_data_preprocessed, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_preprocessed.drop(columns=['ID']), drop_first=True)

# Align columns of test with training data
test_data_encoded = test_data_encoded.reindex(columns=train_data_encoded.drop(columns=['income>50K']).columns, fill_value=0)

# Separate features and target
X = train_data_encoded.drop(columns=['income>50K'])
y = train_data_encoded['income>50K']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize continuous features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data_encoded)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# Train and Evaluate Function
def train_and_evaluate(hidden_sizes, dropout_rate, learning_rate, batch_size, num_epochs=100):
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    best_model_state = None
    
    # # Lists to store loss and AUC
    # epoch_losses = []
    # epoch_aucs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor).squeeze().numpy()
            auc_score = roc_auc_score(y_val, val_predictions)
            
        # # Store metrics for plotting
        # epoch_losses.append(total_loss)
        # epoch_aucs.append(auc_score)
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Validation AUC: {auc_score:.4f}")
    
    print(f"Best Validation AUC: {best_auc:.4f}")
    model.load_state_dict(best_model_state)
    
    return model, best_auc
    # return model, best_auc, epoch_losses, epoch_aucs

# Perform Grid Search
hidden_sizes_list = [[32], [64], [128], [256], [32, 32], [64, 32], [64, 64], [128, 64], [128, 128], [256, 128], [256, 256], [256, 128, 64]]  # Number of hidden layers and neurons
dropout_rates = [0.1, 0.2, 0.3, 0.5]  # Dropout probabilities
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # Learning rates
batch_sizes = [32, 64, 128, 256]  # Batch sizes
# hidden_sizes_list = [[64, 32]]  # Number of hidden layers and neurons
# dropout_rates = [0.1]  # Dropout probabilities
# learning_rates = [0.005]  # Learning rates
# batch_sizes = [128]  # Batch sizes
# num_epochs = 100
best_model = None
best_auc = 0
best_params = None

for hidden_sizes in hidden_sizes_list:
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                print(f"Training with hidden_sizes={hidden_sizes}, dropout_rate={dropout_rate}, "
                      f"learning_rate={learning_rate}, batch_size={batch_size}")
                model, auc_score = train_and_evaluate(hidden_sizes, dropout_rate, learning_rate, batch_size)
                # model, best_auc, losses, aucs = train_and_evaluate(hidden_sizes, dropout_rate, learning_rate, batch_size, num_epochs)
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_model = model
                    best_params = {
                        'hidden_sizes': hidden_sizes,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size
                    }
                    
# # Plot Loss and AUC
# plt.figure(figsize=(12, 5))

# # Loss Plot
# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_epochs + 1), losses)
# plt.title('Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid()

# # AUC Plot
# plt.subplot(1, 2, 2)
# plt.plot(range(1, num_epochs + 1), aucs, color='orange')
# plt.title('AUC per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('AUC')
# plt.grid()

# plt.tight_layout()
# plt.show()

print()
print(f"Best Parameters: {best_params}")
print(f"Best Validation AUC: {best_auc:.4f}")

# Test Predictions
best_model.eval()
with torch.no_grad():
    test_predictions = best_model(X_test_tensor).squeeze().numpy()

# Save the submission
submission = pd.DataFrame({
    'ID': test_data_preprocessed['ID'],
    'Prediction': test_predictions
})
submission.to_csv('./data/submission_predictions_nn.csv', index=False)
print("Submission saved to './data/submission_predictions_nn.csv'")