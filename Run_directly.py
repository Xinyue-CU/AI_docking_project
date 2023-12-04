import torch
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.data_preprocessing import padding, get_data
from model.LSTM_model import *
from sklearn.metrics import r2_score

# 3. Data preprocessing: load the embedding data
pickle_file_path = 'embedding_data.pkl'
tqdm.write("Loading the embedding data...")
df_smiles = pd.read_pickle(pickle_file_path)
tqdm.write("Embedding data loaded successfully!")

original_smiles = df_smiles['SMILES'].tolist()
smiles2seq_list = df_smiles['embedding'].tolist()  # X in the model
docking_score = df_smiles['docking score'].tolist()  # y in the model
original_length = [len(seq) for seq in smiles2seq_list]


# find one element with 0 length, can will cause error in the later training process
# we need to remove it
index_list = [i for i in range(len(original_length)) if original_length[i] == 0]
smiles2seq_list = [element for index, element in enumerate(smiles2seq_list) if index not in index_list]
docking_score = [element for index, element in enumerate(docking_score) if index not in index_list]
original_length = [element for index, element in enumerate(original_length) if index not in index_list]
print('Get the embedding result successfully！')


# 4. Data preprocessing: padding and train test split
max_seq_length, X_padding = padding(smiles2seq_list, num_features=100)
X = np.array(X_padding)
y = np.array(docking_score)
train_loader, val_loader, test_loader, y_train_tensor, \
y_val_tensor, y_test_tensor, smiles_train, smiles_val, smiles_test = get_data(original_length, X, y)
print('Prepare the data successfully!')


# 5. Train the model
input_dim = 100
LSTM_hidden_size1 = 100
LSTM_hidden_size2 = 100
LSTM_output_dim = 64
linear_hidden_size1 = 32
linear_hidden_size2 = 16
output_size = 1
dropout_prob = 0.5
num_epochs = 10
learning_rate = 0.001

model = TwoLayerLSTMModel(input_dim, LSTM_hidden_size1, LSTM_hidden_size2, LSTM_output_dim, linear_hidden_size1, linear_hidden_size2, output_size, dropout_rate=0.5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

for epoch in range(num_epochs):
    for inputs, labels, length in train_loader:
        labels = labels.unsqueeze(1)
        output = model(inputs, length)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


# Save the model
torch.save(model, 'LSTM_model.pth')
loaded_model = torch.load('LSTM_model.pth')
loaded_model.eval()
print('Training completed.')


# 6. Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for inputs, labels, length in test_loader:
        output = model(inputs, length)
        predictions.append(output.detach().cpu().numpy())

predictions = np.concatenate(predictions)
r2 = r2_score(y_test_tensor, predictions)
print(f'R-squared Score on Test Set: {r2:.4f}')

# Save the result
print(y_test_tensor.flatten()[0:10])
print(predictions.flatten()[0:10])
df_test = pd.DataFrame({'SMILES': smiles_test, 'docking score': y_test_tensor.flatten(), 'predictions': predictions.flatten()})
df_test.to_csv('Test_result.csv', index=False)