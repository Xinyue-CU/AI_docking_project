import torch
import tqdm
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.LSTM1 import LSTM
from model.data_preprocessing import padding, get_data
from sklearn.metrics import r2_score

pickle_file_path = 'embedding_data.pkl'
tqdm.write("Loading the embedding data...")
df_smiles = pd.read_pickle(pickle_file_path)
tqdm.write("Embedding data loaded successfully!")

# sampled_df = df_smiles.sample(n=10000, random_state=42)
# sampled_df = sampled_df.reset_index(drop=True)

original_smiles = df_smiles['SMILES'].tolist()
smiles2seq_list = df_smiles['embedding'].tolist()  # X in the model
docking_score = df_smiles['docking score'].tolist()  # y in the model
print('Get the embedding result successfully！')

# 4. Data preprocessing: padding and train test split
max_seq_length, X_padding = padding(smiles2seq_list, num_features=100)

X = np.array(X_padding)
y = np.array(docking_score)
pickle_file_path = 'traing_data.pkl'
tqdm.write("Saving X and y...")
with open(pickle_file_path, 'wb') as pickle_file:
    data_to_save = {'X': X, 'y': y}
    pickle.dump(data_to_save, pickle_file)
tqdm.write("Embedding data saved successfully!")

# Loading X and y from the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_data = pickle.load(pickle_file)

X = loaded_data['X']
y = loaded_data['y']
tqdm.write("X and y loaded successfully!")

train_loader, val_loader, test_loader, y_train_tensor, \
y_val_tensor, y_test_tensor, smiles_train, smiles_val, smiles_test = get_data(original_smiles, X, y)
print('Prepare the data successfully!')

num_classes = 1
input_size = 100
hidden_size1 = 64
hidden_size2 = 32
num_layers = 1
seq_length = max_seq_length
dropout_prob = 0.2  
learning_rate = 0.001
num_epochs = 100

model = LSTM(num_classes, input_size, hidden_size1, hidden_size2, num_layers, seq_length, dropout_prob)

# model = LSTM(num_classes, input_size, hidden_size1, hidden_size2, hidden_size3, num_layers, seq_length, dropout_prob)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
print('Start training.....')
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Save the model
torch.save(model, 'LSTM_model.pth')
loaded_model = torch.load('LSTM_model.pth')
loaded_model.eval()
print('Training completed.')


# 6. Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.concatenate(predictions)
print(predictions)
r2 = r2_score(y_test_tensor, predictions)
print(f'R-squared Score on Test Set: {r2:.4f}')

# Save the result
print(smiles_test[0:10])
print(y_test_tensor.flatten()[0:10])
print(predictions.flatten()[0:10])
df_test = pd.DataFrame({'SMILES': smiles_test, 'docking score': y_test_tensor.flatten(), 'predictions': predictions.flatten()})
df_test.to_csv('Test_result.csv', index=False)

