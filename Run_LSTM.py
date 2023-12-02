import codecs
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from SmilesPE.learner import *
from SmilesPE.tokenizer import *
from SmilesPE.spe2vec import *
from model.LSTM_model import LSTM
from model.data_preprocessing import padding, get_data
from sklearn.metrics import r2_score

print('Start running the program...')
# 1. Prepare the vocabulary
dataset_name = 'resampled_data.csv'
df_smiles = pd.read_csv(dataset_name)
df_smiles = df_smiles.dropna(how='any')
SMILES = df_smiles['SMILES'].tolist()

# Build up the vocabulary of tokens
vocab_output = codecs.open('SPE_token_vocab.txt', 'w')  # Create a directory for the output files
learn_SPE(SMILES, vocab_output, 30000, min_frequency=200,
          augmentation=1, verbose=True, total_symbols=True)  # learn_SPE is a function in learner.py in SmilesPE
vocab_output.close()
print('Build up the vocabulary successfully!')



# 2. Build up SPE tokenizer(Train the model with skip-gram algorithm)
# Load the vocabulary
spe_vocab = codecs.open('SPE_token_vocab.txt')
# # Build a tokenizer model with the vocabulary
spe = SPE_Tokenizer(spe_vocab)  # SPE_Tokenizer is a class in tokenizer.py in SmilesPE

# Train the SPE2VEC model
current_directory = os.getcwd()  # indir is the file path of the SMIlES csv
file_name = dataset_name
corpus = Corpus(file_name, tokenizer=spe, isdir=True, dropout=0.2)
model = learn_spe2vec(corpus=corpus, outfile=None, vector_size=100,
                      min_count=10, n_jobs=4,
                      method='skip-gram')  # learn_spe2vec is a function in spe2vec.py in SmilesPE
print('Build up spe2vec model successfully!')


# Save the trained model
model_path = 'spe_model.bin'
model.save(model_path)
spe2vec = SPE2Vec(model_path, spe)  # create SPE2Vec object


# 3. get SMILES embedding string using SPE tokenizer
df_smiles['tokenize'] = df_smiles['SMILES'].apply(lambda x: spe.tokenize(x))
df_smiles['embedding'] = df_smiles['SMILES'].apply(lambda x: spe2vec.smiles2vec(x))
# # save the result
# df_smiles.to_csv('SAMPLE_SMILE1_embedding_result.csv', index=False)
original_smiles = df_smiles['SMILES'].tolist()
smiles2seq_list = df_smiles['embedding'].tolist()  # X
docking_score = df_smiles['docking score'].tolist()  # y
print('Get the embedding result successfully！')



# 4. Data preprocessing: padding and train test split
max_seq_length, X_padding = padding(smiles2seq_list, num_features=100)
X = np.array(X_padding)
y = np.array(docking_score)
train_loader, val_loader, test_loader, y_train_tensor, \
y_val_tensor, y_test_tensor, smiles_train, smiles_val, smiles_test = get_data(original_smiles, X, y)
print('Get the data successfully!')



# 5. Build up the model
input_size = 100
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
num_layers = 2  # Number of LSTM layers
num_classes = 1  # Number of output classes
learning_rate = 0.01
num_epochs = 100  # Number of training epochs
seq_length = max_seq_length  # Length of the input sequence
dropout_prob = 0.2  # Dropout rate

model = LSTM(num_classes, input_size, hidden_size1, hidden_size2, hidden_size3, num_layers, seq_length, dropout_prob)
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

