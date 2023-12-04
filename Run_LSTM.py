import codecs
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from SmilesPE.learner import *
from SmilesPE.tokenizer import *
from SmilesPE.spe2vec import *
from model.LSTM_model import *
from model.data_preprocessing import padding, get_data
from sklearn.metrics import r2_score

# 1. Prepare the vocabulary
dataset_name = 'resampled_data.csv'
df_smiles = pd.read_csv(dataset_name)
SMILES = df_smiles['SMILES'].tolist()
print('Get the SMILES list successfully!')

# # Build up the vocabulary of tokens
# vocab_output = codecs.open('SPE_token_vocab.txt', 'w')  # Create a directory for the output files
# learn_SPE(SMILES, vocab_output, 30000, min_frequency=50,
#           augmentation=1, verbose=True, total_symbols=True)  # learn_SPE is a function in learner.py in SmilesPE
# vocab_output.close()
# print('Build up the vocabulary successfully!')


# 2. Build up SPE tokenizer(Train the model with skip-gram algorithm)
# Load the vocabulary
spe_vocab = codecs.open('SPE_ChEMBL.txt')
# Build a tokenizer model with the vocabulary
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
print('Get spe2vec model successfully!')

# 3. get SMILES embedding string using SPE tokenizer
df_smiles['tokenize'] = df_smiles['SMILES'].apply(lambda x: spe.tokenize(x))
print('Get tokenize result successfully!')


def process_embedding(smiles):
    embedding = spe2vec.smiles2vec(smiles)
    return embedding


tqdm.pandas(desc="Processing SMILES embeddings")
df_smiles['embedding'] = df_smiles['SMILES'].progress_apply(process_embedding)

# save the result
# to_csv
df_smiles.to_csv('SAMPLE_SMILE1_embedding_result.csv', index=False)
original_smiles = df_smiles['SMILES'].tolist()
smiles2seq_list = df_smiles['embedding'].tolist()  # X
docking_score = df_smiles['docking score'].tolist()  # y

# to_pickle
pickle_file_path = 'embedding_data.pkl'
df_smiles.to_pickle(pickle_file_path)
tqdm.write("Embedding data saved successfully!")

# 4. Data preprocessing: load the embedding data
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

# 5. Data preprocessing: padding and train test split
max_seq_length, X_padding = padding(smiles2seq_list, num_features=100)
X = np.array(X_padding)
y = np.array(docking_score)
train_loader, val_loader, test_loader, y_train_tensor, \
y_val_tensor, y_test_tensor, smiles_train, smiles_val, smiles_test = get_data(original_length, X, y)
print('Prepare the data successfully!')

# 6. Train the model
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

model = TwoLayerLSTMModel(input_dim, LSTM_hidden_size1, LSTM_hidden_size2, LSTM_output_dim, linear_hidden_size1,
                          linear_hidden_size2, output_size, dropout_rate=0.5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
df_test = pd.DataFrame(
    {'SMILES': smiles_test, 'docking score': y_test_tensor.flatten(), 'predictions': predictions.flatten()})
df_test.to_csv('Test_result.csv', index=False)
