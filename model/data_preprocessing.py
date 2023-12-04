import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def padding(smiles2seq_list, num_features=200):
    print('Start padding!')
    '''
    :param smiles2seq_list:
    A list of all the smiles sequences after embedding
    The length of each smiles sequence is different
    input dimension (n*seq_length*num_features)
        n: number of smiles
        seq_length: number of the tokens in each smiles
        num_features: The seq length of each token

    :param num_features: The seq length of each token

    :return: A list of all the smiles sequences
    The length of each smiles sequence is the same
    input dimension (n*seq_length*num_features)
    seq_length = max(len(seq) for seq in smiles2seq_list)
    '''


    len_list = [len(seq) for seq in smiles2seq_list]
    max_seq_length = max(len_list)
    all_result = []
    for i in tqdm(range(len(smiles2seq_list))):
        seq = smiles2seq_list[i]
        padding_result = seq.copy()
        padding_rows = max_seq_length - len(seq)
        if padding_rows > 0:
            # Creating a padding array
            padded_array = padding_rows * [[0.0] * num_features]
            padding_result.extend([list(map(float, row)) for row in padded_array])
        else:
            # If no padding is needed, the padded array is the same as the original array
            padding_result = padding_result
        all_result.append(padding_result)
    return max_seq_length, all_result


def get_data(original_length, X, y):
    # Split the data into training, validation, and test sets
    print('Start Train test split.....')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # we also want to reserve the original smiles
    len_train, len_temp, y_train, y_temp = train_test_split(original_length, y, test_size=0.2, random_state=42)
    len_val, len_test, y_val, y_test = train_test_split(len_temp, y_temp, test_size=0.5, random_state=42)

    # Convert data to PyTorch tensors
    print('Start converting to tensor.....')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    batch_size = 32  # Batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_train_tensor, y_val_tensor, y_test_tensor, len_train, len_val, len_test