import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seed = 42
torch.manual_seed(seed)

# LSTM version 1
# single layer LSTM with padding
class LSTM_1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size1, hidden_size2, hidden_size3, num_layers, seq_length, dropout_prob):
        super(LSTM_1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.seq_length = seq_length
        self.dropout_prob = dropout_prob

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size1,
                            num_layers=num_layers, batch_first=True, dropout=self.dropout_prob)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, num_classes)
        self.activation = nn.ReLU()  # Changed to ReLU activation function
        self.dropout = nn.Dropout(self.dropout_prob)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size1)
        out = self.activation(hn)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# LSTM version 2
# two layer LSTM with padding
class LSTM_2layer(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size1, hidden_size2, num_layers, dropout_prob):
        super(LSTM_2layer, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_prob = dropout_prob

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1,
                             num_layers=num_layers, batch_first=True, dropout=self.dropout_prob)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2,
                             num_layers=num_layers, batch_first=True, dropout=self.dropout_prob)

        # Additional linear layers
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, num_classes)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        h_01 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        c_01 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        output1, (hn1, cn1) = self.lstm1(x, (h_01, c_01))

        h_02 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        c_02 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        output2, (hn2, cn2) = self.lstm2(output1, (h_02, c_02))

        hn2 = hn2.view(-1, self.hidden_size2)
        out = self.activation(hn2)
        out = self.dropout(out)

        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)

        return out

# LSTM version 3
# two layer LSTM without padding
# use pad_packed_sequence to inverse the padding process
class TwoLayerLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim, dropout_rate=0.5):
        super(TwoLayerLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_dim)
        self.linear1 = nn.Linear(output_dim, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, lengths):
        # Pack the sequence
        packed_input1 = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # First LSTM layer with dropout
        packed_output1, _ = self.lstm1(packed_input1)
        output1, lengths1 = pad_packed_sequence(packed_output1, batch_first=True)

        # Second LSTM layer with dropout
        packed_input2 = pack_padded_sequence(output1, lengths1, batch_first=True, enforce_sorted=False)
        packed_output2, _ = self.lstm2(packed_input2)
        output2, lengths2 = pad_packed_sequence(packed_output2, batch_first=True)

        # Fully connected layer
        output = self.fc(output2[:, -1, :])

        # Additional linear layers with dropout
        output = self.dropout(output)
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.linear3(output)

        return output
