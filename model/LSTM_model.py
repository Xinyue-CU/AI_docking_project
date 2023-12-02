import torch
import torch.nn as nn

seed = 42
torch.manual_seed(seed)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size1, hidden_size2, hidden_size3, num_layers, seq_length, dropout_prob):
        super(LSTM, self).__init__()
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
