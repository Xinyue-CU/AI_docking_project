import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)

        decoder_input = x[:, -1, :].unsqueeze(1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        output = self.fc(decoder_output[:, -1, :]).squeeze()
        return output