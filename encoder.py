from torch import nn
from torch.nn import LSTM


class Encoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = LSTM(input_size, hidden_size)

    def forward(self, input_sentence):
        embedded_input = self.embedding(input_sentence).view(len(input_sentence), 1, -1)
        return self.lstm(embedded_input)