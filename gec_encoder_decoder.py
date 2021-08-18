from torch import nn

from decoder import Decoder
from encoder import Encoder


class GECEncoderDecoder(nn.Module):

    def __init__(self, input_vocab, output_vocab, input_size, hidden_size, device):
        super(GECEncoderDecoder, self).__init__()
        self.device = device
        self.encoder = Encoder(input_vocab.words_count, input_size, hidden_size)
        self.decoder = Decoder(output_vocab, input_size, hidden_size)

    def forward(self, input_sentence, output_sentence=None, beam_width=None):
        _, hidden = self.encoder(input_sentence)
        return self.decoder(hidden, output_sentence, beam_width)




