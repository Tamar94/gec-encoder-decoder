from operator import attrgetter

import torch
from torch import nn
from torch.nn import LSTM, Linear, Softmax


class Decoder(nn.Module):
    def __init__(self, vocab, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.max_output_len = 50
        self.embedding = nn.Embedding(vocab.words_count, input_size)
        self.lstm = LSTM(input_size, hidden_size)
        self.out = Linear(hidden_size, vocab.words_count)

    def forward(self, hidden, output_sentence=None, beam_width=None):
        embedded_output = None
        outputs = []
        max_range = self.max_output_len
        start_token_id = self.vocab.word_to_id_mapping.get(self.vocab.START)
        end_token_id = self.vocab.word_to_id_mapping.get(self.vocab.END)
        start_token_embedding = self._get_embeddings(torch.tensor([start_token_id]))

        if beam_width:
            return self._beam_search(hidden, beam_width, start_token_embedding, end_token_id)

        if output_sentence is not None:
            max_range = len(output_sentence)
            embedded_output = self._get_embeddings(output_sentence)

        decoder_input = start_token_embedding
        for i in range(max_range):
            decoder_output, hidden = self._decoder_step(decoder_input, hidden)
            outputs.append(decoder_output)
            decoder_best_output = decoder_output.argmax(0)
            if output_sentence is not None:
                decoder_input = embedded_output[i].view(1, 1, -1)
            else:
                if decoder_best_output == end_token_id:
                    break
                decoder_input = self._get_embeddings(decoder_best_output.view(1))
        outputs = torch.stack(outputs)
        if output_sentence is not None:
            return outputs
        return outputs.argmax(1)

    def _beam_search(self, hidden, beam_width, start_token_embedding, end_token_id):
        outputs = [BeamSearchNode(hidden, start_token_embedding)]
        max_range = self.max_output_len
        softmax_layer = Softmax()
        for i in range(max_range):
            curr_results = []
            for beam_search_node in outputs:
                if not beam_search_node.ended:
                    decoder_output, hidden = self._decoder_step(beam_search_node.decoder_input,
                                                                beam_search_node.hidden)
                    probs, indices = torch.topk(softmax_layer(decoder_output), beam_width)
                    for j in range(beam_width):
                        ended = False
                        if indices[j] == end_token_id:
                            ended = True
                        decoder_input = self._get_embeddings(indices[j].view(1))
                        curr_results.append(BeamSearchNode(hidden, decoder_input,
                                                           beam_search_node.calc_prob(probs[j]),
                                                           beam_search_node.seq + [indices[j]],
                                                           ended))
                else:
                    curr_results.append(beam_search_node)
            outputs = list(sorted(curr_results, reverse=True, key=attrgetter('prob')))[:beam_width]
        return torch.tensor(max(outputs, key=attrgetter('prob')).seq)

    def _decoder_step(self, decoder_input, hidden):
        decoder_output, hidden = self.lstm(decoder_input, hidden)
        decoder_output = self.out(decoder_output)[0][0]
        return decoder_output, hidden

    def _get_embeddings(self, sentence):
        return self.embedding(sentence).view(len(sentence), 1, -1)


class BeamSearchNode(object):
    def __init__(self, hidden, decoder_input, prob=1, seq=[], ended=False):
        self.hidden = hidden
        self.decoder_input = decoder_input
        self.seq = seq
        self.prob = prob
        self.ended = ended

    def calc_prob(self, new_prob):
       return self.prob * new_prob
