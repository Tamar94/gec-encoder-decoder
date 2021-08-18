import time

import torch
import yaml
from nltk.translate.gleu_score import corpus_gleu
from torch import optim
from torch.nn import CrossEntropyLoss

from m2_data import prepare_data, read_data
from gec_encoder_decoder import GECEncoderDecoder
from torch import nn
from vocab import Vocab

with open('settings.yaml', 'r') as ymlfile:
    SETTINGS = yaml.load(ymlfile)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(model, train_sequence, epochs):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    index = 0
    torch.autograd.set_detect_anomaly(True)
    for e in range(epochs):
        for (input_sentence, output_sentence) in train_sequence:
            optimizer.zero_grad()
            model_output, input_tensor, output_tensor = _get_model_output(model, input_sentence, output_sentence)
            loss = criterion(model_output, output_tensor)
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                print(f'Loss: {loss}\n')
            index += 1

    torch.save(model, SETTINGS.get('SAVED_MODEL_PATH'))


def evaluate(model, test_sequence, test_data, vocab, beam_width):
    start_time = time.time()
    hypotheses = []
    list_of_references = []

    for i in range(len(test_sequence)):
        input_sentence = test_sequence[i][0]
        model_output, _, _ = _get_model_output(model, input_sentence, beam_width=beam_width)
        hypotheses.append([vocab.id_to_word_mapping.get(word_id) for word_id in model_output.tolist()])
        list_of_references.append([test_data[i][1]])

    if beam_width:
        print(f'\nBeam search: beam width {beam_width}')
    else:
        print(f'Greedy search')
    print(f'Time: {time.time() - start_time}')
    print(f'GLEU score: {corpus_gleu(list_of_references, hypotheses)*100}')


def _get_model_output(model, input_sentence, output_sentence=None, beam_width=None):
    input_tensor = torch.tensor(input_sentence).to(DEVICE)
    output_tensor = torch.tensor(output_sentence).to(DEVICE) if output_sentence else None
    return model.forward(input_tensor, output_tensor, beam_width), input_tensor, output_tensor


if __name__ == '__main__':
    input_vocab = Vocab()
    output_vocab = Vocab()
    train_data = read_data(SETTINGS.get('TRAIN_DATA_PATH'), input_vocab.END)
    test_data = read_data(SETTINGS.get('TEST_DATA_PATH'), input_vocab.END)
    train_sequence = prepare_data(train_data, input_vocab, output_vocab)
    test_sequence = prepare_data(test_data, input_vocab, output_vocab)

    embedding_size = SETTINGS.get('EMBEDDING_SIZE')
    hidden_size = SETTINGS.get('HIDDEN_SIZE')
    epochs = SETTINGS.get('EPOCHS')

    model = GECEncoderDecoder(input_vocab, output_vocab, embedding_size, hidden_size, DEVICE).to(DEVICE)
    if SETTINGS.get('USE_SAVED_MODEL'):
        model = torch.load(SETTINGS.get('SAVED_MODEL_PATH'))
    else:
        train_loop(model, train_sequence, epochs)

    evaluate(model, test_sequence, test_data, output_vocab, None)
    evaluate(model, test_sequence, test_data, output_vocab, 3)
    evaluate(model, test_sequence, test_data, output_vocab, 5)
    evaluate(model, test_sequence, test_data, output_vocab, 8)
    evaluate(model, test_sequence, test_data, output_vocab, 10)
    evaluate(model, test_sequence, test_data, output_vocab, 20)
    evaluate(model, test_sequence, test_data, output_vocab, 30)




