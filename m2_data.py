
def read_data(path, end_token, annotator_id=0):
    data = []
    skip = {"noop", "UNK", "Um"}

    with open(path) as input_file:
        m2 = input_file.read().strip().split("\n\n")
        for sent in m2:
            sent = sent.split("\n")
            cor_sent = sent[0].split()[1:]  # Ignore "S "
            original_sentence = (cor_sent.copy() + [end_token])
            edits = sent[1:]
            offset = 0
            for edit in edits:
                edit = edit.split("|||")
                if edit[1] in skip: continue  # Ignore certain edits
                coder = int(edit[-1])
                if coder != annotator_id: continue  # Ignore other coders
                span = edit[0].split()[1:]  # Ignore "A "
                start = int(span[0])
                end = int(span[1])
                cor = edit[2].split()
                cor_sent[start + offset:end + offset] = cor
                offset = offset - (end - start) + len(cor)
            corrected_sentence = (cor_sent + [end_token])
            data.append((original_sentence, corrected_sentence))

    return data


def prepare_data(data, input_vocab, output_vocab):
    data_sequences = []
    for (original_sentence, corrected_sentence) in data:
        data_sequences.append((input_vocab.index_words(original_sentence),
                               output_vocab.index_words(corrected_sentence)))
    return data_sequences