class Vocab:
    def __init__(self):
        self.START = "<START>"
        self.END = "<END>"
        self.word_to_id_mapping = {}
        self.id_to_word_mapping = {}
        self.words_count = 0
        self.index_word(self.START)
        self.index_word(self.END)

    def index_words(self, words):
        word_indexes = [self.index_word(w) for w in words]
        return word_indexes

    def get_words(self, tags):
        tag_indexes = [self.id_to_word_mapping[t] for t in tags]
        return tag_indexes

    def index_word(self, word):
        if word not in self.word_to_id_mapping:
            self.word_to_id_mapping[word] = self.words_count
            self.id_to_word_mapping[self.words_count] = word
            self.words_count += 1
        return self.word_to_id_mapping[word]