import nltk
import numpy as np

def tokenize_sentences(sentences):
    return [nltk.word_tokenize(s) for s in sentences]

def average_word_length(tokenized_sentences):
    length = [np.array(list(map(len, tokenized_sentence))).mean() for tokenized_sentence in tokenized_sentences]
    return length
    ##token_length = []
    #[list(map(len, tokenized_sentence)) for tokenized_sentence in tokenize_sentences()]
    #for tokenized_sentence in tokenized_sentences:
    #    [len(token) for token in tokenized_sentence]
    #return [len(token) for token in tokenized_sentence]

def test():
    sentences = ["I lost my wallet", "You got to be kidding me"]
    tokenize_sentences(sentences)
    print(average_word_length(tokenize_sentences(sentences)))

if __name__ == '__main__':
    test()