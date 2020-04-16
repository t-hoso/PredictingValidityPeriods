import nltk
import numpy as np

def tokenize_sentences(sentences):
    """
    tokenizes sentences using nltk
    :param sentences: a list of sentences
    :return: a list of tokenized sentences
    """
    return [nltk.word_tokenize(s) for s in sentences]

def average_word_length(tokenized_sentences):
    """
    returns average word length of each tokenized sentences
    :param tokenized_sentences: a list of tokenized sentences, which is provided by tokenize_sentences
    :return: np.array, average_word_length of each sentences
    """
    length = [np.array(list(map(len, tokenized_sentence))).mean() for tokenized_sentence in tokenized_sentences]
    return np.array(length)

def test():
    sentences = ["I lost my wallet", "You got to be kidding me"]
    tokenize_sentences(sentences)
    print(average_word_length(tokenize_sentences(sentences)))

if __name__ == '__main__':
    test()