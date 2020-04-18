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
    length = [[np.array(list(map(len, tokenized_sentence))).mean()] for tokenized_sentence in tokenized_sentences]
    return np.array(length)

def sentence_length(tokenized_sentences):
    """
    returns sentence length
    it is not specified what sentence length is on the paper
    so I define sentence length as the number of words
    :param tokenized_sentences: a list of sentences
    :return: a list of the numbers of words
    """
    length = [[len(tokenized_sentence)] for tokenized_sentence in tokenized_sentences]
    return np.array(length)

def pos_tag(tokenized_sentences):
    """
    returns pos_tag matrices of each tokenized sentence
    :param sentences: a list of tokenized sentences
    :return: the count of pos_tag matrices
    """
    tags = np.array(["CC", "CD", "DT", "EX", "FW",
                     "IN", "JJ", "JJR", "JJS", "LS",
                     "MD", "NN", "NNS", "NNP", "NNPS",
                     "PDT", "POS", "PRP", "PRP$", "RB",
                     "RBR", "RBS", "RP", "TO", "UH",
                     "VB", "VBD", "VBG", "VBN", "VBP",
                     "VBZ", "WDT", "WP", "WP$", "WRB"])

    sentences = ["I lost my wallet", "You got to be kidding me"]
    tokenized_sentences = tokenize_sentences(sentences)
    sentence_pos = [np.array(list(map(list, s)))[:, 1] for s in nltk.pos_tag_sents(tokenized_sentences)]
    matrices = []
    for s_p in sentence_pos:
        tag_matrix = np.zeros(tags.shape)
        for p in s_p:
            tag_matrix[tags == p] += 1
        matrices.append(tag_matrix)

    return np.array(matrices)

def test():
    tags = np.array(["CC", "CD", "DT", "EX", "FW",
            "IN", "JJ", "JJR", "JJS", "LS",
            "MD", "NN", "NNS", "NNP", "NNPS",
            "PDT", "POS", "PRP", "PRP$", "RB",
            "RBR", "RBS", "RP", "TO", "UH",
            "VB", "VBD", "VBG", "VBN", "VBP",
            "VBZ", "WDT", "WP", "WP$", "WRB"])

    sentences = ["I lost my wallet", "You got to be kidding me"]
    tokenized_sentences = tokenize_sentences(sentences)
    sentence_pos = [np.array(list(map(list, s)))[:, 1] for s in nltk.pos_tag_sents(tokenized_sentences)]
    matrices = []
    for s_p in sentence_pos:
        tag_matrix = np.zeros(tags.shape)
        for p in s_p:
            tag_matrix[tags == p] += 1
        matrices.append(tag_matrix)

    return np.array(matrices)

if __name__ == '__main__':
    print(test())