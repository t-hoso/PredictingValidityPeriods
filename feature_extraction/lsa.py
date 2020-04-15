from gensim import corpora
from gensim.models import LsiModel, TfidfModel
import nltk
import os
import sys
import numpy as np

"""
I simplified the model not using wikipedia dump to create Tf-idf model but using dataset
"""

def create_tokenized_sentence(data_dir, files, output_dir, output_names):
    """
    :param files: refers to files that include texts of each class
    :return: tokenized texts
    """
    for file in files:
        with open(os.path.join(data_dir, file), encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line:
                basename = os.path.splitext(os.path.basename(file))[0]
                if basename[-4:] == "temp":
                    num = basename[-6]
                else:
                    num = basename[-1]

                sentence = line[:-3]
                tokenized_sentence = nltk.word_tokenize(sentence)

class LsiSvd():
    def __init__(self):
        self.lsi_svd = None
        self.dct = None
        self.tfidf = None
    def train(self, sentences, num_topics=400, truncated_topics=200):
        '''
        :param sentences: train set
        :param num_topics: the number of topics, which is not fixed in the paper
        :param truncated_topics: the number of major topics, which is set to 200 according to paper
        :return: Nothing but become able to access U of truncated svd acquired from LSI
        '''

        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        self.dct = corpora.Dictionary(tokenized_sentences)
        corpus = [self.dct.doc2bow(s) for s in tokenized_sentences]
        self.tfidf = TfidfModel(corpus)
        tfidf_corpus = self.tfidf[corpus]
        lsi = LsiModel(corpus=tfidf_corpus, num_topics=num_topics, id2word=self.dct)
        self.lsi_svd = lsi.projection.u[:, :truncated_topics]

    def get_lsi_u(self, sentences):
        '''
        transform sentences to the vector of topics
        :param sentences: sentence
        :return: topics
        '''
        lsi_list = []
        for sentence in sentences:
            tfidf_sentence = self.tfidf[[self.dct.doc2bow(nltk.word_tokenize(sentence))]]
            tfidf_sentence = np.array([[t[0], t[1]] for t in tfidf_sentence])

            shape = tfidf_sentence[:, 1].shape
            shape = list(shape)
            shape.append(1)

            lsi_list.append(self.lsi_svd[tfidf_sentence[:, 0].astype(np.int)] * tfidf_sentence[:, 1].reshape(shape))
        return np.array(lsi_list)
        #tfidf_sentences = self.tfidf[self.dct.doc2bow(nltk.word_tokenize(s)) for s in sentences]
        #tfidf_sentences = np.array([[t[0], t[1]] for t in tfidf_sentences])
        #print(tfidf_sentences)
        #return self.lsi_svd[tfidf_sentences[:,0].astype(np.int)]*tfidf_sentences[:,1]


if __name__ == '__main__':
    num_topics = 400 # we truncate this to 200, which is truncated_topics
    truncated_topics = 200

    np.random.seed(920)

    # retrieve filenames
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_dir = "data"
    data_dir = os.path.join(parent, data_dir)
    dataset_dir = "dataset"
    data_dir = os.path.join(data_dir, dataset_dir)
    files = os.listdir(data_dir)

    # for each file, tokenize the sentences and add them to corpora.Dictionary
    dct = corpora.Dictionary()
    for file in files:
        with open(os.path.join(data_dir,file), encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line:
                basename = os.path.splitext(os.path.basename(file))[0]
                if basename[-4:] == "temp":
                    num = basename[-6]
                else:
                    num = basename[-1]

                sentence = line[:-3]
                tokenized_sentence = nltk.word_tokenize(sentence)
                dct.add_documents([tokenized_sentence])
                break
                line = f.readline()

    # get tokenized sentences again
    # should be replaced with loading dump file of tokenized sentences
    # and then add them to corpus
    corpus = []
    for file in files:
        with open(os.path.join(data_dir, file), encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line:
                basename = os.path.splitext(os.path.basename(file))[0]
                if basename[-4:] == "temp":
                    num = basename[-6]
                else:
                    num = basename[-1]

                sentence = line[:-3]
                tokenized_sentence = nltk.word_tokenize(sentence)
                corpus.append(dct.doc2bow(tokenized_sentence))

                break
                line = f.readline()

    tfidf_model = TfidfModel(corpus)
    tfidf_corpus = tfidf_model[corpus]
    lsi = LsiModel(corpus=tfidf_corpus, id2word=dct, num_topics=10)

    #print(lsi[tfidf_model[dct.doc2bow(nltk.word_tokenize("I am a dog"))]])
    #print(tfidf_model[dct.doc2bow(nltk.word_tokenize("I am a dog"))])
    s = np.zeros_like((lsi.projection.u[0]))
    sentences = ["I am a dog", "I have been teaching English for a long time"]
    #print([tfidf_model[dct.doc2bow(nltk.word_tokenize(s))] for s in sentences])
    print("model",tfidf_model[dct.doc2bow(nltk.word_tokenize(sentences[0]))])
    print("model",tfidf_model[dct.doc2bow(nltk.word_tokenize(sentences[1]))])
    tfidf_s  = [tfidf_model[dct.doc2bow(nltk.word_tokenize(s))] for s in sentences] # atteru
    print("tfidfs",tfidf_s)
#    tfidf_l = [[[t[0], t[1]] for t in tfid] for tfid in tfidf_s] # atteru
    tfidf_l = [np.array([[t[0], t[1]] for t in tfid]) for tfid in tfidf_s] # atteru
    print("test",tfidf_l)
    print(lsi.projection.u[tfidf_l[:,0].astype(np.int)])
    print(lsi.projection.u[tfidf_l[:,0]]*tfidf_l[:,1])
    for t in tfidf_model[dct.doc2bow(nltk.word_tokenize("I am a dog"))]:
        s += lsi.projection.u[t[0]] * t[1]
    print(s)
#    print(lsi.projection.u[tfidf_model[dct.doc2bow(nltk.word_tokenize("I am a dog"))]])
    #print(len(dct))
    #print(type(lsi))
    #print(type(lsi.projection.u))
    #print(lsi.projection.u[:,:5].shape)
    #print(lsi[tfidf_model[dct.doc2bow(nltk.word_tokenize("I am a dog"))]])
    #print(lsi.projection.u[])
#    for i in range(10):
##      print(lsi.print_topic(topicno=i,topn=5,))
#    for topics in lsi[tfidf_corpus]:
#        print(topics)

