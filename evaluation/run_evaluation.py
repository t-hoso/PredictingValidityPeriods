from sklearn.metrics import f1_score
from training.train_models import train_dnn
from models.models import DNN
from feature_extraction.lsa import LsiSvd
from feature_extraction.read_files import read_all_sentences
from feature_extraction.feature_extraction import tokenize_sentences, average_word_length
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

def evaluate_model(model, X_test, t_test):

    """
    evaluate the model built by tf
    evaluation is done by only f1 for now while I should include many metrics
    I should also consider models from both tf and sklearn
    :param model: the predictive model
    :param X_test:
    :param t_test:
    :return:
    """

    preds = model(X_test)
    one_hot_preds = np.eye(5)[np.argmax(preds.numpy(), axis=1).tolist()].reshape(-1,5)
    f1 = f1_score(t_test, one_hot_preds, average='micro')
    print('f1', f1)

def average_length():
    tf.random.set_seed(920)
    np.random.seed(920)

    test_size=0.3
    model = DNN()
    X, t = read_all_sentences()
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size)
    X_train_tokenized = tokenize_sentences(sentences=X_train.tolist())
    X_train_awl = average_word_length(X_train_tokenized)
    print(X_train_awl.shape)
    train_dnn(model, X_train_awl, t_train, epochs=20, batch_size=30)

    X_test_awl = average_word_length(X_test)
    print(X_test_awl)

    evaluate_model(model, X_test_awl, t_test)


def main():
    tf.random.set_seed(920)
    np.random.seed(920)

    test_size=0.3
    model = DNN()
    X, t = read_all_sentences()
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size)
    lsi = LsiSvd()
    lsi.train(sentences=X_train.tolist())
    X_train_lsi = lsi.get_lsi_u(sentences=X_train)
    X_train_tokenized = tokenize_sentences(sentences=X_train.tolist())
    X_train_awl = average_word_length(X_train_tokenized)
    train_dnn(model, np.concatenate([X_train_lsi, X_train_awl], axis=1), t_train, epochs=100, batch_size=30)

    X_test_lsi = lsi.get_lsi_u(sentences=X_test.tolist())
    X_test_tokenized = tokenize_sentences(sentences=X_test.tolist())
    X_test_awl = average_word_length(X_test_tokenized)

    evaluate_model(model, np.concatenate([X_test_lsi, X_test_awl], axis=1), t_test)

if __name__ == '__main__':
    main()
    #average_length()