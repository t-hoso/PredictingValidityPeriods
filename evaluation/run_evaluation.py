from sklearn.metrics import f1_score
from training.train_models import train_dnn
from models.models import DNN
from feature_extraction.lsa import LsiSvd
from feature_extraction.read_files import read_all_sentences
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

if __name__ == '__main__':
    tf.random.set_seed(920)
    np.random.seed(920)

    test_size=0.3
    model = DNN()
    X, t = read_all_sentences()
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size)
    lsi = LsiSvd()
    lsi.train(sentences=X_train.tolist())
    X = lsi.get_lsi_u(sentences=X_train)
    train_dnn(model, X, t_train, epochs=100, batch_size=30)

    X_test = lsi.get_lsi_u(sentences=X_test.tolist())

    evaluate_model(model, X_test, t_test)
