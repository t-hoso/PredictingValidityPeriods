from training.train_models import train_dnn
from models.models import DNN
from feature_extraction.lsa import LsiSvd
from feature_extraction.read_files import read_all_sentences
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

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
    train_dnn(model, X, t_train, epochs=15, batch_size=30)