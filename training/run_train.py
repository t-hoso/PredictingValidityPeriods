from training.train_models import train_dnn
from models.models import DNN
from feature_extraction.lsa import LsiSvd
from feature_extraction.read_files import read_all_sentences

if __name__ == '__main__':
    model = DNN()
    X, t = read_all_sentences()
    lsi = LsiSvd()
    lsi.train(sentences=X.tolist())
    X = lsi.get_lsi_u(sentences=X)
    train_dnn(model, X, t, epochs=15, batch_size=30)