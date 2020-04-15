import tensorflow as tf
from tensorflow import losses, metrics, optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from training.callbacks import EarlyStopping

def train_dnn(model, X, y, epochs, batch_size):
    criterion = losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-07, amsgrad=False)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    es = EarlyStopping(patience=5)

    hist = {'val_loss': [],
            'val_acc': [],
            'train_loss': [],
            'train_acc': []}

    batches = X_train.shape[0] // batch_size
    for epoch in range(epochs):
        train_loss = metrics.Mean()
        train_acc = metrics.CategoricalAccuracy()
        val_loss = metrics.Mean()
        val_acc = metrics.CategoricalAccuracy()

        def train_step(x, y):
            with tf.GradientTape() as tape:
                preds = model(x)
                loss = criterion(y, preds)
            grads = tape.gradient(loss, model.trainable_variables, output_gradients=None)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)
            train_acc(y, preds)
            return loss

        def val_step(x, y):
            preds = model(x)
            loss = criterion(y, preds)
            val_loss(loss)
            val_acc(y, preds)

        X_, y_ = shuffle(X_train, y_train)

        for batch in range(batches):
            start = batch*batch_size
            end = batch_size + start
            train_step(X[start:end], y[start:end])

        val_step(X_val, y_val)

        hist['val_acc'].append(val_acc.result())
        hist['val_loss'].append(val_loss.result())
        hist['train_acc'].append(train_acc.result())
        hist['train_loss'].append(train_loss.result())

        print('epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}'
              ', train_acc: {:.3f}, val_acc: {:.3f}'.format(
            epoch+1,
            train_loss.result(),
            val_loss.result(),
            train_acc.result(),
            val_loss.result()
        ))
        if es(val_loss.result()):
            break

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1, len(hist['train_loss'])+1), hist['train_loss'], color=[1,0,0], linewidth=1, label="train_loss")
    ax1.plot(range(1, len(hist['val_loss'])+1), hist['val_loss'], color=[1,0,1], linewidth=1, label='val_loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, len(hist['train_acc'])+1), hist['train_acc'], color=[1,0,0], linewidth=1, label="train_loss")
    ax2.plot(range(1, len(hist['val_acc'])+1), hist['val_acc'], color=[1,0,1], linewidth=1, label='val_loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.set_title('Accuarcy')
    ax2.legend()
    plt.show()