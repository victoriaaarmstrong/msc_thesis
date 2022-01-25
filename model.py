from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, Model, models, datasets

import numpy as np
import pandas as pd
import tensorflow as tf

def SVM(X, Y, model_name):
    """
    Support vector machine
    :param X: features
    :param Y: labels
    :param model_name: name to save model as
    :return:
    """
    ## Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.70, test_size=0.30,
                                                        random_state=101)

    ## Create classifier
    classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1, multi_class='ovr',
                           fit_intercept=True, intercept_scaling=1, class_weight='balanced', verbose=0,
                           random_state=101)

    ## Compile and fit
    classifier = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    ## Metrics
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    return


def feedforward(X, Y, model_name):
    """
    Fully connected feedforward neural network
    :param X: features
    :param Y: labels
    :param model_name: name to save model as
    :return: None
    """
    ## Split out testing
    temp_x, test_x, temp_y, test_y = train_test_split(X, Y,
                                                  test_size=0.15,
                                                  random_state=101)
    ## Split out training and validation
    train_x, val_x, train_y, val_y = train_test_split(temp_x, temp_y,
                                                  test_size=0.15,
                                                  random_state=42)
    ## Build model
    inputs = Input(shape=(7,))
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dense(32, activation='relu', name="dense_3")(x)
    outputs = Dense(2, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    ## Compile and fit model
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.fit(x=train_x,
              y=train_y,
              epochs=100,
              batch_size=100,
              validation_data=(val_x, val_y))

    ## Metrics
    predictions = model.evaluate(test_x, test_y)
    print('\t accuracy: ' + str(predictions[1]))

    model.save("trained_models/" + model_name)

    return predictions[1]


def deep_flow_arch(features, labels, window_size, model_name):
    """

    :param features:
    :param labels:
    :param window_size:
    :param model_name:
    :return:
    """

    ## Make a 70, 30 training testing split
    train_x, test_x, train_y, test_y = train_test_split(features, labels,
                                                    test_size=0.30,
                                                    random_state=101, ## maybe you could experiment with different random states?
                                                    shuffle=True)

    model = models.Sequential()

    ## Conv1D Block 1
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(window_size, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.1, input_shape=(2,)))

    ## Conv1D Block 2
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.1, input_shape=(2,)))

    ## Conv1D Block 3
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.1, input_shape=(2,)))

    ## Conv1D Block 4
    #model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(32, 32, 3)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling1D(pool_size=2))
    #model.add(layers.Dropout(0.1, input_shape=(2,)))

    ## Final Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5, input_shape=(2,)))
    model.add(layers.Dense(1, activation='sigmoid')) #don't use softmax!!

    print(model.summary())

    ## Compile and fit model
    opt = keras.optimizers.Adam(learning_rate=0.0001) ##was 0.0001
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=opt,
                  metrics=[keras.metrics.BinaryAccuracy()])
                  
    model.fit(x=train_x,
              y=train_y,
              epochs=10,
              batch_size=256) ##could add a validation set you want, then you should change to 0.15, 0.15, 0.7 test, validation, train split

    ## Metrics
    metrics = model.evaluate(test_x, test_y)
    pred_y = model.predict(test_x)
    pred_y = np.argmax(pred_y, 1) ## to smooth non-integer output
    print(pred_y)
    print('\t accuracy: ' + str(metrics[1]))
    print(confusion_matrix(test_y, pred_y))

    model.save("trained_models/" + model_name)

    return metrics[1]


def model_to_tflite(model_path):
    """
    Converts a tensorflow model to a tflite model that can be converted to C using
    xxd -i model.tflite > model_data.cc
    :param model_path:
    :return:
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
      f.write(tflite_model)

    return



