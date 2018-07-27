import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.saved_model import utils

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info


import os
import urllib

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    import tensorflow as tf
    sess = tf.Session()

    from keras import backend as K
    K.set_session(sess)


    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "w") as f:
            f.write(raw)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    """
    model = keras.Sequential([
        keras.layers.Dense(6, activation=tf.nn.relu, input_shape=(4,), name='irisin'), 
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax, name='irisout')
    ])
    """

    _in = Input(shape=(4, ), name='irisin')
    lyr1 = Dense(6, activation='relu')(_in)
    lyr2 = Dense(10, activation='relu')(lyr1)
    out = Dense(3, activation='softmax', name='irisout')(lyr2)
    model = Model(input=_in, output=out)

    for op in sess.graph.get_operations():
        print(op.name)  

    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(training_set.data, training_set.target, epochs=500)

    test_loss, test_acc = model.evaluate(test_set.data, test_set.target)

    print('Test accuracy:', test_acc)



    # Pick out the model input and output
    input_tensor = sess.graph.get_tensor_by_name('irisin:0')
    output_tensor = sess.graph.get_tensor_by_name('irisout/Softmax:0')

    model_input = build_tensor_info(input_tensor)
    model_output = build_tensor_info(output_tensor)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={'irisin': model_input},
        outputs={'irisout': model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder('./models/iris/1')

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()

    print('Done exporting!')


if __name__ == "__main__":
    main()