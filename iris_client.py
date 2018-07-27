"""Manual test client for tensorflow_model_server."""
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.python.platform import flags

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

from tensorflow.python.saved_model import signature_constants

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'iris'
    request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    X = np.asarray([np.asarray([5.0,2.3,3.3,1.0])]).astype(np.float32)
    request.inputs['irisin'].CopyFrom(
        tf.contrib.util.make_tensor_proto(X, shape=X.shape))
    request.output_filter.append('irisout')
    # Send request

    channel = implementations.insecure_channel('localhost', int(9000))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    print(stub.Predict(request, 5.0))


if __name__ == '__main__':
    tf.app.run()