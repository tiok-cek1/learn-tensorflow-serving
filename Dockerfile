FROM tensorflow/serving:latest-devel

ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

ENV MODEL_NAME=iris

COPY models /models

ENTRYPOINT tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}