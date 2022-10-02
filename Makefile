# Variables
TF_LOG_LEVEL=1
MODEL_URL="https://padistorage.blob.core.windows.net/models/v1/full/model.h5"
CLASS_NAMES_URL="https://padistorage.blob.core.windows.net/models/v1/full/class_names.z"
BASE_PATH="/home/fahmi/padiscanner_be"

APPINSIGHTS_CS="InstrumentationKey=;"
AZURESTORAGE_CS=""

IMAGE_REPO=fahminlb33
IMAGE_TAG=latest
IMAGE_NAME=padiscanner.ml
IMAGE_FULLNAME=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}
DOCKER_CONTAINER_NAME="padiscanner"

# Directives
.PHONY: download run debug docker-build docker-run docker-stop docker-rm

.DEFAULT_GOAL := docker-run

# Commands
download:
	mkdir -p model
	curl -z -nc -L -o ${BASE_PATH}/model/model.h5 ${MODEL_URL}
	curl -z -nc -L -o ${BASE_PATH}/model/class_names.z ${CLASS_NAMES_URL}
	echo "Model and class names downloaded"

run:
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	model_path="${BASE_PATH}/model/model.h5" \
	class_names_path="${BASE_PATH}/model/class_names.z" \
	applicationinsights_connection_string=${APPINSIGHTS_CS} \
	azure_storage_connection_string=${AZURESTORAGE_CS} \
	uvicorn app:app --port 8080

debug:
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	model_path="${BASE_PATH}/model/model.h5" \
	class_names_path="${BASE_PATH}/model/class_names.z" \
	applicationinsights_connection_string=${APPINSIGHTS_CS} \
	azure_storage_connection_string=${AZURESTORAGE_CS} \
	uvicorn app:app --port 8080 --reload

docker-build:
	docker build --build-arg MODEL_URL="${MODEL_URL}" --build-arg CLASS_NAMES_URL="${CLASS_NAMES_URL}" -t ${IMAGE_FULLNAME} .

docker-run: docker-stop docker-rm docker-build
	docker run --name ${DOCKER_CONTAINER_NAME} -d -p 8080:80 ${IMAGE_FULLNAME}

docker-stop:
	-docker stop ${DOCKER_CONTAINER_NAME}

docker-rm: docker-stop
	-docker rm -f ${DOCKER_CONTAINER_NAME}
