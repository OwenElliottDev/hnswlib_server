#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
IMAGE_NAME="owenpelliott/hnswlib_server"
TAG="latest"
FULL_IMAGE="$IMAGE_NAME:$TAG"

# docker build -t $IMAGE_NAME:amd64 --platform linux/amd64 .
# docker build -t $IMAGE_NAME:arm64 --platform linux/arm64 .
# docker manifest create $IMAGE_NAME:latest \
#     --amend $IMAGE_NAME:amd64 \
#     --amend $IMAGE_NAME:arm64
# docker manifest push $IMAGE_NAME:latest

docker build -t $IMAGE_NAME:latest --platform linux/arm64 .
docker push $IMAGE_NAME:latest

echo "Successfully built and pushed $FULL_IMAGE"
