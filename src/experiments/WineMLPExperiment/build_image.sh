#!/bin/sh
#
# Usage: ./build_image.sh [tag]

EXPERIMENT_NAME=$(basename $PWD)
REPO_NAME=$(echo $EXPERIMENT_NAME | tr '[:upper:]' '[:lower:]' | tr ' _' '--')

# Determine the full image tag
if [ -z "$1" ]; then
    TAG="$REPO_NAME:latest"
else
    case "$1" in
        *:*)
            TAG=$1
            ;;
        *)
            TAG="$REPO_NAME:$1"
            ;;
    esac
fi

echo Building image for $EXPERIMENT_NAME as $TAG

# Build from repository root (3 levels up: WineMLPExperiment -> experiments -> src -> repo)
docker build -t $TAG -f Dockerfile --progress=plain ../../..
