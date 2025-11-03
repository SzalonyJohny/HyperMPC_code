docker rm hyper_prediction_models
xhost + local:root

docker run -it -d \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="$(pwd):/hyper_prediction_models" \
    --privileged \
    --network=host \
    --gpus=all\
    --name=hyper_prediction_models \
    hyper_prediction_models
