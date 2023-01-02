DOCKER
docker build -t unit_infer docker
docker run --gpus all -it --shm-size=16gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /home/:/unit_inference/data --name unit_infer_01.02 unit_infer

DETECTION

POSE ESTIMATION

CLASSIFICATION
