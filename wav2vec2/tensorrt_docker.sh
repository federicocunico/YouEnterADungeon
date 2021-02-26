cd ..
docker run --gpus all -it -v `pwd`:/home/fc -w /home/fc nvcr.io/nvidia/tensorrt:20.12-py3
