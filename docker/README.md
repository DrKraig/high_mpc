# For building the image
docker build -t high_mpc_image -f high_mpc.Dockerfile .

# For running the container
docker run -d -it --name high_mpc -e interface=eht1 -e DISPLAY=$DISPLAY -e LOCAL_USER_ID=1000 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v ~/Documents/Projects/high_mpc:/home/user/workspace/src/high_mpc:rw --security-opt seccomp=unconfined --network=host --pid=host --gpus all --privileged --device=/dev:/dev --runtime=nvidia high_mpc_image:latest

# To start the container
Either start from the portainer or run the command below

docker start CONTAINER high_mpc

# To port into the container
docker exec -u user -it high_mpc /bin/bash