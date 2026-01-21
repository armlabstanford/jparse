# Docker Setup for J-PARSE

To build the Docker image for our environment, we use VNC docker, which allows for a graphical user interface displayable in the browser.

## Use the Public Docker Image (Recommended)

We have created a public Docker image that you can pull!

```sh
docker pull peasant98/jparse
docker run --privileged -p 6080:80 --shm-size=512m -v <path to jparse repo>:/home/ubuntu/Desktop/jparse_ws/src peasant98/jparse
```

## Build the Image Yourself

You can build the docker image yourself! To do so, follow the below steps:

```sh
cd Docker
docker build -t jparse .
docker run --privileged -p 6080:80 --shm-size=512m -v <path to jparse repo>:/home/ubuntu/Desktop/jparse_ws/src jparse
```

## Accessing the GUI

Once the container is running, open your browser and navigate to:

```
http://localhost:6080
```

You should see a desktop environment where you can run the ROS examples.
