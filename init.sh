docker rmi -f jcadic/asyncdnn
docker build . -t jcadic/asyncdnn
docker run --rm --gpus all -it  --shm-size=70g  -v /mnt/.cdata:/mnt/.cdata jcadic/asyncdnn bash
