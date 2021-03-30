docker build . -t zakuro/sakura
docker run --rm \
  --gpus all \
  --shm-size=70g \
  -v $(pwd):/workspace -it zakuro/sakura python demo.py
