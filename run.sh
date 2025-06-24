docker build -t bam-contener .

docker run -it --rm -t -d \
    --name bam-contener\
    --network="host" \
    -e DISPLAY=$DISPLAY \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v "/tmp/.X11-unix-cv2425g26:/tmp/.X11-unix:rw" \
    --privileged \
    --runtime=nvidia \
    --gpus '"device=0,2",capabilities=utility' \
    -p 9765:9765 \
    -v "$(pwd):/workspace" \
    -v "$HOME/.tmp:/tmp-video/" \
    bam-contener

echo "Connecting to container..."
echo "REMEMBER TO RUN source setup.fish INSIDE THE CONTAINER"
docker exec -it bam-contener fish

