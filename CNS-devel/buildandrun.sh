docker build . -t cns_sim

docker run --rm -t -d \
    --name cns_sim \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --mount src="$(pwd)",target=/root/,type=bind \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -w /root \
    --runtime=nvidia \
    --gpus "device=2" \
    --privileged \
    cns_sim

# -v $(pwd):/workspace \
# -v $(pwd)/checkpoints/:/root/checkpoints/ \

docker exec -it cns_sim bash
clear
read -p "Do you want to stop container? (y/N):" answer

if [[ "$answer" == [Yy] ]]; then
    echo "Stopping"
    docker stop cns_sim
    echo "Container stopped."
else
    echo "Container will continue running."
fi
