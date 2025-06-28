docker build . -t cns_sim_container

docker run --rm -t -d \
    --name cns_sim_container \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --mount src="$(pwd)",target=/root/,type=bind \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v $(pwd)/output_images:/workspace/output_images \
    -w /root \
    --runtime=nvidia \
    --gpus "device=1" \
    --privileged \
    cns_sim_container

# -v $(pwd):/workspace \
# -v $(pwd)/checkpoints/:/root/checkpoints/ \

docker exec -it cns_sim_container bash
clear
read -p "Do you want to stop container? (y/N):" answer

if [[ "$answer" == [Yy] ]]; then
    echo "Stopping"
    docker stop cns_sim_container
    echo "Container stopped."
else
    echo "Container will continue running."
fi
