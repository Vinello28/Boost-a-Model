#!/usr/bin/env fish
source .env.fish
echo "Starting BAM sender..."
echo "SSH_USER and SSH_MACHINE are set to: $SSH_USER@$SSH_MACHINE via .env.fish"
echo "REMEMBER: sender must be run on the machine with the camera connected, NOT IN THE REMOTE!!!"
echo "NOTE: you will be prompted for a password, the prompt will probably be hidden somewhere in the terminal, so be sure to type it in and press Enter."
read -p "Are you sure you run this on the machine with the camera? (y/N): " -n 1 -s confirm

if test $confirm != 'y'
    echo "\nExiting..."
    exit 1
end

# ffmpeg -re -i /dev/video0 -f mpegts - | ssh -p $SSH_PORT $SSH_USER@$SSH_MACHINE 'docker exec -i bam-contener ffplay -i -'
set TIME ( date +%Y-%m-%d_%H-%M-%S )
# This command captures video from a camera and sends it over SSH to a remote machine and saves every frame as a raw video file.
ffmpeg -f v4l2 -framerate 10 -video_size 640x480 -i /dev/video0 \
    -vcodec libx264 -preset ultrafast -tune zerolatency -f mp4 -movflags frag_keyframe+empty_moov+default_base_moof - \
| ssh $SSH_USER@$SSH_MACHINE "cat > \$HOME/.tmp/stream-$TIME.mp4"