#!/usr/bin/env fish
source .env.fish
echo "Starting BAM sender..."

# ffmpeg -re -i /dev/video0 -f mpegts - | ssh -p $SSH_PORT $SSH_USER@$SSH_MACHINE 'docker exec -i bam-contener ffplay -i -'
set TIME ( date +%Y-%m-%d_%H-%M-%S )
# This command captures video from a camera and sends it over SSH to a remote machine and saves every frame as a raw video file.
ffmpeg -f v4l2 -framerate 10 -video_size 640x480 -i /dev/video0 \
    -vcodec libx264 -preset ultrafast -tune zerolatency -f mp4 -movflags frag_keyframe+empty_moov+default_base_moof - \
| ssh $SSH_USER@$SSH_MACHINE "cat > \$HOME/.tmp/stream-$TIME.mp4"