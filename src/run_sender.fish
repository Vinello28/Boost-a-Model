#!/usr/bin/env fish
source .env.fish
ffmpeg -re -i /dev/video0 -f mpegts - | ssh -p $SSH_PORT $SSH_USER@$SSH_MACHINE 'docker exec -i bam-contener ffplay -i -'