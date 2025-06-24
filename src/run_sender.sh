source .env
# Create a new tmux session named 'camera_stream'
tmux new-session -s camera_stream -d

# Split the window vertically (creates two panes)
tmux split-window -v -t camera_stream

# Configure pane layout (optional)
tmux select-layout -t camera_stream tiled

tmux send-keys -t camera_stream.0 "ssh -R $CAMERA_PORT:localhost:$CAMERA_PORT $SSH_USER@$SSH_MACHINE -p $SSH_PORT" C-m

tmux send-keys -t camera_stream.1 "python3 sender.py --camera /dev/video0 --width 640 --height 480 --fps 30 --host $SSH_MACHINE --port $SSH_PORT" C-m

# Attach to the session to see/interact with it
tmux attach-session -t camera_stream