source .env
ssh -R $CAMERA_PORT:localhost:$CAMERA_PORT $SSH_USER@$SSH_MACHINE -p $SSH_PORT