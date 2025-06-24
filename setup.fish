cd src/
python -m venv venv
source venv/bin/activate.fish
pip install -r requirements.txt
deactivate
echo "Virtual environment setup for BAM completed. To activate, run: source venv/bin/activate.fish"
cd /workspace
