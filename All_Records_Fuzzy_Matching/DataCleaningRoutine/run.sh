# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install necessary dependencies
pip install -r requirements.txt

# Run the main script
python sanitize.py
