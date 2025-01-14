import sys
import os

# add your project directory to the sys.path
path = '<your_site>'

if path not in sys.path:
    sys.path.append(path)

def get_pipenv_path():
    return os.popen('pipenv --venv').read().strip()

activate_env = '<your_site>/.local/bin/pipenv'
os.environ['PIPENV_ACTIVE'] = '1'
os.environ['VIRTUAL_ENV'] = '<your_site>/.local/share/virtualenvs/myapp-env'
sys.executable = '<your_site>/.local/share/virtualenvs/myapp-env/bin/python'

from app import app as application
