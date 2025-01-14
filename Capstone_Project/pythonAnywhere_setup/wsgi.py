import sys
import os

# add your project directory to the sys.path
path = '/home/rdtgeo65/mysite'

if path not in sys.path:
    sys.path.append(path)

def get_pipenv_path():
    return os.popen('pipenv --venv').read().strip()

site_packages = os.path.join(get_pipenv_path(), 'lib', 'python3.10', 'site-packages')
if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)

from app import app as application
