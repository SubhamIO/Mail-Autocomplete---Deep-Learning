#flaskapp.wsgi
activate_this = '/var/www/html/cerebrumXalgorithms/algos_venv/bin/activate_this.py'
with open(activate_this) as f:
	exec(f.read(), dict(__file__=activate_this))
import sys
sys.path.insert(0, '/var/www/html/cerebrumXalgorithms/pythonapi_algos')
import logging
logging.basicConfig(stream=sys.stderr)

from apis.app import app as application
