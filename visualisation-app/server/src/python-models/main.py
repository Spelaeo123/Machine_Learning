# This module uses ZeroRPC to run a Server and bind it to localhost.
# This allows us to communicate with our Python models from our NodeJS server.

from dotenv import load_dotenv
import zerorpc
import os

# Loads dotenv so we can acces environment variables.
load_dotenv()

# Public API for interfaceing with the Python Models
class PublicApi(object):
    def hello(self, name):
        return "Hello, %s" % name

host = os.getenv("PYTHON_SERVER_ADDRESS")
print("Starting on host " + host)

server = zerorpc.Server(PublicApi())
server.bind(host)
server.run()