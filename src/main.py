import zerorpc

class HelloRPC(object):
    def hello(self, name):
        return "Hello, %s" % name

s = zerorpc.Server(HelloRPC())
host = "tcp://0.0.0.0:4242"
s.bind(host)
print("Starting on host " + host)
s.run()