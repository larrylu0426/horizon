import sys

class Logger(object):
    def __init__(self, file_name) -> None:
        self.terminal = sys.stdout
        self.log = open(file_name, "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        pass