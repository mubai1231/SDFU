import os
import sys


class StreamToLogger:
    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, 'w')
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()
