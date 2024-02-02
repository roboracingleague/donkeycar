from donkeycar.utils import Singleton
import time

class RobocarsExtensibleRecord(metaclass=Singleton):

    def __init__(self, cfg):
        self.key = 'Ext'
        self.reset_data()

    def reset_data (self):
        self.record={}
        self.add_data('ts',time.time())

    def add_data (self, key, data):
        self.record[key] = data

    def get_data (self, request):
        if request=='key':
            return self.key
        return self.record
    
    def update(self):
        # not implemented
        pass

    def run_threaded(self, throttle, angle, mode, sl):
        # not implemented
        pass

    def run (self):
        self.reset_data()

    def shutdown(self):
        # indicate that the thread should be stopped
        pass