from donkeycar.utils import Singleton
import time

class RobocarsExtensibleRecord(metaclass=Singleton):

    def __init__(self, cfg):
        self.key = 'Ext'
        self.reset_data()

    def reset_data (self):
        self.record={}

    def add_data (self, key, data):
        self.record[key] = {'ts':time.time(), 'data':data}

    def get_data (self, request):
        if request=='key':
            return self.key
        return self.record