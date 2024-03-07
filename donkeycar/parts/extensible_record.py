from donkeycar.utils import Singleton
import time

class RobocarsExtensibleRecord(metaclass=Singleton):

    def __init__(self):
        self.data = {}
        self.reset_data()

    def reset_data (self):
        for key in self.data.keys() :
            self.data[key]['data']={}

    def register_data (self, key, type):
        if key in self.data.keys() :
            print(f"RobocarsExtensibleRecord : Key {key} already registered !")
            return None
        self.data[key]={}
        self.data[key]['type']=type
        return key
    
    def record_data (self, key, data):
        if (key == None):
            return
        if self.data[key]['type'] == 'float':
            # Handle np.float() types gracefully
            self.data[key]['data'] = float(data)
        elif self.data[key]['type'] == 'str':
            self.data[key]['data'] = data
        elif self.data[key]['type'] == 'int':
            self.data[key]['data'] = int(data)
        elif self.data[key]['type'] == 'boolean':
            self.data[key]['data'] = bool(data)
        elif self.data[key]['type'] == 'nparray':
            self.data[key]['data'] = data.tolist()
        elif self.data[key]['type'] == 'list' or input_type == 'vector':
            self.data[key]['data'] = list(data)
        else:
            self.data[key]['data'] = 'type error'

    def flatten_data (self, kind, contents):
        if (kind=='contents'):
            for k in self.data.keys():
                contents[k] = self.data[k]['data']
        if (kind=='labels'):
            for k in self.data.keys():
                contents.append(k)
        if (kind=='types'):
            for k in self.data.keys():
                contents.append(self.data[k]['type'])

    
    def update(self):
        # not implemented
        pass

    def run_threaded(self):
        # not implemented
        pass

    def run (self):
        self.reset_data()

    def shutdown(self):
        # indicate that the thread should be stopped
        pass