from donkeycar.utils import Singleton
import time

class RobocarsExtensibleRecord(metaclass=Singleton):

    def __init__(self, cfg):
        self.reset_data()

    def reset_data (self):
        self.record={}
        self.add_data('ts',time.time_ns() / (10 ** 9), 'float')

    def add_data (self, key, data, input_type):
        if input_type == 'float':
            # Handle np.float() types gracefully
            self.record[key] = float(data)
        elif input_type == 'str':
            self.record[key] = data
        elif input_type == 'int':
            self.record[key] = int(data)
        elif input_type == 'boolean':
            self.record[key] = bool(data)
        elif input_type == 'nparray':
            self.record[key] = data.tolist()
        elif input_type == 'list' or input_type == 'vector':
            self.record[key] = list(data)
        else:
            self.record[key] = 'type error'

    def flatten_data (self, kind, contents):
        if (kind=='contents'):
            for k,v in self.record:
                contents[k] = v
        if (kind=='labels'):
            for k,v in self.record:
                contents.append(k)

    
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