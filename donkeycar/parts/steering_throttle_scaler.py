import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def map_range_float(x, X_min, X_max, Y_min, Y_max):
    '''
    Same as map_range but supports floats return, rounded to 2 decimal places
    '''
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range/Y_range

    x=max(min(x, X_max), X_min)
    y = ((x-X_min) / XY_ratio + Y_min)

    # print("y= {}".format(y))

    return round(y,2)

class SteeringThrottleScaler:
    def __init__(self, cfg):
        self.steering_factor = cfg.STEERING_FACTOR
        self.throttle_factor = cfg.THROTTLE_FACTOR
        self.steering_on_throttle_factor = cfg.STEERING_ON_THROTTLE_FACTOR
        self.min_throttle = cfg.AUX_FEATURE_LOCAL_ANGLE_FIX_THROTTLE_MIN
        self.max_throttle = cfg.AUX_FEATURE_LOCAL_ANGLE_FIX_THROTTLE_MAX
        self.running = True    

    def run(self, steering_angle, throttle):
        scaled_throttle = throttle * self.throttle_factor
        dyn_steering_factor = map_range_float(throttle, self.min_throttle, self.max_throttle, 1.0, self.steering_on_throttle_factor)
        #scaled_steering = steering_angle * self.steering_factor * self.steering_on_throttle_factor * self.throttle_factor
        scaled_steering = steering_angle * dyn_steering_factor
        return scaled_steering, scaled_throttle
        
    def shutdown(self):
        self.running = False
        logger.info('Stopping Steering Throttle Scaler')
