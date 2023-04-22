import logging
import time
import numpy as np
from donkeycar.utils import deg2rad, wrap_to_pi


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PathPilot:
    def __init__(self, max_steering_angle, vehicle_length, fix_throttle, brake_throttle, Kc=2.0, Ks=0.1):
        self.max_steering_angle_rad = deg2rad(max_steering_angle)
        self.vehicle_length = vehicle_length
        self.fix_throttle = fix_throttle
        self.brake_throttle = brake_throttle
        self.Kc = Kc
        self.Ks = Ks
        logger.info("PathPilot ready.")
        
    def run(self, path, x, y, yaw, speed):
        if path is not None:
            angle, throttle, end_of_path, stanley_metrics = self.control(path, x, y, yaw, speed)
            if not end_of_path:
                return angle, throttle, end_of_path, stanley_metrics
        
        # no path or end of path reached: let's brake
        if abs(speed) > 0.1:
            return 0.0, self.brake_throttle, True, None
        return 0.0, 0.0, True, None

    def find_nearest_waypoints(self, path, position):
        end_of_path = False

        distances = np.linalg.norm(path - position, axis=1)
        nearest_index = np.argmin(distances)

        # find which from previous or next path point is the nearest
        if nearest_index <= 0:
            previous_index = 0
            next_index = 1
        elif nearest_index >= path.shape[0] - 1:
            previous_index = path.shape[0] - 2
            next_index = path.shape[0] - 1
            end_of_path = True
        elif distances[nearest_index - 1] <= distances[nearest_index + 1]:
            previous_index = nearest_index - 1
            next_index = nearest_index
        else:
            previous_index = nearest_index
            next_index = nearest_index + 1

        return path[previous_index], path[next_index], end_of_path

    def crosstrack_error(self, waypoint, path_vector, position):
        # when trajectory line is ax + by + c = 0 then cte = (ax + by + c) / sqrt(a^2 + b^2)
        a = path_vector[1]
        b = - path_vector[0]
        c = - (a * waypoint[0] + b * waypoint[1])
        return (a * position[0] + b * position[1] + c) / np.linalg.norm(path_vector)

    def stanley_control(self, previous_waypoint, next_waypoint, position, yaw, speed):
        path_vector = next_waypoint - previous_waypoint

        crosstrack_error = self.crosstrack_error(previous_waypoint, path_vector, position)
        crosstrack_steer = np.arctan2(self.Kc * crosstrack_error, self.Ks + abs(speed))

        trajectory_yaw = np.arctan2(path_vector[1], path_vector[0])
        trajectory_steer = trajectory_yaw - yaw

        # log some metrics to visualize and help debug
        stanley_metrics = [crosstrack_error, crosstrack_steer, trajectory_steer]

        return wrap_to_pi(trajectory_steer + crosstrack_steer), stanley_metrics
    
    def control(self, path, x, y, yaw, speed):
        # convert from vehicule center to front axle center
        position = np.array([x, y])
        position = position + self.vehicle_length / 2 * np.array([np.cos(yaw), np.sin(yaw)])

        previous_waypoint, next_waypoint, end_of_path = self.find_nearest_waypoints(path, position)
        
        steer_output, stanley_metrics = self.stanley_control(previous_waypoint, next_waypoint, position, yaw, speed)

        # convert to [-1, 1] range
        angle = min(1.0, max(-1.0, steer_output / self.max_steering_angle_rad))

        throttle = self.fix_throttle

        return angle, throttle, end_of_path, stanley_metrics

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')

    def shutdown(self):
        logger.info('Stopping PathPilot')
