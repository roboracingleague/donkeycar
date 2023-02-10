import logging
import time
from collections import deque
from pathlib import Path
import numpy as np
import depthai as dai
from donkeycar.utils import deg2rad, rad2deg, wrap_to_pi
import numpy as np
import math


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestTrajectory():
    def __init__(self):
        self.path = self.shift_x(self.generate_path_w_trapeze(), 15)
        self.sent = False
        logger.info('Starting TestTrajectory')

    def generate_path_straight_50cm(self):
        x = np.array([x for x in range(50)]) / 100
        y = np.zeros((50,))
        p = np.vstack((x,y)).T
        return p

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def generate_path_sigmoid(self):
        x = np.array([x for x in range(150)]) / 100
        y = np.array([ self.sigmoid((x-75)/10) for x in range(150)]) / 2
        p = np.vstack((x,y)).T
        return p
    
    def generate_path_w_trapeze(self):
        x = np.array([x for x in range(150)]) / 100
        y = [0]
        for i in range(1, 150):
                y.append(y[-1] + (i if i < 75 else 150 - i))
        y = np.array(y) / (2 * max(y))
        p = np.vstack((x,y)).T
        return p
    
    def shift_x(self, p, l_cm):
        # add l cm to the beginning of the trajectory
        n = l_cm
        x = np.concatenate([np.array([x for x in range(n)])/100, p[:,0] + n/100])
        y = np.concatenate([np.array([0] * n), p[:,1]])
        p = np.vstack((x,y)).T
        return p

    def run(self, run_pilot):
        if run_pilot:
            if not self.sent:
                self.sent = True
                return time.time(), self.path
        else:
            self.sent = False
        return None, None

    def run_threaded(self):
        pass

    def update(self):
        pass

    def shutdown(self):
        logger.info('Stopping TestTrajectory')


class PathFollower:
    def __init__(self, max_steering_angle, vehicle_length, fix_throttle):
        self.on = False
        self.max_steering_angle_rad = deg2rad(max_steering_angle)
        self.vehicle_length = vehicle_length
        self.fix_throttle = fix_throttle
        self.poses = deque([], maxlen=20)

        self.path = None
        self.time = None
        self.path_origin = None
        self.throttle = None
        self.angle = None

        logger.info("PathFollower ready.")
        self.on = True

        
    def run(self, path_time, path, pose_time, pose_x, pose_y, pose_theta, velocity): # pose = [time, x, y, theta, v] ; path = [[x1, y1],[x2, y2], ...] ; pose relative to rear axle
        pose = np.array([pose_time, pose_x, pose_y, pose_theta, velocity])
        self.poses.append(pose)

        if path_time is not None and path is not None:
            self.path = path
            self.time = path_time
            self.path_origin = self.find_pose_with_nearest_time(path_time)
            print('got new path')
        
        if self.path is not None and self.path_origin is not None:
            angle, end_of_path, crosstrack_error, crosstrack_steer, trajectory_steer = self.lateral_control(pose)
            return angle, self.fix_throttle if not end_of_path else -0.20, end_of_path, crosstrack_error, crosstrack_steer, trajectory_steer, self.path_origin[1], self.path_origin[2], self.path_origin[3]

        return 0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def find_pose_with_nearest_time(self, time):
        min_delta = self.poses[0][0] - time
        min_index = 0
        for i in range(len(self.poses)):
            delta = self.poses[i][0] - time
            if delta < min_delta:
                min_delta = delta
                min_index = i
        return self.poses[min_index]

    def find_nearest_waypoint_index(self, waypoints, position):
        distances = np.linalg.norm(waypoints - position, axis=1)
        index = np.argmin(distances)
        return index, distances[index]

    # def get_path_for_pose(self, pose):
    #     translation = self.path_pose[1:3] - pose[1:3]
    #     rotation_angle = self.path_pose[3] - pose[3]
    #     R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    #     return (self.path + translation) @ R
    
    def lateral_control(self, pose):
        # control is done in path frame (ie origin is pose when path was generated)
        position = pose[1:3] - self.path_origin[1:3]
        yaw = pose[3] - self.path_origin[3]
        v = pose[4]

        # use front axle center as ref
        front_position = position + self.vehicle_length / 2 * np.array([np.cos(yaw), np.sin(yaw)]) # from vehicule center to center of front axis
        position = front_position

        waypoints = self.path # self.get_path_for_pose(pose)
        nearest_waypoint_index, nearest_waypoint_distance = self.find_nearest_waypoint_index(waypoints, position)

        # find which from previous or next waypoint is the nearest
        end_of_path = False
        if nearest_waypoint_index <= 0:
            previous_index = 0
            next_index = 1
        elif nearest_waypoint_index >= waypoints.shape[0]-1:
            previous_index = waypoints.shape[0]-2
            next_index = waypoints.shape[0]-1
            end_of_path = True
        elif np.linalg.norm(waypoints[nearest_waypoint_index-1] - position) <= np.linalg.norm(waypoints[nearest_waypoint_index+1] - position):
            previous_index = nearest_waypoint_index-1
            next_index = nearest_waypoint_index
        else:
            previous_index = nearest_waypoint_index
            next_index = nearest_waypoint_index+1

        previous_waypoint = np.array(waypoints[previous_index])
        next_waypoint = np.array(waypoints[next_index])
        trajectory = next_waypoint - previous_waypoint
        #front_position = position + self._L_center_front * np.array([np.cos(yaw), np.sin(yaw)]) # center of front axis
        
        # trajectory line: ax + by + c = 0
        # e = (ax + by + c) / sqrt(a^2 + b^2)
        a = trajectory[1]
        b = - trajectory[0]
        c = - (a * previous_waypoint[0] + b * previous_waypoint[1])
        crosstrack_error = (a * front_position[0] + b * front_position[1] + c) / np.linalg.norm(trajectory)
        # not signed...
        # crosstrack_error = nearest_waypoint_distance

        trajectory_yaw = np.arctan2(trajectory[1], trajectory[0])
        
        # Change the steer output with the lateral controller. 
        Kc = 2.0
        Ks = 0.1
        crosstrack_steer = np.arctan2(Kc * crosstrack_error, Ks + abs(v))
        trajectory_steer = trajectory_yaw - yaw
        steer_output = wrap_to_pi(trajectory_steer + crosstrack_steer)

        angle = steer_output / self.max_steering_angle_rad
        angle = min(1.0, max(-1.0, angle))

        return angle, end_of_path, crosstrack_error, crosstrack_steer, trajectory_steer

    def run_threaded(self):
        raise RuntimeError('Not implemented')
        return self.throttle, self.angle

    def update(self):
        raise RuntimeError('Not implemented')
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.run()

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        logger.info('Stopping PathFollower')
        time.sleep(.5)
