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


class PathPilot:
    def __init__(self, max_steering_angle, vehicle_length, fix_throttle, brake_throttle):
        self.BRAKE_DURATION_S = 1
        self.max_steering_angle_rad = deg2rad(max_steering_angle)
        self.vehicle_length = vehicle_length
        self.fix_throttle = fix_throttle
        self.brake_throttle = brake_throttle
        self.brake_until = time.time() - 1
        self.poses = deque([], maxlen=20)
        self.path = None
        self.path_origin = None
        self.stanley_metrics = None

        logger.info("PathPilot ready.")
        
    def run(self, path_time, path, pos_time, x, y, yaw, speed):
        pose = np.array([pos_time, x, y, yaw, speed])
        self.poses.append(pose)

        if path_time is not None and path is not None:
            self.path_origin = self.find_pose_with_nearest_time(path_time)
            #self.path = path
            self.path = self.change_frame2(self.path_origin[:2], self.path_origin[2], path)
            logger.info('Received a new path')
        
        if self.path is not None and self.path_origin is not None:
            angle, end_of_path = self.lateral_control(self.path, self.path_origin, pose)
            throttle = self.fix_throttle

            if not end_of_path:
                return angle, throttle, end_of_path, self.stanley_metrics, self.path_origin[0], self.path_origin[1], self.path_origin[2]
            else:
                self.brake_until = time.time() + self.BRAKE_DURATION_S
                self.path = None
                self.path_origin = None
            
        return 0.0, self.brake_throttle if self.brake_until >= time.time() else 0.0, True, None, 0.0, 0.0, 0.0

    def find_pose_with_nearest_time(self, time):
        pose_times = np.array([p[0] for p in self.poses])
        index = np.argmin(np.absolute(pose_times - time))
        return self.poses[index][1:4] # [x, y, yaw]

    # def get_path_for_pose(self, path, x0, y0, yaw0):
    #     translation = np.array([x0, y0])
    #     rotation_angle = yaw0
    #     R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    #     new_path = np.zeros_like(path)
    #     for i in range(path.shape[0]):
    #         new_path[i,:] = R @ path[i,:].T + translation
    #     return new_path

    # change points in F1 to F2 with change_frame(-origin2, -yaw2, points_in_F1)
    def change_frame1(self, translation, rotation, points):
        R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        return (R @ (points + translation).T).T # .T to accept multiple points at once and .T again to get original shape (n,2)
    
    # change points in F2 to F1 with change_frame(origin2, yaw2, points_in_F2)
    def change_frame2(self, translation, rotation, points):
        R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        return (R @ points.T).T + translation# .T to accept multiple points at once and .T again to get original shape (n,2)

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

    def stanley_control(self, previous_waypoint, next_waypoint, position, yaw, velocity, Kc, Ks):
        path_vector = next_waypoint - previous_waypoint

        crosstrack_error = self.crosstrack_error(previous_waypoint, path_vector, position)
        crosstrack_steer = np.arctan2(Kc * crosstrack_error, Ks + abs(velocity))

        trajectory_yaw = np.arctan2(path_vector[1], path_vector[0])
        trajectory_steer = trajectory_yaw - yaw

        # log some metrics to visualize and help debug
        self.stanley_metrics = [crosstrack_error, crosstrack_steer, trajectory_steer]

        return wrap_to_pi(trajectory_steer + crosstrack_steer)
    
    def lateral_control(self, path, path_origin, pose):
        # control is done in path frame (ie origin is pose when path was generated)
        # position = pose[1:3] - path_origin[0:2]
        # rotation_angle = - path_origin[2]
        # R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        # position = R @ position
        # yaw = pose[3] - path_origin[2]

        # control is done in path frame (ie origin is pose when path was generated)
        # position = self.change_frame(- path_origin[0:2], - path_origin[2], pose[1:3])
        # position = pose[1:3]
        # yaw = pose[3] - path_origin[2]

        # control is done in ref frame
        position = pose[1:3]
        yaw = pose[3]
        v = pose[4]

        # convert from vehicule center to front axle center
        position = position + self.vehicle_length / 2 * np.array([np.cos(yaw), np.sin(yaw)])

        previous_waypoint, next_waypoint, end_of_path = self.find_nearest_waypoints(path, position) # path=self.get_path_for_pose(pose)
        
        steer_output = self.stanley_control(previous_waypoint, next_waypoint, position, yaw, v, Kc=2.0, Ks=0.1)

        # convert to [-1, 1] range
        angle = min(1.0, max(-1.0, steer_output / self.max_steering_angle_rad))

        return angle, end_of_path

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')
        # keep looping infinitely until the thread is stopped
        # while self.on:
        #     self.run()

    def shutdown(self):
        logger.info('Stopping PathPilot')
        # indicate that the thread should be stopped
        # self.on = False
        # time.sleep(.5)
