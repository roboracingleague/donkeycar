import logging
import time
from collections import deque
from pathlib import Path
import numpy as np
import depthai as dai


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SegmentationPilot:
    def __init__(self):
        self.on = False
        self.poses = deque([], maxlen=20)

        self.path = None
        self.time = None
        self.path_pose = None
        self.throttle = None
        self.angle = None

        logger.info("Segmentation Pilot ready.")
        self.on = True

        
    def run(self, segmentation_time, path, pose): # pose = [x, y, theta, v, yaw, t]
        if segmentation_time is not None and path is not None:
            self.path = path
            self.time = segmentation_time
            self.path_pose = self.find_pose_with_nearest_time(segmentation_time)
        
        if pose is not None:
            self.poses.append(pose)
            if self.path is not None and self.path_pose is not None:
                angle = self.lateral_control(pose)
                return 0, angle

        return 0,0

    def find_pose_with_nearest_time(self, time):
        min_delta = self.poses[0][3] - time
        min_index = 0
        for i in range(len(self.poses)):
            delta = self.poses[i][3] - time
            if delta < min_delta:
                min_delta = delta
                min_index = i
        return self.poses[min_index]

    def find_nearest_waypoint_index(self, waypoints, position):
        distances = waypoints - position
        index = np.argmin(distances)
        return index, distances[index]

    def get_path_for_pose(self, pose):
        translation = self.path_pose[0:2] - pose[0:2]
        rotation_angle = self.path_pose[2] - pose[2]
        R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        return (self.path + translation) @ R
    
    def lateral_control(self, pose):
        position = pose[0:2]
        yaw = pose[4]
        v = pose[3]
        waypoints = self.get_path_for_pose(pose)
        nearest_waypoint_index, nearest_waypoint_distance = self.find_nearest_waypoint_index(waypoints, position)

        # find with from previous or next waypoint is the nearest
        if nearest_waypoint_index <= 0:
            previous_index = 0
            next_index = 1
        elif nearest_waypoint_index >= waypoints.shape[0]-1:
            previous_index = waypoints.shape[0]-2
            next_index = waypoints.shape[0]-1
        elif np.linalg.norm(waypoints[nearest_waypoint_index-1] - position) <= np.linalg.norm(waypoints[nearest_waypoint_index+1] - position):
            previous_index = nearest_waypoint_index-1
            next_index = nearest_waypoint_index
        else:
            previous_index = nearest_waypoint_index
            next_index = nearest_waypoint_index+1

        previous_waypoint = np.array(waypoints[previous_index][:2])
        next_waypoint = np.array(waypoints[next_index][:2])
        trajectory = next_waypoint - previous_waypoint
        #front_position = position + self._L_center_front * np.array([np.cos(yaw), np.sin(yaw)]) # center of front axis
        
        # trajectory line: ax + by + c = 0
        # e = (ax + by + c) / sqrt(a^2 + b^2)
        # a = trajectory[1]
        # b = - trajectory[0]
        # c = - (a * previous_waypoint[0] + b * previous_waypoint[1])
        # crosstrack_error = (a * front_position[0] + b * front_position[1] + c) / np.linalg.norm(trajectory)
        crosstrack_error = nearest_waypoint_distance

        trajectory_yaw = np.arctan2(trajectory[1], trajectory[0])
        
        # Change the steer output with the lateral controller. 
        Kc = 0.7
        Ks = 0.001
        crosstrack_steer = np.arctan2(Kc * crosstrack_error, Ks + v)
        trajectory_steer = trajectory_yaw - yaw
        steer_output = trajectory_steer + crosstrack_steer

        max_steer_angle = 30 * np.pi / 180
        angle = steer_output / max_steer_angle
        angle = np.min(1.0, np.max(-1.0, angle))
        return angle

    def run_threaded(self):
        return self.throttle, self.angle

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.run()

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        logger.info('Stopping Segmentation Pilot')
        time.sleep(.5)
