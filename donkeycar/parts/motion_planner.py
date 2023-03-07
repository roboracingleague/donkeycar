import logging
from collections import deque
import numpy as np
import math


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# change points given in F1 to F2: translate first and then rotate
def change_frame_1to2(origin2, yaw2, points_in_f1):
    rotation = -yaw2
    R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    return (R @ (points_in_f1 - origin2).T).T

# change points given in F2 to F1: rotate first and then translate
def change_frame_2to1(origin2, yaw2, points_in_f2):
    R = np.array([[np.cos(yaw2), -np.sin(yaw2)], [np.sin(yaw2), np.cos(yaw2)]])
    return (R @ points_in_f2.T).T + origin2


class LocalPlanner:
    '''
    Local Planner class. Takes environment mapping inputs and evaluate a position and velocity trajectory.
    '''
    def __init__(self):
        logger.info("LocalPlanner ready")

    def run(self, pos_time, x, y, yaw, speed, lane_center, left_lane, right_lane, occupancy_grid=None, signs=None):

        trajectory = lane_center

        return trajectory

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')
    
    def shutdown(self):
        logger.info('Stopping LocalPlanner')


class TestTrajectory():
    def __init__(self):
        self.trajectory = self.shift_x(self.generate_trapeze(), 15)
        self.sent = False
        self.poses = deque([], maxlen=20)
        logger.info('Starting TestTrajectory')

    def generate_path_straight_50cm(self):
        x = np.array([x for x in range(50)]) / 100
        y = np.zeros((50,))
        p = np.vstack((x,y)).T
        return p

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def generate_sigmoid(self):
        x = np.array([x for x in range(150)]) / 100
        y = np.array([ self.sigmoid((x-75)/10) for x in range(150)]) / 2
        return np.vstack((x,y)).T
    
    def generate_trapeze(self):
        x = np.array([x for x in range(150)]) / 100
        y = [0]
        for i in range(1, 150):
            y.append(y[-1] + (i if i < 75 else 150 - i))
        y = np.array(y) / (2 * max(y))
        return np.vstack((x,y)).T
    
    def shift_x(self, trajectory, delta):
        # add delta (int) cm to the beginning of the trajectory
        x = np.concatenate([np.array([x for x in range(delta)]) / 100, trajectory[:,0] + delta / 100])
        y = np.concatenate([np.array([0] * delta), trajectory[:,1]])
        return np.vstack((x,y)).T
    
    def find_pose_with_nearest_time(self, time):
        pose_times = np.array([p[0] for p in self.poses])
        index = np.argmin(np.absolute(pose_times - time))
        return self.poses[index][1:4] # [x, y, yaw]

    def run(self, run_pilot, pos_time, x, y, yaw):
        pose = np.array([pos_time, x, y, yaw])
        self.poses.append(pose)

        if run_pilot:
            if not self.sent:
                self.sent = True
                trajectory_origin = self.find_pose_with_nearest_time(pos_time)
                trajectory = change_frame_2to1(trajectory_origin[:2], trajectory_origin[2], self.trajectory)
                logger.debug('Sending a new path')
                return trajectory, trajectory_origin[0] # need to send 2 values at least to record None value in vehicle mem
        else:
            self.sent = False
        
        return None, None

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')

    def shutdown(self):
        logger.info('Stopping TestTrajectory')