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

    def run(self, pos_time, x, y, yaw, speed, left_lane, lane_center, right_lane, occupancy_grid=None, signs=None):
        trajectory = self.plan_trajectory(lane_center)
        return trajectory

    def plan_trajectory(self, lane_center):
        return lane_center

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')
    
    def shutdown(self):
        logger.info('Stopping LocalPlanner')


class TestTrajectory():
    def __init__(self):
        self.updated = False
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

    def update_trajectory(self, pose):
        trajectory = self.shift_x(self.generate_trapeze(), 15)
        return change_frame_2to1(pose[1:3], pose[3], trajectory)

    def run(self, run_pilot, pos_time, x, y, yaw):
        if run_pilot:
            if not self.updated:
                pose = np.array([pos_time, x, y, yaw])
                trajectory = self.update_trajectory(pose)
                self.updated = True
                logger.debug('Sending a new path')
                return trajectory, pos_time # need to send 2 values at least to record None value in vehicle mem
        else:
            self.updated = False
        
        return None, None

    def run_threaded(self):
        raise RuntimeError('Not implemented')

    def update(self):
        raise RuntimeError('Not implemented')

    def shutdown(self):
        logger.info('Stopping TestTrajectory')
