from datetime import datetime
import donkeycar as dk
import re
import time
import logging
from donkeycar.utils import Singleton, bound
import numpy as np
from collections import deque

from donkeycar.parts.actuator import RobocarsHat
from donkeycar.parts.robocars_hat_ctrl import RobocarsHatInCtrl

from donkeycar.utilities.logger import init_special_logger
from transitions.extensions import HierarchicalMachine
from collections import deque
import numpy as np

drivetrainlogger = init_special_logger ("DrivetrainCtrl")
drivetrainlogger.setLevel(logging.INFO)
logging.getLogger('transitions').setLevel(logging.INFO)

class RobocarsHatDriveCtrl(metaclass=Singleton):

    LANE_LEFT=0
    LANE_CENTER=1
    LANE_RIGHT=2    
    
    LANE_LABEL=["left","center","right"]

    states = [
            {'name':'stopped'}, 
            {'name':'driving'}
            ]

    def set_regularspeed(self):
        self.fix_throttle = self.cfg.ROBOCARS_TRAIN_CTRL_REGULAR_SPEED

    def __init__(self, cfg):

        self.cfg = cfg
        self.hatInCtrl = None
        if (self.cfg.USE_ROBOCARSHAT_AS_CONTROLLER):
            self.hatInCtrl = RobocarsHatInCtrl(self.cfg)
        self.fix_throttle = 0
        self.lane = 0
        self.on = True
        self.requested_lane = self.cfg.DEFAULT_LANE_CENTER

        self.last_angles=deque(maxlen=self.cfg.ROBOCARS_ANGLES_HISTORY_SIZE)

        self.machine = HierarchicalMachine(self, states=self.states, initial='stopped', ignore_invalid_triggers=True)
        self.machine.add_transition (trigger='drive', source='stopped', dest='driving', before='set_regularspeed')
        self.machine.add_transition (trigger='stop', source='driving', dest='stopped')

        drivetrainlogger.info('starting RobocarsHatLaneCtrl Hat Controller')

    def update_angles_history (self,angle):
        if (angle != None) :
            self.last_angles.append(angle)

    def is_driving_straight(self):
        angles_arr = np.array(list(self.last_angles))
        angles_avg = np.mean(angles_arr)
        if angles_avg <= self.cfg.ROBOCARS_ANGLES_AVG_TRESH_STRAIGHT:
            return True
        return False

    def adjust_steering_to_lane(self, angle, lane, requested_lane):
        drivetrainlogger.debug(f"Change lane from {self.LANE_LABEL[lane]} to {self.LANE_LABEL[requested_lane]}")    
        needed_adjustment = int(lane-requested_lane)
        drivetrainlogger.debug(f"LaneCtrl     -> adjust needed {needed_adjustment}")      
        needed_steering_adjustment = self.cfg.ROBOCARS_TRAIN_CTRL_LANE_STEERING_ADJUST_STEPS[abs(needed_adjustment)]
        if (needed_adjustment)>0:
            needed_steering_adjustment = - needed_steering_adjustment
        drivetrainlogger.debug(f"LaneCtrl     -> adjust steering by {needed_steering_adjustment}")      
        angle=bound(angle+needed_steering_adjustment,-1,1)
        return angle

    def processState(self, throttle, angle, mode, lane):
            
        if self.is_stopped(allow_substates=True):
            if (mode != 'user') :
                self.drive()

        if self.is_driving(allow_substates=True):
            throttle=self.fix_throttle
            self.update_angles_history (angle)
            if self.cfg.ROBOCARS_DRIVE_ON_LANE:
                if (is_driving_straight()):
                    self.requested_lane = self.hatInCtrl.getRequestedLane()
                else:
                    self.requested_lane = self.cfg.DEFAULT_LANE_CENTER
                new_angle = self.adjust_steering_to_lane (angle, lane, self.requested_lane)
                angle=new_angle    
            if (mode == 'user') :
                self.stop()

        return throttle, angle
 
    def update(self):
        # not implemented
        pass

    def run_threaded(self, throttle, angle, mode, lane):
        # not implemented
        pass

    def run (self,throttle, angle, mode, lane):
        throttle, angle = self.processState (throttle, angle, mode, lane)
        return throttle, angle
    

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        drivetrainlogger.info('stopping RobocarsHatLaneCtrl Hat Controller')
        time.sleep(.5)

