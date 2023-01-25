import logging
import time
from numpy import cos, sin, tan, pi, floor
from donkeycar.utils import Singleton, deg2rad, rad2deg, wrap_to_pi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FeedForwardLocalization:
    '''
    A class to estimate current pose based on commands (throttle, angle).
    '''
    def __init__(self, frequency, vehicle_length, center_length, max_velocity, max_steering_angle, odometry=None):
        self.poll_delay = 1.0 / frequency
        self.vehicle_length = vehicle_length # distance from rear axle to front axle
        self.center_length = center_length # distance from rear axle to localization frame origin (gravity center)
        self.max_velocity = max_velocity
        self.max_steering_angle_rad = deg2rad(max_steering_angle)
        self.odometry = odometry # odometry part with a run method that returns speed in m/s
        
        self.commands = FeedForwardLocalizationCommands()

        self.pose = (time.time(), 0.0, 0.0, 0.0, 0.0)
        self.on = True
        logger.info("FeedForwardLocalization ready.")
        
    def get_commands(self):
        current_time = time.time()
        command_time, throttle, angle = self.commands.get_commands()

        # if given use odometry instead of feed forward estimate
        # else approximate velocity with throttle: works if throttle is fed into a longitudinal asserv
        velocity = self.odometry.run() if self.odometry else throttle * self.max_velocity

        # approximate steering with angle: probably not linear and true only at steady state...
        steering = angle * self.max_steering_angle_rad
        
        return current_time, velocity, steering

    def run(self):
        current_time, velocity, steering = self.get_commands()

        last_time, last_x, last_y, last_theta, last_velocity = self.pose
        dt = current_time - last_time
        theta = last_theta + dt * velocity * tan(steering) / self.vehicle_length # TODO works only for front axle
        theta = wrap_to_pi(theta)
        x = last_x + dt * velocity * cos(theta)
        y = last_y + dt * velocity * sin(theta)
        self.pose = (current_time, x, y, theta, velocity)

        # console output for debugging and calibration
        if(logger.isEnabledFor(logging.DEBUG)):
            logger.debug('Steering: {:>6,.1f} deg, x: {:>9,.3f} m, y: {:>9,.3f} m, theta: {:>6,.1f} deg, velocity: {:>7,.3f} m/s'.format(rad2deg(steering), x, y, rad2deg(theta), velocity))

        return x, y, theta, velocity

    def run_threaded(self):
        return self.pose[1:] # remove time

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.run()
            delay = self.pose[0] - time.time() + self.poll_delay
            if delay > 0:
                time.sleep(delay)

    def shutdown(self):
        self.on = False
        logger.info('Stopping FeedForwardLocalization')
 

class FeedForwardLocalizationCommands(metaclass=Singleton):
    '''
    Helper class to register last throttle and angle and keep them ready for FeedForwardLocalization class
    '''
    def __init__(self):
        self.commands = (time.time(), 0.0, 0.0)
        self.on = True
        logger.info("FeedForwardLocalizationCommands ready.")
        
    def run(self, throttle, angle):
        self.commands = (time.time(), throttle, angle)

    def get_commands(self):
        return self.commands

    def shutdown(self):
        self.on = False
        logger.info('Stopping FeedForwardLocalizationCommands')
