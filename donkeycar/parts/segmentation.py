import logging
import time
from collections import deque
from pathlib import Path
import numpy as np
import depthai as dai
from depthai_sdk import toTensorResult
from donkey.parts.lane_detection import detect_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CameraError(Exception):
    pass

class OakSegmentationCamera:
    def __init__(self, model_blol_path, framerate=4):
        self.blob_path = Path(model_blol_path)
        self.frame = None
        self.frame_time = None
        self.segmentation = None
        self.segmentation_time = None
        self.path = None
        self.on = False
        self.device = None
        self.preview_queue = None
        self.segmentation_queue = None
        self.latencies = deque([], maxlen=10)
        self.segmentation_latencies = deque([], maxlen=10)
        self.lane_detection_latencies = deque([], maxlen=10)

        self.pipeline = dai.Pipeline()
        # This might improve reducing the latency on some systems
        self.pipeline.setXLinkChunkSize(0)

        # Define sources
        cam = self.pipeline.create(dai.node.ColorCamera)

        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # 1920 x 1080
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.initialControl.setManualFocus(0) # from calibration data
        cam.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT) # CLOUDY_DAYLIGHT FLUORESCENT

        cam.setFps(framerate) # heavily influence e2e latency

        cam.setPreviewKeepAspectRatio(False)
        resolution = (896, 512)
        cam.setPreviewSize(resolution) # wich means cropping if aspect ratio kept
        cam.setIspScale(9, 19) # "scale" sensor size, (9,19) = 910x512 ; seems very slightly faster eg. from 30.48ms to 28.83ms
        # see https://docs.google.com/spreadsheets/d/153yTstShkJqsPbkPOQjsVRmM8ZO3A6sCqm7uayGF-EE/edit#gid=0

        # Segmentation model
        seg = self.pipeline.create(dai.node.NeuralNetwork)
        seg.setBlobPath(self.blob_path)

        cam.preview.link(seg.input)

        seg.input.setQueueSize(1)
        seg.input.setBlocking(False)

        # Send NN out to the host via XLink
        outSegmentation = self.pipeline.create(dai.node.XLinkOut)
        outSegmentation.setStreamName("outSegmentation")
        seg.out.link(outSegmentation.input)

        outPreview = self.pipeline.create(dai.node.XLinkOut)
        outPreview.setStreamName("outPreview")
        cam.preview.link(outPreview.input)

        try:
            # Connect to device and start pipeline
            logger.info('Starting OAK camera')
            self.device = dai.Device(self.pipeline)
            self.preview_queue = self.device.getOutputQueue("outPreview", maxSize=1, blocking=False)
            self.segmentation_queue = self.device.getOutputQueue("outSegmentation", maxSize=1, blocking=False)
            try:
                logger.info("OAK camera USB speed: " + str(self.device.getUsbSpeed()))
            except:
                logger.info("ERROR while retrieving OAK camera USB speed")

            # get the first frame or timeout
            warming_time = time.time() + 5  # seconds
            while self.frame is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("Unable to start OAK camera.")

            logger.info("OAK camera ready.")
            self.on = True
        except:
            self.shutdown()
            raise
        
    def run(self):
        # grab the frame from the stream 
        if self.preview_queue is not None:
            preview_frame = self.preview_queue.tryGet()

            if preview_frame is not None:
                self.frame = preview_frame.getFrame()
                self.frame_time = preview_frame.getTimestamp()

                if logger.isEnabledFor(logging.DEBUG):
                    # Latency in miliseconds
                    self.latencies.append((dai.Clock.now() - preview_frame.getTimestamp()).total_seconds() * 1000)
                    logger.debug('Image latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}' \
                        .format(self.latencies[-1], np.average(self.latencies), np.std(self.latencies)))
        
        if self.segmentation_queue is not None:
            segmentation_frame = self.segmentation_queue.tryGet()

            if segmentation_frame is not None:
                start_time = dai.Clock.now()
                if logger.isEnabledFor(logging.DEBUG):
                    # Latency in milliseconds
                    self.segmentation_latencies.append((start_time - segmentation_frame.getTimestamp()).total_seconds() * 1000)
                    logger.debug('Segmentation latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}' \
                        .format(self.segmentation_latencies[-1], np.average(self.segmentation_latencies), np.std(self.segmentation_latencies)))

                data = np.squeeze(toTensorResult(segmentation_frame)["L0317_ReWeight_SoftMax"])

                # background, road, curb, mark
                classes = [0, 1, 2, 3]
                classes = np.asarray(classes, dtype=np.uint8)
                indices = np.argmax(data, axis=0)

                segmentation = np.take(classes, indices, axis=0)
                segmentation_time = segmentation_frame.getTimestamp()

                path = detect_path(segmentation)

                self.segmentation = segmentation
                self.segmentation_time = segmentation_time
                self.path = path

                if logger.isEnabledFor(logging.DEBUG):
                    # Latency in milliseconds
                    self.lane_detection_latencies.append((dai.Clock.now() - start_time).total_seconds() * 1000)
                    logger.debug('Lane detection latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}' \
                        .format(self.lane_detection_latencies[-1], np.average(self.lane_detection_latencies), np.std(self.lane_detection_latencies)))

        return self.frame, self.frame_time, self.segmentation, self.segmentation_time, self.path

    def run_threaded(self):
        frame = self.frame
        frame_time = self.frame_time
        segmentation = self.segmentation
        segmentation_time = self.segmentation_time
        path = self.path
        self.frame = None
        self.frame_time = None
        self.segmentation = None
        self.segmentation_time = None
        self.path = None
        return frame, frame_time, segmentation, segmentation_time, path

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.run()

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        logger.info('Stopping OAK camera')
        time.sleep(.5)
        if self.device is not None:
            self.device.close()
        self.device = None
        self.preview_queue = None
        self.segmentation_queue = None
        self.pipeline = None


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
