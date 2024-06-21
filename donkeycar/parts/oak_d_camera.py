import logging
import time
from collections import deque
import numpy as np
import depthai as dai
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CameraError(Exception):
    pass

class OakDCameraBuilder:
    def __init__(self):
        self.width = None
        self.height = None
        self.depth = 3
        self.isp_scale = None
        self.framerate = 30
        self.enable_depth = False
        self.depth_crop_rect = None
        self.enable_obstacle_dist = False
        self.rgb_resolution = "1080p"
        self.rgb_apply_cropping = False
        self.rgb_sensor_crop_x = 0.0
        self.rgb_sensor_crop_y = 0.125
        self.rgb_video_size = (1280,600)
        self.rgb_apply_manual_conf = False
        self.rgb_exposure_time = 2000
        self.rgb_sensor_iso = 1200
        self.rgb_wb_manual = 2800
        self.use_camera_tuning_blob = False
        self.enable_undistort_rgb = False
        self.pixel_crop_height = 35

    def with_width(self, width):
        self.width = width
        return self

    def with_height(self, height):
        self.height = height
        return self

    def with_depth(self, depth):
        self.depth = depth
        return self

    def with_isp_scale(self, isp_scale):
        self.isp_scale = isp_scale
        return self

    def with_framerate(self, framerate):
        self.framerate = framerate
        return self

    def with_enable_depth(self, enable_depth):
        self.enable_depth = enable_depth
        return self
    
    def with_depth_crop_rect(self, depth_crop_rect):
        self.depth_crop_rect = depth_crop_rect
        return self

    def with_enable_obstacle_dist(self, enable_obstacle_dist):
        self.enable_obstacle_dist = enable_obstacle_dist
        return self

    def with_rgb_resolution(self, rgb_resolution):
        self.rgb_resolution = rgb_resolution
        return self

    def with_rgb_apply_cropping(self, rgb_apply_cropping):
        self.rgb_apply_cropping = rgb_apply_cropping
        return self

    def with_rgb_sensor_crop_x(self, rgb_sensor_crop_x):
        self.rgb_sensor_crop_x = rgb_sensor_crop_x
        return self

    def with_rgb_sensor_crop_y(self, rgb_sensor_crop_y):
        self.rgb_sensor_crop_y = rgb_sensor_crop_y
        return self

    def with_rgb_video_size(self, rgb_video_size):
        self.rgb_video_size = rgb_video_size
        return self

    def with_rgb_apply_manual_conf(self, rgb_apply_manual_conf):
        self.rgb_apply_manual_conf = rgb_apply_manual_conf
        return self

    def with_rgb_exposure_time(self, rgb_exposure_time):
        self.rgb_exposure_time = rgb_exposure_time
        return self

    def with_rgb_sensor_iso(self, rgb_sensor_iso):
        self.rgb_sensor_iso = rgb_sensor_iso
        return self

    def with_rgb_wb_manual(self, rgb_wb_manual):
        self.rgb_wb_manual = rgb_wb_manual
        return self

    def with_use_camera_tuning_blob(self, use_camera_tuning_blob):
        self.use_camera_tuning_blob = use_camera_tuning_blob
        return self

    def with_enable_undistort_rgb(self, enable_undistort_rgb):
        self.enable_undistort_rgb = enable_undistort_rgb
        return self
    
    def with_pixel_crop_height(self, pixel_crop_height):
        self.pixel_crop_height = pixel_crop_height
        return self
    
    def with_device_id(self, device_id):
        self.device_id = device_id
        return self

    def build(self):
        return OakDCamera(
            width=self.width, 
            height=self.height, 
            depth=self.depth, 
            isp_scale=self.isp_scale, 
            framerate=self.framerate, 
            enable_depth=self.enable_depth, 
            depth_crop_rect=self.depth_crop_rect,
            enable_obstacle_dist=self.enable_obstacle_dist, 
            rgb_resolution=self.rgb_resolution,
            rgb_apply_cropping=self.rgb_apply_cropping,
            rgb_sensor_crop_x=self.rgb_sensor_crop_x,
            rgb_sensor_crop_y=self.rgb_sensor_crop_y,
            rgb_video_size=self.rgb_video_size,
            rgb_apply_manual_conf=self.rgb_apply_manual_conf,
            rgb_exposure_time=self.rgb_exposure_time,
            rgb_sensor_iso=self.rgb_sensor_iso,
            rgb_wb_manual=self.rgb_wb_manual,
            use_camera_tuning_blob=self.use_camera_tuning_blob,
            enable_undistort_rgb=self.enable_undistort_rgb,
            pixel_crop_height=self.pixel_crop_height,
            mxid=self.device_id
        )

class OakDCamera:
    def __init__(self, 
                 width, 
                 height, 
                 depth=3, 
                 isp_scale=None, 
                 framerate=30, 
                 enable_depth=False,
                 depth_crop_rect=None,
                 enable_obstacle_dist=False, 
                 rgb_resolution="1080p",
                 rgb_apply_cropping=False,
                 rgb_sensor_crop_x=0.0,
                 rgb_sensor_crop_y=0.125,
                 rgb_video_size=(1280,600),
                 rgb_apply_manual_conf=False,
                 rgb_exposure_time = 2000,
                 rgb_sensor_iso = 1200,
                 rgb_wb_manual= 2800,
                 use_camera_tuning_blob = False,
                 enable_undistort_rgb = False,
                 pixel_crop_height = 35,
                 mxid = None):
        
        self.width = width
        self.height = height
        self.depth = depth
        self.isp_scale = isp_scale
        self.framerate = framerate
        self.enable_depth = enable_depth
        self.depth_crop_rect = depth_crop_rect
        self.enable_obstacle_dist = enable_obstacle_dist
        self.rgb_resolution = rgb_resolution
        self.rgb_apply_cropping = rgb_apply_cropping
        self.rgb_sensor_crop_x = rgb_sensor_crop_x
        self.rgb_sensor_crop_y = rgb_sensor_crop_y
        self.rgb_video_size = rgb_video_size
        self.rgb_apply_manual_conf = rgb_apply_manual_conf
        self.rgb_exposure_time = rgb_exposure_time
        self.rgb_sensor_iso = rgb_sensor_iso
        self.rgb_wb_manual = rgb_wb_manual
        self.use_camera_tuning_blob = use_camera_tuning_blob
        self.enable_undistort_rgb = enable_undistort_rgb
        self.pixel_crop_height = pixel_crop_height
        # depth config
        self.extended_disparity = True # Closer-in minimum depth, disparity range is doubled (from 95 to 190)
        self.subpixel = False # Better accuracy for longer distance, fractional disparity 32-levels
        self.lr_check = True # Better handling for occlusions
        
        self.on = False
        self.device = None
        self.queue_xout = None
        self.queue_xout_depth = None
        self.queue_xout_spatial_data = None
        self.frame_xout = None
        self.frame_time = None
        self.frame_xout_depth = None
        self.frame_undistorted_rgb = None
        self.roi_distances = []
        self.latencies = deque([], maxlen=20)

        self.alpha = 0

        if mxid:
            self.device_info = dai.DeviceInfo(mxid) # MXID
        else:
            self.device_info = dai.DeviceInfo()

        # Create pipeline
        self.pipeline = dai.Pipeline()
        # self.pipeline.setCameraTuningBlobPath('/tuning_color_ov9782_wide_fov.bin')
        if self.use_camera_tuning_blob == True:
            self.pipeline.setCameraTuningBlobPath('/home/donkey/tuning_exp_limit_8300us.bin')
        
        # self.pipeline.setXLinkChunkSize(0) # default = 64*1024  To adjust latency on some systems if needed RRL
        if self.depth == 3:
            self.create_color_pipeline()
        elif self.depth == 1:
            self.create_mono_pipeline()
        else:
            raise ValueError("'depth' parameter must be either '3' (RGB) or '1' (GRAY)")
        
        if self.enable_depth:
            self.create_depth_pipeline()
        elif self.enable_obstacle_dist:
            self.create_obstacle_dist_pipeline()
        try:
            # Connect to device and start pipeline
            logger.info(f'Starting OAK-D camera id {mxid}')
            self.device = dai.Device(self.pipeline, self.device_info)

            calibData = self.device.readCalibration2()
            rgbCamSocket = dai.CameraBoardSocket.CAM_A
            rgb_w = self.width  # camRgb.getResolutionWidth()
            rgb_h = self.height  # camRgb.getResolutionHeight()
            rgbIntrinsics = np.array(calibData.getCameraIntrinsics(rgbCamSocket, rgb_w, rgb_h))
            rgb_d = np.array(calibData.getDistortionCoefficients(rgbCamSocket))

            rgb_new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(rgbIntrinsics, rgb_d, (rgb_w, rgb_h), self.alpha)

            self.map_x, self.map_y = cv2.initUndistortRectifyMap(rgbIntrinsics, rgb_d, None, rgb_new_cam_matrix, (rgb_w, rgb_h), cv2.CV_32FC1)

            # Create queues
            self.queue_xout = self.device.getOutputQueue("xout", maxSize=1, blocking=False)
            if enable_depth:
                self.queue_xout_depth = self.device.getOutputQueue("xout_depth", maxSize=1, blocking=False)
            elif enable_obstacle_dist:
                self.queue_xout_spatial_data = self.device.getOutputQueue("spatialData", maxSize=1, blocking=False)
            
            # Get the first frame or timeout
            warming_time = time.time() + 5  # seconds
            while self.frame_xout is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.run()
                time.sleep(0.2)

            if self.frame_xout is None:
                raise CameraError("Unable to start OAK-D camera.")

            self.on = True
            logger.info("OAK-D camera ready.")
        except:
            self.shutdown()
            raise

    def create_color_pipeline(self):
        # Source
        camera = self.pipeline.create(dai.node.ColorCamera)
        camera.setFps(self.framerate)
        if self.rgb_resolution == "800p":
            camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        elif self.rgb_resolution == "1080p":
            camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        else:
            camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camera.setInterleaved(False)
        camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        if self.isp_scale:
            # see https://docs.google.com/spreadsheets/d/153yTstShkJqsPbkPOQjsVRmM8ZO3A6sCqm7uayGF-EE/edit#gid=0
            camera.setIspScale(self.isp_scale)
        
        if self.rgb_apply_cropping:
            # setSensorCrop sets the upper left corner (in noramlized format) of the video window extracted from the sensor stream
            camera.setSensorCrop(self.rgb_sensor_crop_x, self.rgb_sensor_crop_y) # When croping to keep only smaller video

            # setVideoSize sets the video window size, this window upper left corner corresponds to the previously set upper left corner
            camera.setVideoSize(self.rgb_video_size) # Desired video size = ispscale result or smaller if croping
            #  x sensor crop
            # -->|
            #    | y sensor crop
            #    |
            #    ---------------------|
            #    |                    |
            #    | video window size  |
            #    |                    |
            #    |--------------------|

        # Resize image
        camera.setPreviewKeepAspectRatio(False)
        camera.setPreviewSize(self.width, self.height) # wich means cropping if aspect ratio kept
        
        camera.setIspNumFramesPool(1)
        camera.setVideoNumFramesPool(1)
        camera.setPreviewNumFramesPool(1)

        if self.rgb_apply_manual_conf:
            camera.initialControl.setManualExposure(self.rgb_exposure_time, self.rgb_sensor_iso)
            camera.initialControl.setManualWhiteBalance(self.rgb_wb_manual)
        else:
            camera.initialControl.SceneMode(dai.CameraControl.SceneMode.SPORTS)
            # camera.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)

        # Define output and link
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("xout")
        camera.preview.link(xout.input)

    def create_mono_pipeline(self):
        # Source
        camera = self.pipeline.create(dai.node.MonoCamera)
        camera.setFps(self.framerate)
        camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # Resize image
        manip = self.pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height)
        manip.initialConfig.setResize(self.width, self.height)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        # Define output and link
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("xout")
        camera.out.link(manip.inputImage)
        manip.out.link(xout.input)

    def create_depth_pipeline(self):
        # Create mono left image source node
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        
        # Set mono left image source node properties
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setNumFramesPool(2)
        monoLeft.setFps(self.framerate)

        # Create mono right image source node
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        
        # Set mono right image source node properties
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setNumFramesPool(2)
        monoRight.setFps(self.framerate)
        
        if (self.height != 400):
            # Create resize manip left node
            stereo_manip_left = self.pipeline.create(dai.node.ImageManip)
            
            # Set resize manip left node properties
            # stereo_manip_left.initialConfig.setResize(self.width, self.height)
            stereo_manip_left.initialConfig.setResize(self.width, self.height)
            stereo_manip_left.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            stereo_manip_left.initialConfig.setInterpolation(dai.Interpolation.DEFAULT_DISPARITY_DEPTH)
            stereo_manip_left.setNumFramesPool(2)

            # Create resize manip right node
            stereo_manip_right = self.pipeline.create(dai.node.ImageManip)

            # Set resize manip left node properties
            # stereo_manip_right.initialConfig.setResize(self.width, self.height)
            stereo_manip_right.initialConfig.setResize(self.width, self.height)
            stereo_manip_right.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            stereo_manip_right.initialConfig.setInterpolation(dai.Interpolation.DEFAULT_DISPARITY_DEPTH)
            stereo_manip_right.setNumFramesPool(2)

            # Crop range
            #    - - > x 
            #    |
            #    y
            if self.depth_crop_rect:
                stereo_manip_left.initialConfig.setCropRect(*self.depth_crop_rect)
                stereo_manip_right.initialConfig.setCropRect(*self.depth_crop_rect)

        # Create stereo_depth node
        stereo = self.pipeline.create(dai.node.StereoDepth)

        # Set stereo_depth node properties
        stereo.setLeftRightCheck(self.lr_check)
        stereo.setExtendedDisparity(self.extended_disparity)
        stereo.setSubpixel(self.subpixel)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        # stereo.initialConfig.setConfidenceThreshold(self.height)
        # stereo.setInputResolution(self.width, self.height)
        stereo.setNumFramesPool(2)
        
        stereo.setAlphaScaling(self.alpha)

        stereo.setInputResolution(self.width,self.height)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Create depth output node
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("xout_depth")
        
        if (self.height != 400):
            # Linking mono node to manip image resize node
            monoLeft.out.link(stereo_manip_left.inputImage)
            monoRight.out.link(stereo_manip_right.inputImage)
            
            # Linking manip node to stereo_depth node
            stereo_manip_left.out.link(stereo.left)
            stereo_manip_right.out.link(stereo.right)
        else:
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)
            
        # Linking stereo_depth node to output
        stereo.depth.link(xout_depth.input)
        

    def create_obstacle_dist_pipeline(self):
        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        # xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

        # xoutDepth.setStreamName("depth")
        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoRight.setFps(self.framerate)
        monoLeft.setFps(self.framerate)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(self.lr_check)
        stereo.setExtendedDisparity(self.extended_disparity)
        stereo.setSubpixel(self.subpixel)
        spatialLocationCalculator.inputConfig.setWaitForMessage(False)

        for i in range(4):
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 200
            config.depthThresholds.upperThreshold = 10000
            # 30 - 40 est le mieux
            config.roi = dai.Rect(dai.Point2f(i*0.1+0.3, 0.35), dai.Point2f((i+1)*0.1+0.3, 0.43))
            spatialLocationCalculator.initialConfig.addROI(config)
            # 4 zones
            # PCLL PCCL PCCR PCRR
            # -.75 -.75 +.75 +.75
            
        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)

        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    def run(self):
        # Grab the frame from the stream 
        if self.queue_xout is not None:
            data_xout = self.queue_xout.get() # blocking
            image_data_xout = data_xout.getFrame()
            if self.depth == 3:
                image_data_xout = np.moveaxis(image_data_xout,0,-1)

            latency = (dai.Clock.now() - data_xout.getTimestamp()).total_seconds()
            self.frame_time = time.time() - latency

            if self.enable_undistort_rgb == True:
                frame_undistorted_rgb_full = cv2.remap(image_data_xout.copy(), self.map_x, self.map_y, cv2.INTER_LINEAR)
                self.frame_undistorted_rgb = frame_undistorted_rgb_full[self.pixel_crop_height:self.height,0:self.width]
            
            image_data_xout = image_data_xout[self.pixel_crop_height:self.height,0:self.width]
            self.frame_xout = image_data_xout

            if logger.isEnabledFor(logging.DEBUG):
                # Latency in miliseconds 
                self.latencies.append(latency * 1000)
                if len(self.latencies) >= self.latencies.maxlen:
                    logger.debug('Image latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}' \
                        .format(self.latencies[-1], np.average(self.latencies), np.std(self.latencies)))
                    self.latencies.clear()

        if self.queue_xout_depth is not None:
            data_xout_depth = self.queue_xout_depth.get()
            frame_xout_depth_full = data_xout_depth.getFrame()
            self.frame_xout_depth = frame_xout_depth_full[self.pixel_crop_height:self.height,0:self.width]

        if self.queue_xout_spatial_data is not None:
            xout_spatial_data = self.queue_xout_spatial_data.get().getSpatialLocations()
            self.roi_distances = []
            for depthData in xout_spatial_data:
                roi = depthData.config.roi
                coords = depthData.spatialCoordinates
                
                self.roi_distances.append(round(roi.topLeft().x,2)) 
                self.roi_distances.append(round(roi.topLeft().y,2))
                self.roi_distances.append(round(roi.bottomRight().x,2))
                self.roi_distances.append(round(roi.bottomRight().y,2))
                self.roi_distances.append(int(coords.x))
                self.roi_distances.append(int(coords.y))
                self.roi_distances.append(int(coords.z))

        # return self.frame

    def run_threaded(self):

        ret_list = [self.frame_xout]
        
        if self.enable_depth == True: ret_list.append(self.frame_xout_depth)
        if self.enable_undistort_rgb == True: ret_list.append(self.frame_undistorted_rgb)
        if self.enable_obstacle_dist == True: ret_list.append(np.array(self.roi_distances))
        
        return ret_list
        # if self.enable_depth:
        #     return self.frame_xout, self.frame_xout_depth
        # elif self.enable_obstacle_dist:
        #     return self.frame_xout, np.array(self.roi_distances)
        # else:
        #     return self.frame_xout

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while self.on:
            self.run()

    def shutdown(self):
        # Indicate that the thread should be stopped
        self.on = False
        logger.info('Stopping OAK-D camera')
        time.sleep(.5)
        if self.device is not None:
            self.device.close()
        self.device = None
        self.queue = None
        self.pipeline = None
        
