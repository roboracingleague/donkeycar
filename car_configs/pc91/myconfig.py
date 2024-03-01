# """
# My CAR CONFIG

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here.
# The update operation will not touch this file.
# """

# import os
#
# #PATHS
# CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = os.path.join(CAR_PATH, 'data')
# MODELS_PATH = os.path.join(CAR_PATH, 'models')
#
# #VEHICLE
DRIVE_LOOP_HZ = 35      # the vehicle loop will pause if faster than this speed.
# MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.
#
# #CAMERA
# # CAMERA_TYPE = "PICAM"   # (OAK|PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
# # IMAGE_W = 160
# # IMAGE_H = 120
# # IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
# # CAMERA_FRAMERATE = DRIVE_LOOP_HZ
# # CAMERA_VFLIP = False
# # CAMERA_HFLIP = False
# # CAMERA_INDEX = 0  # used for 'WEBCAM' and 'CVCAM' when there is more than one camera connected
# # # For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
# # CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)
# # OAK_D_ISP_SCALE = None
#
# # OAK-D-LITE CAMERA SETTINGS
CAMERA_TYPE = "OAK"   # (OAK|PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
#
# # OAK-D-LITE: "1080p" for rgb
# # OAK-D-WIDE: "800p" for rgb
RGB_RESOLUTION = "1080p"
#
RGB_APPLY_CROPPING = True
RGB_SENSOR_CROP_X = 0.0
RGB_SENSOR_CROP_Y = 0.2
RGB_VIDEO_SIZE = (240,108) # (240,108)
#
# RGB_APPLY_MANUAL_CONF = False
# RGB_EXPOSURE_TIME = 2000
# RGB_SENSOR_ISO = 400
# RGB_WB_MANUAL = 2800
#
# # OAK-D-LITE: from 1920/1080 (1,8)>>240/135
# # OAK-D-WIDE: from 1280/800  (1,8)>>160/100 (3,16)>>240/150 5/32>>200/125
OAK_D_ISP_SCALE = (1,8) # (1,8)
#
# # OAK-D-LITE: color cam = 240 ISP 1/8 ou 192 ISP 1/10 ou 224 ISP 7/60
# # OAK-D-WIDE: 240 ou 200 ou 160
IMAGE_W = 240 # 240
# # OAK-D-LITE: color cam = 135 ISP 1/8 ou 108 ISP 1/10 ou 126 ISP 7/60
# # OAK-D-WIDE: 150 ou 125 ou 100
IMAGE_H = 108 # 108
#
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = DRIVE_LOOP_HZ # 35hz
#
OAK_ENABLE_DEPTH_MAP = True # enables depth map output
# OAK_DEPTH_CROP_RECT = None # (top_left_x, top_left_y, bottom_right_x, bottom_right_y) with normalized values ie in [0,1]
# OAK_OBSTACLE_DETECTION_ENABLED = False # enable roi distances output
#
# # OBSTACLE_AVOIDANCE SETTINGS
# OBSTACLE_AVOIDANCE_ENABLED = False
# OBSTACLE_AVOIDANCE_FOR_AUTOPILOT = False # True activates avoidance for autopilot, False for user (manual control)
# CLOSE_AVOIDANCE_DIST_MM = 1000
#
# # OBSTACLE_DETECTOR
# OBSTACLE_DETECTOR_ENABLED = False
# OBSTACLE_DETECTOR_NUM_LOCATIONS = 4
# OBSTACLE_DETECTOR_MODEL_PATH = "~/mycar/models/pilot_23-02-15_29.tflite"
# OBSTACLE_DETECTOR_MODEL_TYPE = "tflite_obstacle_detector"
# OBSTACLE_DETECTOR_BEHAVIOR_LIST = ['NA', 'left', 'middle', 'right']
# BEHAVIOR_LIST = ['left', 'middle', 'right']
# OBSTACLE_DETECTOR_AVOIDANCE_ENABLED = False # To free drive using behavior model
# OBSTACLE_DETECTOR_MANUAL_LANE = False
#
# #CAMERA Settings Vivatech 2022 (nano)
# #CAMERA_TYPE = "CSIC"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
# #IMAGE_W = 160
# #IMAGE_H = 120
# #IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
# #CAMERA_FRAMERATE = 60
# #CAMERA_VFLIP = False
# #CAMERA_HFLIP = False
# #CAMERA_INDEX = 0  # used for 'WEBCAM' and 'CVCAM' when there is more than one camera connected
# # For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
# #CSIC_CAM_GSTREAMER_FLIP_PARM = 2 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)
#
# # For IMAGE_LIST camera
# # PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"
#
# #9865, over rides only if needed, ie. TX2..
# PCA9685_I2C_ADDR = 0x40     #I2C address, use i2cdetect to validate this number
# PCA9685_I2C_BUSNUM = None   #None will auto detect, which is fine on the pi. But other platforms should specify the bus num.
#
# #SSD1306_128_32
# USE_SSD1306_128_32 = False    # Enable the SSD_1306 OLED Display
# SSD1306_128_32_I2C_ROTATION = 0 # 0 = text is right-side up, 1 = rotated 90 degrees clockwise, 2 = 180 degrees (flipped), 3 = 270 degrees
# SSD1306_RESOLUTION = 1 # 1 = 128x32; 2 = 128x64
#
# #
# # DRIVE_TRAIN_TYPE
# # These options specify which chasis and motor setup you are using.
# # See Actuators documentation https://docs.donkeycar.com/parts/actuators/
# # for a detailed explanation of each drive train type and it's configuration.
# # Choose one of the following and then update the related configuration section:
# #
# # "PWM_STEERING_THROTTLE" uses two PWM output pins to control a steering servo and an ESC, as in a standard RC car.
# # "MM1" Robo HAT MM1 board
# # "SERVO_HBRIDGE_2PIN" Servo for steering and HBridge motor driver in 2pin mode for motor
# # "SERVO_HBRIDGE_3PIN" Servo for steering and HBridge motor driver in 3pin mode for motor
# # "DC_STEER_THROTTLE" uses HBridge pwm to control one steering dc motor, and one drive wheel motor
# # "DC_TWO_WHEEL" uses HBridge in 2-pin mode to control two drive motors, one on the left, and one on the right.
# # "DC_TWO_WHEEL_L298N" using HBridge in 3-pin mode to control two drive motors, one of the left and one on the right.
# # "ROBOCARSHAT" using robocars hat
# # "MOCK" no drive train.  This can be used to test other features in a test rig.
# # (deprecated) "SERVO_HBRIDGE_PWM" use ServoBlaster to output pwm control from the PiZero directly to control steering,
# #                                  and HBridge for a drive motor.
# # (deprecated) "PIGPIO_PWM" uses Raspberrys internal PWM
# # (deprecated) "I2C_SERVO" uses PCA9685 servo controller to control a steering servo and an ESC, as in a standard RC car
# #
# DRIVE_TRAIN_TYPE = "ROBOCARSHAT"
#
# #
# # PWM_STEERING_THROTTLE
# #
# # Drive train for RC car with a steering servo and ESC.
# # Uses a PwmPin for steering (servo) and a second PwmPin for throttle (ESC)
# # Base PWM Frequence is presumed to be 60hz; use PWM_xxxx_SCALE to adjust pulse with for non-standard PWM frequencies
# #
# PWM_STEERING_THROTTLE = {
#     "PWM_STEERING_PIN": "PCA9685.1:40.1",   # PWM output pin for steering servo
#     "PWM_STEERING_SCALE": 1.0,              # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,         # True if hardware requires an inverted PWM pulse
#     "PWM_THROTTLE_PIN": "PCA9685.1:40.0",   # PWM output pin for ESC
#     "PWM_THROTTLE_SCALE": 1.0,              # used to compensate for PWM frequence differences from 60hz; NOT for increasing/limiting speed
#     "PWM_THROTTLE_INVERTED": False,         # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,               #pwm value for full left steering
#     "STEERING_RIGHT_PWM": 290,              #pwm value for full right steering
#     "THROTTLE_FORWARD_PWM": 500,            #pwm value for max forward throttle
#     "THROTTLE_STOPPED_PWM": 370,            #pwm value for no movement
#     "THROTTLE_REVERSE_PWM": 220,            #pwm value for max reverse throttle
# }
#
# #
# # I2C_SERVO (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_CHANNEL = 1            #(deprecated) channel on the 9685 pwm board 0-15
# STEERING_LEFT_PWM = 460         #pwm value for full left steering
# STEERING_RIGHT_PWM = 290        #pwm value for full right steering
# THROTTLE_CHANNEL = 0            #(deprecated) channel on the 9685 pwm board 0-15
# THROTTLE_FORWARD_PWM = 500      #pwm value for max forward throttle
# THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
# THROTTLE_REVERSE_PWM = 220      #pwm value for max reverse throttle
#
# #
# # PIGPIO_PWM (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_PWM_PIN = 13           #(deprecated) Pin numbering according to Broadcom numbers
# STEERING_PWM_FREQ = 50          #Frequency for PWM
# STEERING_PWM_INVERTED = False   #If PWM needs to be inverted
# THROTTLE_PWM_PIN = 18           #(deprecated) Pin numbering according to Broadcom numbers
# THROTTLE_PWM_FREQ = 50          #Frequency for PWM
# THROTTLE_PWM_INVERTED = False   #If PWM needs to be inverted
#
# #
# # SERVO_HBRIDGE_2PIN
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_2PIN = {
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # provides forward duty cycle to motor
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # provides reverse duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
#
# #
# # SERVO_HBRIDGE_3PIN
# # - configures a steering servo and an HBridge in 3pin mode (2 ttl pins, 1 pwm pin)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by three pins,
# #   one ttl output for forward, one ttl output
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_3PIN = {
#     "FWD_PIN": "RPI_GPIO.BOARD.18",   # ttl pin, high enables motor forward
#     "BWD_PIN": "RPI_GPIO.BOARD.16",   # ttl pin, high enables motor reverse
#     "DUTY_PIN": "RPI_GPIO.BOARD.35",  # provides duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
#
# #
# # DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" (deprecated in favor of SERVO_HBRIDGE_2PIN)
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Uses ServoBlaster library, which is NOT installed by default, so
# #   you will need to install it to make this work.
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - the pwm pins produce a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# HBRIDGE_PIN_FWD = 18       # provides forward duty cycle to motor
# HBRIDGE_PIN_BWD = 16       # provides reverse duty cycle to motor
# STEERING_CHANNEL = 0       # PCA 9685 channel for steering control
# STEERING_LEFT_PWM = 460    # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
# STEERING_RIGHT_PWM = 290   # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
#
# #VESC controller, primarily need to change VESC_SERIAL_PORT  and VESC_MAX_SPEED_PERCENT
# VESC_MAX_SPEED_PERCENT =.2  # Max speed as a percent of the actual speed
# VESC_SERIAL_PORT= "/dev/ttyACM0" # Serial device to use for communication. Can check with ls /dev/tty*
# VESC_HAS_SENSOR= True # Whether or not the bldc motor is using a hall effect sensor
# VESC_START_HEARTBEAT= True # Whether or not to automatically start the heartbeat thread that will keep commands alive.
# VESC_BAUDRATE= 115200 # baudrate for the serial communication. Shouldn't need to change this.
# VESC_TIMEOUT= 0.05 # timeout for the serial communication
# VESC_STEERING_SCALE= 0.5 # VESC accepts steering inputs from 0 to 1. Joystick is usually -1 to 1. This changes it to -0.5 to 0.5
# VESC_STEERING_OFFSET = 0.5 # VESC accepts steering inputs from 0 to 1. Coupled with above change we move Joystick to 0 to 1
#
# #
# # DC_STEER_THROTTLE with one motor as steering, one as drive
# # - uses L298N type motor controller in two pin wiring
# #   scheme utilizing two pwm pins per motor; one for
# #   forward(or right) and one for reverse (or left)
# #
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_STEER_THROTTLE = {
#     "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # pwm pin produces duty cycle for steering left
#     "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for steering right
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # pwm pin produces duty cycle for forward drive
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # pwm pin produces duty cycle for reverse drive
# }
#
# #
# # DC_TWO_WHEEL pin configuration
# # - configures L298N_HBridge_2pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - each pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL = {
#     "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # pwm pin produces duty cycle for left wheel forward
#     "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for left wheel reverse
#     "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # pwm pin produces duty cycle for right wheel forward
#     "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # pwm pin produces duty cycle for right wheel reverse
# }
#
# #
# # DC_TWO_WHEEL_L298N pin configuration
# # - configures L298N_HBridge_3pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by three pins,
# #   one ttl output for forward, one ttl output
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL_L298N = {
#     "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # TTL output pin enables left wheel forward
#     "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # TTL output pin enables left wheel reverse
#     "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # PWM pin generates duty cycle for left motor speed
#
#     "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # TTL output pin enables right wheel forward
#     "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # TTL output pin enables right wheel reverse
#     "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # PWM pin generates duty cycle for right wheel speed
# }
#
# #ODOMETRY
# HAVE_ODOM = False                   # Do you have an odometer/encoder
# ENCODER_TYPE = 'GPIO'            # What kind of encoder? GPIO|Arduino|Astar|ROBOCARSHAT
# MM_PER_TICK = 12.7625               # How much travel with a single tick, in mm. Roll you car a meter and divide total ticks measured by 1,000
# ODOM_PIN = 13                        # if using GPIO, which GPIO board mode pin to use as input
# ODOM_DEBUG = False                  # Write out values on vel and distance as it runs
#
# # #LIDAR
# USE_LIDAR = False
# LIDAR_TYPE = 'RP' #(RP|YD)
# LIDAR_LOWER_LIMIT = 90 # angles that will be recorded. Use this to block out obstructed areas on your car, or looking backwards. Note that for the RP A1M8 Lidar, "0" is in the direction of the motor
# LIDAR_UPPER_LIMIT = 270
#
# # TFMINI
# HAVE_TFMINI = False
# TFMINI_SERIAL_PORT = "/dev/serial0" # tfmini serial port, can be wired up or use usb/serial adapter
#
# #TRAINING
# # The default AI framework to use. Choose from (tensorflow|pytorch)
# DEFAULT_AI_FRAMEWORK = 'tensorflow'
#
# # The DEFAULT_MODEL_TYPE will choose which model will be created at training
# # time. This chooses between different neural network designs. You can
# # override this setting by passing the command line parameter --type to the
# # python manage.py train and drive commands.
# # tensorflow models: (linear|categorical|tflite_linear|tensorrt_linear)
# # pytorch models: (resnet18)
# DEFAULT_MODEL_TYPE = 'linear'
# BATCH_SIZE = 128                #how many records to use when doing one pass of gradient decent. Use a smaller number if your gpu is running out of memory.
# TRAIN_TEST_SPLIT = 0.8          #what percent of records to use for training. the remaining used for validation.
# MAX_EPOCHS = 100                #how many times to visit all records of your data
# SHOW_PLOT = True                #would you like to see a pop up display of final loss?
# VERBOSE_TRAIN = True            #would you like to see a progress bar with text during training?
# USE_EARLY_STOP = True           #would you like to stop the training if we see it's not improving fit?
# EARLY_STOP_PATIENCE = 5         #how many epochs to wait before no improvement
# MIN_DELTA = .0005               #early stop will want this much loss change before calling it improved.
# PRINT_MODEL_SUMMARY = True      #print layers and weights to stdout
# OPTIMIZER = None                #adam, sgd, rmsprop, etc.. None accepts default
# LEARNING_RATE = 0.001           #only used when OPTIMIZER specified
# LEARNING_RATE_DECAY = 0.0       #only used when OPTIMIZER specified
# SEND_BEST_MODEL_TO_PI = False   #change to true to automatically send best model during training
# CREATE_TF_LITE = False           # automatically create tflite model in training
# CREATE_TENSOR_RT = False        # automatically create tensorrt model in training
# CREATE_ONNX_MODEL = True       # automatically create onnx model in training
#
# PRUNE_CNN = False               #This will remove weights from your model. The primary goal is to increase performance.
# PRUNE_PERCENT_TARGET = 75       # The desired percentage of pruning.
# PRUNE_PERCENT_PER_ITERATION = 20 # Percenge of pruning that is perform per iteration.
# PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # The max amout of validation loss that is permitted during pruning.
# PRUNE_EVAL_PERCENT_OF_DATASET = .05  # percent of dataset used to perform evaluation of model.
#
# # Augmentations and Transformations
# AUGMENTATIONS = []
# TRANSFORMATIONS = []
# # Settings for brightness and blur, use 'MULTIPLY' and/or 'BLUR' in
# # AUGMENTATIONS
# AUG_MULTIPLY_RANGE = (0.5, 3.0)
# AUG_WB_RANGE = (1100,10000)
# AUG_BLUR_RANGE = (0.0, 3.0)
# # Region of interest cropping, requires 'CROP' in TRANSFORMATIONS to be set
# # If these crops values are too large, they will cause the stride values to
# # become negative and the model with not be valid.
# ROI_CROP_TOP = 45               # the number of rows of pixels to ignore on the top of the image
# ROI_CROP_BOTTOM = 0             # the number of rows of pixels to ignore on the bottom of the image
# ROI_CROP_RIGHT = 0              # the number of rows of pixels to ignore on the right of the image
# ROI_CROP_LEFT = 0               # the number of rows of pixels to ignore on the left of the image
# # For trapezoidal see explanation in augmentations.py. Requires 'TRAPEZE' in
# # TRANSFORMATIONS to be set
# ROI_TRAPEZE_LL = 0
# ROI_TRAPEZE_LR = 160
# ROI_TRAPEZE_UL = 20
# ROI_TRAPEZE_UR = 140
# ROI_TRAPEZE_MIN_Y = 60
# ROI_TRAPEZE_MAX_Y = 120
#
# #Model transfer options
# #When copying weights during a model transfer operation, should we freeze a certain number of layers
# #to the incoming weights and not allow them to change during training?
# FREEZE_LAYERS = False               #default False will allow all layers to be modified by training
# NUM_LAST_LAYERS_TO_TRAIN = 7        #when freezing layers, how many layers from the last should be allowed to train?
#
# #WEB CONTROL
# WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # which port to listen on when making a web controller
# WEB_INIT_MODE = "user"              # which control mode to start in. one of user|local_angle|local. Setting local will start in ai mode.
#
# #JOYSTICK
# USE_JOYSTICK_AS_DEFAULT = False      #when starting the manage.py, when True, will not require a --js option to use the joystick
# JOYSTICK_MAX_THROTTLE = 0.5         #this scalar is multiplied with the -1 to 1 throttle value to limit the maximum throttle. This can help if you drop the controller or just don't need the full speed available.
# JOYSTICK_STEERING_SCALE = 1.0       #some people want a steering that is less sensitve. This scalar is multiplied with the steering -1 to 1. It can be negative to reverse dir.
# AUTO_RECORD_ON_THROTTLE = True      #if true, we will record whenever throttle is not zero. if false, you must manually toggle recording with some other trigger. Usually circle button on joystick.
# CONTROLLER_TYPE = 'mock'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the my_joystick.py controller written by the `donkey createjs` command
# USE_NETWORKED_JS = False            #should we listen for remote joystick control over the network?
# NETWORK_JS_SERVER_IP = None         #when listening for network joystick control, which ip is serving this information
# JOYSTICK_DEADZONE = 0.01            # when non zero, this is the smallest throttle before recording triggered.
# JOYSTICK_THROTTLE_DIR = -1.0         # use -1.0 to flip forward/backward, use 1.0 to use joystick's natural forward/backward
# USE_FPV = False                     # send camera data to FPV webserver
# JOYSTICK_DEVICE_FILE = "/dev/input/js0" # this is the unix file use to access the joystick.
#
# #For the categorical model, this limits the upper bound of the learned throttle
# #it's very IMPORTANT that this value is matched from the training PC config.py and the robot.py
# #and ideally wouldn't change once set.
# MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8
#
# #RNN or 3D
# SEQUENCE_LENGTH = 3             #some models use a number of images over time. This controls how many.
#
# #IMU
# HAVE_IMU = False                #when true, this add a Mpu6050 part and records the data. Can be used with a
# IMU_SENSOR = 'mpu6050'          # (mpu6050|mpu9250)
# IMU_ADDRESS = 0x68              # if AD0 pin is pulled high them address is 0x69, otherwise it is 0x68
# IMU_DLP_CONFIG = 0              # Digital Lowpass Filter setting (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)
#
# #SOMBRERO
# HAVE_SOMBRERO = False           #set to true when using the sombrero hat from the Donkeycar store. This will enable pwm on the hat.
#
# #PIGPIO RC control
# STEERING_RC_GPIO = 26
# THROTTLE_RC_GPIO = 20
# DATA_WIPER_RC_GPIO = 19
# PIGPIO_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
# PIGPIO_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
# PIGPIO_STOPPED_PWM = 1500
# PIGPIO_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
# PIGPIO_SHOW_STEERING_VALUE = False
# PIGPIO_INVERT = False
# PIGPIO_JITTER = 0.025   # threshold below which no signal is reported
#
#
#
# #ROBOHAT MM1
# MM1_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
# MM1_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
# MM1_STOPPED_PWM = 1500
# MM1_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
# MM1_SHOW_STEERING_VALUE = False
# # Serial port
# # -- Default Pi: '/dev/ttyS0'
# # -- Jetson Nano: '/dev/ttyTHS1'
# # -- Google coral: '/dev/ttymxc0'
# # -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# # -- MacOS/Linux:please use 'ls /dev/tty.*' to find the correct serial port for mm1
# #  eg.'/dev/tty.usbmodemXXXXXX' and replace the port accordingly
# MM1_SERIAL_PORT = '/dev/ttyS0'  # Serial Port for reading and sending MM1 data.
#
# #ROBOCARSHAT
USE_ROBOCARSHAT_AS_CONTROLLER = True
ROBOCARSHAT_SERIAL_PORT = '/dev/ttyTHS1'
ROBOCARSHAT_SERIAL_SPEED = 250000
#
# USE_ROBOCARSHAT_BATTERY_MONITOR = False
# ROBOCARSHAT_LIPO_CELLS = 2
#
# # Following values must be aligned with values in Hat !
# ROBOCARSHAT_PWM_OUT_THROTTLE_MIN    =   1000
# ROBOCARSHAT_PWM_OUT_THROTTLE_IDLE   =   1500
# ROBOCARSHAT_PWM_OUT_THROTTLE_MAX    =   2000
ROBOCARSHAT_PWM_OUT_STEERING_MIN    =   1070
ROBOCARSHAT_PWM_OUT_STEERING_IDLE   =   1450 # 1460
ROBOCARSHAT_PWM_OUT_STEERING_MAX    =   1850
# ROBOCARSHAT_PWM_OUT_STEERING_INVERT    =   False
#
# # Folowing values can be ajusted to normalized btzeen -1 and 1.
# # # If  ROBOCARSHAT_USE_AUTOCALIBRATION is used, IDLE values are automatically identified by the Hat
ROBOCARSHAT_PWM_IN_THROTTLE_MIN    =   1000
# ROBOCARSHAT_PWM_IN_THROTTLE_IDLE   =   1500
ROBOCARSHAT_PWM_IN_THROTTLE_MAX    =   1980
ROBOCARSHAT_PWM_IN_STEERING_MIN    =   1000
# ROBOCARSHAT_PWM_IN_STEERING_IDLE   =   1500
ROBOCARSHAT_PWM_IN_STEERING_MAX    =   1970
# ROBOCARSHAT_PWM_IN_AUX_MIN    =   1000
# ROBOCARSHAT_PWM_IN_AUX_IDLE   =   1500
# ROBOCARSHAT_PWM_IN_AUX_MAX    =   2000
#
# #ODOM Sensor max value (max matching lowest speed)
# ROBOCARSHAT_ODOM_IN_MAX = 20000
ROBOCARSHAT_PILOT_MODE = 'local_angle' # Which autonomous mode is triggered by Hat : local_angle or local
ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE = 0.15 # For pilot_angle autonomous mode (aka constant throttle), this is the default throttle to apply
# # ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE = None # if set to None, throttle is the one provided by remote control
# ROBOCARSHAT_BRAKE_ON_IDLE_THROTTLE = -0.2
#
# THROTTLE_BRAKE_REV_FILTER = False # ESC is configured in Fw/Rv mode (no braking)
#
# #ROBOCARSHAT_CH3_FEATURE and ROBOCARSHAT_CH4_FEATURE controls the feature attached to radio ch3 and ch4
# # 'none' means aux ch is not used
# # 'record/pilot' means aux ch is used to control either data recording (lower position), either to enable pilot mode (upper position)
# # 'record' means aux ch is used to control data recording
# # 'pilot' means aux ch is used to control pilot mode
# # 'throttle_exploration' means special mode where aux ch is used to increment/decrement a fixed throttle value in user mode
# # 'steering_exploration' means special mode where aux ch is used to increment/decrement a fixed steering value in user mode
# # 'output_steering_trim' means special mode where aux ch is used to increment/decrement a steering idle output for triming direction in user mode, resulting value must be reported in  ROBOCARSHAT_PWM_OUT_STEERING_IDLE
# # 'output_steering_exp' means special mode where aux ch is used to increment/decrement a fixed steering output to calibrate direction in user mode, resulting values must be reported in  ROBOCARSHAT_PWM_IN_STEERING_MIN and ROBOCARSHAT_PWM_IN_STEERING_MAX
# # 'throttle_scalar_exp' means special mode where aux ch is used to explore throttle scalar to apply on throttle when autopilot is engaged
# # 'adaptative_steering_scalar_exp' means special mode where aux ch is used to explore adaptative steering scalar to apply on steering when autopilot is engaged
# # 'drive_by_lane' means special mode where aux ch is used to set the lane the car must use (left, middle, right) (behavior model)
# ROBOCARSHAT_CH3_FEATURE = 'record/pilot'
# ROBOCARSHAT_CH4_FEATURE = 'none'
# ROBOCARSHAT_EXPLORE_THROTTLE_SCALER_USING_THROTTLE_CONTROL = False # specific mode when in pilot, throttle control control throttle scaler
# ROBOCARSHAT_EXPLORE_THROTTLE_SCALER_USING_THROTTLE_CONTROL_INC = 0.01
# ROBOCARS_DRIVE_BY_LANE = False # if true, when in pilot mode, steering is used to feed model as lane to follow
#
# ROBOCARSHAT_THROTTLE_EXP_INC = 0.05
# ROBOCARSHAT_STEERING_EXP_INC = 0.05
# ROBOCARSHAT_OUTPUT_STEERING_TRIM_INC = 10
#
# #ROBOCARSHAT_STEERING_FIX used for steering calibration, enforce a fixed steering value (betzeen -1.0 and 1.0). None means no enforcment
# ROBOCARSHAT_STEERING_FIX = None
#
# # For 'throttle_scalar_exp' feature, specify maximum scalar to apply when aux ch is set to maximum position.
# AUX_FEATURE_THROTTLE_SCALAR_EXP_MAX_VALUE = 1.0 # to report to ROBOCARS_THROTTLE_SCALER or ROBOCARS_THROTTLE_SCALER_ON_SL depending on use case tested
# # For 'adaptative_steering_scalar_exp' feature, specify maximum scalar to apply when aux ch is set to maximum position.
# AUX_FEATURE_ADAPTATIVE_STEERING_SCALAR_EXP_MAX_VALUE = 3.0 # to report on ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER
#
# # ROBOCARSHAT_THROTTLE_DISCRET used to control throttle with discretes values (only in user mode, first value must be 0.0)
# # ROBOCARSHAT_THROTTLE_DISCRET has precedence over ROBOCARSHAT_THROTTLE_FLANGER
# #Example : ROBOCARSHAT_THROTTLE_DISCRET = [0.0, 0.1, 0.2], if not used, set to None
# ROBOCARSHAT_THROTTLE_DISCRET = None
#
# # ROBOCARSHAT_THROTTLE_FLANGER used to control throttle flange (map outputs to given range), ONLY in USER MODE
# # giving a range between -1 and 1, like [-0.1, 0.1]
# #Example : ROBOCARSHAT_THROTTLE_FLANGER = [-0.1, 0.1], if not used, set to None
# #ROBOCARSHAT_THROTTLE_FLANGER = None
ROBOCARSHAT_THROTTLE_FLANGER = [-0.2, 0.25]
#
# # ROBOCARSHAT_USE_AUTOCALIBRATION used to rely on idle coming from autocalibation done by hat
# ROBOCARSHAT_USE_AUTOCALIBRATION = True
#
ROBOCARSHAT_CONTROL_LED = False
ROBOCARSHAT_CONTROL_LED_DEDICATED_TTY = "/dev/ttyACM0"
ROBOCARSHAT_LED_MODEL = 'Duo' #Alpine, Duo,
ROBOCARSHAT_CONTROL_LED_PILOT_ANIM = 3 # 1 = sparkle, 2 = strobe, 3=HAL
#
# # straight line detection model
# ROBOCARS_SL_DETECTION_MODEL=False
# ROBOCARS_NUM_SCEN_CAT=2 #straight line or turn
#
# #
# USE_ROBOCARSHAT_POWERTRAIN_CONTROLLER  = False
# ROBOCARS_THROTTLE_SCALER = 0.0 # extra scalar to apply by default
# ROBOCARS_THROTTLE_SCALER_ON_SL = 0.0 # extra scalar to apply to throttle when straight line is detected
# ROBOCARS_THROTTLE_ON_SL_BRAKE_SPEED = 0.0 # For pilot_angle autonomous mode, throttle for turn entry (brake)
# ROBOCARS_THROTTLE_ON_SL_BRAKE_DURATION = 5
# ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER = 0.0 # scalar to apply to throttle when straight line is detected.THis scalar is proportionnaly applied dependeing on throttle
# ROBOCARS_CTRL_ADAPTATIVE_STEERING_IN_TURN_ONLY = False
# ROBOCARS_THROTTLE_SCALER_ON_SL_FILTER_SIZE=6
# ROBOCARS_SL_FILTER_TRESH_HIGH=4
# ROBOCARS_SL_FILTER_TRESH_LOW=2
#
# ROBOCARS_PROFILES = None # Powetrain Profile, define fix throttle, steering scalar and LED Anim to use in pilot mode, accordingly to profile selector (remote control feature)
# #ROBOCARS_PROFILES = [[0.0, 0.0,3],[0.2,1.1,3],[0.35,1.85,1]]
#
#
# #LOGGING
# HAVE_CONSOLE_LOGGING = True
# LOGGING_LEVEL = 'INFO'          # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
#
# #TELEMETRY
# HAVE_MQTT_TELEMETRY = False
# TELEMETRY_DONKEY_NAME = 'my_robot1234'
# TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
# TELEMETRY_MQTT_JSON_ENABLE = False
# TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
# TELEMETRY_MQTT_BROKER_PORT = 1883
# TELEMETRY_PUBLISH_PERIOD = 1
# TELEMETRY_LOGGING_ENABLE = True
# TELEMETRY_LOGGING_LEVEL = 'INFO' # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# TELEMETRY_LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
# TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
# TELEMETRY_DEFAULT_TYPES = 'float,float'
#
# # PERF MONITOR
# HAVE_PERFMON = False
#
# #RECORD OPTIONS
# RECORD_DURING_AI = False        #normally we do not record during ai mode. Set this to true to get image and steering records for your Ai. Be careful not to use them to train.
# AUTO_CREATE_NEW_TUB = False     #create a new tub (tub_YY_MM_DD) directory when recording or append records to data directory directly
#
# #LED
# HAVE_RGB_LED = False            #do you have an RGB LED like https://www.amazon.com/dp/B07BNRZWNF
# LED_INVERT = False              #COMMON ANODE? Some RGB LED use common anode. like https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B
#
# #LED board pin number for pwm outputs
# #These are physical pinouts. See: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
# LED_PIN_R = 12
# LED_PIN_G = 10
# LED_PIN_B = 16
#
# #LED status color, 0-100
# LED_R = 0
# LED_G = 0
# LED_B = 1
#
# #LED Color for record count indicator
# REC_COUNT_ALERT = 1000          #how many records before blinking alert
# REC_COUNT_ALERT_CYC = 15        #how many cycles of 1/20 of a second to blink per REC_COUNT_ALERT records
# REC_COUNT_ALERT_BLINK_RATE = 0.4 #how fast to blink the led in seconds on/off
#
# #first number is record count, second tuple is color ( r, g, b) (0-100)
# #when record count exceeds that number, the color will be used
# RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
#             (3000, (5, 5, 5)),
#             (5000, (5, 2, 0)),
#             (10000, (0, 5, 0)),
#             (15000, (0, 5, 5)),
#             (20000, (0, 0, 5)), ]
#
#
# #LED status color, 0-100, for model reloaded alert
# MODEL_RELOADED_LED_R = 100
# MODEL_RELOADED_LED_G = 0
# MODEL_RELOADED_LED_B = 0
#
#
# #BEHAVIORS
# #When training the Behavioral Neural Network model, make a list of the behaviors,
# #Set the TRAIN_BEHAVIORS = True, and use the BEHAVIOR_LED_COLORS to give each behavior a color
# TRAIN_BEHAVIORS = False
# # BEHAVIOR_LIST = ['left', 'middle', "right"]
# # BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]  #RGB tuples 0-100 per chanel
#
# #Localizer
# #The localizer is a neural network that can learn to predict its location on the track.
# #This is an experimental feature that needs more developement. But it can currently be used
# #to predict the segement of the course, where the course is divided into NUM_LOCATIONS segments.
# TRAIN_LOCALIZER = False
# # NUM_LOCATIONS = 10
# # BUTTON_PRESS_NEW_TUB = False #when enabled, makes it easier to divide our data into one tub per track length if we make a new tub on each X button press.
#
# #DonkeyGym
# #Only on Ubuntu linux, you can use the simulator as a virtual donkey and
# #issue the same python manage.py drive command as usual, but have them control a virtual car.
# #This enables that, and sets the path to the simualator and the environment.
# #You will want to download the simulator binary from: https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip
# #then extract that and modify DONKEY_SIM_PATH.
# DONKEY_GYM = False
# DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
# DONKEY_GYM_ENV_NAME = "donkey-roboracingleague-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
# GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
# GYM_CONF["racer_name"] = "Your Name"
# GYM_CONF["country"] = "Place"
# GYM_CONF["bio"] = "I race robots."
#
# SIM_HOST = "127.0.0.1"              # when racing on virtual-race-league use host "trainmydonkey.com"
# SIM_ARTIFICIAL_LATENCY = 0          # this is the millisecond latency in controls. Can use useful in emulating the delay when useing a remote server. values of 100 to 400 probably reasonable.
#
# #Donkey Webot
# DONKEY_WEBOT = False
# DONKEY_WEBOT_WORLD_NAME="vivatech_2023"
# WEBOT_CONF={}
#
# # Save info from Simulator (pln)
# SIM_RECORD_LOCATION = False
# SIM_RECORD_GYROACCEL= False
# SIM_RECORD_VELOCITY = False
# SIM_RECORD_LIDAR = False
#
# #publish camera over network
# #This is used to create a tcp service to publish the camera feed
# PUB_CAMERA_IMAGES = False
#
# #When racing, to give the ai a boost, configure these values.
# AI_LAUNCH_DURATION = 0.0            # the ai will output throttle for this many seconds
# AI_LAUNCH_THROTTLE = 0.0            # the ai will output this throttle value
# AI_LAUNCH_ENABLE_BUTTON = 'R2'      # this keypress will enable this boost. It must be enabled before each use to prevent accidental trigger.
# AI_LAUNCH_KEEP_ENABLED = False      # when False ( default) you will need to hit the AI_LAUNCH_ENABLE_BUTTON for each use. This is safest. When this True, is active on each trip into "local" ai mode.
#
# #Scale the output of the throttle of the ai pilot for all model types.
# AI_THROTTLE_MULT = 1.0              # this multiplier will scale every throttle value for all output from NN models
#
# #Path following
# PATH_FILENAME = "donkey_path.pkl"   # the path will be saved to this filename
# PATH_SCALE = 5.0                    # the path display will be scaled by this factor in the web page
# PATH_OFFSET = (0, 0)                # 255, 255 is the center of the map. This offset controls where the origin is displayed.
# PATH_MIN_DIST = 0.3                 # after travelling this distance (m), save a path point
# PID_P = -10.0                       # proportional mult for PID path follower
# PID_I = 0.000                       # integral mult for PID path follower
# PID_D = -0.2                        # differential mult for PID path follower
# PID_THROTTLE = 0.2                  # constant throttle value during path following
# SAVE_PATH_BTN = "cross"             # joystick button to save path
# RESET_ORIGIN_BTN = "triangle"       # joystick button to press to move car back to origin
#
# # Intel Realsense D435 and D435i depth sensing camera
# REALSENSE_D435_RGB = True       # True to capture RGB image
# REALSENSE_D435_DEPTH = True     # True to capture depth as image array
# REALSENSE_D435_IMU = False      # True to capture IMU data (D435i only)
# REALSENSE_D435_ID = None        # serial number of camera or None if you only have one camera (it will autodetect)
#
# # Stop Sign Detector
# STOP_SIGN_DETECTOR = False
# STOP_SIGN_MIN_SCORE = 0.2
# STOP_SIGN_SHOW_BOUNDING_BOX = True
# STOP_SIGN_MAX_REVERSE_COUNT = 10    # How many times should the car reverse when detected a stop sign, set to 0 to disable reversing
# STOP_SIGN_REVERSE_THROTTLE = -0.5     # Throttle during reversing when detected a stop sign
#
# # FPS counter
# SHOW_FPS = False
# FPS_DEBUG_INTERVAL = 10    # the interval in seconds for printing the frequency info into the shell

# PROFILES
ROBOCARS_CONFIG_PROFILE_IARACING_SPRINT_FIXSPEED = {
    'ROBOCARS_PROFILE_NAME' : 'IARACING-SPRINT-FIXSPEED',
    'ROBOCARSHAT_PILOT_MODE' : 'local_angle',
    'ROBOCARS_THROTTLE_DUAL_THROTTLE_MODEL' : 'False',
    'ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE' : 0.18,
    'ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER': 0.0,
    'OBSTACLE_DETECTOR_AVOIDANCE_ENABLED' : False,
    'OBSTACLE_DETECTOR_MANUAL_LANE' : True
}

ROBOCARS_CONFIG_PROFILE_IARACING_SPRINT = {
    'ROBOCARS_PROFILE_NAME' : 'IARACING-SPRINT',
    'ROBOCARSHAT_PILOT_MODE' : 'local',
    'ROBOCARS_THROTTLE_DUAL_THROTTLE_MODEL' : 'True',
    'ROBOCARS_THROTTLE_LOW' : 0.15,
    'ROBOCARS_THROTTLE_HIGH' : 0.20,
    'ROBOCARS_THROTTLE_ON_SL_BRAKE_SPEED' : -0.00,
    'ROBOCARS_THROTTLE_SCALER_ON_SL_FILTER_SIZE' : 10,
    'ROBOCARS_SL_FILTER_TRESH_HIGH':3,
    'ROBOCARS_SL_FILTER_TRESH_LOW':1,
    'ROBOCARS_THROTTLE_ON_SL_BRAKE_DURATION' : 15,
    'ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE' : 0.18,
    'ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER': 0.0,
    'OBSTACLE_DETECTOR_AVOIDANCE_ENABLED' : False,
    'OBSTACLE_DETECTOR_MANUAL_LANE' : True
}

ROBOCARS_CONFIG_PROFILE_IARACING_SLALOM = {
    'ROBOCARS_PROFILE_NAME' : 'IARACING-SLALOM',
    'ROBOCARSHAT_PILOT_MODE' : 'local_angle',
    'ROBOCARS_THROTTLE_DUAL_THROTTLE_MODEL' : 'False',
    'ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE' : 0.12,
    'ROBOCARS_CTRL_STEERING_OFFSET' : 0.0,
    'ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER': 0.1,
    'OBSTACLE_DETECTOR_AVOIDANCE_ENABLED' : True,
    'OBSTACLE_DETECTOR_MANUAL_LANE' : False
}

ROBOCARS_CONFIG_PROFILE_COPILOT_SPRINT = {
    'ROBOCARS_PROFILE_NAME' : 'COPILOT-SPRINT',
    'ROBOCARSHAT_PILOT_MODE' : 'local_angle',
    'ROBOCARS_THROTTLE_DUAL_THROTTLE_MODEL' : 'False',
    'ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER': 0.0,
    'ROBOCARS_CTRL_STEERING_OFFSET' : -0.25,
    'ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE' : None,
    'OBSTACLE_DETECTOR_AVOIDANCE_ENABLED' : False,
    'OBSTACLE_DETECTOR_MANUAL_LANE' : True
}

ROBOCARS_CONFIG_PROFILE_COPILOT_SLALOM = {
    'ROBOCARS_PROFILE_NAME' : 'COPILOT-SLALOM',
    'ROBOCARS_CTRL_STEERING_OFFSET' : 0.0,
    'ROBOCARSHAT_PILOT_MODE' : 'local_angle',
    'ROBOCARS_CTRL_ADAPTATIVE_STEERING_SCALER': 0.1,
    'ROBOCARSHAT_LOCAL_ANGLE_FIX_THROTTLE' : None,
    'OBSTACLE_DETECTOR_AVOIDANCE_ENABLED' : True,
    'OBSTACLE_DETECTOR_MANUAL_LANE' : False
}

#ROBOCARS_CONFIG_PROFILE = ROBOCARS_CONFIG_PROFILE_IARACING_SPRINT_FIXSPEED
#ROBOCARS_CONFIG_PROFILE = ROBOCARS_CONFIG_PROFILE_IARACING_SPRINT
#ROBOCARS_CONFIG_PROFILE = ROBOCARS_CONFIG_PROFILE_IARACING_SLALOM
ROBOCARS_CONFIG_PROFILE = ROBOCARS_CONFIG_PROFILE_COPILOT_SPRINT
#ROBOCARS_CONFIG_PROFILE = ROBOCARS_CONFIG_PROFILE_COPILOT_SLALOM