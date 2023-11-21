import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#from donkeycar.parts.lane_detection.calibration import calibrate_camera, undistort
from donkeycar.parts.lane_detection.binarization import binarize


class Birdeye:
    def __init__(self, height, width, vanishing_point=0.35, crop_top=0.44, crop_corner=0.7, height_ratio=1.0, border_value=0):
        self.height = height
        self.width = width
        self.out_height = int(height * height_ratio)
        self.border_value = border_value
        p = vanishing_point # position of vanishing point, from top, in percent of height (horizontal position is set to width / 2)
        c = crop_top # crop top in percent ; MUST be > p
        e = crop_corner # how much black in bottom corners

        assert 0.0 <= c <= 1.0, "crop_top must be between 0 and 1"
        assert p < c , "vanishing_point_height must be between lower than crop_top"
        assert 0.0 <= e <= 1.0, "crop_corner must be between 0 and 1"

        top = int(c * height)
        # Thales calculations with vanishing point
        bottom_left_x = - e * (1 - c) / (c - p) * width / 2
        top_left_x = (1 - (c - p) / (1 - p) * (1 + e * (1 - c) / (c - p))) * width / 2

        self.source_points = np.float32([[top_left_x, top], [width - top_left_x, top],
                                         [bottom_left_x, height], [width - bottom_left_x, height]])
        
        self.dest_points = np.float32([[0, 0], [width, 0],
                                       [0, self.out_height], [width, self.out_height]])

        # forward and backward transformation matrices
        self.M = cv2.getPerspectiveTransform(self.source_points, self.dest_points)
        self.Minv = cv2.getPerspectiveTransform(self.source_points, self.dest_points)


    def apply(self, image, border_value=None, plot=False):
        """
        Apply perspective transform to input frame to get the bird's eye view.
        :param image: input frame
        :param verbose: if True, show the transformation result
        :return: warped image
        """
        border_value = border_value if border_value is not None else self.border_value
        warped = cv2.warpPerspective(image, self.M, (self.width, self.out_height), flags=cv2.INTER_LINEAR, borderValue=border_value)

        if plot:
            f, ax = plt.subplots(1, 2)
            f.set_facecolor('white')
            ax[0].set_title('Before perspective transform')
            ax[0].imshow(image, cmap='gray')
            for point in self.source_points[:]:
                ax[0].plot(*point, '.')
            ax[1].set_title('After perspective transform')
            ax[1].imshow(warped, cmap='gray')
            for point in self.dest_points:
                ax[1].plot(*point, '.')
            for axis in ax:
                axis.set_axis_off()
            plt.show()

        return warped


if __name__ == '__main__':

    #ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img_undistorted = undistort(img, mtx, dist, plot=False)
        img_undistorted = img

        img_binary = binarize(img_undistorted, plot=False)

        img_birdeye = Birdeye(*img_binary.shape[:2]).apply(img_undistorted, plot=True)

