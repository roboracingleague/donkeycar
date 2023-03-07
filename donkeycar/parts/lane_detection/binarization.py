import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def threshold_HSV(frame, min_values, max_values, plot=False):
    """
    Threshold a color frame in HSV space
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    min_th_ok = np.all(hsv > min_values, axis=2)
    max_th_ok = np.all(hsv < max_values, axis=2)

    out = np.logical_and(min_th_ok, max_th_ok)

    if plot:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out


def threshold_sobel(frame, kernel_size):
    """
    Apply Sobel edge detection to an input frame, then threshold the result
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

    return sobel_mag.astype(bool)


def threshold_equalized_grayscale(frame, threshhold=250):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = frame if len(frame.shape) < 3 or frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=threshhold, maxval=1, type=cv2.THRESH_BINARY)

    return th


def binarize(image, hsv_min=None, hsv_max=None, eq_threshhold=250, sobel_kernel_size=None, close_kernel_size=5, plot=False):
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.

    :param image: input color frame
    :param plot: if True, show intermediate results
    :return: binarized frame
    """
    h, w = image.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    if hsv_min is not None and hsv_max is not None:
        hsv_mask = threshold_HSV(image, hsv_min, hsv_max, plot=False)
        binary = np.logical_or(binary, hsv_mask)

    # highlight white lines by thresholding the equalized frame
    if eq_threshhold is not None:
        eq_white_mask = threshold_equalized_grayscale(image, threshhold=eq_threshhold)
        binary = np.logical_or(binary, eq_white_mask)

    # get Sobel binary mask (thresholded gradients)
    if sobel_kernel_size is not None:
        sobel_mask = threshold_sobel(image, kernel_size=sobel_kernel_size)
        binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    if close_kernel_size is not None:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if plot:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(image)
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 0].set_facecolor('red')

        if eq_threshhold is not None:
            ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        if hsv_min is not None and hsv_max is not None:
            ax[0, 2].imshow(hsv_mask, cmap='gray')
        ax[0, 2].set_title('HSV mask')
        ax[0, 2].set_axis_off()

        if sobel_kernel_size is not None:
            ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        if close_kernel_size is not None:
            ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing if close_kernel_size is not None else binary.astype(np.uint8)


def binarize_grayscale(image, eq_threshhold=240, blur_kernel_size=5, blur_sigma_x=1, close_kernel_size=5):
    if blur_kernel_size is not None and blur_sigma_x is not None:
        image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), blur_sigma_x)
    
    binary = threshold_equalized_grayscale(image, threshhold=eq_threshhold)
    
    # apply a light morphology to "fill the gaps" in the binary image
    if close_kernel_size is not None:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return binary


if __name__ == '__main__':

    test_images = glob.glob('test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # selected threshold to highlight yellow lines
        hsv_min = [0, 70, 70]
        hsv_max = [50, 255, 255]

        binarize(img, hsv_min, hsv_max, sobel_kernel_size=9, plot=True)
