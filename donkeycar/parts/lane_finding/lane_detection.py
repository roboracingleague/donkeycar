import time
import math
import glob
import logging
from queue import SimpleQueue
import matplotlib.colors as mcolors
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import depthai as dai
from depthai_sdk import toTensorResult
from donkeycar.parts.lane_finding.perspective import Birdeye

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Template:
    '''
    Template
    '''
    def __init__(self):
        self.result = None
        self.on = True
        logger.info("Template ready")

    def run(self):

        if(logger.isEnabledFor(logging.DEBUG)):
            logger.debug('{:>6,.1f}'.format(0))

        return None

    def run_threaded(self):
        return self.result

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.result = self.run()
    
    def shutdown(self):
        self.on = False
        logger.info('Stopping Template')



def vis_lanes(image, lane_lines, multi_color=False):
    image_out = image
    colors = [tuple(255 * e for e in mcolors.to_rgb(c)) for n,c in mcolors.TABLEAU_COLORS.items()]
    count = 0
    if lane_lines is not None:
        for line in lane_lines:
            x1, y1, x2, y2 = line.astype(int)
            color = colors[count%len(colors)] if multi_color else (255, 0, 255)

            image_out = cv2.line(image_out.astype(np.uint8), (x1, y1), (x2, y2), color, 7)
            image_out = cv2.circle(image_out, (x1,y1), 5, (255, 0, 255), -1)
            count += 1
    return image_out

def vis_cluster_lines(image, clusters):
    image_out = image
    count = 0
    colors = [tuple(255 * e for e in mcolors.to_rgb(c)) for n,c in mcolors.TABLEAU_COLORS.items()]
    for cluster in clusters:
        color = colors[count%len(colors)]
        count += 1
        for line in cluster:
            x1, y1, x2, y2 = line.astype(int)
            image_out = cv2.line(image_out.astype(np.uint8), (x1, y1), (x2, y2), color, 7)
            image_out = cv2.circle(image_out, (x1,y1), 5, (255, 0, 255), -1)
    return image_out

def vis_points(image, points):
    image_out = image.astype(np.uint8)
    count = 1
    colors = [tuple(255 * e for e in mcolors.to_rgb(c)) for n,c in mcolors.TABLEAU_COLORS.items()]
    for line in points:
        color = colors[count%len(colors)]
        count += 1
        if line is not None:
            for point in line:
                x, y = point.astype(int)
                cv2.circle(image_out, (x,y), 5, color, -1)
    return image_out





def get_slope_intecept(lines):
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
    intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (
        lines[:, 2] + lines[:, 0])) / 2
    return slopes, intercepts


def get_polar(lines):
    x1 = lines[:,0]
    y1 = lines[:,1]
    x2 = lines[:,2]
    y2 = lines[:,3]

    theta = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2

    indices = x1 != x2
    a = np.ones(x1.shape)
    a[indices] = (y1 - y2)[indices] / (x2 - x1)[indices]
    b = np.zeros(a.shape)
    b[indices] = 1
    c = a * x1 + b * y1

    r = c / (a * np.cos(theta) + b * np.sin(theta))

    indices = r < 0
    r[indices] = - r[indices]
    theta[indices] = theta[indices] - np.pi

    indices = theta <= -np.pi
    theta[indices] = theta[indices] + 2 * np.pi

    return r, theta


def project_points(slope, intercept, points):
    # define a line by point a and b is parametrized by a + t * ab
    # t for projection of a point p is ap.dot(ab) / norm(ab)^2
    #  np.sum((ap * ab), axis=1) gives the line by line dot product: same as ap[i,:].dot(ab[i,:]) for all i
    n = points.shape[0]
    a = np.array([0, intercept])
    ab = np.array([1, slope])
    t = np.sum((points - a) * (np.tile(ab, (n, 1))), axis=1) / (slope**2 + 1)
    p = a + np.multiply(np.tile(t, (2,1)).T, ab)
    return (p, t)


def line_to_points(start, end, step=20):
    length = np.linalg.norm(end - start)
    points = [start]
    v = step * (end - start) / length
    current = start

    while length > step / 2:
        current = current + v
        points.append(current)
        length -= step

    return np.array(points)


def binarize_segmentation(segmentation):
    """
    Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    Get road mask by choosing pixels in segmentation output with value 2 or 3
    """
    binary = np.zeros(segmentation.shape, dtype='uint8')
    binary[segmentation == 2] = 1
    binary[segmentation == 3] = 1
    return binary


def detect_hough_lines(binary, gauss_kernel_size=5, gauss_sigma_x=1, canny_threshold_1=0, canny_threshold_2=2, hough_rho=1, hough_theta=np.pi / 180, hough_threshold=80, hough_min_line_length=30, hough_max_line_gap=10):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    binary -- tensor of dimension (H,W), containing a binary representation of lane points
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    # parameters
    #   src         input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    #   dst         output image of the same size and type as src.
    #   ksize       Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
    #   sigmaX      Gaussian kernel standard deviation in X direction.
    #   sigmaY      Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively (see getGaussianKernel for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
    #   borderType  pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported.
    #binary = cv2.GaussianBlur(binary, (gauss_kernel_size, gauss_kernel_size), gauss_sigma_x)

    # parameters:
    #   image         8-bit input image.
    #   threshold1    first threshold for the hysteresis procedure.
    #   threshold2    second threshold for the hysteresis procedure.
    #   apertureSize  aperture size for the Sobel operator.
    #   L2gradient    a flag, indicating whether a more accurate L2 norm =sqrt((dI/dx)2+(dI/dy)2) should be used to calculate 
    #                 the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm =|dI/dx|+|dI/dy| is enough.
    # returns:
    #   edges         output edge map; single channels 8-bit image, which has the same size as image .
    # defaults: apertureSize = 3, L2gradient = false
    #edges = cv2.Canny(binary, 0, 2, apertureSize = 3, L2gradient = True)
    
    # parameters
    #   image           8-bit, single-channel binary source image. The image may be modified by the function.
    #   lines           Output vector of lines. Each line is represented by a 4-element vector (x1,y1,x2,y2) , where (x1,y1) and (x2,y2) are the ending points of each detected line segment.
    #   rho             Distance resolution of the accumulator in pixels.
    #   theta           Angle resolution of the accumulator in radians.
    #   threshold       Accumulator threshold parameter. Only those lines are returned that get enough votes ( >𝚝𝚑𝚛𝚎𝚜𝚑𝚘𝚕𝚍 ).
    #   minLineLength   Minimum line length. Line segments shorter than that are rejected.
    #   maxLineGap      Maximum allowed gap between points on the same line to link them.
    # defaults: rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10
    lines = cv2.HoughLinesP(binary, rho=hough_rho, theta=hough_theta, threshold=hough_threshold, minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)
    lines = np.reshape(lines, (len(lines), 4))
    
    # dimensions of returned lines is (N x 4)
    return lines, binary


def merge_lines(lines, slope_similarity_threshold=0.1, intercept_similarity_threshold=40, min_slope_threshold=0):
    """
    Merges lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    radiuses, thetas = get_polar(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    keep = abs(slopes) >= min_slope_threshold
    
    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    clusters = []
    clusters_indices = []
    
    for i, (theta, r) in enumerate(zip(thetas, radiuses)):
        exists_in_clusters = np.array([i in current for current in clusters_indices])

        if not exists_in_clusters.any():
            slope_cluster = np.logical_and(
                thetas < (theta + slope_similarity_threshold),
                thetas > (theta - slope_similarity_threshold)
            )
            intercept_cluster = np.logical_and(
                radiuses < (r + intercept_similarity_threshold),
                radiuses > (r - intercept_similarity_threshold)
            )
            indices = np.argwhere(slope_cluster & intercept_cluster & keep).T.flatten()
            if indices.size:
                clusters_indices.append(indices)
                clusters.append(lines[indices])

    # Step 4: Merge all lines in clusters using mean averaging
    def merge(cluster, slopes, intercepts, i):
        cluster_len = len(cluster)
        if cluster_len != slopes.shape[0] or cluster_len != intercepts.shape[0]:
            raise ValueError('cluster ({}) and slopes ({}) and intercepts ({}) must be of same length'.format(cluster_len, slopes.shape[0], intercepts.shape[0]))
        points = np.reshape(cluster, (cluster_len * 2, 2))

        slope = np.mean(slopes)
        intercept = np.mean(intercepts)
        r, t = project_points(slope, intercept, points)
        i_min = np.argmin(t)
        i_max = np.argmax(t)

        return np.hstack((r[i_min], r[i_max]))

    merged_lines = [merge(cluster, slopes[clusters_indices[i]], intercepts[clusters_indices[i]], i) for i, cluster in enumerate(clusters)]
    merged_lines = np.array(merged_lines)
    
    # dimensions of returned lines is (N x 4)
    return merged_lines, clusters


def _find_nearest_from_point(lines, point):
    start_dist = np.linalg.norm(lines[:,0:2] - point, axis=1)
    end_dist = np.linalg.norm(lines[:,2:4] - point, axis=1)
    i = np.argmin(start_dist)
    j = np.argmin(end_dist)
    k, d = (i, start_dist[i]) if start_dist[i] < end_dist[j] else (j, end_dist[j])
    # index, start, end, distance
    return (k, lines[k, 0:2], lines[k, 2:4], d)


def _check_regularity(before, after, step=20):
    if len(before) > 1 and len(after) > 1:
        u = before[-1] - before[-2]
        v = after[1] - after[0]
        w = after[0] - before[-1]
        change_dir = np.sign(np.cross(u, v)) != np.sign(np.cross(u, w)) # u cross v = |u|.|v|.sin(angle)
        go_back = np.sign(u.dot(v)) != np.sign(u.dot(w)) # u dot v = |u|.|v|.cos(angle)
        dist = np.linalg.norm(w)
        too_close = dist < (step / 2)
        too_far = dist > (step * 1.5)
        return (change_dir, go_back, too_close, too_far)
    return (False, False, False, False)


def _extend_lane(lane, points, step=20):
    if points.shape[0] == 0:
        return lane
    change_dir, go_back, too_close, too_far = _check_regularity(lane, points, step)
    if change_dir:
        return _extend_lane(lane, points[1:], step)
    if go_back:
        return _extend_lane(lane[:-1], points, step)
    if too_close:
        return _extend_lane(lane, points[1:], step)
    if too_far:
        fill = line_to_points(lane[-1], points[0], step=20)
        lane = _extend_lane(lane, fill[1:-1], step)
        return _extend_lane(lane, points, step)
    return np.vstack((lane, points))


def associate_and_discretize_lane_lines(lines, origin, threshold=50, step=20):
    print(lines.shape)
    print(lines)
    todo = [i for i in range(lines.shape[0])]
    lanes = []
    while len(todo) > 0:
        new_lane = False
        i, start, end, dist = _find_nearest_from_point(lines[todo], lanes[-1][-1] if len(lanes) > 0 else origin)
        if dist > threshold:
            i, start, end, dist = _find_nearest_from_point(lines[todo], origin)
            new_lane = True
        todo.remove(todo[i])
        points = line_to_points(start, end)
        if points is not None:
            if len(lanes) > 0 and not new_lane:
                lanes[-1] =  _extend_lane(lanes[-1], points, step)
            else:
                lanes.append(points)

    return lanes


def associate_lines(lines, origin, threshold=50):
    todo = [i for i in range(lines.shape[0])]
    lanes = []
    while len(todo) > 0:
        new_lane = False
        i, start, end, dist = _find_nearest_from_point(lines[todo], lanes[-1][-1][2:4] if len(lanes) > 0 else origin)
        if dist > threshold:
            i, start, end, dist = _find_nearest_from_point(lines[todo], origin)
            new_lane = True
        todo.remove(todo[i])
        if len(lanes) > 0 and not new_lane:
            lanes[-1].append(np.hstack((start, end)))
        else:
            lanes.append([np.hstack((start, end))])

    return [np.array(lane) for lane in lanes]


def calc_point_to_segment_distance(p1, p2, p):
    # http://paulbourke.net/geometry/pointlineplane/
    u = (p[0] - p1[0]) * (p2[0] - p1[0]) + (p[1] - p1[1]) * (p2[1] - p1[1]) / np.linalg.norm(p2 - p1) ** 2
    u = min(1.0, max(0.0, u))
    d = np.linalg.norm(p1 + u * (p2 - p1) - p)
    return d


def calc_lines_distances(lines):
    distances = np.ones((lines.shape[0], lines.shape[0])) * np.inf
    for i, iline in enumerate(lines):
        for j, jline in enumerate(lines):
            if i != j:
                d1 = calc_point_to_segment_distance(jline[:2], jline[2:], iline[:2])
                d2 = calc_point_to_segment_distance(jline[:2], jline[2:], iline[2:])
                if i < j:
                    distances[i, j] =  min(d1, d2)
                else:
                    distances[i, j] = distances[j, i] = min(d1, d2, distances[j, i])
    return distances


def associate_lines_2(lines, origin, threshold=50):
    distances = calc_lines_distances(lines)

    todo = [i for i in range(lines.shape[0])]
    lanes = []
    open = SimpleQueue()
    while len(todo) > 0:
        i, _, _, _ = _find_nearest_from_point(lines[todo], origin)
        open.put(todo[i])
        current_lane_idx = []
        while not open.empty():
            idx = open.get()
            if idx in todo:
                todo.remove(idx)
                current_lane_idx.append(idx)
                next_idx = np.argwhere(distances[idx, :] <= threshold)
                for i in next_idx.flatten().tolist():
                    open.put(i)
        lanes.append(lines[current_lane_idx])
        
    return [np.array(lane) for lane in lanes]


def discretize_lanes_2(lane_lines, step=20):
    lanes = []
    for lines in lane_lines:
        new_lane = True
        for line in lines:
            points = line_to_points(line[0:2], line[2:4])
            if new_lane:
                lane_points = points
                new_lane = False
            else:
                lane_points = np.vstack((lane_points, points))
        # sort points
        # slopes, intercepts = get_slope_intecept(lines)
        # slope = np.mean(slopes)
        # intercept = np.mean(intercepts)
        # _, t = project_points(slope, intercept, lane_points)
        # indices = np.argsort(t)
        # lanes.append(lane_points[indices])
        lanes.append(lane_points)

    return lanes


def discretize_lanes(lane_lines, step=20):
    lanes = []
    for lines in lane_lines:
        new_lane = True
        for line in lines:
            points = line_to_points(line[0:2], line[2:4])
            if new_lane:
                lane_points = points
                new_lane = False
            else:
                lane_points = _extend_lane(lane_points, points, step)
        # sort points
        # slopes, intercepts = get_slope_intecept(lines)
        # slope = np.mean(slopes)
        # intercept = np.mean(intercepts)
        # _, t = project_points(slope, intercept, lane_points)
        # indices = np.argsort(t)
        # lanes.append(lane_points[indices])
        lanes.append(lane_points)

    return lanes


def find_left_right_lanes(lanes, origin):
    x0, y0 = origin
    positions = []
    for lane in lanes:
        line = np.reshape(lane[0:2,:], (1, 4))
        slopes, intercepts = get_slope_intecept(line)
        if slopes[0] != 0:
            x = (y0-intercepts[0]) / slopes[0]
            positions.append(x)
        else:
            print('WARINING: found slope = 0')

    positions= np.array(positions)
    if positions.size == 0:
        return (None, None)
    ranks = np.argsort(positions)
    [right_indice] = np.digitize([x0], positions[ranks]) # x0 is between right_indice-1 and right_indice

    if ranks.size == 1:
        return (None, lanes[ranks[0]]) if right_indice == 0 else (lanes[ranks[0]], None)
    if right_indice == 0:
        return (lanes[ranks[0]], lanes[ranks[1]]) # all lanes are at right
    if right_indice == len(ranks):
        return (lanes[ranks[-2]], lanes[ranks[-1]]) # all lanes are at left
    return (lanes[ranks[right_indice-1]], lanes[ranks[right_indice]])


def find_left_right_lanes_2(lane_lines, origin):
    distances = []
    nearest_points = []
    for lane in lane_lines:
        points = np.vstack((lane[:,:2], lane[:,2:]))
        dists = np.linalg.norm(points - origin, axis=1)
        min_index = np.argmin(dists)
        distances.append(dists[min_index])
        nearest_points.append(points[min_index])

    index = np.argmin(np.array(distances))
    vect = nearest_points[index] - origin
    distances[index] = np.inf
    lane_1 = lane_lines[index]
    angle_1 = np.arctan2(vect[1], vect[0])

    index = np.argmin(np.array(distances))
    vect = nearest_points[index] - origin
    lane_2 = lane_lines[index]
    angle_2 = np.arctan2(vect[1], vect[0])
    
    return (lane_1, lane_2) if angle_1 > angle_2 else (lane_2, lane_1)



def estimate_path(left_lane, right_lane):
    cross_dist = cdist(left_lane, right_lane)

    if left_lane.size <= right_lane.size:
        indices = np.argmin(cross_dist.T, axis=-1)
        left = left_lane[indices]
        right = right_lane
    else:
        indices = np.argmin(cross_dist, axis=-1)
        left = left_lane
        right = right_lane[indices]

    return left + (right - left) / 2


def regularize_lane(lane, step=30):
    path = [lane[0]]
    remaining = step
    distance = 0
    index = 0

    while True:
        if remaining < distance:
            current = current + remaining * u / distance
            u = next - current
            distance = np.linalg.norm(u)
            remaining = step
            path.append(current)
        else:
            if index + 1 >= lane.shape[0]:
                path.append(current + remaining * u / distance)
                break
            remaining -= distance
            current = lane[index]
            next = lane[index+1]
            u = next - current
            distance = np.linalg.norm(u)
            index += 1

    return np.array(path)


def path_to_vehicule_frame(path, origin, pixel_in_cm):
    return (origin - path) @ np.array([[0, 1], [1, 0]]) * pixel_in_cm


def threshold_equalized_grayscale(frame, threshhold=250):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = frame if len(frame.shape) < 3 or frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=threshhold, maxval=255, type=cv2.THRESH_BINARY)

    return th, eq_global


def detect_path(segmentation, vanishing_point, crop_top, height_ratio, pixel_in_cm=0.333, plot=False):
    birdeye = Birdeye(*segmentation.shape, vanishing_point=vanishing_point, crop_top=crop_top, crop_corner=1, height_ratio=height_ratio)
    rectified_segmentation = birdeye.apply(segmentation)
    binary = binarize_segmentation(rectified_segmentation)

    lines, edges = detect_hough_lines(binary, hough_threshold=30, hough_min_line_length=30, hough_max_line_gap=20)
    merged_lines, lines_clusters = merge_lines(lines, slope_similarity_threshold=0.1, intercept_similarity_threshold=20)

    origin = np.array([int(binary.shape[1] / 2), binary.shape[0]])

    lane_lines = associate_lines(merged_lines, origin, threshold=70)
    lanes = discretize_lanes(lane_lines)
    left_lane, right_lane = find_left_right_lanes(lanes, origin)
    estimated_path = estimate_path(left_lane, right_lane)
    path = regularize_lane(estimated_path)
    path_v = path_to_vehicule_frame(path, origin, pixel_in_cm=pixel_in_cm)

    return path_v


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from donkeycar.parts.lane_finding.binarization import binarize
    from donkeycar.parts.lane_finding.perspective import Birdeye
    
    for test_img in glob.glob('test_images/*.jpg'):

        frame = cv2.imread(test_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        birdeye = Birdeye(*frame.shape[:2], vanishing_point=0.46, crop_top=0.5, crop_corner=0.7) \
            .apply(frame, plot=False)
        
        gauss_kernel_size=5
        gauss_sigma_x=1
        birdeye = cv2.GaussianBlur(birdeye, (gauss_kernel_size, gauss_kernel_size), gauss_sigma_x)
        binary, eq_global = threshold_equalized_grayscale(birdeye, threshhold=245)
        
        lines, edges = detect_hough_lines(binary, hough_threshold=30, hough_min_line_length=30, hough_max_line_gap=20)
        merged_lines, lines_clusters = merge_lines(lines, slope_similarity_threshold=0.1, intercept_similarity_threshold=20)

        origin = np.array([int(binary.shape[1] / 2), binary.shape[0]])

        lane_lines = associate_lines_2(merged_lines, origin, threshold=40)


        left_lane, right_lane = find_left_right_lanes_2(lane_lines, origin)
        left_lane, right_lane = discretize_lanes_2([left_lane, right_lane])

        #lanes = discretize_lanes(lane_lines)


        #left_lane, right_lane = find_left_right_lanes(lanes, origin)
        estimated_path = estimate_path(left_lane, right_lane)
        path = regularize_lane(estimated_path)
        path_v = path_to_vehicule_frame(path, origin, pixel_in_cm=0.333)


        fig = plt.figure(figsize=(11, 10))
        (rows, columns) = (4,3)

        ax = fig.add_subplot(rows, columns, 1)
        plt.imshow(frame)
        ax.set_title("image")

        ax = fig.add_subplot(rows, columns, 2)
        plt.imshow(birdeye)
        ax.set_title("birdeye image")

        ax = fig.add_subplot(rows, columns, 3)
        plt.imshow(binary, cmap='gray', vmin=0, vmax=255)
        ax.set_title("edges")

        ax = fig.add_subplot(rows, columns, 4)
        plt.imshow(vis_cluster_lines(birdeye, lines_clusters))
        ax.set_title("clusters")
        print('merged lines count: ' + str(merged_lines.shape[0]))

        ax = fig.add_subplot(rows, columns, 5)
        plt.imshow(vis_lanes(birdeye, merged_lines, multi_color=True))
        ax.set_title("merged lines")

        ax = fig.add_subplot(rows, columns, 6)
        plt.imshow(vis_cluster_lines(birdeye, lane_lines))
        ax.set_title("lanes lines")
        print('lanes count: ' + str(len(lane_lines)))



        # ax = fig.add_subplot(rows, columns, 7)
        # plt.imshow(vis_points(birdeye, lanes))
        # ax.set_title("lanes")

        ax = fig.add_subplot(rows, columns, 8)
        plt.imshow(vis_points(birdeye, [left_lane]))
        ax.set_title("left lane")

        ax = fig.add_subplot(rows, columns, 9)
        plt.imshow(vis_points(birdeye, [right_lane]))
        ax.set_title("right lane")

        ax = fig.add_subplot(rows, columns, 10)
        plt.imshow(vis_points(birdeye, [estimated_path]))
        ax.set_title("central lane")

        ax = fig.add_subplot(rows, columns, 11)
        plt.imshow(vis_points(birdeye, [path]))
        ax.set_title("regularized central lane")

        plt.show()

        # break