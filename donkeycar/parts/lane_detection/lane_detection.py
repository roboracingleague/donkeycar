import time
import math
import glob
import logging
from collections import deque
import cv2
import numpy as np
from queue import SimpleQueue
from scipy.spatial.distance import cdist
from donkeycar.parts.lane_detection.perspective import Birdeye
from donkeycar.parts.lane_detection.binarization import binarize_grayscale
from donkeycar.parts.motion_planner import change_frame_2to1


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class PoseHistoSample():
    def __init__(self):
        self.trajectory = np.array([0.0, 0.0])
        self.trajectory_origin = np.array([0.0, 0.0, 0.0, 0.0])
        self.poses = deque([], maxlen=20)
        logger.info('Starting TestTrajectory')
    
    def find_pose_with_nearest_time(self, time):
        pose_times = np.array([p[0] for p in self.poses])
        index = np.argmin(np.absolute(pose_times - time))
        return self.poses[index] # [t, x, y, yaw]

    def update_trajectory(self, time):
        trajectory = ...
        self.trajectory_origin = self.find_pose_with_nearest_time(time)
        self.trajectory = change_frame_2to1(self.trajectory_origin[1:3], self.trajectory_origin[3], trajectory)
        logger.debug('Sending a new path')

    def run(self, run_pilot, pos_time, x, y, yaw):
        pose = np.array([pos_time, x, y, yaw])
        self.poses.append(pose)
        return self.trajectory, self.trajectory_origin[0] # need to send 2 values at least to record None value in vehicle mem
    

class LaneDetection:
    '''
    Lane detection part based on opencv and image analysis, returns 2 numpy arrays of shape [N, 2] for left and right (sorted) points.
    '''
    def __init__(self, image_width, image_height, vanishing_point=0.46, crop_top=0.5, crop_corner=0.7,
                 eq_threshhold=240, blur_kernel_size=5, blur_sigma_x=1, close_kernel_size=5,
                 left_anchor=np.array([0.0, -1.0, 250, 370]), right_anchor=np.array([0.0, -1.0, 390, 370]),
                 width_m_per_pix=1.0, height_m_per_pix=1.0):
        self.thread_input = None
        self.thread_output = None
        self.on = True
        self.eq_threshhold = eq_threshhold
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma_x = blur_sigma_x
        self.close_kernel_size = close_kernel_size
        self.left_anchor = left_anchor
        self.right_anchor = right_anchor
        self.width_m_per_pix = width_m_per_pix
        self.height_m_per_pix = height_m_per_pix
        self.birdeye = Birdeye(image_height, image_width, vanishing_point, crop_top, crop_corner, border_value=0)
        self.poses = deque([], maxlen=20)
        logger.info("LaneDetection ready")

    def find_pose_with_nearest_time(self, time):
        if len(self.poses) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0])
        pose_times = np.array([p[0] for p in self.poses])
        index = np.argmin(np.absolute(pose_times - time))
        return self.poses[index] # [t, x, y, yaw]
    
    def detect_lanes(self, image_time, image):
        image_origin = self.find_pose_with_nearest_time(image_time)
        birdeye = self.birdeye.apply(image)
        binary = binarize_grayscale(birdeye, self.eq_threshhold, self.blur_kernel_size, self.blur_sigma_x, self.close_kernel_size)
        left_points, right_points, left_anchors, right_anchors, scan_points, unassigned_groups = \
            scan_for_lane(binary, self.left_anchor, self.right_anchor)
        
        # TODO : change frame (image x, y ?) and scale
        left_points_vframe = change_frame_2to1(image_origin[1:3], image_origin[3], left_points)
        right_points_vframe = change_frame_2to1(image_origin[1:3], image_origin[3], right_points)

        # TODO : eval center_points
        center_points_vframe = None
        
        return left_points_vframe, center_points_vframe, right_points_vframe, \
               left_points, right_points, birdeye, binary, left_anchors, right_anchors, scan_points, unassigned_groups
    
    def run(self, pos_time, x, y, yaw, image_time, image):
        self.poses.append(np.array([pos_time, x, y, yaw]))
        return self.detect_lanes(image_time, image)[:3]

    def run_threaded(self, pos_time, x, y, yaw, image_time, image):
        self.poses.append(np.array([pos_time, x, y, yaw]))
        if image is not None and image_time is not None:
            self.thread_input = (image_time, image)
        return self.thread_output

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            if self.thread_input is not None:
                frame_time, frame = self.thread_input
                self.thread_input = None
                self.thread_output = self.detect_lanes(frame_time, frame)[:3]
            else:
                time.sleep(0.005)
    
    def shutdown(self):
        self.on = False
        logger.info('Stopping LaneDetection')


def scan_for_lane(binary, left_anchor, right_anchor, lane_width = 100, scan_iters=2, scan_radius=180, scan_steps=60, group_threshold=50):
    """
    Apply a ray tracing algorithme to find points from left and right lane borders and sort them 
    """
    left_points = np.empty((0,2))
    right_points = np.empty((0,2))
    left_anchors = []
    right_anchors = []
    scan_points = []
    unassigned_groups = []

    for i in range(scan_iters):
        scan_point = np.mean([left_anchor[2:], right_anchor[2:]], axis=0)
        points = scan(binary, scan_point, scan_radius=scan_radius, search_value=1, scan_steps=scan_steps)

        groups = greedy_group(points, threshold=group_threshold)
        current_left, current_right, current_unassigned = assign_groups_to_anchors(groups, left_anchor, right_anchor)

        left_points, current_left = move_overlap(left_points, current_left)
        right_points, current_right = move_overlap(right_points, current_right)

        current_left = sort_points(current_left, left_anchor)
        current_right = sort_points(current_right, right_anchor)

        # save results of current slice
        left_points = np.vstack((left_points, current_left)) if current_left is not None else left_points
        right_points = np.vstack((right_points, current_right)) if current_right is not None else right_points
        left_anchors.append(left_anchor)
        right_anchors.append(right_anchor)
        scan_points.append(scan_point)
        unassigned_groups.extend(current_unassigned)

        # find new anchors for next iter
        left_anchor = find_new_anchor(left_points, left_anchor)
        right_anchor = find_new_anchor(right_points, right_anchor)

        if left_anchor is None and right_anchor is None:
            break

        if left_anchor is None:
            vx, vy, x, y = right_anchor
            n = np.linalg.norm(right_anchor[:2])
            # estimate anchor by shifting existing one to the other side of the lane
            left_anchor = np.array([vx, vy, int(x + vy * lane_width / n), int(y - vx * lane_width / n)])

        if right_anchor is None:
            vx, vy, x, y = left_anchor
            n = np.linalg.norm(left_anchor[:2])
            right_anchor = np.array([vx, vy, int(x - vy * lane_width / n), int(y + vx * lane_width / n)])

    return left_points, right_points, np.array(left_anchors), np.array(right_anchors), np.array(scan_points), unassigned_groups


def scan(binary, origin, scan_radius, search_value, scan_steps=360):
    """
    Apply ray tracing algorithm.
    """
    (h, w) = np.shape(binary)
    x, y = origin
    points = []
    angles = [i * 2 * np.pi / scan_steps for i in range(scan_steps)]
    ranges = range(1, scan_radius)
    
    for i in range(len(angles)):
        cos = math.cos(angles[i])
        sin = math.sin(angles[i])
        for r in ranges:
            # determine the coordinates of the cell.
            xi = int(x + r * cos)
            yi = int(y + r * sin)
            
            # if not in the map, stop going further
            if (xi < 0 or xi > w-1 or yi < 0 or yi > h-1):
                break
            # if in the map, but hitting an obstacle, retain point and stop ray tracing
            if binary[yi, xi] == search_value:
                points.append([xi, yi])
                break
            
    return np.array(points)


def greedy_group(points, threshold):
    """
    Group points so that when to points are spaced by less than threshold, they share the same group.
    """
    distances = cdist(points, points)

    groups = []
    remaining_indexes = [i for i in range(points.shape[0])]
    open_indexes = SimpleQueue()
    # loop until all points have been added to a group
    while len(remaining_indexes) > 0:
        open_indexes.put(remaining_indexes[0])
        current_group_indexes = []
        # iterate over current neighbours
        while not open_indexes.empty():
            idx = open_indexes.get()
            # and if it has not been done
            if idx in remaining_indexes:
                current_group_indexes.append(idx)
                remaining_indexes.remove(idx)
                # add it's own neighbours to current neighbours
                neighbours_indexes = np.argwhere(distances[idx, :] <= threshold)
                for i in neighbours_indexes.flatten().tolist():
                    open_indexes.put(i)
        # when there is no more neighbours, close current group
        groups.append(points[current_group_indexes])
    return [np.array(group) for group in groups]


def assign_groups_to_anchors(groups, left_anchor, right_anchor):
    """
    Takes many array of points and find which is nearer to each anchor.
    """
    if len(groups) < 1:
        return None, None, []
    
    left_dists = []
    right_dists = []
    for group in groups:
        left_dists.append(np.min(np.linalg.norm(group - left_anchor[2:], axis=1)))
        right_dists.append(np.min(np.linalg.norm(group - right_anchor[2:], axis=1)))
    left_dists = np.array(left_dists)
    right_dists = np.array(right_dists)

    left_indexes = np.argsort(left_dists)
    left_index = left_indexes[0]
    right_indexes = np.argsort(right_dists)
    right_index = right_indexes[0]

    # check if same group is the nearest from both sides
    if left_index == right_index:
        if left_dists[left_index] < right_dists[right_index]:
            right_index = right_indexes[1] if len(right_indexes) > 1 else None
        else:
            left_index = left_indexes[1] if len(left_indexes) > 1 else None

    left_points = groups[left_index] if left_index is not None else None
    right_points = groups[right_index] if right_index is not None else None

    unassigned_indexes = [i for i in range(len(groups)) if i not in [left_index, right_index]]
    unassigned_groups = [groups[i] for i in unassigned_indexes]

    if len(unassigned_groups) > 0:
       logger.debug('WARNING found points that where not assigned to left or right lane')
    
    return left_points, right_points, unassigned_groups


def move_overlap(points_a, points_b):
    """
    Move points from points_a to points_b if they are inside a rect containing points_b.
    """
    if points_a is None or points_b is None:
        return points_a, points_b
    p_min = np.min(points_b, axis=0)
    p_max = np.max(points_b, axis=0)
    has_overlap = np.logical_and(np.all(points_a >= p_min, axis=1), np.all(points_a <= p_max, axis=1))
    no_overlap_indexes = np.argwhere(np.logical_not(has_overlap)).flatten()
    no_overlap = points_a[no_overlap_indexes,:]
    overlap_indexes = np.argwhere(has_overlap).flatten()
    overlap = points_a[overlap_indexes,:]
    t = np.vstack((overlap, points_b))
    return no_overlap, t


def sort_points(points, anchor):
    """
    Sort points according to anchor vector direction.
    """
    if points is None or points.shape[0] < 2:
        return points
    
    line = fit_line(points, anchor, pos_to_anchor=True)
    # calc parametric position of point projection on the line, see: http://paulbourke.net/geometry/pointlineplane/
    u = (points[:,0] - line[2]) * line[0] + (points[:,1] - line[3]) * line[1] / np.linalg.norm(line[:2]) ** 2

    sorted_points = points[np.argsort(u),:]
    return sorted_points


def find_new_anchor(points, last_anchor, point_count=5):
    """
    Create an anchor at the position of the last point and with a vector that fit the orientation
    of the last 'point_count' points and shares direction with 'last_anchor'.
    """
    if points is None or points.shape[0] < 2:
        return None
    
    n = min(points.shape[0], point_count) if point_count > 0 else points.shape[0]
    last_points = points[-n:,:]
    line = fit_line(last_points, last_anchor, pos_to_anchor=False)

    return line


def fit_line(points, anchor, pos_to_anchor=False):
    """
    Fit a line, sharing orientation with the given anchor vector and make it goes through last point (or anchor point if asked so).
    """
    # fit a line [vx, vy, x0, y0]
    line = cv2.fitLine(points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    # correct orientation to correspond to given anchor
    line[:2] = line[:2] if anchor[:2].dot(line[:2]) > 0 else -line[:2]
    # and take line that goes through anchor point or last point
    line[2:] = np.reshape(anchor[2:] if pos_to_anchor else points[-1,:], (2,1))
    return np.squeeze(line)


def mask_to_color(gray):
    max_val = np.max(gray)
    scale = int(255.0 / max_val) if max_val > 0 else 1
    return np.dstack([gray * scale,] * 3)


def draw_points(image, points, size=4, color_index=1):
    import matplotlib.colors as mcolors

    image_out = image.astype(np.uint8)
    colors = [tuple(255 * e for e in mcolors.to_rgb(c)) for n,c in mcolors.TABLEAU_COLORS.items()]
    color = colors[color_index%len(colors)]
    if points is not None:
        for point in points:
            x, y = point.astype(int)
            cv2.circle(image_out, (x,y), size, color, -1)
        image_out = cv2.circle(image_out, points[-1].astype(int), size, (255, 0, 255), -1)
    return image_out


def draw_points_groups(image, groups, size=4, color_index=1):
    if groups is not None:
        for i in range(len(groups)):
            image = draw_points(image, groups[i], size, color_index+i)
    return image


def draw_anchor(image, anchor, size=6):
    image_out = image
    if anchor is not None:
        x, y = anchor[2:].astype(np.int32)
        ux, uy = anchor[:2] * 6 * size / np.linalg.norm(anchor[:2])
        color = (255, 0, 255)
        image_out = cv2.line(image_out.astype(np.uint8), (x, y), (int(x + ux), int(y + uy)), color, thickness=int(size/2))
        image_out = cv2.circle(image_out, (x, y), size, color, -1)
    return image_out


def draw_anchors(image, anchors, size=6):
    if anchors is not None:
        for i in range(anchors.shape[0]):
            image = draw_anchor(image, anchors[i,:], size)
    return image


def main_from_frame(frame):
    from matplotlib import pyplot as plt

    part = LaneDetection(frame.shape[1], frame.shape[0], vanishing_point=0.46, crop_top=0.5, crop_corner=0.7,
                 eq_threshhold=240, blur_kernel_size=5, blur_sigma_x=1, close_kernel_size=5,
                 left_anchor=np.array([0.0, -1.0, 250, 370]), right_anchor=np.array([0.0, -1.0, 390, 370]))
    
    start = time.time()
    left_points_v, center_points_v, right_points_v, left_points, right_points, birdeye, \
        binary, left_anchors, right_anchors, scan_points, unassigned_groups = part.detect_lanes(0.0, frame)
    print('execution time (s) :', time.time() - start)

    scan_mask = np.zeros((binary.shape[0], binary.shape[1], 3))
    scan_mask = draw_points_groups(scan_mask, [left_points, right_points] + unassigned_groups)

    fig = plt.figure(figsize=(14, 16))
    (rows, columns) = (4,3)

    ax = fig.add_subplot(rows, columns, 1)
    plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
    ax.set_title("frame")

    ax = fig.add_subplot(rows, columns, 2)
    plt.imshow(birdeye)
    ax.set_title("birdeye")

    ax = fig.add_subplot(rows, columns, 3)
    plt.imshow(binary, cmap='gray')
    ax.set_title("binary")

    ax = fig.add_subplot(rows, columns, 4)
    plt.imshow(draw_points(scan_mask, scan_points, color_index=1))
    ax.set_title("scan points")

    ax = fig.add_subplot(rows, columns, 5)
    plt.imshow(draw_anchors(draw_points(birdeye, left_points, color_index=3), left_anchors))
    ax.set_title("left points")

    ax = fig.add_subplot(rows, columns, 6)
    plt.imshow(draw_anchors(draw_points(birdeye, right_points, color_index=2), right_anchors))
    ax.set_title("right points")

    ax = fig.add_subplot(rows, columns, 7)
    plt.imshow(draw_points_groups(birdeye, unassigned_groups, color_index=4))
    ax.set_title("unassigned points")

    plt.show()


def main(image_name):
    print(image_name)
    frame = cv2.imread(image_name)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    main_from_frame(frame)


if __name__ == '__main__':
    for test_img in glob.glob('test_images/*.jpg'):
        main(test_img)
