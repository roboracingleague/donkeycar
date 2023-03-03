import time
import math
import glob
import logging
from queue import SimpleQueue
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import depthai as dai
from depthai_sdk import toTensorResult
from donkeycar.parts.lane_finding.perspective import Birdeye
from donkeycar.parts.lane_finding.binarization import threshold_equalized_grayscale

import matplotlib.colors as mcolors


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mask_to_color(gray):
    max_val = np.max(gray)
    scale = int(255.0 / max_val) if max_val > 0 else 1
    return np.dstack([gray * scale,] * 3)


def draw_points(image, points, size=4, color_index=1):
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


def draw_anchor(image, anchor, size=6, color_index=1):
    image_out = image
    if anchor is not None:
        x, y = anchor[2:].astype(np.int32)
        ux, uy = anchor[:2] * 6 * size / np.linalg.norm(anchor[:2])
        color = (255, 0, 255)
        image_out = cv2.line(image_out.astype(np.uint8), (x, y), (int(x + ux), int(y + uy)), color, thickness=int(size/2))
        image_out = cv2.circle(image_out, (x, y), size, color, -1)
    return image_out


def draw_anchors(image, anchors, size=6, color_index=1):
    if anchors is not None:
        for i in range(anchors.shape[0]):
            image = draw_anchor(image, anchors[i,:], size, color_index)
    return image


def scan(binary, origin, rmax, search_value, angle_steps=360):
    """
    Use ray tracing algorithm
    """
    (h, w) = np.shape(binary)
    x, y = origin
    angles = [i * 2 * np.pi / angle_steps for i in range(angle_steps)]
    points = []
    scan_mask = np.zeros((h,w), dtype=np.uint8)
    
    for i in range(len(angles)):
        for r in range(1, rmax+1):
            # determine the coordinates of the cell.
            xi = int(round(x + r * math.cos(angles[i])))
            yi = int(round(y + r * math.sin(angles[i])))
            
            # if not in the map, stop going further
            if (xi < 0 or xi > w-1 or yi < 0 or yi > h-1):
                break
            # if in the map, but hitting an obstacle, retain point and stop ray tracing
            elif binary[yi, xi] == search_value:
                points.append([xi, yi])
                scan_mask[yi, xi] = 1
                break
            elif binary[yi, xi] != 0:
                break
                
    return np.array(points), scan_mask


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


def extract_overlap(points_a, points_b):
    """
    Split points_a set in points that are outside of the rect containing points_b and points that are inside
    """
    p_min = np.min(points_b, axis=0)
    p_max = np.max(points_b, axis=0)
    has_overlap = np.logical_and(np.all(points_a >= p_min, axis=1), np.all(points_a <= p_max, axis=1))
    no_overlap_indexes = np.argwhere(np.logical_not(has_overlap)).flatten()
    no_overlap = points_a[no_overlap_indexes,:]
    overlap_indexes = np.argwhere(has_overlap).flatten()
    overlap = points_a[overlap_indexes,:]
    return no_overlap, overlap


def fit_line(points, anchor, pos_to_anchor=False):
    # fit a line [vx, vy, x0, y0]
    line = cv2.fitLine(points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    # correct orientation to correspond to given anchor
    line[:2] = line[:2] if anchor[:2].dot(line[:2]) > 0 else -line[:2]
    # and take line that goes through anchor point or last point
    line[2:] = np.reshape(anchor[2:] if pos_to_anchor else points[-1,:], (2,1))
    return np.squeeze(line)


def sort_points(points, anchor):
    if points is None or points.shape[0] == 1:
        return points
    
    line = fit_line(points, anchor, pos_to_anchor=True)
    # calc parametric position of point projection on the line, see: http://paulbourke.net/geometry/pointlineplane/
    u = (points[:,0] - line[2]) * line[0] + (points[:,1] - line[3]) * line[1] / np.linalg.norm(line[:2]) ** 2

    sorted_points = points[np.argsort(u),:]
    return sorted_points


def second_min_index(array):
    if array.shape[0] < 2:
        return None
    return np.argpartition(array, 2)[:2][1]


def assign_groups_to_anchors(groups, left_anchor, right_anchor):
    left_dists = []
    right_dists = []
    for group in groups:
        left_dists.append(np.min(np.linalg.norm(group - left_anchor[2:], axis=1)))
        right_dists.append(np.min(np.linalg.norm(group - right_anchor[2:], axis=1)))
    left_dists = np.array(left_dists)
    right_dists = np.array(right_dists)

    left_index = np.argmin(left_dists)
    right_index = np.argmin(right_dists)

    if left_index == right_index:
        if left_dists[left_index] < right_dists[right_index]:
            right_index = second_min_index(right_dists)
        else:
            left_index = second_min_index(left_dists)

    left_points = groups[left_index] if left_index is not None else None
    right_points = groups[right_index] if right_index is not None else None

    unassigned_indexes = [i for i in range(len(groups)) if i not in [left_index, right_index]]
    unassigned_groups = [groups[i] for i in unassigned_indexes]

    if len(unassigned_groups) > 0:
       print('WARNING found points that where not assigned to left or right lane')

    return left_points, right_points, unassigned_groups


def find_new_anchor(points, last_anchor, point_count=5):
    if points is None or points.shape[0] < 2:
        return None
    
    n = min(points.shape[0], point_count) if point_count > 0 else points.shape[0]
    last_points = points[-n:,:]
    line = fit_line(last_points, last_anchor, pos_to_anchor=False)

    return line


def scan_for_lane(binary, left_anchor, right_anchor, lane_width = 100, scan_iters=4):
    left_points_groups = []
    right_points_groups = []
    left_anchors = []
    right_anchors = []
    scan_points = []
    full_unassigned_groups = []
    full_scan_mask = np.zeros(binary.shape)

    for i in range(scan_iters):
        print('---', i)
        scan_point = np.mean([left_anchor[2:], right_anchor[2:]], axis=0)
        points, scan_mask = scan(binary, scan_point, rmax=120, search_value=1, angle_steps=72)
        groups = greedy_group(points, threshold=50)
        left_points, right_points, unassigned_groups = assign_groups_to_anchors(groups, left_anchor, right_anchor)
        left_points = sort_points(left_points, left_anchor)
        right_points = sort_points(right_points, right_anchor)

        left_anchors.append(left_anchor)
        right_anchors.append(right_anchor)
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

        # TODO: overlap

        # save results of current slice
        # full_left_points = np.vstack((full_left_points, left_points)) if full_left_points is not None else left_points
        # full_right_points = np.vstack((full_right_points, left_points)) if full_right_points is not None else right_points
        left_points_groups.append(left_points)
        right_points_groups.append(right_points)
        scan_points.append(scan_point)
        full_unassigned_groups.extend(unassigned_groups)
        full_scan_mask[scan_mask > 0] = 1

    return left_points_groups, right_points_groups, np.array(left_anchors), np.array(right_anchors), np.array(scan_points), full_unassigned_groups, full_scan_mask



def main(image_name):
    from matplotlib import pyplot as plt

    print(image_name)
    frame = cv2.imread(image_name)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start = time.time()

    birdview = Birdeye(*frame.shape[:2], vanishing_point=0.46, crop_top=0.5, crop_corner=0.7, border_value=0)
    birdeye = birdview.apply(frame)
    mask = birdview.apply(np.zeros(frame.shape[:2], dtype=np.uint8), border_value=255)

    gauss_kernel_size=5
    gauss_sigma_x=1
    birdeye = cv2.GaussianBlur(birdeye, (gauss_kernel_size, gauss_kernel_size), gauss_sigma_x)
    binary = threshold_equalized_grayscale(birdeye, threshhold=240)
    binary[binary>0] = 1
    binary[mask>0] = 2

    left_anchor = np.array([0.0, -1.0, 250, 370])
    right_anchor = np.array([0.0, -1.0, 390, 370])

    left_points_groups, right_points_groups, left_anchors, right_anchors, scan_points, unassigned_groups, scan_mask = \
        scan_for_lane(binary, left_anchor, right_anchor, scan_iters=2)
    
    print(time.time() - start)


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
    plt.imshow(draw_points(mask_to_color(scan_mask), scan_points, color_index=1))
    ax.set_title("scan mask")

    ax = fig.add_subplot(rows, columns, 5)
    plt.imshow(draw_anchors(draw_points_groups(birdeye, left_points_groups, color_index=3), left_anchors, color_index=0))
    ax.set_title("left points")

    ax = fig.add_subplot(rows, columns, 6)
    plt.imshow(draw_anchors(draw_points_groups(birdeye, right_points_groups, color_index=2), right_anchors))
    ax.set_title("right points")

    ax = fig.add_subplot(rows, columns, 7)
    plt.imshow(draw_points_groups(birdeye, unassigned_groups, color_index=4))
    ax.set_title("unassigned points")

    plt.show()

if __name__ == '__main__':
    for test_img in glob.glob('test_images/*.jpg'):
        main(test_img)
