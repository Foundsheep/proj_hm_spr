import torch
import math
import numpy as np
from copy import deepcopy
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import datetime
import skimage

BACKGROUND = [0, 0, 0]
LOWER = [255, 0, 0]
MIDDLE = [255, 255, 0]
RIVET = [0, 255, 0]
UPPER = [0, 0, 255]
COLOUR_NAMES = {
    "BACKGROUND": BACKGROUND,
    "LOWER": LOWER,
    "MIDDLE": MIDDLE,
    "RIVET": RIVET,
    "UPPER": UPPER,
    }
RIVET_DIAMETER = 7.75
HEAD_DIAMETER = 13.5


def adjust_ratio_and_resize_image(path, to_h, to_w, save_folder_path=None):
    # load an image
    img = Image.open(path)

    # set the basic info
    target_ratio = 0.75
    h = img.height
    w = img.width
    ratio = h / w
    new_ratio = 0.00

    # get new length according to the target ratio
    if ratio != target_ratio:
        if ratio < target_ratio:
            new_length = int(w * target_ratio)
            half = (new_length - h) // 2
        elif ratio > target_ratio:
            new_length = int(h / target_ratio)
            half = (new_length - w) // 2

        another_half = new_length - half - h if 2 * half != new_length else half

        # adjust ratio by padding in reflection mode
        img_np = np.array(img)
        if ratio < target_ratio:
            img_padded = np.pad(img_np, ((half, another_half), (0, 0), (0, 0)), "reflect")
        elif ratio > target_ratio:
            img_padded = np.pad(img_np, ((0, 0), (half, another_half), (0, 0)), "reflect")

        new_ratio = img_padded.shape[0] / img_padded.shape[1]
        assert f"{target_ratio :.2f}" == f"{new_ratio :.2f}", f"{target_ratio =}, {new_ratio =}"
        img = Image.fromarray(img_padded)
        
    # downsize the image
    img = img.resize((to_w, to_h), Image.Resampling.NEAREST)
    if save_folder_path is None:
        return img
        
    # if to save
    save_folder = Path(save_folder_path)
    if save_folder.exists():
        # print("... folder exists already")
        pass
    else:
        save_folder.mkdir()
        # print(f"...{save_folder} made!")

    # save
    img.save(save_folder / path.parts[-1])
    return img

def get_image_array(path):
    img_arr = None
    
    img = Image.open(path)
    if img is not None:
        img_arr = np.array(img)
    
    return img_arr

def get_line_coords_via_corners(img_arr, colour, is_upper_corner=True):
    condition_1 = img_arr[:, :, 0] == colour[0]
    condition_2 = img_arr[:, :, 1] == colour[1]
    condition_3 = img_arr[:, :, 2] == colour[2]
    
    channel_1_coords, channel_2_coords = np.where(condition_1 & condition_2 & condition_3)
    
    # 모서리 기준점 좌표 설정
    h = img_arr.shape[0]
    w = img_arr.shape[1]
    if is_upper_corner:
        left_coord = [0, 0]
        right_coord = [w - 1, 0]
    else:
        left_coord = [0, h - 1]
        right_coord = [w - 1, h - 1]
    
    # 최대 거리 설정
    max_diff_left = h * w
    max_diff_right = h * w
    
    # 찾아낸 좌표들 기본값 설정
    left_x = None
    left_y = None

    right_x = None
    right_y = None

    # 좌표 값마다 모서리 기준점과 거리(L1) 비교
    # --- 이미지 표현에서 첫번째 채널 값은 y축, 두번째 채널 값은 x축을 표현
    for y, x in zip(channel_1_coords, channel_2_coords):
        left_distance = np.abs(left_coord[0] - x) + np.abs(left_coord[1] - y)
        right_distance = np.abs(right_coord[0] - x) + np.abs(right_coord[1] - y)
        
        if left_distance < max_diff_left:
            left_x = x
            left_y = y
            # print(f"left coords found [{left_y}, {left_x}]")
            max_diff_left = left_distance
            
        elif right_distance < max_diff_right:
            right_x = x
            right_y = y
            max_diff_right = right_distance
            # print(f"right coords found [{right_y}, {right_x}]")

    # visualise(img_arr, [left_x, right_x], [left_y, right_y], fig_save_folder, True)    
    return left_x, left_y, right_x, right_y

def get_matrix(x1, y1, x2, y2):
    matrix = np.array([
        [x1, x2],
        [y1, y2]
    ])
    
    return matrix

def rotate_vector_around_origin(vector, rad):
    x, y = vector
    if rad > 0:
        xx = x * math.cos(rad) + y * math.sin(rad)
        yy = -x * math.sin(rad) + y * math.cos(rad)
    elif rad < 0:
        xx = x * math.cos(rad) + -y * math.sin(rad)
        yy = x * math.sin(rad) + y * math.cos(rad)
    else:
        xx = x
        yy = y        
    
    rotated_vector = np.array([xx, yy])
    # print("====== rotate ======")
    # print(f"original vector: \n{vector}\nrad: {rad}\n")
    # print(f"rotated vector: \n{rotated_vector}")
    # print("====================")
    return rotated_vector

def rotate_vector(vector, two_y_points, rad):
    x, y = vector
    left_y, right_y = two_y_points
    c, s = np.cos(rad), np.sin(rad)
    
    # y축에서 0이 더 높이 위치한 이미지상에서 left_y 가 right_y보다 더 높은 값을 갖는다면,
    # 더 낮게 위치하고 있으므로 clockwise로 진행 필요
    if left_y > right_y:
        j = np.matrix([
            [c, -s],
            [s, c]
        ])
        
    # 반대로 counter-clockwise
    elif left_y < right_y:
        j = np.matrix([
            [c, s],
            [-s, c]
        ])
    else:
        j = np.matrix([
            [1, 0],
            [0, 1]
        ])
        
    m = np.dot(j, [x, y])
    rotated_vector = np.array([m.T[0].item(), m.T[1].item()], dtype=float)

    # print("====== rotate ======")
    # print(f"original vector: \n{vector}\nrad: {rad}\n")
    # print(f"rotated vector: \n{rotated_vector}")
    # print("====================")
    
    return rotated_vector

def get_angle_between_two_vectors_from_origin(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    deg = np.degrees(rad)
    
    # apply minus multiplication if the image should be rotated clockwise
    if u[1] < v[1]:
        rad = -rad
        deg = -deg
    
    # print("--- get angle between two vectors ---")
    # print(f"u: \n{u}\nv: \n{v}")
    # print(f"rad: {rad}, deg: {deg}")
    # print("-------------------------------------")
    return rad, deg

def move_to_origin(matrix, x1, y1):
    matrix_copy = deepcopy(matrix)

    # 0,0 으로 origin(x1, y1) 옮기기
    matrix_copy[0, 0] -= x1
    matrix_copy[0, 1] -= x1
    matrix_copy[1, 0] -= y1
    matrix_copy[1, 1] -= y1
    # print("--- move to origin ---")
    # print(f"before: \n{matrix}\nafter: \n{matrix_copy}")
    # print("----------------------")
    return matrix_copy

def move_to_x_axis_with_origin_anchored(matrix):
    # x축으로 거리 유지하며 vector 옮기기
    vector_length = np.sqrt((matrix[0, 1] - matrix[0, 0]) ** 2 + (matrix[1, 1] - matrix[1, 0]) ** 2)
    matrix_on_x_axis = [
        [0, vector_length],
        [0, 0]
    ]

    matrix_on_x_axis = np.array(matrix_on_x_axis)

    # print("--- move to x axis ---")
    # print(f"length: {vector_length}\nafter: \n{matrix_on_x_axis}")
    # print("----------------------")
    
    return matrix_on_x_axis

def get_rotated_image_rivet_coords(img_arr):
    # 리벳 직경 좌표 구하기
    left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet = get_line_coords_via_corners(img_arr, RIVET, True)

    # 리벳 직경 좌표 매트릭스로 변형
    rivet_matrix = get_matrix(left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet)

    # origin으로 보낼 때 필요한 x축, y축 차이
    x_diff = rivet_matrix[0, 0]
    y_diff = rivet_matrix[1, 0]

    # origin으로 보내진 리벳 직경 매트릭스
    rivet_origin = move_to_origin(rivet_matrix, x_diff, y_diff)

    # origin을 기점으로 x축으로 내려온 리벳 직경 매트릭스
    rivet_on_x_axis = move_to_x_axis_with_origin_anchored(rivet_origin)

    # 두 벡터 간 각도 계산
    rad, deg = get_angle_between_two_vectors_from_origin(rivet_origin.T[1], rivet_on_x_axis.T[1])

    # rotate
    img = Image.fromarray(img_arr)
    img_rotated = img.rotate(deg)
    img_rotated_arr = np.array(img_rotated)
    
    # 리벳 직경 좌표 rotated된 이미지에서 구하기
    left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet = get_line_coords_via_corners(img_rotated_arr, RIVET, True)
    return img_rotated_arr, (left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet)

def get_thickness_of_plate(coords, img_arr, colour):
    result_coords = []
    result_lengths = []
    
    for x, y in coords:
        original_x = x
        original_y = y

        tmp_length = 0
        tmp_coords = (x, y)
        reached = False
        
        while not reached:
            
            # assume drawing a line needs to be done from top to bottom
            y += 1
            
            if all(img_arr[y, x, :] == colour):
                tmp_coords = (x, y)
            else:
                tmp_length = abs((y - 1) - original_y)
                
                if tmp_length > 0:
                    result_lengths.append(tmp_length)
                    result_coords.append(tmp_coords)
                else:
                    result_lengths.append(None)
                    result_coords.append(None)
                
                reached = True
    # if result_coords[0] is not None:
    #     for idx, (c, rc) in enumerate(zip(coords, result_coords)):
    #         if idx == 0:
    #             visualise(img_arr, [c[0], rc[0]], [c[1], rc[1]], fig_save_folder, True)
    #         else:
    #             visualise(img_arr, [c[0], rc[0]], [c[1], rc[1]], fig_save_folder, False)
    return result_coords, result_lengths

def calculate_weighted_mm_per_pixel(img_rotated_arr: np.array, 
                                    plates: list | tuple, 
                                    mms: list | tuple,
                                    weights: list | tuple=None,
                                    ):
    """
    This function assumes the elements for calculation put in as an argument
    are ordered where the rivet info is placed at first.
    Other elements don't need to be considered in the order.
    """
    weighted_mm_per_pixel = 0.0
    
    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    # 가중치가 주어지지 않은 경우 균등한 가중치 적용    
    # print(f"...weights received: {weights = }")
    if weights is None:
        value = 1 / len(mms)
        weights = [value] * len(mms)
        
        # to sum up to 1
        if sum(weights) != 1:
            weights[0] = 1 - sum(weights[1:])

        # print(f"...weights modified: {weights = }")

    for idx, plate_colour in enumerate(plates):        
        lx, ly, rx, ry = get_line_coords_via_corners(img_rotated_arr, plate_colour, True)
        left_xs.append(lx)
        left_ys.append(ly)
        right_xs.append(rx)
        right_ys.append(ry)

        weight = weights[idx]
        mm = mms[idx]
        
        # 리벳일 경우 직경을 통해 mm per pixel 산출
        if plate_colour == RIVET and idx == 0:
            pixel = abs(rx - lx)
        # 그 외의 경우 두께를 통해 mm per pixel 산출
        else:
            tmp_coords, tmp_lengths = get_thickness_of_plate([[lx, ly], [rx, ry]], img_rotated_arr, plate_colour)
            tmp_lengths = [x for x in tmp_lengths if x is not None]
            pixel = max(tmp_lengths)
        
        tmp_mm_per_pixel = mm / pixel
        # print(f"...{idx = } / {plate_colour = } / {tmp_mm_per_pixel = }")
        tmp_weighted_mm_per_pixel = tmp_mm_per_pixel * weight
        # print(f"...{idx = } / {plate_colour = } / {tmp_weighted_mm_per_pixel = }")
        weighted_mm_per_pixel += tmp_weighted_mm_per_pixel
    
    # print(f"...... {weighted_mm_per_pixel = }")
    return weighted_mm_per_pixel

def get_middle_coords(lx, ly, rx, ry):
    middle_x = int((lx + rx) / 2)
    middle_y = int((ly + ry) / 2)
    return middle_x, middle_y

def get_coords_from_centre_for_both_sides(diameter_mm, middle_x, weighted_mm_per_pixel):
    """
    This function assumes the aim is to get coords that are distanced
    from the center by a certain mm, and when trying to distance the coordinates,
    y-axis value doesn't need to be considered, because they are all aligned in parallel
    to x-axis
    """
    half_mm = diameter_mm / 2
    half_pixel = half_mm / weighted_mm_per_pixel

    left_x_further = int(middle_x - half_pixel)
    right_x_further = int(middle_x + half_pixel)
    return left_x_further, right_x_further

def decide_x_value_and_moving_direction(img_rotated_arr, x):
    towards_left = None
    if x < 0:
        x = 0
        towards_left = False
    elif x >= img_rotated_arr.shape[1]:
        x = img_rotated_arr.shape[1] - 1
        towards_left = True
    else:
        pass
    return x, towards_left

def get_colour_touching_coords_by_moving_on_x_axis(img_rotated_arr, x, y, colour, towards_left):
    if towards_left is not None:
        is_not_on_colour = any(img_rotated_arr[y, x, :] != colour)
        if towards_left:
            while is_not_on_colour:
                x -= 1
                try:
                    is_not_on_colour = any(img_rotated_arr[y, x, :] != colour)
                
                # 오른쪽에서부터 계속 1씩 빼왔는데 못 찾을 경우 그냥 맨 오른쪽 UPPER 코너 값으로 할당
                except:
                    left_x, left_y, right_x, right_y = get_line_coords_via_corners(img_rotated_arr, UPPER)
                    x = right_x
                    y = right_y
                    break
        else:
            while is_not_on_colour:
                x += 1
                try:
                    is_not_on_colour = any(img_rotated_arr[y, x, :] != colour)
                
                # 왼쪽에서부터 계속 1씩 더해왔는데 못 찾을 경우 그냥 맨 왼쪽 UPPER 코너 값으로 할당
                except IndexError:
                    left_x, left_y, right_x, right_y = get_line_coords_via_corners(img_rotated_arr, UPPER)
                    x = left_x
                    y = left_y
                    break
    return x, y    

def get_colour_touching_coords_by_moving_on_y_axis(img_rotated_arr, x, y, colour):
    is_hovering = any(img_rotated_arr[y, x, :] != colour)
    
    if is_hovering:
        while is_hovering:
            y += 1
            is_hovering = any(img_rotated_arr[y, x, :] != colour)
    else:
        is_on_colour = not is_hovering
        while is_on_colour:
            y -= 1
            is_on_colour = all(img_rotated_arr[y, x, :] == colour)
    return x, y

def create_base_image(path):

    # 이미지 어레이 가져오기
    img_arr = None
    if isinstance(path, str) or isinstance(path, Path):
        img_arr = get_image_array(path)
    elif isinstance(path, np.ndarray):
        img_arr = path
    elif isinstance(path, torch.Tensor):
        img_arr = path.numpy().transpose(1, 2, 0)
    
    # 이미지를 rivet 직경에 수평하게 rotate
    img_rotated_arr, (left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet) = get_rotated_image_rivet_coords(img_arr)
    return img_rotated_arr, (left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet)

def select_plates_for_mm_calculation(upper_type, 
                                     upper_thickness,
                                     middle_type, 
                                     middle_thickness,
                                     lower_type,
                                     lower_thickness):
    plates = [RIVET]
    mms = [RIVET_DIAMETER]
    
    if upper_type.lower().startswith("s"):
        plates.append(UPPER)
        mms.append(upper_thickness)
    
    if middle_type is not None and middle_type.lower().startswith("s"):
        plates.append(MIDDLE)
        mms.append(middle_thickness)
    
    if lower_type.lower().startswith("s"):
        plates.append(LOWER)
        mms.append(lower_thickness)
    
    plate_names = [list(COLOUR_NAMES.keys())[list(COLOUR_NAMES.values()).index(p)] for p in plates]
    # print(f"...selected: {plate_names = } / {mms = }")
    return plates, mms

def get_head_height(img_rotated_arr, 
                    lx_rivet, 
                    ly_rivet, 
                    rx_rivet, 
                    ry_rivet, 
                    diameter_mm, 
                    weighted_mm_per_pixel,
                    fig_save_folder=None):    
    
    # 리벳 중간점 구하기
    middle_x, middle_y = get_middle_coords(lx_rivet, ly_rivet, rx_rivet, ry_rivet)
    if fig_save_folder:
        visualise(img_rotated_arr, [middle_x], [middle_y], fig_save_folder, True)
    
    # 리벳에서 양쪽으로 떨어진 두 점 구하기
    lx_further, rx_further = get_coords_from_centre_for_both_sides(diameter_mm, middle_x, weighted_mm_per_pixel)
    if fig_save_folder:
        visualise(img_rotated_arr, [lx_further, rx_further], [middle_y, middle_y], fig_save_folder, True)

    # 리벳 양쪽 두 점이 upper와 만나는 점 구하기
    # -- 1) x좌표 먼저 구하기
    lx_further, towards_left = decide_x_value_and_moving_direction(img_rotated_arr, lx_further)
    lx_touching, middle_y = get_colour_touching_coords_by_moving_on_x_axis(img_rotated_arr, lx_further, middle_y, UPPER, towards_left)

    rx_further, towards_left = decide_x_value_and_moving_direction(img_rotated_arr, rx_further)
    rx_touching, middle_y = get_colour_touching_coords_by_moving_on_x_axis(img_rotated_arr, rx_further, middle_y, UPPER, towards_left)
    
    # -- 2) y좌표 구하기
    lx_touching, ly_touching = get_colour_touching_coords_by_moving_on_y_axis(img_rotated_arr, lx_touching, middle_y, UPPER)
    rx_touching, ry_touching = get_colour_touching_coords_by_moving_on_y_axis(img_rotated_arr, rx_touching, middle_y, UPPER)
    if fig_save_folder:
        visualise(img_rotated_arr, [lx_touching, rx_touching], [ly_touching, ry_touching], fig_save_folder, True)
    
    # upper 위 두 점의 중간점 구하기
    upper_middle_x_on_line, upper_middle_y_on_line = get_middle_coords(lx_touching, ly_touching, rx_touching, ry_touching)
    if fig_save_folder:
        visualise(img_rotated_arr, [upper_middle_x_on_line], [upper_middle_y_on_line], fig_save_folder, False)

    # upper 기준선 위 중간점이 리벳에 닿는 점 구하기
    rivet_x_touching, rivet_y_touching = get_colour_touching_coords_by_moving_on_y_axis(img_rotated_arr, upper_middle_x_on_line, upper_middle_y_on_line, RIVET)
    if fig_save_folder:
        visualise(img_rotated_arr, [upper_middle_x_on_line, rivet_x_touching], [upper_middle_y_on_line, rivet_y_touching], fig_save_folder, False, True)
    
    # 계산
    pixel_length = (upper_middle_y_on_line - rivet_y_touching)
    # print(f"...{pixel_length = }")
    head_height = pixel_length * weighted_mm_per_pixel
    # print(head_height)
    return head_height


def get_perp_coords(left_x, left_y, right_x, right_y, length, divider):
    x_length = right_x - left_x
    y_length = right_y - left_y

    mag = math.sqrt(x_length**2 + y_length**2)    
    x_length_norm = x_length / mag
    y_length_norm = y_length / mag
    
    # middle point
    middle_x = round((left_x + right_x) / 2)
    middle_y = round((left_y + right_y) / 2)
    
    # switch
    temp = x_length_norm
    x_length_norm = -1 * y_length_norm
    y_length_norm = temp
    
    # get coords
    ratio = 1 / divider
    new_x_1 = middle_x + x_length_norm * length * ratio
    new_y_1 = middle_y + y_length_norm * length * ratio
    new_x_2 = middle_x - x_length_norm * length * (1 - ratio)
    new_y_2 = middle_y - y_length_norm * length * (1 - ratio)
    
    # if coords goes beyond the original coordinate system
    if new_x_1 < 0 or new_y_1 < 0 or new_x_2 < 0 or new_y_2 < 0:
        ratio = 1 - ratio
        new_x_1 = middle_x + x_length_norm * length * ratio
        new_y_1 = middle_y + y_length_norm * length * ratio
        new_x_2 = middle_x - x_length_norm * length * (1 - ratio)
        new_y_2 = middle_y - y_length_norm * length * (1 - ratio)
        
    return int(new_x_1), int(new_y_1), int(new_x_2), int(new_y_2)

def count_image_files(folder_name):
    folder = Path(folder_name)
    image_files = list(folder.glob("*.png"))
    image_count = len(image_files)
    # print(f"... current {image_count =}")
    return image_count

def get_zfill_index(image_count):
    return f"{image_count + 1}".zfill(3)

def visualise(img_rotated_arr: np.array, 
              xs: list | tuple, 
              ys: list | tuple,
              fig_save_folder: str,
              is_new_figure: bool=True,
              is_no_dot=False) -> None:
    # 시각화
    if is_new_figure:
        plt.figure()
    plt.imshow(img_rotated_arr)    
    plt.plot(xs, ys, "w" if is_no_dot else "wo", linestyle="--")
    
    # get index
    idx = get_zfill_index(count_image_files(fig_save_folder))
    plt.savefig(f"{str(fig_save_folder)}/{idx}.png")
    
    # print(f"...fig saved with {idx} number")

def crop_image(image, ratio=0.8):
    y_length = int(image.shape[0] * ratio)
    x_length = int(image.shape[1] * ratio)

    y_margin = int((image.shape[0] - y_length) / 2)
    x_margin = int((image.shape[1] - x_length) / 2)

    return image[y_margin:y_margin+y_length, x_margin:x_margin+x_length], x_margin, y_margin

def get_edges(image, colour):
    colour_extracted_grey = np.where(np.all(image == colour, axis=-1), 255, 0).astype(np.uint8)
    edges = skimage.feature.canny(image=colour_extracted_grey)
    return edges

def get_distance(x1, x2, y1, y2):
    return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))

def get_minimum_thickness_at_bottom(img_rotated_arr, weighted_mm_per_pixel, fig_save_folder=None):
    
    # crop the center. ratio = 0.8, by default
    img_rotated_cropped, x_margin, y_margin = crop_image(img_rotated_arr)
    
    left_x_lower, left_y_lower, _, _ = get_line_coords_via_corners(img_rotated_cropped, LOWER, False)
    
    # get edges    
    colour = LOWER
    edges = get_edges(img_rotated_cropped, colour)

    if fig_save_folder:
        visualise(edges, [], [], fig_save_folder)

    # edge's coords
    edge_ys, edge_xs = np.where(edges)
    edge_points = set(zip(edge_xs, edge_ys))

    # get bottom line in the edges
    boundary = 2
    bottom_points = set()
    prev_added_bottom_points = set()
    prev_added_bottom_points.add((left_x_lower, left_y_lower))
    
    while len(prev_added_bottom_points) != 0:

        # update previously added points
        bottom_points |= prev_added_bottom_points

        # temporarily store the data
        tmp_points = deepcopy(prev_added_bottom_points)

        # initialisation
        prev_added_bottom_points = set()
        
        for target_x, target_y in tmp_points:
            for sample_x, sample_y in edge_points:
                distance = get_distance(target_x, sample_x, target_y, sample_y)
                if distance <= boundary:
                    prev_added_bottom_points.add((sample_x, sample_y))
            
            edge_points -= prev_added_bottom_points
    
    bottom_xs = [coords[0] for coords in list(bottom_points)]
    bottom_ys = [coords[1] for coords in list(bottom_points)]
    
    # if the max(x) doesn't reach almost the end of x-axis, it means broken
    broken_margin = 10
    if max(bottom_xs) < img_rotated_cropped.shape[1] - broken_margin:
        return -1

    if fig_save_folder:
        upper_xs = [coords[0] for coords in list(edge_points)]
        upper_ys = [coords[1] for coords in list(edge_points)]
        
        tmp_canvas = np.zeros_like(edges)
        tmp_canvas[upper_ys, upper_xs] = 1
        visualise(tmp_canvas, [], [], fig_save_folder)        
        
        tmp_canvas = np.zeros_like(edges)
        tmp_canvas[bottom_ys, bottom_xs] = 1
        visualise(tmp_canvas, [], [], fig_save_folder)

    # get the points with the minumum distance
    pixel_distance_min = 100_000
    point_tuple = tuple()
    for lx, ly in bottom_points:
        for ux, uy in edge_points:
            distance = np.sqrt(((lx - ux) ** 2 + (ly - uy) ** 2))
            if distance < pixel_distance_min:
                point_tuple = (lx, ly, ux, uy)
                pixel_distance_min = distance
    
    # get the coords in the original image
    lx_in_full = point_tuple[0] + x_margin
    ly_in_full = point_tuple[1] + y_margin
    ux_in_full = point_tuple[2] + x_margin
    uy_in_full = point_tuple[3] + y_margin
    bottom_thickness = weighted_mm_per_pixel * pixel_distance_min

    if fig_save_folder:
        visualise(img_rotated_arr, [lx_in_full, ux_in_full], [ly_in_full, uy_in_full], fig_save_folder)
    
    return bottom_thickness

def get_interlock(img_rotated_arr, weighted_mm_per_pixel, fig_save_folder=None):
    # crop the center. ratio = 0.8, by default
    img_cropped, x_margin, y_margin = crop_image(img_rotated_arr)

    # get rivet edges    
    colour = RIVET
    rivet_edges = get_edges(img_cropped, colour)

    # get corners of rivet close to the bottom
    left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet = get_line_coords_via_corners(img_cropped, colour, False)
    
    if fig_save_folder:
        visualise(rivet_edges, [left_x_rivet, right_x_rivet], [left_y_rivet, right_y_rivet], fig_save_folder)
    
    # rivet edge's coords
    rivet_edge_ys, rivet_edge_xs = np.where(rivet_edges)

    # get the centre of rivet
    x_length = rivet_edge_xs.max() - rivet_edge_xs.min()
    y_length = rivet_edge_ys.max() - rivet_edge_ys.min()
    x_centre = rivet_edge_xs.min() + x_length // 2
    y_centre = rivet_edge_ys.min() + y_length // 2

    if fig_save_folder:
        visualise(rivet_edges, [x_centre], [y_centre], fig_save_folder)

    # # get lower quadrants
    # quadrant_3 = edges[y_centre:, :x_centre]
    # quadrant_4 = edges[y_centre:, x_centre:]
    # plt.imshow(quadrant_3)
    # plt.show()

    # plt.imshow(quadrant_4)
    # plt.show()

    # get lower quadrants
    quadrant_3_cond = (rivet_edge_xs < x_centre) & (rivet_edge_ys > y_centre)
    quadrant_4_cond = (rivet_edge_xs > x_centre) & (rivet_edge_ys > y_centre)

    # get the furthest point in the quadrant 3
    quadrant_3_x_far = rivet_edge_xs[quadrant_3_cond].min()
    quadrant_3_min_mask = rivet_edge_xs[quadrant_3_cond] == quadrant_3_x_far
    quadrant_3_y_far = rivet_edge_ys[quadrant_3_cond][quadrant_3_min_mask].max()

    if fig_save_folder:
        visualise(rivet_edges, [quadrant_3_x_far], [quadrant_3_y_far], fig_save_folder)

    # get the furthest point in the quadrant 4
    quadrant_4_x_far = rivet_edge_xs[quadrant_4_cond].max()
    quadrant_4_max_mask = rivet_edge_xs[quadrant_4_cond] == quadrant_4_x_far
    quadrant_4_y_far = rivet_edge_ys[quadrant_4_cond][quadrant_4_max_mask].max()

    if fig_save_folder:
        visualise(rivet_edges, [quadrant_4_x_far], [quadrant_4_y_far], fig_save_folder)
        visualise(
            img_rotated_arr, 
            [quadrant_3_x_far+x_margin, quadrant_4_x_far+x_margin], 
            [quadrant_3_y_far+y_margin, quadrant_4_y_far+y_margin], 
            fig_save_folder
        )
    
    # get lower plate edges    
    colour = LOWER
    lower_edges = get_edges(img_cropped, colour)

    # lower plate edge's coords
    lower_edge_ys, lower_edge_xs = np.where(lower_edges)

    # get corners of lower plate close to the top
    left_x_lower, left_y_lower, right_x_lower, right_y_lower = get_line_coords_via_corners(img_cropped, colour, True)

    # get the most distant points from the corners
    # by finding the closest, but the most distant point gradually
    boundary = 2
    far_lower_points = []

    # get the points twice from the left and the right
    for standard_x, standard_y in zip([left_x_lower, right_x_lower], [left_y_lower, right_y_lower]):
        # lower plate edge's coords
        edge_points = set(zip(lower_edge_xs, lower_edge_ys))

        just_added_points = set()
        just_added_points.add((standard_x, standard_y))
        max_distance = -1
        point_x = None
        point_y = None
        
        # if any newly found points
        while just_added_points:
            
            # get the furthest from the newly found points
            for sample_x, sample_y in just_added_points:
                distance = get_distance(standard_x, sample_x, standard_y, sample_y)
                if distance > max_distance:
                    point_x = sample_x
                    point_y = sample_y
                    max_distance = distance

            # initialisation before adding
            just_added_points = set()

            # get the points within the boundary
            for edge_x, edge_y in edge_points:
                distance = get_distance(edge_x, point_x, edge_y, point_y)
                if distance <= boundary:
                    just_added_points.add((edge_x, edge_y))
            
            # remove detected points from the original set
            edge_points -= just_added_points

        far_lower_points.extend([point_x, point_y])
    
    left_lower_far_x = far_lower_points[0]
    left_lower_far_y = far_lower_points[1]
    right_lower_far_x = far_lower_points[2]
    right_lower_far_y = far_lower_points[3]
    
    if fig_save_folder:
        visualise(lower_edges, [], [], fig_save_folder)
        visualise(img_cropped, [left_lower_far_x], [left_lower_far_y], fig_save_folder)
        visualise(img_cropped, [right_lower_far_x], [right_lower_far_y], fig_save_folder)
        visualise(
            img_rotated_arr,
            [left_lower_far_x+x_margin, right_lower_far_x+x_margin],
            [left_lower_far_y+y_margin, right_lower_far_y+y_margin],
            fig_save_folder
        )
        visualise(
            img_rotated_arr,
            [quadrant_3_x_far+x_margin, quadrant_4_x_far+x_margin], 
            [quadrant_3_y_far+y_margin, quadrant_4_y_far+y_margin],
            fig_save_folder
        )
        visualise(
            img_rotated_arr,
            [left_lower_far_x+x_margin, right_lower_far_x+x_margin], 
            [left_lower_far_y+y_margin, right_lower_far_y+y_margin],
            fig_save_folder
        )

    # if interlock didn't occur, give it -1    
    left_interlock = (
        (left_lower_far_x - quadrant_3_x_far) * weighted_mm_per_pixel 
        if left_lower_far_x > quadrant_3_x_far 
        else -1
    )
    right_interlock = (
        (quadrant_4_x_far - right_lower_far_x) * weighted_mm_per_pixel 
        if quadrant_4_x_far > right_lower_far_x 
        else -1
    )
    return left_interlock, right_interlock

def get_ms_metrics(
    path,
    upper_type,
    upper_thickness,
    middle_type,
    middle_thickness,
    lower_type,
    lower_thickness,
    weights,
    head_diameter,
    to_save_fig=False
    ):

    fig_save_folder = None
    if to_save_fig:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_save_folder = Path(f"figs/{timestamp}")
        if not fig_save_folder.exists():
            fig_save_folder.mkdir(parents=True)
    
    img_rotated_arr, (left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet) = create_base_image(path)
    plates, mms = select_plates_for_mm_calculation(
        upper_type,
        upper_thickness,
        middle_type,
        middle_thickness,
        lower_type,
        lower_thickness
    )
    weighted_mm_per_pixel = calculate_weighted_mm_per_pixel(
        img_rotated_arr,
        plates,
        mms,
        weights,
    )
    head_height = get_head_height(
        img_rotated_arr, 
        left_x_rivet, 
        left_y_rivet, 
        right_x_rivet, 
        right_y_rivet,
        head_diameter,
        weighted_mm_per_pixel,
        fig_save_folder
    )
    
    bottom_thickness = get_minimum_thickness_at_bottom(
        img_rotated_arr,
        weighted_mm_per_pixel,
        fig_save_folder
    )
    
    left_interlock, right_interlock = get_interlock(
        img_rotated_arr,
        weighted_mm_per_pixel,
        fig_save_folder
    )
    
    return head_height, bottom_thickness, left_interlock, right_interlock