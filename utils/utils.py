import os
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use('TkAgg')

original_contour_idx = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78,
                        80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148,
                        149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185,
                        191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296,
                        297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362,
                        365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398,
                        400, 402, 405, 409, 415, 454, 466]

original_contour_connections = [(72, 124), (60, 45), (8, 0), (27, 5), (91, 94), (80, 96), (114, 113), (1, 57), (7, 66),
                                (5, 89), (104, 109), (68, 117), (118, 103), (70, 71), (81, 77), (117, 99), (59, 39),
                                (7, 1), (106, 47), (52, 51), (123, 92), (14, 33), (69, 127), (94, 85), (21, 36),
                                (124, 79), (51, 59), (126, 93), (116, 115), (23, 64), (44, 60), (87, 86), (9, 8),
                                (67, 118), (41, 42), (122, 91), (24, 25), (86, 125), (48, 49), (112, 120), (119, 102),
                                (65, 37), (33, 21), (74, 81), (98, 83), (107, 121), (37, 56), (92, 105), (105, 79),
                                (90, 122), (25, 26), (49, 50), (30, 62), (96, 82), (83, 95), (71, 72), (46, 40),
                                (36, 2), (99, 126), (16, 38), (88, 87), (47, 44), (125, 85), (55, 54), (82, 97),
                                (19, 15), (17, 43), (23, 32), (109, 110), (120, 101), (100, 78), (66, 55), (56, 6),
                                (0, 70), (26, 3), (38, 31), (89, 123), (2, 98), (61, 28), (115, 114), (110, 111),
                                (22, 18), (17, 63), (4, 90), (34, 20), (84, 80), (111, 101), (29, 61), (63, 10),
                                (11, 13), (76, 68), (121, 106), (40, 58), (93, 100), (3, 88), (6, 14), (58, 16),
                                (103, 104), (57, 41), (73, 75), (13, 12), (102, 108), (108, 107), (43, 30), (69, 67),
                                (75, 74), (28, 4), (42, 48), (50, 39), (31, 65), (20, 35), (32, 29), (53, 52),
                                (12, 19), (95, 76), (10, 9), (64, 24), (18, 34), (62, 27), (127, 116), (45, 46),
                                (78, 119), (54, 53), (113, 112)]

jaw_idx = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

left_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
right_eye_idx = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]

left_eyebrow_idx = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
right_eyebrow_idx = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]

contour_idx = sorted(jaw_idx + lips_idx + left_eyebrow_idx + right_eyebrow_idx + left_eye_idx + right_eye_idx)

contour_connections = [(63, 109), (52, 38), (6, 0), (23, 4), (80, 83), (70, 84), (100, 99), (1, 49), (5, 58), (4, 78),
                       (90, 95), (103, 89), (61, 62), (71, 67), (51, 32), (5, 1), (92, 40), (45, 44), (108, 81),
                       (60, 112), (83, 74), (109, 69), (44, 51), (111, 82), (102, 101), (19, 56), (37, 52), (76, 75),
                       (7, 6), (59, 103), (34, 35), (107, 80), (20, 21), (75, 110), (41, 42), (98, 105), (104, 88),
                       (65, 71), (93, 106), (81, 91), (91, 69), (79, 107), (21, 22), (42, 43), (26, 54), (84, 72),
                       (62, 63), (39, 33), (13, 31), (77, 76), (40, 37), (110, 74), (48, 47), (72, 85), (16, 12),
                       (14, 36), (19, 28), (95, 96), (105, 87), (86, 68), (58, 48), (0, 61), (22, 2), (31, 27),
                       (78, 108), (53, 24), (101, 100), (96, 97), (18, 15), (14, 55), (3, 79), (29, 17), (73, 70),
                       (97, 87), (25, 53), (55, 8), (9, 11), (106, 92), (33, 50), (82, 86), (2, 77), (50, 13),
                       (89, 90), (49, 34), (64, 66), (11, 10), (88, 94), (94, 93), (36, 26), (60, 59), (66, 65),
                       (24, 3), (35, 41), (43, 32), (27, 57), (17, 30), (28, 25), (46, 45), (10, 16), (8, 7), (56, 20),
                       (15, 29), (54, 23), (112, 102), (38, 39), (68, 104), (47, 46), (99, 98)]

mesh2contour = {0: 0, 7: 1, 13: 2, 14: 3, 17: 4, 33: 5, 37: 6, 39: 7, 40: 8, 46: 9, 52: 10, 53: 11, 55: 12, 58: 13,
                61: 14, 63: 15, 65: 16, 66: 17, 70: 18, 78: 19, 80: 20, 81: 21, 82: 22, 84: 23, 87: 24, 88: 25, 91: 26,
                93: 27, 95: 28, 105: 29, 107: 30, 132: 31, 133: 32, 136: 33, 144: 34, 145: 35, 146: 36, 148: 37,
                149: 38, 150: 39, 152: 40, 153: 41, 154: 42, 155: 43, 157: 44, 158: 45, 159: 46, 160: 47, 161: 48,
                163: 49, 172: 50, 173: 51, 176: 52, 178: 53, 181: 54, 185: 55, 191: 56, 234: 57, 246: 58, 249: 59,
                263: 60, 267: 61, 269: 62, 270: 63, 276: 64, 282: 65, 283: 66, 285: 67, 288: 68, 291: 69, 293: 70,
                295: 71, 296: 72, 300: 73, 308: 74, 310: 75, 311: 76, 312: 77, 314: 78, 317: 79, 318: 80, 321: 81,
                323: 82, 324: 83, 334: 84, 336: 85, 361: 86, 362: 87, 365: 88, 373: 89, 374: 90, 375: 91, 377: 92,
                378: 93, 379: 94, 380: 95, 381: 96, 382: 97, 384: 98, 385: 99, 386: 100, 387: 101, 388: 102, 390: 103,
                397: 104, 398: 105, 400: 106, 402: 107, 405: 108, 409: 109, 415: 110, 454: 111, 466: 112}

def get_contour_idx_and_connections():
    """
    Retrieve the MediaPipe Contour connections and reorder the index
    :return: All contour node indices and their connections after reordering
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    connections = mp_face_mesh.FACEMESH_CONTOURS
    contour = []
    connections = list(connections)
    for c in connections:
        contour.extend(c)
    contour_idx = sorted(list(set(contour)))
    idx_mesh2contour = {i: contour_idx.index(i) for i in contour_idx}
    contour_connections = [(idx_mesh2contour[i], idx_mesh2contour[j]) for i, j in connections]
    return contour_idx, contour_connections


def get_key_contour_connections():
    """
    Get a subset of contour connection from the original MediaPipe Contour connections and reorder the index
    :return: All contour node indices and their connections after reordering
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    connections = mp_face_mesh.FACEMESH_CONTOURS
    connections = list(connections)
    idx_mesh2contour = {i: contour_idx.index(i) for i in contour_idx}
    contour_connections = [(idx_mesh2contour[i], idx_mesh2contour[j]) for i, j in connections
                           if i in idx_mesh2contour and j in idx_mesh2contour]
    print(idx_mesh2contour)
    return contour_connections


def area_of_polygon(x, y, use_torch):
    """
    Numpy implementation of shoelace formula that computes area of polygon given x, y coordinates
    All points must be arranged in clockwise/counter-clockwise manner
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    :param x: x coordinates of all points
    :param y: y coordinates of all points
    :return: area of polygon
    """
    if use_torch:
        xs = torch.bmm(x[:, None, :], torch.roll(y, 1, dims=1)[:, :, None]).flatten()
        ys = torch.bmm(y[:, None, :], torch.roll(x, 1, dims=1)[:, :, None]).flatten()
        return 0.5 * torch.abs(xs - ys)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_inner_lip_area_torch(face):
    """
    Calculate inner lip area
    :param face: x, y coordinates of a frame of face contour landmark
    :return: inner_lip_area of the face
    """
    contour_inner_lips_idx = [mesh2contour[i] for i in inner_lips_idx]
    inner_lip_coord = face[:, :, contour_inner_lips_idx]
    inner_lip_area = area_of_polygon(inner_lip_coord[:, 0], inner_lip_coord[:, 1], use_torch=True)
    return inner_lip_area


def get_inner_lip_area(face):
    """
    Calculate inner lip area
    :param face: x, y coordinates of a frame of face contour landmark
    :return: inner_lip_area of the face
    """
    contour_inner_lips_idx = [mesh2contour[i] for i in inner_lips_idx]
    inner_lip_coord = face[:, contour_inner_lips_idx]
    inner_lip_area = area_of_polygon(inner_lip_coord[0], inner_lip_coord[1], use_torch=False)
    return inner_lip_area


def get_eye_area(face):
    """
    Calculate eye area
    :param face: x, y coordinates of a frame of face contour landmark
    :return: eye area of the face
    """
    contour_left_eye_idx = [mesh2contour[i] for i in left_eye_idx]
    contour_right_eye_idx = [mesh2contour[i] for i in right_eye_idx]
    left_eye_coord = face[:, contour_left_eye_idx]
    right_eye_coord = face[:, contour_right_eye_idx]
    left_eye_area = area_of_polygon(left_eye_coord[0], left_eye_coord[1], use_torch=False)
    right_eye_area = area_of_polygon(right_eye_coord[0], right_eye_coord[1], use_torch=False)
    return left_eye_area + right_eye_area


def get_eye_area_torch(face):
    """
    Calculate eye area
    :param face: x, y coordinates of a frame of face contour landmark
    :return: eye area of the face
    """
    contour_left_eye_idx = [mesh2contour[i] for i in left_eye_idx]
    contour_right_eye_idx = [mesh2contour[i] for i in right_eye_idx]
    left_eye_coord = face[:, :, contour_left_eye_idx]
    right_eye_coord = face[:, :, contour_right_eye_idx]
    left_eye_area = area_of_polygon(left_eye_coord[:, 0], left_eye_coord[:, 1], use_torch=True)
    right_eye_area = area_of_polygon(right_eye_coord[:, 0], right_eye_coord[:, 1], use_torch=True)
    return left_eye_area + right_eye_area


def first_order_difference(face, cuda=True):
    """
    Computer first order difference over time (velocity)
    :param cuda: torch version
    :param face: face landmarks
    :return: velocity
    """
    if cuda:
        return torch.diff(face, dim=0)
    return np.diff(face, axis=0)


def get_lips_and_jaw(face):
    """
    Get lips and jaw related landmarks by their index
    :param face: all face landmark
    :return: lips and jaw landmark on the input face
    """
    mesh_target_idx = lips_idx + jaw_idx
    contour_target_idx = [mesh2contour[i] for i in mesh_target_idx]
    return face[:, :, contour_target_idx]


def batched_average_euclidean_distance(pred, gold, lengths):
    """
    Computer euclidean distance of two sequence averaged over time and batch size
    :param pred: pred sequence
    :param gold: gold sequence
    :param lengths: sequence length
    :return: batched euclidean distance
    """
    distances = np.sqrt((pred - gold) ** 2).sum(axis=(1, 2))
    start_idx = 0
    sum_distances = 0
    for l in lengths:
        sum_distances += np.sum(distances[start_idx:start_idx + l]) / l
        start_idx += l
    return sum_distances / len(lengths)


def batched_average_euclidean_distance_torch(pred, gold, lengths):
    """
    Computer euclidean distance of two sequence averaged over time and batch size, torch version
    :param pred: pred sequence
    :param gold: gold sequence
    :param lengths: sequence length
    :return: batched euclidean distance
    """
    distances = torch.sqrt((pred - gold) ** 2).sum(dim=(1, 2))
    start_idx = 0
    sum_distances = 0
    for l in lengths:
        sum_distances += torch.sum(distances[start_idx:start_idx + l]) / l
        start_idx += l
    return sum_distances / len(lengths)


def batched_average_area_diff(pred, gold, lengths, region='mouth'):
    """
    Compute l1 area difference of two sequence averaged over time and batch size of a particular region
    :param pred: pred sequence
    :param gold: gold sequence
    :param lengths: sequence length
    :param region: the region to compute area, only accept 'mouth' or 'eye'
    :return: batched euclidean distance
    """
    assert region == 'mouth' or region == 'eye', "area region should be either 'mouth' or 'eye'"
    area_function = get_inner_lip_area if region == 'mouth' else get_eye_area
    pred_area = np.array([area_function(f) for f in pred])
    gold_area = np.array([area_function(f) for f in gold])
    area_diff = pred_area - gold_area
    start_idx = 0
    sum_distances = 0
    for l in lengths:
        sum_distances += np.sum(np.abs(area_diff[start_idx:start_idx + l])) / l
        start_idx += l
    return sum_distances / len(lengths)


def batched_average_area_diff_torch(pred, gold, lengths, region='mouth'):
    """
    Compute l1 area difference of two sequence averaged over time and batch size of a particular region
    :param pred: pred sequence
    :param gold: gold sequence
    :param lengths: sequence length
    :param region: the region to compute area, only accept 'mouth' or 'eye'
    :return: batched euclidean distance
    """
    assert region == 'mouth' or region == 'eye', "area region should be either 'mouth' or 'eye'"
    if region == 'mouth':
        pred_area = get_inner_lip_area_torch(pred)
        gold_area = get_inner_lip_area_torch(gold)
    else:
        pred_area = get_eye_area_torch(pred)
        gold_area = get_eye_area_torch(gold)
    area_diff = pred_area - gold_area
    start_idx = 0
    sum_distances = 0
    for l in lengths:
        sum_distances += torch.sum(torch.abs(area_diff[start_idx:start_idx + l])) / l
        start_idx += l
    return sum_distances / len(lengths)


def smooth_landmarks(faces, w):
    """
    Compute moving average with width w along the time dimension
    :param faces: facial landmarks, time x dimension x num_of_keypoints
    :param w: width of the moving average
    :return: smoothed landmarks
    """
    faces = np.concatenate((np.repeat(np.expand_dims(faces[0], 0), w - 1, axis=0), faces), axis=0)
    ret = np.cumsum(faces, axis=0)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:] / w


def get_displaced_ref_contour(displace, static_face):
    """
    Get contour displaced on a 3d static face, default the mean face
    :param displace: displacements
    :param static_face: static face that is going get displaced from
    :return: the displaced landmarks
    """
    ref_contour = face_3d_to_2d(mash_to_contour(static_face))
    displaced_contour = displace + ref_contour
    return displaced_contour


def get_displaced_static_contour(displace, static_face, consecutive_displace=False):
    """
    Get contour displaced on a 2d static face
    :param displace: displacements
    :param static_face: static face that is going get displaced from, the first frame face
    :param consecutive_displace: True: displacement wrt the previous frame, False: wrt the first frame
    :return: the displaced landmarks
    """
    if consecutive_displace:
        faces = []
        curr_face = static_face
        for d in displace:
            faces.append(curr_face)
            curr_face += d
        return np.stack(faces)
    else:
        displaced_contour = displace + static_face
        return displaced_contour


def face_3d_to_2d(landmark, seq=False):
    """
    Turn the metric 3d face to 2d
    :param landmark: metric 3d landmark
    :param seq: whether input is a sequence
    :return: 2d landmark
    """
    if seq:
        return landmark[:, :2, :]
    return landmark[:2, :]


def mash_to_contour(landmark, seq=False):
    """
    Given face mash landmarks (468 points), reduce to contour only (128 points)
    :param landmark: face mesh landmarks
    :param seq: whether input is a sequence
    :return: contour landmarks
    """
    if seq:
        return landmark[:, :, contour_idx]
    return landmark[:, contour_idx]


def plot_2d_face(landmark, title=None, save=None):
    """
    Plot 2d landmarks
    :param landmark: 2d face landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(landmark[0], landmark[1])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


def plot_2d_contour(landmark, title=None, save=None):
    """
    Given 2d contour landmarks, plot 2d landmarks and contour connections
    :param landmark: 2d contour landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(landmark[0], landmark[1], c='tab:blue', zorder=10)
    for c in contour_connections:
        point_1 = landmark[:, c[0]]
        point_2 = landmark[:, c[1]]
        plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:orange', zorder=0)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


def plot_2d_face_depth(landmark, title=None, save=None):
    """
    A 2d scatter plot of 3d landmarks, opacity represents z-axis (depth)
    :param landmark: 3d landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig = plt.figure(figsize=(6, 6))
    depth = landmark[2]
    depth = 0.2 + 0.8 * ((depth - depth.min()) / (depth.max() - depth.min()))
    plt.scatter(landmark[0], landmark[1], alpha=depth)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


def plot_3d_face(landmark, title=None, save=None):
    """
    plot 3d landmarks on 3d axes
    :param landmark: 3d landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.view_init(azim=-90, elev=90)
    ax.scatter3D(
        xs=[landmark[0]],
        ys=[landmark[1]],
        zs=[landmark[2]],
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


def save_images_to_video(source, filename, cwd_dir, audio=None):
    source_path = os.path.join(source, '%d.png')
    if audio is None:
        p = subprocess.Popen(
            ['ffmpeg', '-r', '25', '-f', 'image2', '-i', source_path, '-vcodec', 'libx264', '-crf', '25',
             '-pix_fmt', 'yuv420p', filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            cwd=cwd_dir)
    else:
        p = subprocess.Popen(
            ['ffmpeg', '-r', '25', '-f', 'image2', '-i', source_path, '-i', audio, '-vcodec', 'libx264', '-crf', '25',
             '-pix_fmt', 'yuv420p', filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            cwd=cwd_dir)
    p.wait()


if __name__ == '__main__':
    face_3d = np.load('data\\ref_face.npy')
    # # # plot_2d_face_depth(face_3d, title='3d plot')
    face_3d_contour = mash_to_contour(face_3d)
    face_2d_contour = face_3d_to_2d(face_3d_contour)
    # inner_lip_area(face_2d_contour)
    plot_2d_contour(face_2d_contour, title='2d contour')
    # # plot_2d_face_depth(face_3d)
    # contour_idx, contour_connection = get_contour_idx_and_connections()
    # contour_connection = get_key_contour_connections()
    # print(contour_connection)
    # print(contour_connection)
    # face_3d_contour = mash_to_contour(face_3d)
    # face_2d_contour = face_3d_to_2d(face_3d_contour)
    # plot_2d_contour(face_2d_contour)
    # plot_2d_face_depth(face_3d_contour)

    # lm = face_3d_to_2d(mash_to_contour(np.load('data\\landmarks\\id00071___0JDAt0VQxT4___00008.npy'), seq=True),
    #                    seq=True)
    # import os

    # os.mkdir('id00071')
    # for idx, l in enumerate(lm):
    #     lip_area = get_inner_lip_area(l)
    #     eye_area = get_eye_area(l)
    #     plot_2d_contour(l, title=f"Frame {idx} Lip Area {lip_area:.4f} Eye Area {eye_area:.4f}",
    #                     save=f'id00348\\{idx}.png')
