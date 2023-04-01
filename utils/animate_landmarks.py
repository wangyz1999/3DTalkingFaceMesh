import os

import cv2
import numpy as np

from drawing_utils import draw_landmarks, rotation_matrix, do_nothing


if __name__ == '__main__':
    file_id = '0tEHBVzZY4E.npy'
    landmark_path = os.path.join('..\\data\\landmarks', file_id)
    valid_path = os.path.join('..\\data\\validity', file_id)
    mean_face = np.load('..\\data\\slient_mean_face_0.0025.npy')
    face_landmarks = np.load(landmark_path)
    validity = np.load(valid_path)
    face_landmarks[~validity] = mean_face


    face_landmarks = face_landmarks.transpose((0, 2, 1))
    face_landmarks = face_landmarks / 22 + 0.5
    face_landmarks[..., 1] = 1 - face_landmarks[..., 1]
    face_landmarks[..., 1] -= 0.05

    cv2.namedWindow(file_id)
    cv2.createTrackbar("vert", file_id, 180, 360, do_nothing)
    cv2.createTrackbar("hori", file_id, 180, 360, do_nothing)

    audio_options = {
        'loop': 0,
        'framedrop': True,
        # 'sync': 'video',
    }
    fps = 25
    sleep_ms = int(np.round((1 / fps) * 1000))

    idx = 0
    while True:
        landmark = face_landmarks[idx]
        image = np.zeros((600, 600, 3), dtype="uint8")

        # freely rotate landmarks
        rot_x = cv2.getTrackbarPos('vert', file_id)
        rot_y = cv2.getTrackbarPos('hori', file_id)

        rot_x_rad = (rot_x - 180) * np.pi / 180
        rot_y_rad = (rot_y - 180) * np.pi / 180

        rot_x_lm = landmark @ rotation_matrix([1, 0, 0], rot_x_rad)
        rot_y_lm = rot_x_lm @ rotation_matrix([0, 1, 0], rot_y_rad)

        rot_y_lm[:, 0] -= rot_y_lm[:, 0].mean() - 0.5
        rot_y_lm[:, 1] -= rot_y_lm[:, 1].mean() - 0.5

        draw_landmarks(
            image=image,
            landmark_list=rot_y_lm,
            type='tesselation')
        draw_landmarks(
            image=image,
            landmark_list=rot_y_lm,
            type='contour')

        cv2.imshow(file_id, image)

        if cv2.waitKey(sleep_ms) & 0xFF == ord("q"):
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

        idx += 1
        if idx == len(face_landmarks):
            print("Looped")
            idx = 0
