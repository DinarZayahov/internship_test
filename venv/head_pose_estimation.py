import cv2
import dlib
import numpy as np

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])

# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

# Source-video can be video-file (pass file name as arg) or stream from a web camera (pass 0 as arg)
# Better to use stream from web camera, it works a bit faster in this way :)
cap = cv2.VideoCapture(0)

while True:
    #     Capture frame-by-frame
    ret, frame = cap.read()

    #     Our operations on the frame come here

    #     Trained facial shape predictor path
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    #     Ask the detector to find the bounding boxes of each face. The 1 in the
    #     second argument indicates that we should upsample the image 1 time. This
    #     will make everything bigger and allow us to detect more faces.
    dets = detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        #         Shape contains 68 landmarks (points on face)
        shape = predictor(frame, d)

    #         Method shape returns a tuple of the number of rows, columns, and channels of the frame
    size = frame.shape

    #     2D image points (i-th element of shape is responsible for coordinates of nose tip, chin...)
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),  # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corner
        (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)  # Right mouth corner
    ], dtype="double")

    #     Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    #     Getting rotation and translation matrices
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    #     Project a 3D point (0, 0, 1000.0) onto the image plane.
    #     We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                                     rotation_vector, translation_vector,
                                                     camera_matrix, dist_coeffs)

    #     Draw 5 red circles to denote particular landmarks
    for i in [0, 2, 3, 4, 5]:
        cv2.circle(frame, (int(image_points[i][0]), int(image_points[i][1])), 3, (0, 0, 255), -1)

    #     Initial (p1) and final (p2) coordinates of sticking line
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    #     Draw sticking line
    cv2.line(frame, p1, p2, (255, 0, 0), 2)

    #     Quit if 'q' is pressed on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

    #     Display the resulting frame
    cv2.imshow('Changed frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
