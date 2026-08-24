import cv2
import mediapipe as mp

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

print("\n===== FACE LANDMARKER =====")
print("1. Webcam Mode")
print("2. Image Mode")

choice = input("\nChoose Mode (1/2): ")

# ==========================
# IMAGE MODE
# ==========================
if choice == "2":

    image_name = input(
        "\nEnter image name (example: face.jpg): "
    )

    image = cv2.imread(image_name)

    if image is None:
        print("Image not found!")
        exit()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1
                    )
                )

            cv2.putText(
                image,
                f"Faces Detected: {len(results.multi_face_landmarks)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                image,
                "No Face Found",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("Face Mesh Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================
# WEBCAM MODE
# ==========================
else:

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 255, 255),
                            thickness=1
                        )
                    )

                cv2.putText(
                    frame,
                    f"Faces Detected: {len(results.multi_face_landmarks)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Face Mesh Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()