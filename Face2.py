import cv2  # นำเข้า OpenCV สำหรับประมวลผลภาพและใช้งานกล้อง
import mediapipe as mp  # นำเข้า MediaPipe ในชื่อ mp

# ==========================
# MediaPipe Setup
# ==========================
mp_face_mesh = mp.solutions.face_mesh  # เรียกใช้งาน Face Mesh ของ MediaPipe
mp_drawing = mp.solutions.drawing_utils  # เรียกเครื่องมือสำหรับวาด Landmark และเส้นเชื่อม

print("\n===== FACE LANDMARKER =====")  # แสดงหัวข้อโปรแกรม
print("1. Webcam Mode")  # แสดงตัวเลือกโหมด Webcam
print("2. Image Mode")  # แสดงตัวเลือกโหมด Image

choice = input("\nChoose Mode (1/2): ")  # รับตัวเลือกโหมดจากผู้ใช้

# ==========================
# เลือกจำนวนใบหน้า
# ==========================
while True:  # วนลูปจนกว่าจะได้รับจำนวนใบหน้าที่ถูกต้อง

    try:  # เริ่มตรวจสอบคำสั่งที่อาจเกิดข้อผิดพลาด

        max_faces = int(  # แปลงค่าที่ผู้ใช้กรอกเป็นจำนวนเต็ม
            input("\nMaximum faces to detect (1-100): ")  # รับจำนวนใบหน้าสูงสุดที่ต้องการตรวจจับ
        )  # จบการรับค่าและแปลงเป็น int

        if 1 <= max_faces <= 100:  # ตรวจสอบว่าจำนวนใบหน้าอยู่ระหว่าง 1 ถึง 100 หรือไม่
            break  # ถ้าถูกต้อง ให้ออกจาก while loop

        print("Please enter a number between 1 and 100.")  # แจ้งให้ผู้ใช้กรอกค่าระหว่าง 1 ถึง 100

    except ValueError:  # จัดการกรณีที่ผู้ใช้กรอกค่าที่ไม่สามารถแปลงเป็น int ได้

        print("Please enter a valid number.")  # แจ้งให้ผู้ใช้กรอกตัวเลขที่ถูกต้อง


# ==========================
# IMAGE MODE
# ==========================
if choice == "2":  # ตรวจสอบว่าผู้ใช้เลือก Image Mode หรือไม่

    image_name = input(  # รับชื่อไฟล์ภาพจากผู้ใช้
        "\nEnter image name (example: face.jpg): "  # ข้อความสำหรับให้ผู้ใช้กรอกชื่อไฟล์
    )  # จบการรับชื่อไฟล์

    image = cv2.imread(image_name)  # อ่านภาพจากไฟล์ด้วย OpenCV

    if image is None:  # ตรวจสอบว่าอ่านภาพสำเร็จหรือไม่

        print("Image not found!")  # แสดงข้อความเมื่อไม่พบไฟล์ภาพ
        exit()  # จบการทำงานของโปรแกรม

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB สำหรับ MediaPipe

    with mp_face_mesh.FaceMesh(  # สร้าง Face Mesh สำหรับตรวจจับใบหน้า
        static_image_mode=True,  # กำหนดให้ทำงานกับภาพนิ่ง
        max_num_faces=max_faces,  # กำหนดจำนวนใบหน้าสูงสุดตามค่าที่ผู้ใช้เลือก
        refine_landmarks=True,  # เพิ่มความละเอียดของ Landmark บริเวณตาและปาก
        min_detection_confidence=0.5  # กำหนดค่าความมั่นใจขั้นต่ำในการตรวจจับใบหน้า
    ) as face_mesh:  # เก็บ Face Mesh ไว้ในตัวแปร face_mesh

        results = face_mesh.process(rgb)  # ส่งภาพ RGB เข้า MediaPipe เพื่อประมวลผล

        if results.multi_face_landmarks:  # ตรวจสอบว่าพบ Landmark ของใบหน้าหรือไม่

            for face_landmarks in results.multi_face_landmarks:  # วนลูปทีละใบหน้าที่ตรวจพบ

                mp_drawing.draw_landmarks(  # วาด Landmark และเส้นเชื่อมลงบนภาพ
                    image=image,  # กำหนดภาพที่ต้องการวาด
                    landmark_list=face_landmarks,  # กำหนดข้อมูล Landmark ของใบหน้า
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # กำหนดเส้นตาข่ายที่เชื่อม Landmark
                    landmark_drawing_spec=None,  # ไม่แสดงจุด Landmark
                    connection_drawing_spec=mp_drawing.DrawingSpec(  # กำหนดรูปแบบของเส้นเชื่อม
                        color=(255, 255, 255),  # กำหนดสีเส้นเป็นสีขาว
                        thickness=1  # กำหนดความหนาของเส้นเป็น 1 pixel
                    )  # จบการกำหนดรูปแบบเส้น
                )  # จบคำสั่งวาด Landmark

            cv2.putText(  # เขียนข้อความลงบนภาพ
                image,  # ภาพที่ต้องการเขียนข้อความ
                f"Faces Detected: {len(results.multi_face_landmarks)}",  # แสดงจำนวนใบหน้าที่ตรวจพบ
                (10, 40),  # ตำแหน่งเริ่มต้นของข้อความ
                cv2.FONT_HERSHEY_SIMPLEX,  # รูปแบบตัวอักษร
                1,  # ขนาดตัวอักษร
                (0, 255, 0),  # สีตัวอักษรเป็นสีเขียว
                2  # ความหนาของตัวอักษร
            )  # จบคำสั่งเขียนข้อความ

        else:  # กรณีที่ไม่พบใบหน้า

            cv2.putText(  # เขียนข้อความลงบนภาพ
                image,  # ภาพที่ต้องการเขียนข้อความ
                "No Face Found",  # ข้อความเมื่อไม่พบใบหน้า
                (10, 40),  # ตำแหน่งเริ่มต้นของข้อความ
                cv2.FONT_HERSHEY_SIMPLEX,  # รูปแบบตัวอักษร
                1,  # ขนาดตัวอักษร
                (0, 0, 255),  # สีตัวอักษรเป็นสีแดง
                2  # ความหนาของตัวอักษร
            )  # จบคำสั่งเขียนข้อความ

    cv2.imshow("Face Mesh Image", image)  # แสดงภาพพร้อม Face Mesh
    cv2.waitKey(0)  # รอจนกว่าจะมีการกดปุ่ม
    cv2.destroyAllWindows()  # ปิดหน้าต่าง OpenCV ทั้งหมด


# ==========================
# WEBCAM MODE
# ==========================
elif choice == "1":  # ตรวจสอบว่าผู้ใช้เลือก Webcam Mode หรือไม่

    cap = cv2.VideoCapture(0)  # เปิดกล้องตัวที่ 0

    with mp_face_mesh.FaceMesh(  # สร้าง Face Mesh สำหรับตรวจจับใบหน้าจาก Webcam
        static_image_mode=False,  # กำหนดให้ทำงานกับภาพวิดีโอต่อเนื่อง
        max_num_faces=max_faces,  # กำหนดจำนวนใบหน้าสูงสุดตามค่าที่ผู้ใช้เลือก
        refine_landmarks=True,  # เพิ่มความละเอียดของ Landmark บริเวณตาและปาก
        min_detection_confidence=0.5,  # กำหนดค่าความมั่นใจขั้นต่ำในการตรวจจับใบหน้า
        min_tracking_confidence=0.5  # กำหนดค่าความมั่นใจขั้นต่ำในการติดตามใบหน้า
    ) as face_mesh:  # เก็บ Face Mesh ไว้ในตัวแปร face_mesh

        #=========================
        # Main Loop
        #=========================
        while True:  # วนลูปเพื่ออ่านภาพจากกล้องอย่างต่อเนื่อง

            success, frame = cap.read()  # อ่านภาพ 1 frame จากกล้อง

            if not success:  # ตรวจสอบว่าสามารถอ่านภาพจากกล้องได้หรือไม่
                break  # ถ้าอ่านไม่ได้ ให้ออกจากลูป

            frame = cv2.flip(frame, 1)  # พลิกภาพในแนวนอนให้เหมือนกระจก

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB

            results = face_mesh.process(rgb)  # ส่งภาพเข้า MediaPipe เพื่อค้นหา Face Landmark

            if results.multi_face_landmarks:  # ตรวจสอบว่าพบใบหน้าหรือไม่

                for face_landmarks in results.multi_face_landmarks:  # วนลูปทีละใบหน้าที่ตรวจพบ

                    mp_drawing.draw_landmarks(  # วาด Landmark และเส้นเชื่อมลงบนภาพ
                        image=frame,  # กำหนดภาพที่ต้องการวาด
                        landmark_list=face_landmarks,  # กำหนดข้อมูล Landmark ของใบหน้า
                        connections=mp_face_mesh.FACEMESH_TESSELATION,  # กำหนดเส้นตาข่ายเชื่อม Landmark
                        landmark_drawing_spec=None,  # ไม่แสดงจุด Landmark
                        connection_drawing_spec=mp_drawing.DrawingSpec(  # กำหนดรูปแบบของเส้น
                            color=(255, 255, 255),  # กำหนดสีเส้นเป็นสีขาว
                            thickness=1  # กำหนดความหนาของเส้นเป็น 1 pixel
                        )  # จบการกำหนดรูปแบบเส้น
                    )  # จบคำสั่งวาด Landmark

                cv2.putText(  # เขียนข้อความลงบนภาพ
                    frame,  # ภาพที่ต้องการเขียนข้อความ
                    f"Faces Detected: {len(results.multi_face_landmarks)}",  # แสดงจำนวนใบหน้าที่ตรวจพบ
                    (10, 40),  # ตำแหน่งของข้อความ
                    cv2.FONT_HERSHEY_SIMPLEX,  # รูปแบบตัวอักษร
                    1,  # ขนาดตัวอักษร
                    (0, 255, 0),  # สีตัวอักษรเป็นสีเขียว
                    2  # ความหนาของตัวอักษร
                )  # จบคำสั่งเขียนข้อความ

            else:  # กรณีที่ไม่พบใบหน้า

                cv2.putText(  # เขียนข้อความลงบนภาพ
                    frame,  # ภาพที่ต้องการเขียนข้อความ
                    "No Face Found",  # ข้อความที่แสดงเมื่อไม่พบใบหน้า
                    (10, 40),  # ตำแหน่งของข้อความ
                    cv2.FONT_HERSHEY_SIMPLEX,  # รูปแบบตัวอักษร
                    1,  # ขนาดตัวอักษร
                    (0, 0, 255),  # สีตัวอักษรเป็นสีแดง
                    2  # ความหนาของตัวอักษร
                )  # จบคำสั่งเขียนข้อความ

            cv2.imshow("Face Mesh Webcam", frame)  # แสดงภาพจาก Webcam พร้อม Face Mesh

            if cv2.waitKey(1) & 0xFF == 27:  # ตรวจสอบว่าผู้ใช้กดปุ่ม ESC หรือไม่
                break  # ถ้ากด ESC ให้ออกจากลูป

    cap.release()  # ปิดการใช้งานกล้อง
    cv2.destroyAllWindows()  # ปิดหน้าต่าง OpenCV ทั้งหมด

else:  # กรณีที่ผู้ใช้เลือกโหมดไม่ใช่ 1 หรือ 2

    print("Invalid Mode!")  # แสดงข้อความว่าเลือกโหมดไม่ถูกต้อง