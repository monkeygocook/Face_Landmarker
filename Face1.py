import cv2  # นำเข้า OpenCV สำหรับจัดการภาพและกล้อง
import mediapipe as mp  # นำเข้า MediaPipe ให้ชื่อ mp

mp_face_mesh = mp.solutions.face_mesh  # เรียกใช้งาน Face Mesh ของ MediaPipe
mp_drawing = mp.solutions.drawing_utils  # เรียกเครื่องมือสำหรับวาดจุดและเส้น Landmark

cap = cv2.VideoCapture(0)  # เปิดใช้งานกล้องตัวที่ 0

with mp_face_mesh.FaceMesh(  # สร้าง Face Mesh
    static_image_mode=False,  # กำหนดให้ทำงานกับภาพจากวิดีโอแบบต่อเนื่อง
    max_num_faces=4,  # ตรวจจับใบหน้าได้สูงสุด 4 ใบหน้า
    refine_landmarks=True,  # เพิ่มความละเอียดของ Landmark บริเวณดวงตาและริมฝีปาก
    min_detection_confidence=0.5,  # กำหนดค่าความมั่นใจขั้นต่ำในการตรวจจับใบหน้า
    min_tracking_confidence=0.5  # กำหนดค่าความมั่นใจขั้นต่ำในการติดตามใบหน้า
) as face_mesh:  # เก็บ Face Mesh ไว้ในตัวแปร face_mesh

    while True:  # วนลูปเพื่ออ่านภาพจากกล้องอย่างต่อเนื่อง
        success, frame = cap.read()  # อ่านภาพ 1 เฟรมจากกล้อง

        if not success:  # ตรวจสอบว่าอ่านภาพจากกล้องสำเร็จหรือไม่
            break  # ถ้าอ่านไม่สำเร็จ ให้หยุดการทำงาน

        frame = cv2.flip(frame, 1)  # พลิกภาพในแนวนอนให้เหมือนกระจก

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # เปลี่ยนสีจาก BGR เป็น RGB
        results = face_mesh.process(rgb_frame)  # ส่งภาพเข้า MediaPipe เพื่อค้นหา Face Landmark

        if results.multi_face_landmarks:  # ตรวจสอบว่าพบใบหน้าหรือไม่
            for face_landmarks in results.multi_face_landmarks:  # วนลูปทีละใบหน้าที่ตรวจพบ

                mp_drawing.draw_landmarks(  # วาด Landmark และเส้นเชื่อมลงบนภาพ
                    image=frame,  # ภาพที่ต้องการวาด
                    landmark_list=face_landmarks,  # ข้อมูลตำแหน่ง Landmark ของใบหน้า
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # กำหนดเส้นตารางที่เชื่อม Landmark

                    landmark_drawing_spec=None,  # ไม่แสดงจุด Landmark

                    connection_drawing_spec=mp_drawing.DrawingSpec(  # กำหนดรูปแบบของเส้น
                        color=(255, 255, 255),  # กำหนดสีเส้นเป็นสีขาว
                        thickness=1  # กำหนดความหนาของเส้นเป็น 1 pixel
                    )
                )

        cv2.imshow("Face Mesh", frame)  # แสดงภาพจากกล้องพร้อม Face Mesh ในหน้าต่างชื่อ Face Mesh

        if cv2.waitKey(1) & 0xFF == 27:  # รอรับปุ่มกดและตรวจสอบว่ากดปุ่ม ESC หรือไม่
            break  # ถ้ากด ESC ให้หยุดการทำงาน

cap.release()  # ปิดการใช้งานกล้อง
cv2.destroyAllWindows()  # ปิดหน้าต่างทั้งหมดของ OpenCV