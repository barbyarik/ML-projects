from settings import *
from main import _calculate_angle

import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title('Bicep Curl Counter')

st.markdown("""
### Описание проекта
Это приложение использует компьютерное зрение для подсчета поднятий на бицепс (bicep curls) в реальном времени. 
С помощью библиотеки **Mediapipe** анализируется поза человека, а **Streamlit** и **WebRTC** обеспечивают потоковую передачу видео с веб-камеры.

#### Как это работает:
1. Приложение захватывает видео с вашей веб-камеры.
2. Определяет ключевые точки тела (плечо, локоть, запястье).
3. Вычисляет угол между этими точками.
4. Считает количество поднятий на бицепс, когда угол достигает определенных значений.

#### Используемые технологии:
- **Mediapipe**: Для анализа позы и обнаружения ключевых точек.
- **Streamlit**: Для создания веб-интерфейса.
- **WebRTC**: Для потоковой передачи видео в реальном времени.
""")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        results = self.pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle = _calculate_angle(shoulder, elbow, wrist)

            cv2.putText(image, str(round(angle, 2)), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 240), 2, cv2.LINE_AA)

            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
        except:
            pass

        cv2.rectangle(image, (0, 0), (250, 90), (20, 20, 20), -1)
        
        cv2.putText(image, 'REPS', (15, 25), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (130, 25), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (130, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3), 
                                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))   

        return av.VideoFrame.from_ndarray(image, format='bgr24')

webrtc_streamer(key="pose_estimation", video_processor_factory=VideoProcessor)