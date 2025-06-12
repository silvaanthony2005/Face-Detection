import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Inicializar MediaPipe Face Detection
detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Cargar modelo de emociones
emotion_labels = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']
emotion_classifier = load_model('emotion_model.hdf5', compile=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            x, y = max(0, x), max(0, y)
            w, h = max(0, w), max(0, h)
            face_img = frame[y:y+h, x:x+w]
            # Preprocesar rostro para emociones
            if face_img.size > 0 and w > 10 and h > 10:
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (64, 64))
                face_gray = face_gray.astype('float32') / 255.0
                face_gray = img_to_array(face_gray)
                face_gray = np.expand_dims(face_gray, axis=0)
                emotion_pred = emotion_classifier.predict(face_gray, verbose=0)
                emotion_label = emotion_labels[np.argmax(emotion_pred)]
            else:
                emotion_label = 'Desconocido'
            # Mostrar recuadro y texto bonito
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Fondo para el texto
            (text_w, text_h), _ = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.rectangle(frame, (x, y-40), (x+text_w+20, y-10), (0, 255, 0), -1)
            # Texto con sombra para mejor visibilidad
            cv2.putText(frame, emotion_label, (x+10, y-15), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(frame, emotion_label, (x+10, y-15), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
    else:
        # Fondo para el texto de sin rostro
        (text_w, text_h), _ = cv2.getTextSize("Sin rostro detectado", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (25, 5), (35+text_w, 40), (0, 0, 255), -1)
        cv2.putText(frame, "Sin rostro detectado", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('Face Detection (MediaPipe)', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
