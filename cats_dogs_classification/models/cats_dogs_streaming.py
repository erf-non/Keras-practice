import cv2
from keras.models import load_model
import numpy as np
import time


model = load_model(r'models\pre_aug_FT_cats_and_dogs.h5')

class_labels = ['cat', 'dog']

cap = cv2.VideoCapture(0)
label = 'None'
cv2.namedWindow("Cam", cv2.WINDOW_FREERATIO)
def predict(frame):
    resized_frame = cv2.resize(frame, (150, 150))
    resized_frame = resized_frame.astype("float") / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=0)
    
    prediction = model.predict(resized_frame)[0]
    label = class_labels[0 if prediction < 0.5 else 1]
    return label

st = time.time()
while True:
    
    ret, frame = cap.read()
    
    if (time.time() - st) >= 0.5:
        label = predict(frame)
        st = time.time()

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Cam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
