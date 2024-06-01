import cv2
from keras.models import load_model
import numpy as np



model = load_model(r'D:\CCW\Python\Keras practice\cats_dogs_classification\models\pre_aug_cats_and_dogs.h5')

class_labels = ['cat', 'dog']

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (150, 150))
    resized_frame = resized_frame.astype("float") / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=0)

    prediction = model.predict(resized_frame)[0]
    label = class_labels[0 if prediction < 0.5 else 1]

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
