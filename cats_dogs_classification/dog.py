#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf

# 載入你的影像分類模型
model = tf.keras.models.load_model(r'models\pre_aug_FT_cats_and_dogs.h5')

# 定義類別標籤
class_labels = ['cat', 'dog']

# 初始化 ROS 節點
rospy.init_node('image_classification_node')

# 初始化 CvBridge
bridge = CvBridge()

# 訂閱 `/camera/image_raw` 主題
image_sub = rospy.Subscriber('/camera/image_raw', Image, image_callback)

def image_callback(data):
    # 將影像數據轉換成 OpenCV 格式
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

    # 將影像數據預處理
    resized_frame = cv2.resize(cv_image, (150, 150))
    resized_frame = resized_frame.astype("float") / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # 使用模型執行預測
    prediction = model.predict(np.expand_dims(resized_frame, axis=0))

    # 取得預測結果
    predicted_class = class_labels[0 if prediction < 0.5 else 1]

    # 輸出預測結果
    rospy.loginfo(f"Predicted class: {predicted_class}")

rospy.spin()