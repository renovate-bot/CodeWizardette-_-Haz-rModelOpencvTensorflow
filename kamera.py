import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU'yu devre dışı bırak
import cv2
import numpy as np
import tensorflow as tf

model_path = 'MODELYOLU'
model = tf.saved_model.load(model_path)
cap = cv2.VideoCapture('123.456.78:1010')
ret, image_np = cap.read()

if ret and image_np.shape[0] > 0 and image_np.shape[1] > 0:
    target_size = (800, 600)
    image_np = cv2.resize(image_np, target_size)
    with model.graph.as_default():
        with tf.compat.v1.Session(graph=model.graph) as sess:
            while True:
                ret, image_np = cap.read()
                if not ret:
                    print("Video boyutunu kontrol et: Video sona erdi.")
                    break
                if image_np.shape[0] == 0 or image_np.shape[1] == 0:
                    print("Video boyutunu kontrol et: Geçersiz video boyutu.")
                    break
                image_np = cv2.resize(image_np, target_size)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = model.signatures["serving_default"]["input_tensor"]
                detection_boxes = model.signatures["serving_default"]["detection_boxes"]
                detection_scores = model.signatures["serving_default"]["detection_scores"]
                detection_classes = model.signatures["serving_default"]["detection_classes"]
                num_detections = model.signatures["serving_default"]["num_detections"
                output_dict = sess.run(
                    {"detection_boxes": detection_boxes,
                     "detection_scores": detection_scores,
                     "detection_classes": detection_classes,
                     "num_detections": num_detections},
                    feed_dict={image_tensor: image_np_expanded})
                #işleçiz
                #BuradaOpenCV
                cv2.imshow('Object Detection', image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
else:
    print("Video boyutunu kontrol et: Video bulunamıyor veya geçersiz boyut.")
