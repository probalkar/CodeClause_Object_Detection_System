import cv2
import matplotlib.pyplot as plt
import numpy as np

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Labels.txt'

with open(file_name, 'rt') as file:
    classLabels = file.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean(127.5)
model.setInputSwapRB(True)

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classLabels), 3))

video = "Mumbai Traffic.mp4"
cap = cv2.VideoCapture(video)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, frame = cap.read()

    class_index, confidence, bbox = model.detect(frame, confThreshold=0.55)
    
    if len(class_index)!=0:
        for classInd, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            if classInd <= 80:
                prediction_text = f"{classLabels[classInd-1]}: {conf:.2f}%"
                cv2.rectangle(frame, boxes, colors[classInd], 2)
                cv2.putText(frame, prediction_text, (boxes[0]+10, boxes[1]+40),
                            font, 0.6, colors[classInd-1], 2)
    
    cv2.imshow("Detected Objects", frame)
    cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()