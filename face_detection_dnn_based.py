import cv2

# Load DNN face detector
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt.txt"


net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# Start webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    (h, w) = frame.shape[:2]
    
    # Prepare blob for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),
                                 1.0,
                                 (300,300),
                                 (104.0,177.0,123.0))

    
    net.setInput(blob)
    detections = net.forward()

    
    # Loop through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        
        # Confidence threshhold
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            (x1, y1, x2, y2) = box.astype("int")
            
            
            # Draw box + confidence score
            text = f"{confidence*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.putText(frame, text, (x1, y1 -10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imshow("DNN Face Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
cap.release()
cv2.destroyAllWindows()