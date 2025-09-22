import cv2

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read each frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale (Haar cascades work better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Face Detection", frame)
    
    # Press "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()        