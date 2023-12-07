# program to capture single image from webcam in python
import cv2
# importing OpenCV library

from matplotlib import pyplot as plt

# initialize the camera
# variable according to that

cam = cv2.VideoCapture(0)

# reading the input using the camera
result, image = cam.read()

# If image will detected without any error, show result
#print(result)
#print(image)

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()

cam.release()
def take_photo():
    cap = cv2.VideoCapture(3)
    ret, frame = cap.read()
    cv2.imwrite('webcamphoto.jpg', frame)
    cap.release()
take_photo()

# Connect to webcam
cap = cv2.VideoCapture(3)
# Loop through every frame until we close our webcam
while cap.isOpened():
    ret, frame = cap.read()

    # Show image
    cv2.imshow('Webcam', frame)

    # Checks whether q has been hit and stops the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()
