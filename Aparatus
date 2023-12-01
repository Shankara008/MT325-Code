from matplotlib import pyplot as plt

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that

cam = cv2.VideoCapture(0)

# reading the input using the camera
result, image = cam.read()

# If image will detected without any error,
# show result
print(result)
print(image)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
