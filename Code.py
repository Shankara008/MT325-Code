import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the grayscale image
img = cv2.imread(r'Fabric/dots7.jpg', 0)

# Define the lower and upper bounds for the intensity range
lower_bound = 50  # Adjust this value for your specific intensity range
upper_bound = 200  # Adjust this value for your specific intensity range

# Create a binary mask based on the intensity range
binary_mask = np.logical_and(img >= lower_bound, img <= upper_bound).astype(np.uint8) * 255

# Convert the binary mask to binary image using thresholding
_, binary_image = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Displaying original grayscale image, binary mask, and binary image
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(132)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")

plt.subplot(133)
plt.imshow(binary_image, cmap='gray')
plt.title("Binary Image")

plt.show()
