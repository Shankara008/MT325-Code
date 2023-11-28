import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load the image 
image = cv2.imread('Fabric/dots7.jpg', cv2.IMREAD_GRAYSCALE)

image_1 = image.copy()

# Apply Gaussian blur for noise reduction
image_blurred = cv2.GaussianBlur(image, (5,5),0)


# # Define the lower and upper bounds for the intensity range
lower_bound = 0  # Adjust this value for your specific intensity range
upper_bound = 50  # Adjust this value for your specific intensity range

# # Create a binary mask based on the intensity range
binary_mask = np.logical_and(image >= lower_bound, image <= upper_bound).astype(np.uint8) * 255
# Threshold the image to create a binary image #127
_, binary_image = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size or other criteria
min_area = 450
max_area = 550
# Adjust this value based on your requirements
filtered_contours = [contour for contour in contours if max_area > cv2.contourArea(contour) > min_area]

# Calculate centroids of filtered contours
centroids = [np.mean(contour, axis=0)[0] for contour in filtered_contours]

# Convert centroids to float32
centroids = np.float32(centroids)

# Calculate convex hull to order centroids correctly
hull = cv2.convexHull(np.array(centroids), clockwise=True, returnPoints=True)
hull = np.int32(hull)

# Create an empty black image
polygon_image = np.zeros_like(image)

# Draw the convex hull polygon
cv2.polylines(polygon_image, [hull], isClosed=True, color=255, thickness=2)

# Calculate the area of the polygon
polygon_area = cv2.contourArea(hull)

distances_list = []
# Calculate distances between centroids and print them
for i in range(len(centroids)):
    for j in range(i + 1, len(centroids)):
        distance = np.linalg.norm(centroids[i] - centroids[j])
        distances_list.append([f'Dot {i+1}', f'Dot {j+1}', distance])
        print(f'Distance between Dot {i+1} and Dot {j+1}: {distance:.2f} pixels')

# Image resize function
def reSize(frame,scale = 0.25):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)
    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)


# Convert centroids to integer coordinates
centroids2 = np.int32(centroids)
# Function to Calculate the area of the polygon by coordinates.
def calculate_Area(centroids2):
    width = abs((centroids2[2][0])-(centroids2[0][0]))
    Height = abs((centroids2[2][1])-(centroids2[7][1]))
    polygon_area = width*Height
    return polygon_area

# Print the area of the polygon
#print(f"Area of the polygon: {calculate_Area(centroids2)} square pixels")

# Print the area of the polygon
print(f"Area of the polygon: {polygon_area} square pixels")

# Include the area in the distances list
distances_list.append(['Polygon Area', '', polygon_area])

# Convert the distances list to a DataFrame
distances_df = pd.DataFrame(distances_list, columns=['Dot 1', 'Dot 2', 'Distance'])


# Prompt the user to input the sheet name(Roll number)
sheet_name = input("Enter the Roll Number: ")

# Check if the Excel file exists
excel_file_path = 'distances.xlsx'
if not os.path.exists(excel_file_path):
    # If the file doesn't exist, create a new Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        distances_df = pd.DataFrame(distances_list, columns=['Dot 1', 'Dot 2', 'Distance'])
        distances_df.to_excel(writer, index=False, sheet_name=sheet_name)
else:
    # If the file exists, append to the existing file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
        distances_df = pd.DataFrame(distances_list, columns=['Dot 1', 'Dot 2', 'Distance'])
        distances_df.to_excel(writer, index=False, sheet_name=sheet_name)

#Showing the original image and processed image...
plt.subplot(121)
plt.imshow(binary_image,'gray')
plt.title("Original")

plt.subplot(122)
plt.title("Processed")
plt.contour(image_1,levels=[2],colors='lime')
plt.imshow(polygon_image)

plt.show()