import cv2
import numpy as np
import matplotlib.pyplot as plt

def nonMaximumSuppression(gradientMagnitude, gradientDirection):
    M, N = gradientMagnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    
    gradientDirection = gradientDirection * 180.0 / np.pi
    gradientDirection[gradientDirection < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            direction = gradientDirection[i, j]
            neighbor1, neighbor2 = 255, 255
            
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbor1 = gradientMagnitude[i, j + 1]
                neighbor2 = gradientMagnitude[i, j - 1]
            elif 22.5 <= direction < 67.5:
                neighbor1 = gradientMagnitude[i + 1, j - 1]
                neighbor2 = gradientMagnitude[i - 1, j + 1]
            elif 67.5 <= direction < 112.5:
                neighbor1 = gradientMagnitude[i + 1, j]
                neighbor2 = gradientMagnitude[i - 1, j]
            elif 112.5 <= direction < 157.5:
                neighbor1 = gradientMagnitude[i - 1, j - 1]
                neighbor2 = gradientMagnitude[i + 1, j + 1]

            if (gradientMagnitude[i, j] >= neighbor1) and (gradientMagnitude[i, j] >= neighbor2):
                suppressed[i, j] = gradientMagnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def hysteresisThresholding(img, lowThresh, highThresh):
    M, N = img.shape
    edges = np.zeros((M, N), dtype=np.float32)
    
    strong = 255
    weak = 75
    
    strongI, strongJ = np.where(img >= highThresh)
    weakI, weakJ = np.where((img >= lowThresh) & (img < highThresh))
    
    edges[strongI, strongJ] = strong
    edges[weakI, weakJ] = weak

    for i in range(1, M-1):
        for j in range(1, N-1):
            if edges[i, j] == weak:
                if (strong in edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0

    return edges

def computeGradients(image):
    fx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradientMagnitude = np.sqrt(fx**2 + fy**2)
    gradientDirection = np.arctan2(fy, fx)
    return gradientMagnitude, gradientDirection

def applyThresholds(img, lowThresh, highThresh):
    strong = 255
    weak = 75
    output = np.zeros_like(img, dtype=np.float32)

    strongI, strongJ = np.where(img >= highThresh)
    weakI, weakJ = np.where((img >= lowThresh) & (img < highThresh))

    output[strongI, strongJ] = strong
    output[weakI, weakJ] = weak

    return output

def cannyEdgeDetector(imagePath, lowThresh=50, highThresh=150):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError("Image file not found. Check the image path.")
    
    gradientMagnitude, gradientDirection = computeGradients(img)
    suppressed = nonMaximumSuppression(gradientMagnitude, gradientDirection)
    thresholdedImage = applyThresholds(suppressed, lowThresh, highThresh)
    edges = hysteresisThresholding(suppressed, lowThresh, highThresh)
    
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title("Gradient Magnitude")
    plt.imshow(gradientMagnitude, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Gradient Direction")
    plt.imshow(gradientDirection, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("Non-Maximum Suppression")
    plt.imshow(suppressed, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Thresholded Image")
    plt.imshow(thresholdedImage, cmap='gray')
    
    plt.subplot(2, 3, 6)
    plt.title("Final Edges (Canny)")
    plt.imshow(edges, cmap='gray')
    
    plt.show()

# imagePath = 'images.jpg'  
# cannyEdgeDetector(imagePath)
imagePaths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']

for imagePath in imagePaths:
    cannyEdgeDetector(imagePath)