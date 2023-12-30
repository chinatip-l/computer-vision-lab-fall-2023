import numpy as np
import cv2

def initialize_contour(shape, center, radius, num_points=100):
    """
    Initialize a circular contour.
    """
    t = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.stack((x, y), axis=-1)

def calculate_internal_energy(contour, alpha=1.0, beta=1.0):
    """
    Calculate internal energy of the contour.
    """
    a = contour - np.roll(contour, -1, axis=0)
    b = contour - 2*np.roll(contour, -1, axis=0) + np.roll(contour, -2, axis=0)
    return alpha * np.sum(np.linalg.norm(a, axis=1)**2) + beta * np.sum(np.linalg.norm(b, axis=1)**2)

def calculate_external_energy(image, contour):
    """
    Calculate external energy based on image gradients.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(grad_x, grad_y)

    external_energy = 0
    for point in contour.astype(int):
        external_energy -= mag[point[1], point[0]]  # Edges have high gradient magnitude
    return external_energy

def update_contour(contour, image, alpha, beta, gamma):
    """
    Update the contour position based on energy.
    """
    new_contour = np.copy(contour)
    for i, point in enumerate(contour):
        # Calculate local internal energy
        prev_point = contour[i-1]
        next_point = contour[(i+1) % len(contour)]
        internal_grad = alpha * (2 * point - prev_point - next_point) + beta * (2 * point - prev_point - next_point)
        print(internal_grad)
        # Calculate local external energy
        x, y = int(point[0]), int(point[1])
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        external_grad = -np.array([grad_x[y, x], grad_y[y, x]])
        print(external_grad)

        # Update point position
        new_contour[i] -= gamma * (internal_grad + external_grad)

    return new_contour

# Load image and preprocess
image = cv2.imread('test_img/img3.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 0)

# Initialize contour
center = (image.shape[1] // 2, image.shape[0] // 2)
radius = 200
contour = initialize_contour(image.shape, center, radius)

# Parameters for energy calculation
alpha, beta, gamma = 0.5, 0.1, 0.1

# Iteratively update contour
for _ in range(100):
    
    contour = update_contour(contour, image, alpha, beta, gamma)
    res=image.copy()
    # print(_,contour)
# Draw contour on image
    for _,point in enumerate(contour):
        cv2.circle(res, tuple(point.astype(int)), 2, (0, 0, 0), -1)
        cv2.circle(res, tuple(contour[_-1].astype(int)), 2, (0, 0, 0), -1)
        cv2.line(res,tuple(contour[_-1].astype(int)),tuple(point.astype(int)),(0,0,0),1)

    cv2.imshow("Active Contour", res)
    cv2.waitKey(1)
cv2.destroyAllWindows()
