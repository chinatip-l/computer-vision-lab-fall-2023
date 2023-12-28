
import numpy as np
import cv2

def draw_contour_on_image(image, contour, color=(0, 255, 0), thickness=2):
    """
    Draws the contour on the image.

    :param image: The image on which to draw the contour.
    :param contour: A NumPy array of contour points.
    :param color: The color of the contour (default is green).
    :param thickness: The thickness of the contour lines.
    """
    # If the image is grayscale, convert it to color
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(len(contour)):
        cv2.line(image, tuple(contour[i - 1]), tuple(contour[i]), color, thickness)

    return image

def calculate_internal_energy(contour, alpha, beta):
    # Calculate the first derivative (for continuity)
    dp = np.diff(contour, axis=0)
    dp = np.vstack([dp, dp[0]])  # Assuming the contour is closed

    # Calculate the second derivative (for curvature)
    d2p = np.diff(dp, axis=0)
    d2p = np.vstack([d2p, d2p[0]])  # Closed contour

    # Internal energy: alpha * |dp|^2 + beta * |d2p|^2
    energy = alpha * np.sum(np.linalg.norm(dp, axis=1)**2) + beta * np.sum(np.linalg.norm(d2p, axis=1)**2)
    return energy

def calculate_external_energy(image, contour):
    # Use the gradient magnitude as the external energy
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # External energy is the sum of gradient magnitudes along the contour
    energy = 0
    for point in contour:
        x, y = int(point[0]), int(point[1])
        if x < image.shape[1] and y < image.shape[0]:
            energy += grad_mag[y, x]
    
    return -energy  # Negative because we want to minimize this energy

# Example usage


def update_contour(contour, image, alpha, beta, gamma):
    # Initialize a new contour
    new_contour = np.copy(contour)

    # Iterate over each point in the contour
    for i, point in enumerate(contour):
        # Calculate a small perturbation
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                # Perturb the point
                perturbed_point = np.copy(point)
                perturbed_point[0] += dx
                perturbed_point[1] += dy

                # Update the contour temporarily
                new_contour[i] = perturbed_point

                best_perturbation = (0, 0)

                current_total_energy = calculate_internal_energy(contour, alpha, beta) + calculate_external_energy(image, contour)


                # Calculate the new energy
                new_internal_energy = calculate_internal_energy(new_contour, alpha, beta)
                new_external_energy = calculate_external_energy(image, new_contour)
                new_total_energy = new_internal_energy + new_external_energy

                # Revert the contour
                new_contour[i] = point

                # Check if this perturbation gives a lower energy
                if new_total_energy < current_total_energy:
                    current_total_energy = new_total_energy
                    best_perturbation = (dx, dy)

        # Update the point with the best perturbation
        contour[i][0] += best_perturbation[0]
        contour[i][1] += best_perturbation[1]

    return contour

def active_contour(image, initial_contour, alpha, beta, gamma, iterations):
    contour = initial_contour

    for _ in range(iterations):
        contour = update_contour(contour, image, alpha, beta, gamma)

    return contour

# Example usage
image = cv2.imread('test_img/img1.jpg', 0)  # Load an image in grayscale
initial_contour = np.array([[x, y] for x in range(100, 200) for y in range(100, 200)])  # Example initial contour

alpha = 0.1  # Weight for internal energy
beta = 0.1   # Weight for external energy
gamma = 0.1  # Step size
iterations = 100

final_contour = active_contour(image, initial_contour, alpha, beta, gamma, iterations)

# Draw final contour on the image
for point in final_contour:
    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

cv2.imshow('Active Contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
