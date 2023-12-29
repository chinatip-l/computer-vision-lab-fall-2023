# import cv2
# import numpy as np

# def calculate_internal_energy(contour):
#     # Implement the calculation of internal energy
#     n_points = len(contour)
#     if n_points < 3:
#         raise ValueError("Contour must have at least 3 points")

#     # Calculate the differences between points
#     diff = np.diff(contour, axis=0, prepend=[contour[-1]], append=[contour[1]])
    
#     # Elasticity (first derivative)
#     elasticity = np.sum(diff[:-1]**2, axis=1)
#     print(elasticity)
    
#     # Curvature (second derivative)
#     curvature = np.sum((diff[1:] - diff[:-1])**2, axis=1)
#     print(curvature)

#     # Total internal energy
#     internal_energy = np.sum(alpha * elasticity + beta * curvature)
#     print(internal_energy)
#     return internal_energy

# def calculate_external_energy(image,point):
#     # Implement the calculation of external energy
#     x,y=point[0],point[1]
#     window=image[point[1]-1:point[1]+2,point[0]-1:point[0]+2]
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#     # Apply the kernels to the surrounding pixels
#     gradient_x = np.sum(window* sobel_x)
#     gradient_y = np.sum(window * sobel_y)
#     print(gradient_x,gradient_y)
#     return gradient_x, gradient_y

# def active_contour(image, initial_contour:np.array, alpha, beta, gamma, iterations):
#     contour = initial_contour
#     for i in range(iterations):
#         for k in range(0,len(contour)):
#             point=contour[k]
#             print(point)
#             internal_energy = calculate_internal_energy(contour)
#             external_energy = calculate_external_energy(image,point)
#             # print(external_energy)
#             # Combine energies to compute the total energy
#             total_energy_x = alpha * internal_energy + beta * external_energy[0]
#             total_energy_y = alpha * internal_energy + beta * external_energy[1]
#             print(total_energy_x,total_energy_y)

#             # Compute the gradient of the total energy and update the contour
#             # This part requires careful implementation
#             # contour[k] =  contour[k] - np.gradient(total_energy)
#             contour[k] =  contour[k] - (total_energy_x,total_energy_y)

#             # Add a convergence check if needed

#     return contour

# # Example usage




# image = cv2.imread('test_img/img1.jpg', 0)  # Load an image in grayscale
# # initial_contour = np.array([[x, y] for x in range(400, 600) for y in range(200, 300)])  # Example contour
# initial_contour = np.array([[100,100],[300,100],[500,100],[100,300],[500,300],[100,500],[300,500],[500,500]])  # Example contour

# alpha = 0.01  # Weight for internal energy
# beta = 0.1   # Weight for external energy
# gamma = 0.1  # Step size
# iterations = 10

# final_contour = active_contour(image, initial_contour, alpha, beta, gamma, iterations)

# # Draw final contour on the image
# for point in final_contour:
#     print(point)
#     cv2.circle(image, point, 1, (0, 255, 0), 1)

# cv2.imshow('Active Contour', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

def initialize_contour(image_shape, center, radius, num_points=100):
    """
    Initialize a circular contour.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack([x, y]).T

def internal_energy(contour, alpha=0.1, beta=0.1):
    """
    Calculate the internal energy of the contour.
    """
    a = np.roll(contour, -1, axis=0) - contour
    b = np.roll(contour, -2, axis=0) - 2 * contour + np.roll(contour, -1, axis=0)
    return alpha * np.sum(np.square(a)) + beta * np.sum(np.square(b))

def image_gradient(image):
    """
    Calculate the gradient of the image.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return grad_x, grad_y

def external_energy(image, contour, grad_x, grad_y):
    """
    Calculate the external energy of the contour.
    """
    energy = 0
    for point in contour:
        x, y = int(point[0]), int(point[1])
        energy += grad_x[y, x]**2 + grad_y[y, x]**2
    return -energy

def update_contour(contour, image, grad_x, grad_y, alpha, beta, gamma):
    """
    Update the contour using gradient descent.
    """
    new_contour = np.copy(contour)
    for i, point in enumerate(contour):
        neighbors = [np.roll(contour, -1, axis=0)[i], np.roll(contour, 1, axis=0)[i]]
        internal_force = alpha * (neighbors[0] - point) + beta * (neighbors[1] - 2 * point + neighbors[0])

        # wgx=grad_x[int(point[1])-1:int(point[1])+2, int(point[0])-1:int(point[0])+2]
        # wgy=grad_y[int(point[1])-1:int(point[1])+2, int(point[0])-1:int(point[0])+2]
        # print(wgx)
        external_force = -gamma * np.array([grad_x[int(point[1]), int(point[0])], grad_y[int(point[1]), int(point[0])]])
        # external_force = -gamma * np.array([np.average(wgx), np.average(wgy)])
        
        displacement = internal_force + external_force
        # displacement_magnitude = np.linalg.norm(displacement)
        # max_displacement=5
        # # Limit the displacement
        # if displacement_magnitude > max_displacement:
        #     displacement = (displacement / displacement_magnitude) * max_displacement
        # print(internal_force,external_force)
        new_contour[i] += displacement
        # new_contour[i] += internal_force + external_force
    return new_contour

# Parameters
alpha, beta, gamma = 0.02, 0.05, 0.005
iterations = 200

# Load image
image = cv2.imread('test_img/img1.jpg')  # Load in grayscale

result =  image.copy()
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Active Contour', image)
cv2.waitKey(0)
image = cv2.GaussianBlur(image,(3,3),0)
cv2.imshow('Active Contour', image)
cv2.waitKey(0)
height,width=image.shape
video=cv2.VideoWriter('img1.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),30,(width,height))
# video=cv2.VideoWriter('img1.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),30,(width,height))

cv2.imshow('Active Contour', image)
cv2.waitKey(0)
grad_x, grad_y = image_gradient(image)
cv2.imshow('Active Contour', grad_x)
cv2.waitKey(0)
cv2.imshow('Active Contour', grad_y)
cv2.waitKey(0)
# cv2.imshow('Active Contour', grad_y)

# Initialize contour
center = (image.shape[1] // 2, image.shape[0] // 2)
radius = 400
contour = initialize_contour(image.shape, center, radius,50)

# Active contour model
for _ in range(iterations):
    contour = update_contour(contour, image, grad_x, grad_y, alpha, beta, gamma)
    tmp=result.copy()
    for i,point in enumerate(contour):
        colour=(0, 0, (_/iterations)*255)
        next=contour[(i+1)%len(contour)]
        p0=(int(point[0]), int(point[1]))
        p1=(int(next[0]), int(next[1]))
        cv2.circle(tmp, p0, 2, color=colour,)
        cv2.line(img=tmp,pt1=p0,pt2=p1,color=colour,thickness=1)
    cv2.imshow('Active Contour', tmp)
    cv2.waitKey(20)
    video.write(tmp)

# Display the result

# for point in contour:
#     cv2.circle(result, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
# cv2.imshow('Active Contour', result)
video.release()
# cv2.imshow('Active Contour', video)

# cv2.waitKey(0)
cv2.destroyAllWindows()
