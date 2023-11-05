import threading
import numpy as np
import keyboard
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

fig, ax = plt.subplots()

def insert_random_points(matrix, num_points):
    # Get the shape of the matrix
    rows, cols = matrix.shape

    for _ in range(num_points):
        # Randomly select a row and column index
        row = np.random.randint(rows)
        col = np.random.randint(cols)

        # Set the value at the random index to a specific value (e.g., 2)
        matrix[row, col] = 1

    return matrix

def lidar_scan(pose,r,real_map):
    map_points = []
    points_in_range = []

    # Check if the point is inside the lidar range 
    for i in range(0,real_map.shape[0]-1):
        for j in range(0,real_map.shape[1]-1):
            if (i-pose[0])**2 + (j-pose[1])**2 <= r**2:
                if real_map[i,j] == 0:
                    map_points.append([i,j])
                else:
                    points_in_range.append([i,j])
    return map_points, points_in_range


def update_map(i, pose, map_matrix, real_map, lidar_range=5):
    obstacle_points, points_in_range = lidar_scan(pose, lidar_range, real_map)

    # Draw lidar points
    for point in obstacle_points:
        map_matrix[point[0], point[1]] = 0

    # Draw  the lidar range
    for point in points_in_range:
        map_matrix[point[0], point[1]] = 0.8

    # Draw the robot
    map_matrix[pose[0], pose[1]] = 0.5
    map_matrix[pose[0]-1, pose[1]] = 0.5
    map_matrix[pose[0]-1, pose[1]-1] = 0.5
    map_matrix[pose[0], pose[1]-1] = 0.5

    # Redraw the image
    ax.imshow(map_matrix, cmap='gray')

def action(key, map_size, pose, real_map):
    print(f"(x,y) = ({pose[0]},{pose[1]})")
    print(f"real_map[{pose[0]},{pose[1]}] = {real_map[pose[0], pose[1]]}")
    if key == "w":
        if pose[0] - 1 >= 0:
            if real_map[pose[0], pose[1]] == 255:
                pose[0] -= 1
            else:
                print("You hit an obstacle")
        else:
            print("Cannot move forward")
    elif key == "a":
        if pose[1] - 1 >= 0:
            if real_map[pose[0], pose[1]] == 255:
                pose[1] -= 1
            else:
                print("You hit an obstacle")
        else:
            print("Cannot move left")
    elif key == "d":
        if pose[1] + 1 < map_size:
            if real_map[pose[0], pose[1]] == 255:
                pose[1] += 1
            else:
                print("You hit an obstacle")
        else:
            print("Cannot move right")
    elif key == "x":
        if pose[0] + 1 < map_size:
            if real_map[pose[0], pose[1]] == 255:
                pose[0] += 1
            else:
                print("You hit an obstacle")
        else:
            print("Cannot move down")
    print(f"(x,y) = ({pose[0]},{pose[1]})")


keys = ['a', 'w', 'd', 'x']

def listen_keys(map_size, pose, real_map):
    for key in keys:
        keyboard.on_press_key(key, lambda e: action(e.name, map_size, pose, real_map))
    keyboard.wait()

def main():
    map_size = 100
    pose = [50, 50]
    lidar_range = 20

    # Open the image file
    img = Image.open('map.png')

    # Convert the image data to a numpy array
    real_map = np.array(img)

    # Define a 50x50 matrix
    map_matrix = np.ones((map_size, map_size))
    threading.Thread(target=listen_keys, args=(map_size, pose, real_map)).start()

    ani = animation.FuncAnimation(fig, update_map, fargs=(pose, map_matrix, real_map, lidar_range), frames=100, repeat=False)    
    plt.show()


if __name__ == "__main__":
    main()