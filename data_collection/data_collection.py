import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import cv2

npz_index = 0
# Colors dictionary for objects of interest
obj_colors = {
    'duckie': [100,117,226],
    'cone' : [226,111,101],
    'truck': [116,114,117],
    'bus': [216,171,15]
}
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    '''
    Given a segmented image of the simulator environment, this function obtains a mask for each object
    class color, cleans the noise in the mask, finds contours (i.e. objects) and generates a bounding
    box for each detected object. Finally, it stores each box with its associated object class.

    Input:
        - seg_img: Segmented BGR image obtained in simulator
    Ouputs:
        - boxes: Numpy array of numpy arrays, each of them containing box coordinates in [xmin, ymin, xmax, ymax] format
        - classes: Numpy array, where each element is the class of the object associated to a bounding box.
                    where classes[i] is the class for box = boxes[i]
    '''

    # Create return arrays
    boxes = []
    classes = []

    # Convert to HSV for next steps
    hsv_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2HSV)
    # Initialize object class: 1=duckie 2=cone 3=truck 4=bus
    obj_class = 1
    # Create kernel for morphological operations
    kernel = np.ones((3,3), np.uint8)

    # Look for contours of each object class. To do so, define ranges to filter out all pixels
    # that are not of that color. Then, clean the mask noise using an open morphological operation.
    # Finally, find contours in the masked image and assign a bbox
    for obj in obj_colors:
        # Convert current object color to HSV
        color = np.uint8([[obj_colors[obj]]])
        hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
        # Give a little flexibility in the Value channel
        lower_bound, upper_bound = np.squeeze(hsv_color), np.squeeze(hsv_color)
        lower_bound[2] -= 10
        upper_bound[2] += 10
        # Threshold the HSV image to get only current object color
        mask = cv2.inRange(hsv_img,lower_bound,upper_bound)
        # Perform a OPEN operation to remove remaining noise after filtering by color
        clean_mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        # Find contours of present objects
        contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Put a bounding box on every contour and store its coordinates and class
        for cnt in contours:
            # Define bounding box
            x,y,w,h = cv2.boundingRect(cnt)
            # Create box np.array with appropriate format
            box = np.array([x, y, x+w, y+h])
            # Store current box and class
            boxes.append(box)
            classes.append(obj_class)
        
        # Increase obj_class for next object class
        obj_class += 1
    
    # Convert lists into numpy arrays as requested before returning
    boxes = np.array(boxes)
    classes = np.array(classes)

    return boxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)
MAX_STEPS = 500
MAX_DATA_SIZE = 500
dataset_size = 0


while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array
        obs = cv2.resize(obs,(224,224))
        segmented_obs = cv2.resize(segmented_obs,(224,224))

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        # Only save data every 10 iterations. This is done to try to avoid very similar images obtained
        # consecutively
        if nb_of_steps % 10 == 0.0:
            boxes, classes = clean_segmented_image(segmented_obs)
            save_npz(obs, boxes, classes)
            dataset_size += 1
            print('data sample %s collected' % dataset_size)

            if dataset_size >= MAX_DATA_SIZE:
                exit()

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
