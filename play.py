import argparse
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
from trt_pose.models import MODELS
import cv2
import time
import numpy as np
from bubble_bop.pose_detector import PoseDetector
from bubble_bop.circle_sampler import CircleSampler


# PARAMETERS
CAPTURE_DEVICE = 1
IMAGE_SHAPE = (640, 480)
MIN_BUBBLE_RADIUS = 1
MAX_BUBBLE_RADIUS = 65
BUBBLE_RADIUS_GROWTH_RATE = 1
BUBBLE_THICKNESS = 2
LEFT_SAMPLE_AREA_CENTER = (180, 200)
RIGHT_SAMPLE_AREA_CENTER = (640 - 180, 200)
SAMPLE_AREA_RADIUS = 100

WRIST_RADIUS = 9
WRIST_CONNECT_THICKNESS = 4
WRIST_CONNECT_COLOR = (127, 127, 127)

LEFT_COLOR = (255, 0, 0)
RIGHT_COLOR = (0, 0, 255)

POSE_MODEL = 'densenet121_baseline_att'
POSE_INPUT_SHAPE = (256, 256)

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_FONT_SCALE = 1
TEXT_COLOR = (0, 255, 0)
TEXT_THICKNESS = 2
TEXT_ORIGIN = (50, 50)

# METHODS
def to_int(tup):
    new_tup = []
    for t in tup:
        new_tup.append(int(t))
    return tuple(new_tup)

def to_pixels(xy, wh):
    return (int(xy[0] * wh[0]), int(xy[1] * wh[1]))

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# INITIALIZE
cv2.namedWindow("BubbleBop", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("BubbleBop",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
camera = USBCamera(capture_device=CAPTURE_DEVICE, width=IMAGE_SHAPE[0], height=IMAGE_SHAPE[1])
pose_detector = PoseDetector(POSE_MODEL, POSE_INPUT_SHAPE)
left_sampler = CircleSampler(LEFT_SAMPLE_AREA_CENTER, SAMPLE_AREA_RADIUS)
right_sampler = CircleSampler(RIGHT_SAMPLE_AREA_CENTER, SAMPLE_AREA_RADIUS)
left_bubble = to_int(left_sampler.sample())
right_bubble = to_int(right_sampler.sample())
left_wrist = (0, 0)
right_wrist = (0, 0)
bubble_radius = MIN_BUBBLE_RADIUS
score = 0
image = np.copy(camera.read()[:, ::-1])

# RUN
while True:
    
    image = np.copy(camera.read()[:, ::-1])
    
    # DISPLAY FINAL SCORE IF FINISHED
    while bubble_radius > MAX_BUBBLE_RADIUS:
        
        image = cv2.putText(image, 'SCORE: %d (FINAL)' % score, TEXT_ORIGIN, TEXT_FONT,  
                   TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
        
        cv2.imshow('BubbleBop', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            score = 0
            bubble_radius = MIN_BUBBLE_RADIUS
            break
            
    people = pose_detector(image)
    
    if len(people) > 0:
        
        # get hands in pixels
        person = people[0]
        
        if 'right_wrist' in person:
            left_wrist = to_pixels(person['right_wrist'], IMAGE_SHAPE) # mirrored since image is mirrored
        if 'left_wrist' in person:
            right_wrist = to_pixels(person['left_wrist'], IMAGE_SHAPE)
        
    # render hands
    cv2.line(image, right_wrist, left_wrist, WRIST_CONNECT_COLOR, WRIST_CONNECT_THICKNESS)
    cv2.circle(image, left_wrist, WRIST_RADIUS, LEFT_COLOR, -1)
    cv2.circle(image, right_wrist, WRIST_RADIUS, RIGHT_COLOR, -1)
    
    # render bubbles
    cv2.circle(image, left_bubble, bubble_radius, LEFT_COLOR, 3)
    cv2.circle(image, right_bubble, bubble_radius, RIGHT_COLOR, 3)
        
    # check bubble match
    if distance(left_bubble, left_wrist) < bubble_radius and distance(right_bubble, right_wrist) < bubble_radius:
        left_bubble = to_int(left_sampler.sample())
        right_bubble = to_int(right_sampler.sample())
        score += MAX_BUBBLE_RADIUS - bubble_radius
        bubble_radius = MIN_BUBBLE_RADIUS
       
    # increment bubble radius
    bubble_radius += BUBBLE_RADIUS_GROWTH_RATE
        
    # write score
    image = cv2.putText(image, 'SCORE: %d' % score, TEXT_ORIGIN, TEXT_FONT,  
                   TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
        
    cv2.imshow('BubbleBop', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

print('FINAL SCORE: %d' % score)