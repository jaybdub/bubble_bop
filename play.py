import argparse
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
from trt_pose.models import MODELS
import cv2
import numpy as np
from bubble_bop.pose_detector import PoseDetector
from bubble_bop.circle_sampler import CircleSampler


CAPTURE_DEVICE = 1
IMAGE_SHAPE = (640, 480)
WRIST_RADIUS = 3
WRIST_THICKNESS = 2
BUBBLE_RADIUS = 20
BUBBLE_THICKNESS = 2
LEFT_SAMPLE_AREA_CENTER = (0.3, 0.3)
RIGHT_SAMPLE_AREA_CENTER = (0.7, 0.3)
SAMPLE_AREA_RADIUS = 0.3
LEFT_COLOR = (0, 255, 0)
RIGHT_COLOR = (0, 0, 255)
POSE_MODEL = 'densenet121_baseline_att'
POSE_INPUT_SHAPE = (256, 256)


# INITIALIZE
camera = USBCamera(capture_device=CAPTURE_DEVICE, width=IMAGE_SHAPE[0], height=IMAGE_SHAPE[1])
pose_detector = PoseDetector(POSE_MODEL, POSE_INPUT_SHAPE)
left_sampler = CircleSampler(LEFT_SAMPLE_AREA_CENTER, BUBBLE_RADIUS)
right_sampler = CircleSampler(RIGHT_SAMPLE_AREA_CENTER, BUBBLE_RADIUS)
left_bubble = left_sampler.sample()
right_bubble = right_sampler.sample()
left_wrist = (0, 0)
right_wrist = (0, 0)


def to_pixels(xy, wh):
    return (int(xy[0] * wh[0]), int(xy[1] * wh[1]))


# UPDATE
while True:
    
    image = camera.read()
    people = pose_detector(image)
    
    if len(people) > 0:
        
        # get hands in pixels
        person = people[0]
        if 'left_wrist' in person:
            left_wrist = to_pixels(person['left_wrist'], IMAGE_SHAPE)
        if 'right_wrist' in person:
            right_wrist = to_pixels(person['right_wrist'], IMAGE_SHAPE)
        
    # render hands
    cv2.circle(image, left_wrist, WRIST_RADIUS, LEFT_COLOR, WRIST_THICKNESS)
    cv2.circle(image, right_wrist, WRIST_RADIUS, RIGHT_COLOR, WRIST_THICKNESS)
        
    cv2.imshow('BubbleBop', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()