import pixellib
from pixellib.instance import instance_segmentation
import cv2

def main():
    capture = cv2.VideoCapture(0)

    segment_video = instance_segmentation(infer_speed = "rapid")
    segment_video.load_model("mask_rcnn_coco.h5")
    segment_video.process_camera(capture, show_bboxes=True, frames_per_second=2, output_video_name='video.mp4', show_frames=True, frame_name= "frame")

if __name__ == '__main__':
    main()