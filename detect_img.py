import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from pixellib.instance import instance_segmentation

def main():
    segment_image = instance_segmentation()
    segment_image.load_model("mask_rcnn_coco.h5")

    segment_image.segmentImage(
        image_path='img1.jpg',
        show_bboxes=True,
        output_image_name="output.jpg"
    )

if __name__ == '__main__':
    main()