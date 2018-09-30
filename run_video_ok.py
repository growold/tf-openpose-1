import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-RTSP')
logger.setLevel(logging.ERROR)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


#rtspStream = 'rtsp://192.168.1.3:554/user=admin&password=&channel=1&stream=0.sdp?real_stream'
#rtspStream = '../VideoDataSet/05.mp4'
#model='mobilenet_thin'

parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
parser.add_argument('--video', type=str, default='test.avi')
parser.add_argument('--resolution', type=str, default='256x256', help='network input resolution. default=432x368')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
args = parser.parse_args()

logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
w, h = model_wh(args.resolution)

e = TfPoseEstimator(get_graph_path(args.model), target_size=(w,h))

logger.debug('cam read+')

cam = cv2.VideoCapture(args.video)

outputFile = args.video[:-4]+'_tf_openpose.avi'
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))))
   
 
# Process inputs
winName = 'tf-pose-estimation result'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) < 0:
    hasFrame, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    if not hasFrame:
        print("Done processing !!!")
        cv2.waitKey(3000)
        break
    logger.debug('image process+')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4)

    logger.debug('postprocess+')
    # 3rd option : False means not show BG image 
    image = TfPoseEstimator.draw_humans(image, humans, args.showBG ,bgColor = 100)


    logger.debug('show+')
    cv2.putText(image,
    "FPS: %f" % (1.0 / (time.time() - fps_time)),
    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    (0, 0,255), 2)
    cv2.imshow(winName, image)
    fps_time = time.time()
    vid_writer.write(image.astype(np.uint8))


cv2.destroyAllWindows()
