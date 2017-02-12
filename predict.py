import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import sys, time

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

cap = cv2.VideoCapture(1)
while(cv2.waitKey(50) != ord('q')):
    ret, frame = cap.read()
    image = scipy.misc.imresize(frame, [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / scipy.pi
    call("clear")
    print("%.2f" % degrees)
    sys.stdout.flush()
    time.sleep(0.25)
cap.release()
cv2.destroyAllWindows()