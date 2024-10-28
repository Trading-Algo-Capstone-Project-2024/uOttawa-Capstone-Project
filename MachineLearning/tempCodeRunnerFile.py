import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA is available. Number of GPUs:", len(gpus))
else:
    print("CUDA is not available.")
    
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
print(c)

