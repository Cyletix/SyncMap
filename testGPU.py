import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())

