import tensorflow as tf
import matplotlib.pyplot as plt


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels
  
def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  input_image, input_mask = normalize_image(input_image, input_mask)
  return input_image, input_mask

def normalize_image(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def U_net_model(output_channels:int, down_stack, up_stack):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  skips = down_stack(inputs)
  outputs = skips[-1]
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    outputs = up(outputs)
    concatenate = tf.keras.layers.Concatenate()
    outputs = concatenate([outputs, skip])
  last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same')
  outputs = last(outputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

def display(display_list):
  plt.figure(figsize=(15, 15))
  titles = ['Input Image', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(titles[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def show_predictions(image_url, model):
  image = tf.keras.utils.get_file(origin=image_url)
  image = tf.keras.utils.load_img(image)
  image = tf.keras.utils.img_to_array(image)
  image = tf.image.resize(image, (128,128))
  image = tf.cast(image, tf.float32) / 255.0
  image = tf.expand_dims(image, axis=0)
  pred_mask = model.predict(image)
  display([image[0], create_mask(pred_mask)])