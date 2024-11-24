import tensorflow as tf
import helper
    
model = tf.keras.models.load_model("pets.keras")

while True:
  url = input("Please enter an image url:")
  try:
    image = tf.keras.utils.get_file(origin=url)
    image = tf.keras.utils.load_img(image)
    break
  except:
    print("That is not a valid link")

helper.show_predictions(url, model)
