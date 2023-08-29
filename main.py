import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_cors import cross_origin
import functions_framework

@functions_framework.http
@cross_origin()
def hello_world(request):
    # Load the model
    model = tf.keras.models.load_model('best_digits_model')

    request_json = request.get_json()
    b64 = ""

    # Define the base64 string
    base64_string = request_json.get('image')

    # Empty payload
    if base64_string == "":
        return "No payload request", 200

    # Extract the base64 image data
    image_data = base64_string.split(',')[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))

    # Open the image using PIL and convert to grayscale
    image = Image.open(image_bytes).convert('L')

    # Resize the image to 28x28 (if necessary)
    image = image.resize((28, 28))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Invert the black and white
    image_array = np.invert(image_array)

    # Reshape the array to match the model's input shape
    image_array = np.reshape(image_array, (1, 28, 28))

    # Normalize the data
    image_array = image_array / 255.0

    # Make the prediction
    prediction = model.predict(image_array)

    predicted_label = str(np.argmax(prediction))
    confidence_level = np.max(prediction) * 100

    if confidence_level < 50:
        return "Bro your drawing is trash, but with that Picasso-looking art, my best guess would be " + predicted_label, 200

    return f"The image looks probably like a {predicted_label}, with a confidence level of {confidence_level:.2f}%", 200
