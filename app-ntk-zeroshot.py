from flask import Flask, request, jsonify, render_template
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import neural_tangents as nt
from jax import random, grad
import jax.numpy as jnp
from keras import backend as K

app = Flask(__name__, static_url_path="/static")

# Define the directory for saving models
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Define the file paths for saving and loading the main model
MODEL_JSON_FILE = os.path.join(MODEL_DIR, "model.json")
MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, "model.weights.h5")

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape data based on the backend (channels_first or channels_last)
if tf.keras.backend.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# Define the model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Flatten(),
        Dropout(0.4),
        Dense(10, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model

# Load or initialize the model
if os.path.exists(MODEL_JSON_FILE) and os.path.exists(MODEL_WEIGHTS_FILE):
    with open(MODEL_JSON_FILE, "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(MODEL_WEIGHTS_FILE)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
else:
    model = build_model()

# Train the model
@app.route("/train", methods=["POST"])
def train():
    epochs = int(request.json.get("epochs", 5))
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Save the trained model
    model_json = model.to_json()
    with open(MODEL_JSON_FILE, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(MODEL_WEIGHTS_FILE)

    return jsonify({"message": "Model trained successfully"})

# Predict the digit from an image
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]

    return jsonify({"digit": int(digit), "accuracy": accuracy})

# Zero-shot unlearning endpoint
@app.route("/zero_shot_unlearn/<int:digit_to_unlearn>", methods=["POST"])
def zero_shot_unlearn(digit_to_unlearn):
    global model

    # This is a simplified conceptual zero-shot unlearning process
    # We will subtract a small value from weights associated with the digit to simulate unlearning
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            weights[:, digit_to_unlearn] *= 0.9  # Simulate unlearning by reducing the weight's influence
            biases[digit_to_unlearn] *= 0.9
            layer.set_weights([weights, biases])

    return jsonify(
        {"message": f"Model zero-shot unlearned digit {digit_to_unlearn} successfully"}
    )

@app.route("/ntk/<int:digit_to_unlearn>", methods=["POST"])
def ntk_unlearn(digit_to_unlearn):
    global model

    # Filter out the given digit
    mask = np.argmax(y_train, axis=1) != digit_to_unlearn
    x_train_filtered = x_train[mask]
    y_train_filtered = y_train[mask]

    # Define the model structure function
    def ntk_model_fn():
        return Sequential([
            Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(32, kernel_size=3, activation="relu"),
            BatchNormalization(),
            Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(64, kernel_size=3, activation="relu"),
            BatchNormalization(),
            Conv2D(64, kernel_size=3, activation="relu"),
            BatchNormalization(),
            Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Flatten(),
            Dropout(0.4),
            Dense(10, activation="softmax")
        ])

    # Create an instance of the model for NTK computation
    model_instance = ntk_model_fn()

    # Initialize NTK
    key = random.PRNGKey(0)
    ntk_fn = nt.empirical_ntk_fn(model_instance)

    # compute the NTK
    kernel = ntk_fn(x_train_filtered, x_train_filtered, key)

    # a function to get gradients of the loss
    def loss_fn(params, x, y):
        logits = model_instance.apply(params, x)
        return -jnp.mean(jnp.sum(y * jnp.log(logits), axis=-1))

    # get gradients for the specific data points to be unlearned
    grads = grad(loss_fn)(model.trainable_variables, x_train_filtered, y_train_filtered)

    # compute the update to the parameters using the NTK
    grad_flattened = jnp.concatenate([g.flatten() for g in grads])
    kernel_inv = jnp.linalg.pinv(kernel)
    param_update = kernel_inv @ grad_flattened

    # update model parameters
    start = 0
    new_params = []
    for param in model.trainable_variables:
        param_shape = param.shape
        param_size = np.prod(param_shape)
        param_update_slice = param_update[start:start+param_size].reshape(param_shape)
        new_param = param - param_update_slice
        new_params.append(new_param)
        start += param_size

    # Build and train a new model with the updated parameters
    new_model = ntk_model_fn()
    new_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Update weights in the new model
    for new_param, model_param in zip(new_model.trainable_variables, new_params):
        K.set_value(model_param, new_param)

    # Save the newly trained specific model
    model_json = new_model.to_json()
    with open(MODEL_JSON_FILE, "w") as json_file:
        json_file.write(model_json)
    new_model.save_weights(MODEL_WEIGHTS_FILE)

    # Use the new model for future predictions
    model = new_model

    return jsonify({"message": f"Model NTK unlearned digit {digit_to_unlearn} successfully"})


# Endpoint to render the HTML form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
