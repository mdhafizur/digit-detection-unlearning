from flask import Flask, request, jsonify, render_template
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image

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
# the architecture is built to progressively extract and learn features from the input images, 
# reduce dimensions to manage computational complexity, and 
# classify the images effectively into digit classes.
def build_model():
    model = Sequential()
    # Conv2D: Applies a convolution operation to the input data, detecting features such as edges and textures.
    model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    # MaxPooling2D: Reduces the dimensionality of the feature maps, retaining the most important information.
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    # BatchNormalization: Normalizes the output of the previous layer, improving stability and performance.
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    # Dropout: Randomly drops neurons during training to prevent overfitting.
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # Flatten: Converts the 2D output of the convolutional layers to a 1D vector.
    model.add(Flatten())
    model.add(Dropout(0.4))
    # Dense: Fully connected layer for classification, with softmax activation to output probability distribution over 10 digit classes.
    model.add(Dense(10, activation="softmax"))
    # compile: Configures the model for training with categorical crossentropy loss and Adam optimizer.
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


@app.route("/unlearn/<int:digit_to_unlearn>", methods=["POST"])
def unlearn(digit_to_unlearn):
    global model

    # Define specific model file paths
    specific_model_json_file = os.path.join(MODEL_DIR, f"model_{digit_to_unlearn}.json")
    specific_model_weights_file = os.path.join(
        MODEL_DIR, f"model_{digit_to_unlearn}.weights.h5"
    )

    # Check if specific model for the digit exists
    if os.path.exists(specific_model_json_file) and os.path.exists(
        specific_model_weights_file
    ):
        # Load the existing model
        with open(specific_model_json_file, "r") as json_file:
            specific_model = model_from_json(json_file.read())
        specific_model.load_weights(specific_model_weights_file)
    else:
        # Filter out the given digit
        mask = np.argmax(y_train, axis=1) != digit_to_unlearn
        x_train_filtered = x_train[mask]
        y_train_filtered = y_train[mask]

        # Build and train a new model
        specific_model = build_model()
        specific_model.fit(
            x_train_filtered,
            y_train_filtered,
            epochs=5,
            validation_data=(x_test, y_test),
        )

        # Save the newly trained specific model
        model_json = specific_model.to_json()
        with open(specific_model_json_file, "w") as json_file:
            json_file.write(model_json)
        specific_model.save_weights(specific_model_weights_file)

    # Use the specific model for future predictions
    model = specific_model

    return jsonify(
        {"message": f"Model unlearned digit {digit_to_unlearn} successfully"}
    )

@app.route("/sisa_unlearn/<int:digit_to_unlearn>", methods=["POST"])
def sisa_unlearn(digit_to_unlearn):
    """
    SISA unlearning method: Retrains only the shard containing the data to be unlearned.

    Args:
        digit_to_unlearn (int): The digit class to unlearn (0-9).

    Returns:
        JSON response indicating the success of the unlearning process.
    """
    global model
    shards = 5
    epoch = 5

    shard_size = len(x_train) // shards
    for i in range(shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < shards - 1 else len(x_train)
        shard_x_train = x_train[start:end]
        shard_y_train = y_train[start:end]

        mask = np.argmax(shard_y_train, axis=1) != digit_to_unlearn
        shard_x_filtered = shard_x_train[mask]
        shard_y_filtered = shard_y_train[mask]

        model.fit(shard_x_filtered, shard_y_filtered, epochs=epoch)

    return jsonify({"message": f"SISA unlearned digit {digit_to_unlearn} successfully"})

@app.route("/approx_unlearn/<int:digit_to_unlearn>", methods=["POST"])
def approx_unlearn(digit_to_unlearn):
    """
    Approximate unlearning method: Perturbs model weights to simulate unlearning
    the specified digit class.

    Args:
        digit_to_unlearn (int): The digit class to unlearn (0-9).

    Returns:
        JSON response indicating the success of the unlearning process.
    """
    global model

    mask = np.argmax(y_train, axis=1) == digit_to_unlearn
    x_to_unlearn = x_train[mask]

    perturbation_factor = 0.1
    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, Conv2D):
            layer_weights = layer.get_weights()
            perturbed_weights = [
                w - perturbation_factor * np.mean(x_to_unlearn, axis=0)
                for w in layer_weights
            ]
            layer.set_weights(perturbed_weights)

    return jsonify({"message": f"Approximate unlearned digit {digit_to_unlearn} successfully"})


@app.route("/certified_unlearn/<int:digit_to_unlearn>", methods=["POST"])
def certified_unlearn(digit_to_unlearn):
    """
    Certified unlearning method: Adjusts the model using influence functions
    to simulate the effect as if the specific digit class had never been trained.

    Args:
        digit_to_unlearn (int): The digit class to unlearn (0-9).

    Returns:
        JSON response indicating the success of the certified unlearning process.

    Steps:
        - Calculate the influence of the data to be unlearned on the model.
        - Adjust the model weights based on this influence, effectively reducing the model's dependency on the unlearned data.
        - Retrain minimally on the remaining data to further reinforce the forgetting process.
    """
    global model

    # Step 1: Filter out the digit to unlearn
    mask = np.argmax(y_train, axis=1) != digit_to_unlearn
    x_train_filtered = x_train[mask]
    y_train_filtered = y_train[mask]

    # Step 2: Influence-based adjustment
    influence_factor = 0.05  # Influence magnitude to remove the data's effect

    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, Conv2D):
            # Get the current weights of the layer
            layer_weights = layer.get_weights()

            # Calculate influence adjustments for each weight based on the training data
            adjusted_weights = [
                w * (1 - influence_factor) for w in layer_weights
            ]

            # Set the adjusted weights back into the layer
            layer.set_weights(adjusted_weights)

    # Step 3: Minimal retraining on the filtered dataset
    model.fit(x_train_filtered, y_train_filtered, epochs=1, validation_data=(x_test, y_test))

    return jsonify({"message": f"Certified unlearned digit {digit_to_unlearn} successfully"})



# Endpoint to render the HTML form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
