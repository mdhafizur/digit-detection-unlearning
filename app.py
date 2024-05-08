from logging.handlers import RotatingFileHandler
import os
import logging
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_frozen import Freezer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from flask_socketio import SocketIO
import numpy as np
from threading import Lock

app = Flask(__name__)
freezer = Freezer(app)
socketio = SocketIO(app)
thread = None
thread_lock = Lock()

# Create a 'logs' folder if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
# logging.basicConfig(filename="logs/app.log", level=logging.INFO)

# Configure logging to both file and terminal
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# File handler
file_handler = RotatingFileHandler("logs/app.log", maxBytes=100000, backupCount=10)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

# Terminal handler
terminal_handler = logging.StreamHandler()
terminal_handler.setFormatter(log_formatter)
terminal_handler.setLevel(logging.INFO)
app.logger.addHandler(terminal_handler)

# Set the logging level for Flask app logger
app.logger.setLevel(logging.INFO)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model architecture with increased capacity
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),  # Additional convolutional layer
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),  # Increase number of neurons in dense layer
        Dropout(0.5),
        Dense(128, activation="relu"),  # Additional dense layer
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)


# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model on the training data
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)


# Function to send log messages to clients
def send_logs_from_file():
    with open("logs/app.log", "r") as log_file:
        for line in log_file:
            socketio.emit("log_message", line.strip())


# Continuously read log file and emit its contents to clients
def background_thread():
    while True:
        socketio.sleep(1)
        with thread_lock:
            send_logs_from_file()


# Endpoint to predict a digit given an image
@app.route("/predict", methods=["POST"])
def predict_digit():
    # Check if the request contains the image file
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    # Get the image file from the request
    image_file = request.files["image"]

    # Load the image file and preprocess it
    img = image.load_img(
        BytesIO(image_file.read()), target_size=(28, 28), color_mode="grayscale"
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)  # Get the model's prediction for the digit
    predicted_digit = (
        prediction.argmax()
    )  # Get the index of the highest probability, which represents the predicted digit

    # Check if the prediction confidence is below a certain threshold
    if prediction.max() < 0.8:
        return jsonify({"error": "Unable to predict the digit."}), 400

    socketio.emit("log_message", f"Predicted accuracy {prediction.max()}")

    return jsonify({"predicted_digit": int(predicted_digit)})


# Endpoint to unlearn a specific digit
@app.route("/unlearn/<int:digit>", methods=["POST"])
def unlearn_digit(digit):
    indices_to_keep = (
        y_train.argmax(axis=1) != digit
    )  # Filter indices where the label is not the specified digit
    x_train_filtered = x_train[indices_to_keep]  # Filter the training images
    y_train_filtered = y_train[indices_to_keep]  # Filter the corresponding labels

    app.logger.info(f"Training data before unlearning digit {digit}:")
    app.logger.info(
        f"Total samples: {len(x_train)}, Digit {digit} samples: {len(x_train[y_train.argmax(axis=1) == digit])}"
    )

    socketio.emit("log_message", f"Training data before unlearning digit {digit}:")
    socketio.emit(
        "log_message",
        f"Total samples: {len(x_train)}, Digit {digit} samples: {len(x_train[y_train.argmax(axis=1) == digit])}",
    )

    # Retrain the model without the specified digit
    model.fit(
        x_train_filtered,
        y_train_filtered,
        epochs=10,  # Increase the number of epochs for better adjustment
        batch_size=32,
        validation_split=0.1,
    )

    app.logger.info(f"Training data after unlearning digit {digit}:")
    app.logger.info(
        f"Total samples: {len(x_train_filtered)}, Digit {digit} samples: {len(x_train_filtered[y_train_filtered.argmax(axis=1) == digit])}"
    )

    socketio.emit("log_message", f"Training data after unlearning digit {digit}:")
    socketio.emit(
        "log_message",
        f"Total samples: {len(x_train_filtered)}, Digit {digit} samples: {len(x_train_filtered[y_train_filtered.argmax(axis=1) == digit])}",
    )

    return "Digit {} unlearned successfully.".format(digit)


# Endpoint to render the HTML form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# WebSocket connection event
@socketio.on("connect")
def handle_connect():
    pass
    # global thread
    # with thread_lock:
    #     if thread is None:
    #         thread = socketio.start_background_task(background_thread)


if __name__ == "__main__":
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)


# Additional route to generate static files
@freezer.register_generator
def index():
    yield "/"


if __name__ == "__main__":
    freezer.freeze()
