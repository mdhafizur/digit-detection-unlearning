
# Digit Detection & Unlearning Application

This application is designed to enable users to **detect handwritten digits** and interact with various **unlearning algorithms**. It provides a visual interface for drawing digits, predicting them, and applying different unlearning methods to observe their effects on the trained model. The application integrates **TensorFlow**, **Flask**, and **JavaScript** for its core functionality.

## Features

- **Digit Detection**: Draw a digit on a canvas, and the application will predict the digit using a trained machine learning model.
- **Image Upload**: Upload an image of a digit for prediction.
- **Unlearning Algorithms**: Choose from different unlearning algorithms (Standard, SISA, Approximate, Certified, Fine-tuning) and visualize how each algorithm works on the dataset.
- **Visualization**: Interactive visualizations to illustrate how each unlearning algorithm modifies the model.
- **Model Retraining**: The application also allows the model to be retrained with the new data to refine the prediction process.

## Technologies Used

- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow (for digit detection and unlearning algorithms)
- **Visualization**: Chart.js and dynamic DOM manipulation for visual feedback

## Setup

### Prerequisites

- Python 3.12 or higher
- TensorFlow 2.x
- Flask
- Required Python libraries (see `requirements.txt`)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/digit-detection-unlearning.git
   cd digit-detection-unlearning
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Open the application** in your browser at `http://127.0.0.1:5000`.

### Running in Development

To run the application in development mode, use Flask’s debug mode by setting the following environment variables:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

This will automatically reload the server as changes are made to the codebase.

## Usage

1. **Draw a Digit**: Use the canvas to draw a digit (0-9) and click **"Predict Digit"** to see the model's prediction.
2. **Upload Image**: You can also upload an image of a digit for prediction.
3. **Unlearning Algorithms**: 
   - Select an unlearning algorithm (Standard, SISA, Approximate, Certified, Fine-tuning) from the dropdown.
   - Enter a digit to "unlearn" and click **"Unlearn Digit"** to see how the model is retrained to "forget" this digit.
4. **Visualization**: The visualization panel will show how the selected unlearning algorithm affects the model through interactive graphics.

## Unlearning Algorithms

- **Standard Unlearning**: Basic removal of the selected digit from the dataset.
- **SISA Unlearning**: Uses sharding and selective retraining for efficient unlearning.
- **Approximate Unlearning**: Perturbs the model weights to reduce the influence of specific data.
- **Certified Unlearning**: Systematically reduces the influence of the unlearned data by adjusting parameters based on an influence estimate.
- **Fine-Tune Unlearning**: Retrains parts of the model specifically related to the unlearned data for minimal overall impact.

## Contributing

We welcome contributions to improve this project. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Open a pull request when ready.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact [hafizur.upm@gmail.com].

---

This application is part of a broader initiative to explore and visualize the concept of **machine unlearning** in machine learning models. It is designed for research, educational, and demonstration purposes.
