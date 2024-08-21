# Facial Expression Recognition 
https://youtu.be/wi9gptJcf8s
This project focuses on facial expression recognition using deep learning techniques. It includes a dataset preprocessing script and a CNN model for training.

## Dependencies

The following Python packages are required for this project, along with their versions:

- Matplotlib 3.8.0
- NumPy 1.26.0
- Pandas 2.1.0
- Seaborn 0.12.2
- TensorFlow 2.13.0
- Keras 

You can install these dependencies using `pip`. For example:
Ensure that you have the model.h5 file (the trained model) and the webcam.py file in the same folder.

Install any necessary dependencies (if not already installed) in your local IDE.

Run the main.py script:

bash
Copy code
python webcam.py
The script will use your webcam feed to recognize facial expressions in real-time.

The current model has been trained on the FER (Facial Expression Recognition) dataset and achieves an accuracy of approximately 57%. If you're interested in improving this project, contributions are welcome!

Feel free to fork the repository, make improvements, and submit pull requests to enhance the model's accuracy and functionality.
If you have any questions, suggestions, or would like to collaborate on improving this project, please feel free to contact us at pk20180708248@gmail.com or https://www.linkedin.com/in/pavan-kalyan-39247a213/.

STEPS TO SETUP THE PROJECT IN YOUR LOCAL MACHINE

CREATE A VIRTUAL ENVIRONMENT(RECOMMENDED)
.\venv\Scripts\activate
source venv/bin/activate

python -m venv venv
# Clone the repository
git clone https://github.com/pavankalyan224847/facial-expression-recognition.git

# Install dependencies
cd facial-expression-recognition
pip install -r requirements.txt

#Run the dataset cleaning and model training script:
python modeltrain.py
Ensure that you have the model.h5 file (the trained model) and the webcam.py file in the same folder.

Install any necessary dependencies (if not already installed) in your local IDE.

Run the webcam-based emotion recognition script:
python webcam.py

The current model achieves an accuracy of approximately 57% on the FER (Facial Expression Recognition) dataset. Contributions to improve this project's accuracy and functionality are welcome!

# Run the project
python main.py
pip install matplotlib==3.8.0 numpy==1.26.0 pandas==2.1.0 seaborn==0.12.2 tensorflow keras
The project's main components and usage instructions are as follows:

1. Dataset Exploration: The code includes functionality to explore the dataset. You can visualize sample images from the dataset to understand the emotions present.

2. Data Preprocessing: Data augmentation and preprocessing are performed using the ImageDataGenerator from Keras.

Model Building: A Convolutional Neural Network (CNN) model is built for emotion recognition.

3. Training the Model: The model is trained using the prepared dataset. The training process includes early stopping and model checkpoint callbacks.

4. Visualizing Training Results: The code provides plots for training and validation loss and accuracy.

5. Dataset
This project uses the FER (Facial Expression Recognition) dataset for training and validation. Ensure that you have the dataset properly organized in the specified folder path.

Contributing
If you'd like to contribute to this project, please fork the repository, make improvements, and submit pull requests. Feel free to open issues for bug reports or feature requests as well.




![accuracy](https://github.com/pavankalyan224847/facialexpressionrecognition/assets/124815665/d12c42f2-68dc-48c7-b997-6d04de297984)











