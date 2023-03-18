# Plant_Disease_Detection
## Introduction:
Plant diseases can cause significant damage to crops, leading to economic losses and food shortages. In recent years, there has been a lot of interest in developing plant disease detection systems using computer vision techniques. In this project, we will use Convolutional Neural Networks (CNN) to detect plant diseases and provide information about the cause of the disease, steps to prevent it, and suggest pesticides to control the disease.

### Dataset:
To train our CNN model, we will need a dataset of images of plants with both healthy and diseased states. One of the popular datasets used for this task is the 'PlantVillage' dataset that contains 54,306 images of plants with 38 different diseases and one healthy state.

### Preprocessing:
We will start by preprocessing the images before feeding them to the CNN model. We will first resize the images to a standard size (e.g., 224x224 pixels) and then normalize the pixel values between 0 and 1.

### Model Architecture:
For our CNN model, we will use a pre-trained architecture such as VGG16 or ResNet50 as they have been shown to perform well on image classification tasks. We will remove the top layers of the pre-trained model and add new layers that are specific to our task. The final layer will be a dense layer with 38 output units and a softmax activation function to predict the probability of each disease.

### Training:
We will split the dataset into training and validation sets and train the model using the Adam optimizer and categorical cross-entropy loss function. We will also use data augmentation techniques such as rotation, zoom, and horizontal flip to increase the size of the dataset and reduce overfitting.

### Testing:
To test the performance of our model, we will use a separate test set and calculate the accuracy, precision, recall, and F1-score. If the model achieves satisfactory performance, we can move on to the final step.

### Plant Disease Detection and Information:
We will use Streamlit, a Python library, to create a web application that allows users to upload an image of a plant and detect the disease. Once the disease is detected, the application will provide information about the cause of the disease, steps to prevent it, and suggest pesticides to control the disease. We can use a database that contains information about the different diseases, their causes, and the recommended pesticides.

### Pesticides with Image and Link to Buy:
To suggest pesticides, we will use a database that contains information about the different pesticides and their recommended use for each disease. The application will also display an image of the recommended pesticide and a link to buy it from an online store such as Amazon.

### Conclusion:
In this project, we have developed a plant disease detection system using CNN and provided information about the cause of the disease, steps to prevent it, and suggested pesticides to control the disease. This system can potentially help farmers detect and control plant diseases early, leading to higher crop yields and reduced economic losses. Further improvements can be made by incorporating other sources of data such as soil moisture, temperature, and humidity to enhance the accuracy of the disease detection.

## [Learn Visually - Experience the app here](https://plant-disease-detection.streamlit.app/)
