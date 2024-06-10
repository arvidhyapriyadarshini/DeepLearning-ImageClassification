# DeepLearning-ImageClassification
Data Science Projects - Deep Learning

Bird Species Identification using Deep Learning

The project aimed to develop an automated bird species classification system using deep learning to aid biodiversity research and conservation efforts. Traditional manual methods are labor-intensive and error-prone. The project focused on classifying images into 20 bird species using Convolutional Neural Networks (CNNs) and transfer learning techniques. The dataset for this project is obtained from Kaggle.

Data was organized into training, validation, and test sets. Data augmentation (e.g., rotation, shifting, zooming, flipping) and normalization were applied to enhance model robustness.

Model and Technique Selection
Conventional CNN Model:
Implemented using Keras with sequential layers, achieving training accuracy of 99.67% and validation accuracy of 79.00%, but with overfitting issues. To mitigate these issues, data augmentation was performed.

Transfer Learning Techniques:
Utilized pre-trained models (ResNet101V2, DenseNet121, VGG-16, InceptionV3) with modifications to suit bird classification.

Evaluation Metrics
Training/Validation/Test Accuracy:
Measures the model's predictive performance on respective datasets.

Training/Validation/Test Loss:
Represents the error in model predictions.

Results
ResNet101V2:
Achieved 99% test accuracy with 0.05 test loss, demonstrating the best performance.

DenseNet121:
Achieved 97% test accuracy with 0.09 test loss.

VGG16:
Achieved 91% test accuracy with 0.26 test loss.

InceptionV3:
Achieved 97% test accuracy with 0.08 test loss.

Conventional CNN:
Achieved 77.9% test accuracy with 0.59 test loss, indicating lower performance.

Grad-CAM Process
Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize important image regions influencing the model's predictions, confirming the model's focus on relevant features.

Bird Species Prediction Web App
Developed using Streamlit, the web app allows users to upload bird images and receive species classifications, employing the best-performing ResNet101V2 model.






