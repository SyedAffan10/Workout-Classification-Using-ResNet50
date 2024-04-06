# Workout Classification using ResNet50

This project focuses on exercise classification using the ResNet50 architecture implemented in PyTorch. The primary objective is to classify exercise videos into two categories, enabling users to identify and categorize exercises accurately. The project utilizes an open-source dataset sourced from Kaggle, which will be provided in the repository. Additionally, a Streamlit application is created to allow users to upload exercise images for classification, with predictions displayed upon video upload completion.

![Project Demo](https://github.com/SyedAffan10/Workout-Classification-Using-ResNet50/blob/main/Demo_Image.PNG)


**Key Components:**
1. **Model Architecture:**  
   - ResNet50 is employed as the core architecture for exercise classification, known for its deep learning capabilities.
2. **Training Configuration:**  
   - Batch Size: 32  
   - Epochs: 10  
   - Classes: 2  
   - Learning Rate: 0.001  

**Usage:**
1. **Clone the Repository:**
   ```
   git clone https://github.com/SyedAffan10/Workout-Classification-Using-ResNet50.git
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download Dataset:**
   - Download the exercise dataset from the provided link and place it in the "dataset" directory within the project.

4. **Run the Streamlit Application:**
   ```
   streamlit run app.py
   ```

5. **Upload Image for Prediction:**
   - Upon launching the Streamlit application, users can upload exercise video using the provided interface.

6. **View Prediction:**
   - After video upload, the application promptly processes the video and displays the classification result based on the trained ResNet50 model.

**Dataset:**
- The exercise classification dataset used in this project will be provided in the repository. It contains images belonging to two distinct exercise categories.

**Note:**
- Ensure all necessary Python packages are installed before executing the Streamlit application.
- The trained ResNet50 model weights should be available in the repository or downloaded during runtime for prediction.

**Contributing:**
- Contributions to this project are encouraged. Users are welcome to provide feedback, suggestions, or contribute to the codebase.

**License:**
- This project is licensed under the [MIT License](LICENSE).

**Acknowledgments:**
- Special thanks to Kaggle for providing the dataset essential to this project.
- Credits to the PyTorch community for their contributions to the development of the ResNet50 architecture.

**Contact:**
- For inquiries or collaborations, please contact [syedaffan.dev@gmail.com](mailto:syedaffan.dev@gmail.com).
