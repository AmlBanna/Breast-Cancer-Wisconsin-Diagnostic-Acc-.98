# Breast Cancer Wisconsin (Diagnostic) Classifier

This project implements a neural network classifier to predict whether a breast cancer tumor is malignant or benign based on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Project Overview

The goal of this project is to build and train a deep learning model using TensorFlow and Keras to accurately classify tumors from the WDBC dataset. The notebook covers the complete machine learning workflow, including data loading, preprocessing, exploratory data analysis (EDA), model building, training, and evaluation.

The final model achieves a test accuracy of approximately **96.5%**.

## Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) Data Set**.

  * **Source:** Kaggle (originally from the UCI Machine Learning Repository).
  * **Features:** The dataset contains 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast mass. These features describe characteristics of the cell nuclei present in the image, such as:
      * `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
      * `smoothness_mean`, `compactness_mean`, `concavity_mean`
      * ...and their corresponding standard error (`_se`) and worst (`_worst`) values.
  * **Target:** The `diagnosis` column, which is binary:
      * **M** (Malignant)
      * **B** (Benign)

## Workflow

The project follows these key steps:

1.  **Data Loading & Inspection:** The `data.csv` file is loaded into a pandas DataFrame. Initial inspection is done using `.head()`, `.info()`, and `.describe()`.
2.  **Data Preprocessing:**
      * The unnecessary `id` and `Unnamed: 32` columns are dropped.
      * The categorical `diagnosis` column is mapped to numerical values (M=1, B=0).
3.  **Exploratory Data Analysis (EDA):**
      * The class distribution (Malignant vs. Benign) is visualized using a count plot.
      * Feature distributions (mean, se, worst) are plotted using histograms.
      * A correlation matrix is generated and plotted as a heatmap to understand feature relationships.
4.  **Data Splitting:** The data is split into training (64%), validation (16%), and testing (20%) sets.
5.  **Scaling:** Features are standardized using `StandardScaler` from scikit-learn.
6.  **Model Building:**
      * A Sequential model is built using **TensorFlow/Keras**.
      * The architecture consists of an input layer, followed by Dense layers (128, 64, 32 neurons) with ReLU activation, each followed by BatchNormalization and Dropout layers to prevent overfitting.
      * The output layer is a single Dense neuron with a sigmoid activation function for binary classification.
7.  **Model Training:**
      * The model is compiled with the `Adam` optimizer, `binary_crossentropy` loss, and `accuracy` as the metric.
      * Callbacks (`EarlyStopping` and `ReduceLROnPlateau`) are used to optimize training and prevent overfitting.
      * The model is trained for up to 100 epochs.
8.  **Evaluation:**
      * The trained model is evaluated on the unseen test set, achieving an accuracy of **\~96.5%**.
      * Training & validation accuracy and loss curves are plotted to visualize model performance.
      * A confusion matrix is generated and plotted for a detailed look at the test set predictions.
9.  **Model Saving:** The final trained model is saved to a file named `breast_cancer_model.h5`.

## Results

  * **Test Accuracy:** \~96.5%
  * **Learning Curves:** The training and validation loss/accuracy plots show that the model learns well and does not significantly overfit.
  * **Confusion Matrix:**
  *  ![Confusion Matrix](Confusion_matrix.png)


## How to Use

### Prerequisites

This project uses the following Python libraries:

  * TensorFlow (Keras)
  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib
  * Seaborn

You can install these dependencies using pip:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Download the data:**
    Ensure the `data.csv` file from the Kaggle dataset is placed in the correct directory (e.g., `/kaggle/input/breast-cancer-wisconsin-data/`).
3.  **Run the notebook:**
    Open and run the `breast-cancer-wisconsin-diagnostic-acc-98.ipynb` notebook using Jupyter Notebook or Jupyter Lab.
    ```bash
    jupyter notebook breast-cancer-wisconsin-diagnostic-acc-98.ipynb
    ```
4.  **Use the trained model:**
    The notebook saves the trained model as `breast_cancer_model.h5`, which can be loaded for future predictions.
