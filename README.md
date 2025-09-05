# Hotel Booking Analysis

This notebook performs an exploratory data analysis and preprocessing of hotel booking data to prepare it for potential machine learning model training.

## Notebook Outline:

1.  **Loading the Dataset**: The notebook starts by loading the `hotel_bookings.csv` dataset into a pandas DataFrame.
2.  **Initial Data Exploration**:
    *   Displaying the shape of the dataset.
    *   Showing the first few rows of the DataFrame.
    *   Getting information about the dataset, including data types and non-null counts.
    *   Generating summary statistics for numerical columns.
3.  **Handling Missing Values**:
    *   Calculating and displaying the number and percentage of missing values for each column.
    *   Dropping the 'company' column due to a high percentage of missing values.
    *   Filling missing values in 'children', 'country', and 'agent' columns with appropriate values (mode for 'children', 'Unknown' for 'country', and 0 for 'agent').
4.  **Handling Duplicates**:
    *   Checking for and removing duplicate rows from the dataset.
5.  **Outlier Detection and Handling**:
    *   Identifying numerical columns and generating boxplots to visualize potential outliers.
    *   Calculating the number of outliers for each numerical column using the IQR method.
    *   Capping the outliers in numerical columns to the calculated upper and lower bounds.
6.  **Feature Engineering**:
    *   Converting 'reservation_status_date' and 'arrival_date' columns to datetime objects.
    *   Creating new features: 'total_guests' (sum of adults, children, and babies), 'total_nights' (sum of weekend and week nights), and 'is_family' (binary indicator if children or babies are present).
7.  **Removing Data Leakage**:
    *   Dropping 'reservation_status' and 'reservation_status_date' columns as they contain information about the booking outcome, which would lead to data leakage if used for predicting cancellations.
8.  **Encoding Categorical Variables**:
    *   Identifying categorical and numerical columns.
    *   Applying One-Hot Encoding to the categorical columns using `OneHotEncoder`.
    *   Creating a new DataFrame with the encoded categorical features.
    *   Concatenating the numerical and encoded categorical features to create the final feature matrix `X`.
9.  **Data Splitting**:
    *   Separating features (`X`) and target variable (`y`, 'is_canceled').
    *   Splitting the dataset into training, validation, and test sets (60% train, 20% validation, 20% test) using `train_test_split` with stratification to maintain the proportion of the target variable.

## Usage:

To run this notebook, ensure you have the `hotel_bookings.csv` file in the same directory or provide the correct path. The necessary libraries are imported at the beginning of the notebook.

The notebook provides a clean and preprocessed dataset (`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`) ready for training a classification model to predict hotel booking cancellations.
