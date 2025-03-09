import numpy as np
import polars as pl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ClassifierWrapper:
    def __init__(self, training_data):
        """
        Initialize and train the RandomForest model.

        Parameters:
        -----------
        training_data : str
            Path to the training data file.
        """
        # Save the training data path for future use.
        self.training_data = training_data

        # Read the dataset using Polars.
        df = pl.read_csv(
            training_data,
            separator=" ",
            has_header=False,
        )
        df.columns = ["image_file", "class_id", "x1", "y1", "x2", "y2"]

        # Drop the image_file column as it's not used for training.
        df = df.drop("image_file")
        X = df.drop("class_id")
        y = df["class_id"]

        # Define and train the RandomForestClassifier.
        self.model = RandomForestClassifier(
            n_estimators=30,            # Number of trees
            criterion='entropy',        # Splitting criterion
            max_depth=None,             # Maximum depth of trees
            min_samples_split=2,        # Minimum samples required to split
            min_samples_leaf=0.01,      # Minimum samples required at a leaf (fractional)
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',        # Maximum features per split
            max_leaf_nodes=None,        # Maximum leaf nodes
            min_impurity_decrease=0.0001, # Minimum impurity decrease to split
            bootstrap=True,             # Bootstrapping for sampling
            oob_score=True,             # Out-of-bag score
            n_jobs=-1,                  # Use all available cores
            random_state=20190305,      # Seed for reproducibility
            verbose=1,                  # Verbosity during training
            warm_start=False,           # Do not reuse previous training
            class_weight='balanced'     # Handle imbalanced classes
        )
        self.model.fit(X.to_pandas(), y.to_pandas())
    
    def predict(self, x1, y1, x2, y2) -> tuple[int, float]:
        """
        Predict the class label and confidence score for the given input features.

        Parameters:
        -----------
        x1 : float
        y1 : float
        x2 : float
        y2 : float

        Returns:
        --------
        prediction : int
            The predicted class label.
        confidence : float
            The confidence score (probability) of the prediction.
        """
        sample = pd.DataFrame([[x1, y1, x2, y2]], columns=["x1", "y1", "x2", "y2"])
        prediction = self.model.predict(sample)[0]
        probabilities = self.model.predict_proba(sample)[0]
        confidence = round(max(probabilities), 2)
        return prediction, confidence

    def evaluate_accuracy(self, test_size=0.2) -> float:
        """
        Evaluate the model accuracy using a train-test split on the labeled data.

        Parameters:
        -----------
        test_size : float, optional (default=0.2)
            Fraction of the data to be used as the test set.

        Returns:
        --------
        accuracy : float
            Accuracy score of the model on the test data.
        """
        # Load and prepare the data.
        df = pl.read_csv(
            self.training_data,
            separator=" ",
            has_header=False,
        )
        df.columns = ["image_file", "class_id", "x1", "y1", "x2", "y2"]
        df = df.drop("image_file")
        X = df.drop("class_id")
        y = df["class_id"]

        # Convert to pandas DataFrame for compatibility with scikit-learn.
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()

        # Split the data.
        X_train, X_test, y_train, y_test = train_test_split(
            X_pd, y_pd, test_size=test_size, random_state=42
        )

        # Define a new RandomForest model with the same parameters.
        model = RandomForestClassifier(
            n_estimators=30,
            criterion='entropy',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=0.01,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0001,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=20190305,
            verbose=1,
            warm_start=False,
            class_weight='balanced'
        )
        # Train the model on the training split.
        model.fit(X_train, y_train)

        # Predict on the test set.
        y_pred = model.predict(X_test)

        # Compute accuracy.
        accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy: {:.2f}%".format(accuracy * 100))
        return accuracy

if __name__ == "__main__":
    # Example usage:
    rf = RFModelWrapper("labeled_data.txt")
    # Evaluate accuracy using a train-test split (default 80% training, 20% testing)
    rf.evaluate_accuracy()
    # Example prediction
