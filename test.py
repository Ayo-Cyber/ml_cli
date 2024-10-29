import pandas as pd
import ssl
import requests
import os
import sys
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

def main():
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/refs/heads/master/dataset.csv"
    
    # Attempt to load data with SSL certificate verification
    try:
        print("Trying to load dataset with SSL verification...")
        data = pd.read_csv(url)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset with SSL verification: {e}")

        # Bypass SSL verification if needed
        try:
            print("Bypassing SSL verification...")
            ssl._create_default_https_context = ssl._create_unverified_context
            data = pd.read_csv(url)
            print("Dataset loaded successfully by bypassing SSL verification.")
        except Exception as e:
            print(f"Error loading dataset while bypassing SSL verification: {e}")
            
            # Use requests to fetch the CSV file
            print("Trying to load dataset using requests...")
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for HTTP errors
                # Save the content to a CSV file
                with open('dataset.csv', 'wb') as f:
                    f.write(response.content)
                # Read the CSV file into a DataFrame
                data = pd.read_csv('dataset.csv')
                print("Dataset loaded successfully using requests.")
            except Exception as e:
                print(f"Failed to load dataset using requests: {e}")
                return

    # Now we can proceed to use TPOT for automated ML
    try:
        # Prepare the data for TPOT
        X = data.drop('target', axis=1)  # Replace 'target' with the actual target column name if different
        y = data['target']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the TPOT classifier
        tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

        # Fit the model on the training data
        print("Fitting the TPOT model...")
        tpot.fit(X_train, y_train)

        # Evaluate the model on the test data
        print("Evaluating the TPOT model...")
        print(f"Accuracy: {tpot.score(X_test, y_test)}")

        # Export the best pipeline
        tpot.export('best_pipeline.py')
        print("Best pipeline exported to 'best_pipeline.py'.")

    except Exception as e:
        print(f"An error occurred while running TPOT: {e}")

if __name__ == '__main__':
    # Check if running on macOS to install certificates if needed
    if sys.platform == "darwin":
        try:
            import subprocess
            print("Installing certificates for macOS...")
            subprocess.run(['/Applications/Python 3.x/Install Certificates.command'], check=True)
        except Exception as e:
            print(f"Failed to install certificates: {e}")

    main()
