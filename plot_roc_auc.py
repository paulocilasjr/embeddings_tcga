import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def find_roc_curves(data):
    """
    Recursively searches for 'roc_curve' keys in a nested dictionary.

    Parameters:
        data (dict): JSON-like dictionary.

    Returns:
        dict: A dictionary with keys as labels and their corresponding roc_curve data.
    """
    results = {}

    def recursive_search(sub_data, parent_key=""):
        if isinstance(sub_data, dict):
            for key, value in sub_data.items():
                if key == "roc_curve":
                    # Use the parent key as the label for the curve
                    results[parent_key] = value
                else:
                    # Continue searching in nested dictionaries
                    recursive_search(value, parent_key if parent_key else key)
        elif isinstance(sub_data, list):
            # Search in a list of dictionaries
            for item in sub_data:
                recursive_search(item, parent_key)

    recursive_search(data)
    return results

def plot_roc_auc_curve(file_path, save_name):
    """
    Reads a JSON file containing false positive and true positive rates,
    then plots the ROC-AUC curve and calculates the AUC.

    Parameters:
        file_path (str): Path to the JSON file.
        save_name (str): Name of the file to save the plot.
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Find ROC curves in the nested JSON
        roc_curves = find_roc_curves(data)

        # Iterate over the extracted ROC curves and plot each
        for label, curve_data in roc_curves.items():
            false_positive_rate = curve_data["false_positive_rate"]
            true_positive_rate = curve_data["true_positive_rate"]

            # Calculate AUC
            roc_auc = auc(false_positive_rate, true_positive_rate)

            # Plot the ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(false_positive_rate, true_positive_rate, label=f'{label} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {label}')
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(save_name)
            print(f"Plot saved as: {save_name}")
            plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError as e:
        print(f"Key error: {e}. Ensure the JSON structure matches the expected format.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC-AUC curve from JSON file.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file containing ROC data.")
    parser.add_argument("save_name", type=str, help="File name to save the ROC plot.")
    args = parser.parse_args()

    # Call the function with command-line arguments
    plot_roc_auc_curve(args.file_path, args.save_name)

