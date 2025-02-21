import os
import json
import numpy as np


# Function to read all the JSON files in a directory
def read_json_files(input_directory):
    files_data = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                files_data.append(data)

    return files_data


# Function to calculate standard deviation per topic
def calculate_standard_deviation(files_data):
    # Dictionary to store precision and recall values per topic
    topic_metrics = {}

    # Loop through each file's data
    for file_data in files_data:
        for topic, content in file_data.items():
            metrics = content.get("metrics", {})
            precision = metrics.get("precision")
            recall = metrics.get("recall")

            if topic not in topic_metrics:
                topic_metrics[topic] = {"precision": [], "recall": []}

            # Add precision and recall to their respective lists
            if precision is not None:
                topic_metrics[topic]["precision"].append(precision)
            if recall is not None:
                topic_metrics[topic]["recall"].append(recall)

    # Calculate standard deviation for each topic
    topic_std_dev = {}
    for topic, values in topic_metrics.items():
        precision_std = (
            np.std(values["precision"]) if len(values["precision"]) > 1 else 0.0
        )
        recall_std = np.std(values["recall"]) if len(values["recall"]) > 1 else 0.0
        topic_std_dev[topic] = {
            "precision_std": precision_std,
            "recall_std": recall_std,
        }

    return topic_std_dev


# Function to output the results to a JSON file
def output_results(topic_std_dev, output_file):
    with open(output_file, "w") as file:
        json.dump(topic_std_dev, file, indent=4)


# Main function
def main(input_directory, output_file):
    # Read the JSON files
    files_data = read_json_files(input_directory)

    # Calculate standard deviation for each topic
    topic_std_dev = calculate_standard_deviation(files_data)

    # Output the results to a file
    output_results(topic_std_dev, output_file)
    print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_directory = "/path/to/your/json/files"  # Replace with the directory containing your JSON files
    output_file = (
        "/path/to/output/results.json"  # Replace with the desired output file path
    )
    main(input_directory, output_file)
