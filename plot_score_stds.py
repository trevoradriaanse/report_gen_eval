import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the result JSON files
results_dir = "/home/hltcoe/tadriaanse/SCALE/SCALE2025/report_gen_eval/results/runs/openai_original_prompt_seed"

# Store precision/recall data
topic_metrics = {}

# Read all JSON result files
for filename in os.listdir(results_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(results_dir, filename), "r") as f:
            data = json.load(f)
            for topic_id, metrics in data.items():
                precision = metrics["metrics"]["precision"]
                recall = metrics["metrics"]["recall"]

                if topic_id not in topic_metrics:
                    topic_metrics[topic_id] = {"precision": [], "recall": []}

                # Append precision and recall values
                topic_metrics[topic_id]["precision"].append(precision)
                topic_metrics[topic_id]["recall"].append(recall)

# Compute the mean and std deviation for each topic
means = {}
std_devs = {}

for topic_id, metrics in topic_metrics.items():
    precision_mean = np.mean(metrics["precision"])
    recall_mean = np.mean(metrics["recall"])

    precision_std = np.std(metrics["precision"])
    recall_std = np.std(metrics["recall"])

    means[topic_id] = {"precision": precision_mean, "recall": recall_mean}
    std_devs[topic_id] = {"precision": precision_std, "recall": recall_std}

# Prepare data for plotting
topics = list(means.keys())
precision_means = [means[topic]["precision"] for topic in topics]
recall_means = [means[topic]["recall"] for topic in topics]
precision_stds = [std_devs[topic]["precision"] for topic in topics]
recall_stds = [std_devs[topic]["recall"] for topic in topics]

# Plot the means with standard deviation error bars
fig, ax = plt.subplots(figsize=(10, 6))

# Plot precision
ax.errorbar(
    topics,
    precision_means,
    yerr=precision_stds,
    fmt="o",
    label="Precision",
    capsize=5,
    color="blue",
)

# Annotate precision points
for i, topic in enumerate(topics):
    precision_text = f"{precision_means[i]:.2f}±{precision_stds[i]:.2f}"
    ax.text(
        topic,
        precision_means[i] + precision_stds[i] + 0.01,  # Adjust position
        precision_text,
        ha="center",
        color="blue",
        fontsize=8,
    )

# Plot recall
ax.errorbar(
    topics,
    recall_means,
    yerr=recall_stds,
    fmt="o",
    label="Recall",
    capsize=5,
    color="green",
)

# Annotate recall points
for i, topic in enumerate(topics):
    recall_text = f"{recall_means[i]:.2f}±{recall_stds[i]:.2f}"
    ax.text(
        topic,
        recall_means[i] + recall_stds[i] + 0.01,  # Adjust position
        recall_text,
        ha="center",
        color="green",
        fontsize=8,
    )

# Labels and title
ax.set_xlabel("Topic ID")
ax.set_ylabel("Scores")
ax.set_title("5 runs on dev topics with openai gpt-4o-2024-11-20 old prompt seeded")
ax.legend()

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_plot_openai_new_prompt_seeds.png")  # Save plot as a PNG file
plt.show()
