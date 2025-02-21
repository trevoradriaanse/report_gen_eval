import json
import re


def extract_citations(sentence):
    # Extract citations without brackets, e.g., D2 instead of [D2]
    citations = re.findall(r"D\d+", sentence)
    # Remove citations from the sentence text
    cleaned_sentence = re.sub(r"\[?D\d+\]?", "", sentence).strip()
    return citations, cleaned_sentence


def process_paragraph(paragraph):
    # Decode Unicode escape sequences (if any)
    paragraph = paragraph.encode().decode("unicode_escape")

    # Split paragraph into sentences (naive split by period, more sophisticated parsing can be added)
    sentences = paragraph.split(". ")
    result = []

    for sentence in sentences:
        citations, cleaned_sentence = extract_citations(sentence)
        sentence_data = {
            "text": cleaned_sentence + "."
            if not cleaned_sentence.endswith(".")
            else cleaned_sentence,
            "citations": citations,
        }
        result.append(sentence_data)

    return result


def create_json(request_id, run_id, collection_ids, paragraph):
    sentences_data = process_paragraph(paragraph)
    data = {
        "request_id": request_id,
        "run_id": run_id,
        "collection_ids": collection_ids,
        "sentences": sentences_data,
    }
    return data


def read_paragraph_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Example usage:
file_path = "./report.txt"

# Read paragraph from the file
paragraph_input = read_paragraph_from_file(file_path)

request_id = "301"
run_id = "test"
collection_ids = ["test"]

# Generate the JSON
json_output = create_json(request_id, run_id, collection_ids, paragraph_input)
with open("avengers_report.jsonl", "w") as fp:
    fp.write(json.dumps(json_output))
