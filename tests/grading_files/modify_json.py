import json
import uuid


def update_json_fields(json_data, report_request, questions_data):
    """
    Update the 'query_text' and 'items' based on the report request and question-answer data.
    """
    # Set the 'query_text' field with the report request
    json_data["query_text"] = report_request

    # Update 'items' with question and gold_answers
    json_data["items"] = []
    for q_data in questions_data:
        question_id = uuid.uuid4().hex
        item = {
            "query_id": json_data["query_id"],  # Use the same query_id for all items
            "info": {
                "importance": "vital",  # Set default importance, can be adjusted if needed
                "used": False,  # Default value for 'used'
            },
            "question_id": f"{json_data['query_id']}_{str(question_id)}",  # Generate a unique ID
            "question_text": q_data["question"],
            "gold_answers": [
                {"answer": q_data["answer"], "citations": q_data["citations"]}
            ],
        }
        json_data["items"].append(item)

    return json_data


def read_jsonl_file(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects.
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def write_json_file(file_path, json_data):
    """
    Writes the updated JSON data to a new file.
    """
    with open(file_path, "w") as file:
        json.dump(json_data, file, indent=4)


def main():
    input_jsonl_file = (
        "nuggets.txt"  # Path to the input JSONL file with questions and answers
    )
    output_json_file = "avengers_nuggets.jsonl"  # Path to the output JSON file

    # The report request text
    report_request = "I am a Hollywood reporter writing an article about the highest grossing films Avengers: Endgame and Avatar. My article needs to include when each of these films was considered the highest grossing films and any manipulations undertaken to bring moviegoers back to the box office with the specific goal of increasing the money made on the film."

    # Read the question-answer JSONL data
    questions_data = read_jsonl_file(input_jsonl_file)

    # Initial JSON structure to be updated
    json_data = {
        "query_id": "300",
        "test_collection": "rus_2024",
        "query_text": "",
        "hash": 1111,
        "items": [],
    }

    # Update the JSON structure with the report request and question-answer pairs
    updated_data = update_json_fields(json_data, report_request, questions_data)

    # Write the updated JSON data to the output file
    write_json_file(output_json_file, updated_data)


# Run the main function
if __name__ == "__main__":
    main()
