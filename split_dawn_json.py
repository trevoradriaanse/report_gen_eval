import json

# Input JSONL file
input_file = "neuclir24-dawn-assessor-bank.jsonl"

# Output files for each language
output_files = {
    "fas": open("fas.jsonl", "w", encoding="utf-8"),
    "rus": open("rus.jsonl", "w", encoding="utf-8"),
    "zho": open("zho.jsonl", "w", encoding="utf-8"),
}

try:
    # Read the input file line by line
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                query = json.loads(line)
                src_lang = query.get("info", {}).get("src_lang")
                if src_lang in output_files:
                    # Write to the appropriate file
                    output_files[src_lang].write(json.dumps(query) + "\n")
                else:
                    print(f"Unrecognized src_lang: {src_lang}")
finally:
    # Close all output files
    for f in output_files.values():
        f.close()

print("Queries have been split by src_lang into separate files.")
