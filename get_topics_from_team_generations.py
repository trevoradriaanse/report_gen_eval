import json

# Input JSONL file
input_file = "/home/hltcoe/tadriaanse/SCALE/SCALE2025/rubric-neuclir/neuclir/zho/group1-account-rose"

output_file = "/home/hltcoe/tadriaanse/SCALE/SCALE2025/rubric-neuclir/neuclir/zho/group1-account-rose-only351"

# Open the input file for reading and output file for writing
with (
    open(input_file, "r", encoding="utf-8") as infile,
    open(output_file, "w", encoding="utf-8") as outfile,
):
    for line in infile:
        try:
            # Parse each line as JSON
            data = json.loads(line.strip())
            # Check if query_id equals 307
            if data.get("request_id") == "351":
                # Write the matching object to the output file
                json.dump(data, outfile)
                outfile.write("\n")
        except json.JSONDecodeError:
            # Handle invalid JSON lines
            print(f"Invalid JSON encountered: {line.strip()}")

print(f"Filtered data saved to {output_file}.")
