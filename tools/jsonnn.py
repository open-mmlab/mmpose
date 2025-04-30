import json
import sys

def format_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python format_json.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        try:
            format_json(input_file, output_file)
            print(f"Formatted JSON saved to {output_file}")
        except Exception as e:
            print(f"Error: {e}")
