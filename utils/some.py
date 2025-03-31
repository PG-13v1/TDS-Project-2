import json
import copy

# Path to your GA1_param.json file
input_path = r"c:\Users\Pratul\OneDrive\Desktop\TDS_Project_2\TDS-project-2\utils\GA1_param.json"

with open(input_path, "r") as f:
    data = json.load(f)

# Define the base template (same as vs_code_version)
base_template = {
    "name": None,  # will be set per function object
    "description": "description",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to extract the data from"
            }
        },
        "required": ["text"]
    }
}

# Update each function object to exactly match the base template
for task in data.get("tasks", []):
    if task.get("type") == "function" and "function" in task:
        # Get the original function name
        function_name = task["function"].get("name")
        # Create a copy of the template and assign the function name
        new_func = copy.deepcopy(base_template)
        new_func["name"] = function_name
        # Replace the function object
        task["function"] = new_func

with open(input_path, "w") as f:
    json.dump(data, f, indent=2)

print("GA1_param.json updated successfully.")