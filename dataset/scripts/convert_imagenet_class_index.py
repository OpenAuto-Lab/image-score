import json

# Load the JSON data from the file
with open('../data/imagenet_class_index.json', 'r') as file:
    data = json.load(file)

formatted_data = {}

for key, value in data.items():
    formatted_data[value[0]] = value[1]

with open('../data/imagenet_class_index_formatted.json', 'w') as file:
    json.dump(formatted_data, file, indent=4)

