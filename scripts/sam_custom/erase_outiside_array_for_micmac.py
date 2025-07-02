import os

# Define the directory and limits
directory = "/home/samuelebumbaca/repositories/deep-image-matching-sam/assets/example_plant/micmac_/Homol"
x_limit = 3280
y_limit = 2464


def process_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            x1, y1, x2, y2, _ = map(float, parts)
            if x1 <= x_limit and y1 <= y_limit and x2 <= x_limit and y2 <= y_limit:
                cleaned_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(cleaned_lines)


def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                process_file(file_path)


# Run the script
process_directory(directory)
