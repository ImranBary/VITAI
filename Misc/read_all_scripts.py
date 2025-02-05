import os

# List of directories to read .py files from
directories = [
    '.',
    'vitai_scripts',
    'Finals',
    'Validations',
    'Explain_Xai'
]

# Initialize an empty string to hold the combined content
combined_content = ""

# Iterate over each directory in the list
for directory in directories:
    # Iterate over each file in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a .py file
        if file_name.endswith('.py'):
            file_path = os.path.join(directory, file_name)
            print(f"Reading file {file_path}...")
            # Check if the file exists
            if os.path.exists(file_path):
                # Open the file and read its content
                with open(file_path, 'r', encoding='utf-8') as file:
                    combined_content += file.read() + "\n"
            else:
                print(f"File {file_path} does not exist.")

# Save the combined content to a text file
with open('scripts.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(combined_content)
