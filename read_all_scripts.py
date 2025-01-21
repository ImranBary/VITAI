import os

# List of .py files to read
py_files = [
    # 'data_preprocessing.py',
    # 'health_index.py',
    # 'charlson_comorbidity.py',
    # 'vae_model.py',
    # 'tabnet_model.py'
    'comprehensive_testing_mem_optimized.py',
    'update_comprehensive_results.py',
    'update_model_metrics.py'
]

# Initialize an empty string to hold the combined content
combined_content = ""

# Iterate over each file in the list
for file_name in py_files:
    # Check if the file exists
    if os.path.exists(file_name):
        # Open the file and read its content
        with open(file_name, 'r', encoding='utf-8') as file:
            combined_content += file.read() + "\n"
    else:
        print(f"File {file_name} does not exist.")
        # Save the combined content to a text file

# Print the combined content
#print(combined_content)
with open('scripts.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(combined_content)