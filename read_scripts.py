import os

# Gather .cpp and .h files from the current directory
cpp_file_list = [f for f in os.listdir('.') if (f.endswith('.cpp') or f.endswith('.h'))]

# Output file to store the combined content
output_file = "combined_cpp_scripts.txt"

def read_and_combine_files(file_list, output_file):
    """Read specific C++ files and combine their contents."""
    combined_content = []
    
    for file_path in file_list:
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Add a header to indicate which file this content is from
                    file_header = f"# {'=' * 40}\n# File: {file_path}\n# {'=' * 40}\n"
                    combined_content.append(file_header + content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Write the combined content to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("\n\n".join(combined_content))
    
    print(f"Combined {len(file_list)} C++ files into {output_file}")
    print(f"Files processed: {', '.join(file_list)}")

# Run the function
read_and_combine_files(cpp_file_list, output_file)