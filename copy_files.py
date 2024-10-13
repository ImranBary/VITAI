import shutil
import os

def copy_files(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        # Copy each file to the destination directory
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

# Example usage
source_directory = '/Users/ImranBary/Library/Mobile Documents/com~apple~CloudDocs/Documents/FinalYearProj'
destination_directory = '/Users/ImranBary'
copy_files(source_directory, destination_directory)