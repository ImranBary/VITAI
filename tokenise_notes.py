from transformers import BertTokenizer
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to read and tokenize a text file using BERT
def process_text_file(file_path):
    # Open and read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    # Tokenize the text using BERT tokenizer
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        max_length=512
    )
    
    # Extract token IDs and attention masks
    token_ids = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask']
    
    # Extract patient ID from the file name
    patient_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
    
    # Return a dictionary with patient ID, token IDs, and attention masks
    return {'Id': patient_id, 'Token_IDs': token_ids, 'Attention_Masks': attention_masks}

# Function to process a batch of files concurrently
def process_files_in_batch(file_paths):
    data = []
    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_text_file, file_path) for file_path in file_paths]
        for future in as_completed(futures):
            data.append(future.result())
    return data

def main():
    # Define the folder path containing the text files
    folder_path = '/Users/ImranBary/DataGen/synthea/output/notes/'
    
    # Get a list of all text file paths in the folder
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Define a suitable batch size for processing files
    batch_size = 20000
    results = []

    # Process files in batches for memory efficiency
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_results = process_files_in_batch(batch_paths)
        results.extend(batch_results)
        # Optional: Save intermediate results to disk or database

    # Convert the results to a DataFrame
    notes_df = pd.DataFrame(results)

    # Load your structured data from a CSV file
    structured_df = pd.read_csv('aggregated_patient_data.csv')

    # Merge the tokenized notes data with the structured data
    combined_df = pd.merge(structured_df, notes_df, on='Id', how='left')

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('combined_data.csv', index=False)

# Entry point of the script
if __name__ == "__main__":
    main()