import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to read and process a text file
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and perform lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    return ' '.join(processed_tokens)

# Assuming you have a folder path that contains all text files
folder_path = '/Users/ImranBary/DataGen/synthea/output/notes/'
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
file_paths = file_paths[:3]


# Process each file and store in a dictionary with patient IDs
patient_notes = {}
for file_path in file_paths:
    patient_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
    patient_notes[patient_id] = process_text_file(file_path)


# Convert notes to DataFrame
notes_df = pd.DataFrame(list(patient_notes.items()), columns=['Id', 'Notes'])

notes_df.to_csv('notes_nlp.csv',index=False)



# Load your structured data
# structured_df = pd.read_csv('path_to_your_csv.csv')

# Merge data
# combined_df = pd.merge(structured_df, notes_df, on='Id', how='left')