import pandas as pd
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import ast


# Load the combined data
combined_df = pd.read_csv('combined_data.csv')

# Separate structured and unstructured data
unstructured_cols = [
    'Token_IDs','Attention_Masks'
]

combined_df = combined_df[unstructured_cols]





combined_df['Token_IDs'] = combined_df['Token_IDs'].apply(ast.literal_eval)
combined_df['Attention_Masks'] = combined_df['Attention_Masks'].apply(ast.literal_eval)

# Define maximum sequence length (should be consistent with your tokenizer settings)
MAX_SEQ_LEN = 512

# Pad token sequences and attention masks
token_ids_padded = pad_sequences(
    combined_df['Token_IDs'],
    maxlen=MAX_SEQ_LEN,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
)

attention_masks_padded = pad_sequences(
    combined_df['Attention_Masks'],
    maxlen=MAX_SEQ_LEN,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
)

# Save padded sequences back to the DataFrame
combined_df['Token_IDs_Padded'] = list(token_ids_padded)
combined_df['Attention_Masks_Padded'] = list(attention_masks_padded)

# Save the preprocessed unstructured data
combined_df.to_csv('unstructured_data_preprocessed.csv', index=False)