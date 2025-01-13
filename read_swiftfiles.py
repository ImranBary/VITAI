import os

# Directories containing the .swift files
directories = [
    r'C:\Users\imran\Downloads\6COSC021W_W1843601_Imran_Feisal\6COSC021W_W1843601_Imran_Feisal\WeatherAPP_CWK_Design\WeatherApp\WeatherApp\Views',
    r'C:\Users\imran\Downloads\6COSC021W_W1843601_Imran_Feisal\6COSC021W_W1843601_Imran_Feisal\WeatherAPP_CWK_Design\WeatherApp\WeatherApp\ViewModel',
    r'C:\Users\imran\Downloads\6COSC021W_W1843601_Imran_Feisal\6COSC021W_W1843601_Imran_Feisal\WeatherAPP_CWK_Design\WeatherApp\WeatherApp\Model'
]

# Output file
output_file = r'C:\Users\imran\Downloads\6COSC021W_W1843601_Imran_Feisal\file.txt'

# Open the output file in append mode
with open(output_file, 'a') as outfile:
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.swift'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as infile:
                        content = infile.read()
                        outfile.write(content + '\n')