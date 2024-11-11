import xlsxwriter
from datetime import datetime, timedelta

# Create a new Excel file and add a worksheet
workbook = xlsxwriter.Workbook('project_plan.xlsx')
worksheet = workbook.add_worksheet('Gantt Chart')

# Define the project tasks and timelines
tasks = [
    {'name': 'Proposal Development', 'start': '2024-09-16', 'end': '2024-11-14'},
    {'name': 'Data Generation & Preprocessing', 'start': '2024-11-15', 'end': '2024-11-30'},
    {'name': 'Health Index Calculation', 'start': '2024-12-01', 'end': '2024-12-10'},
    {'name': 'VAE Implementation', 'start': '2024-12-10', 'end': '2024-12-31'},
    {'name': 'TabNet Implementation', 'start': '2025-01-01', 'end': '2025-01-10'},
    {'name': 'Evaluation & Validation', 'start': '2025-01-10', 'end': '2025-01-30'},
    {'name': 'Explainable AI Integration', 'start': '2025-02-01', 'end': '2025-02-15'},
    {'name': 'Clustering & Mapping', 'start': '2025-02-16', 'end': '2025-02-28'},
    {'name': 'Dashboard Development', 'start': '2025-03-01', 'end': '2025-03-15'},
    {'name': 'Final Report & Presentation', 'start': '2025-03-16', 'end': '2025-03-20'},
]

# Convert date strings to datetime objects
for task in tasks:
    task['start_dt'] = datetime.strptime(task['start'], '%Y-%m-%d')
    task['end_dt'] = datetime.strptime(task['end'], '%Y-%m-%d')
    task['duration'] = (task['end_dt'] - task['start_dt']).days + 1

# Set up the worksheet columns
worksheet.set_column('A:A', 30)  # Task names
worksheet.set_column('B:B', 12)  # Start dates
worksheet.set_column('C:C', 12)  # End dates
worksheet.set_column('D:D', 10)  # Duration
worksheet.set_column('E:ZZ', 2)  # Gantt chart timeline

# Write headers
worksheet.write('A1', 'Task')
worksheet.write('B1', 'Start Date')
worksheet.write('C1', 'End Date')
worksheet.write('D1', 'Duration (days)')

# Write tasks and dates
date_format = workbook.add_format({'num_format': 'yyyy-mm-dd', 'align': 'left'})
bold_format = workbook.add_format({'bold': True})

row = 1
for task in tasks:
    worksheet.write_string(row, 0, task['name'])
    worksheet.write_datetime(row, 1, task['start_dt'], date_format)
    worksheet.write_datetime(row, 2, task['end_dt'], date_format)
    worksheet.write_number(row, 3, task['duration'])
    row += 1

# Create the Gantt chart representation
# Determine the earliest and latest dates
start_dates = [task['start_dt'] for task in tasks]
end_dates = [task['end_dt'] for task in tasks]
min_date = min(start_dates)
max_date = max(end_dates)

# Create a list of all dates in the project timeline
timeline = []
current_date = min_date
while current_date <= max_date:
    timeline.append(current_date)
    current_date += timedelta(days=1)

# Write the timeline dates in the header row starting from column 4 (column 'E')
col = 4
date_to_col = {}
for date in timeline:
    col_letter = xlsxwriter.utility.xl_col_to_name(col)
    worksheet.write_datetime(0, col, date, date_format)
    worksheet.set_column(col, col, 2)
    date_to_col[date.strftime('%Y-%m-%d')] = col
    col += 1

# Format for the Gantt chart bars
gantt_format = workbook.add_format({'bg_color': '#4CAF50'})

# Fill in the Gantt chart bars
row = 1
for task in tasks:
    start_col = date_to_col[task['start_dt'].strftime('%Y-%m-%d')]
    end_col = date_to_col[task['end_dt'].strftime('%Y-%m-%d')]
    # Write the bar
    for col in range(start_col, end_col + 1):
        worksheet.write(row, col, '', gantt_format)
    row += 1

# Freeze panes to keep task names and dates visible
worksheet.freeze_panes(1, 4)

# Close the workbook
workbook.close()
