import csv
import os

# Get the current working directory
current_dir = os.getcwd()

# Specify the file path
csv_file_path = os.path.join(current_dir, "example.csv")

# Data to be written to the CSV file
data = [
    ['Name', 'Age', 'City'],
    ['John', 30, 'New York'],
    ['Alice', 25, 'Los Angeles'],
    ['Bob', 35, 'Chicago']
]

# Write data to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)

print(f"CSV file created at: {csv_file_path}")
