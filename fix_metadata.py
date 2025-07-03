
import csv

input_file = "metadata.csv"
output_file = "metadata_fixed.csv"
expected_cols = 9

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

header = rows[0]
data_rows = rows[1:]

fixed_rows = [header]

for row in data_rows:
    if len(row) == expected_cols:
        fixed_rows.append(row)
    elif len(row) == expected_cols - 1:
        row.append("0")  # Add '0' if image_saved column is missing
        fixed_rows.append(row)
    else:
        print(f"❌ Skipped malformed row: {row}")

with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(fixed_rows)

print(f"✅ metadata.csv has been cleaned and now has exactly {expected_cols} columns per row.")
