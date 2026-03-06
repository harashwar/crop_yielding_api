
import pandas as pd

df_raw = pd.read_csv('crop_yielding_predection.csv', header=None)

all_text = []
for i, row in df_raw.iterrows():
    vals = [str(v) for v in row if str(v).strip() != 'nan']
    if vals:
        all_text.append(','.join(vals))

full_text = '\n'.join(all_text)

# Write to file so we can read the full content
with open('raw_data_dump.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
print("Written to raw_data_dump.txt")
print(f"Total lines: {len(all_text)}")
