import csv
import os

# Adjust this input path if your file is in a different spot
input_csv = 'metadata/manifest.csv' 
output_tsv = 'manifest_fixed.tsv'

print("Converting CSV to QIIME 2 Paired-End Format (V2)...")

samples = {}

try:
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=',') 
        
        for row in reader:
            s_id = row['sample-id']
            path = row['absolute-filepath']
            direction = row['direction']
            
            if s_id not in samples:
                samples[s_id] = {'fwd': None, 'rev': None}
            
            if 'forward' in direction:
                samples[s_id]['fwd'] = path
            elif 'reverse' in direction:
                samples[s_id]['rev'] = path

    with open(output_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # --- THE FIX IS HERE: EXACT HEADERS REQUIRED BY QIIME 2 ---
        writer.writerow(['sample-id', 'forward-absolute-filepath', 'reverse-absolute-filepath'])
        
        count = 0
        for s_id, paths in samples.items():
            if paths['fwd'] and paths['rev']:
                writer.writerow([s_id, paths['fwd'], paths['rev']])
                count += 1
                
    print(f"✅ Success! Converted {count} samples to '{output_tsv}'.")

except Exception as e:
    print(f"❌ Error: {e}")
