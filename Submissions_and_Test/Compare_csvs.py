import csv

def load_csv_data(filepath, id_column='row_id', value_column='target_column'):
    """Loads CSV data into a dictionary keyed by id_column."""
    data = {}
    try:
        with open(filepath, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: CSV file {filepath} is empty or has no header.")
                return None
            if id_column not in reader.fieldnames or value_column not in reader.fieldnames:
                print(f"Error: Required columns ('{id_column}', '{value_column}') not found in {filepath}.")
                print(f"Available columns: {reader.fieldnames}")
                return None
            for row in reader:
                data[row[id_column]] = row[value_column]
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return data

def compare_csv_files(file1_path, file2_path, id_column='row_id', value_column='target_column'):
    """
    Compares two CSV files based on a specified ID and value column, and calculates a score.
    File1 is treated as the reference or ground truth.
    """
    print(f"Comparing '{file1_path}' (File 1) and '{file2_path}' (File 2)...")
    data1 = load_csv_data(file1_path, id_column, value_column)
    data2 = load_csv_data(file2_path, id_column, value_column)

    if data1 is None or data2 is None:
        print("Comparison cannot proceed due to file reading errors.")
        return

    matches = 0
    mismatches_value_diff = 0
    mismatch_details = []
    
    file1_ids = set(data1.keys())
    file2_ids = set(data2.keys())

    common_ids = file1_ids.intersection(file2_ids)
    ids_only_in_file1 = file1_ids.difference(file2_ids)
    ids_only_in_file2 = file2_ids.difference(file1_ids)

    for row_id in common_ids:
        if data1[row_id] == data2[row_id]:
            matches += 1
        else:
            mismatches_value_diff += 1
            mismatch_details.append({
                'id': row_id,
                f'{value_column}_file1': data1[row_id],
                f'{value_column}_file2': data2[row_id]
            })

    mismatches_missing_in_file2 = len(ids_only_in_file1)
    total_entries_in_file1 = len(file1_ids)

    if total_entries_in_file1 == 0:
        score = 0.0
        print("File 1 is empty or could not be read. Cannot compute a meaningful score.")
    else:
        # Score is (matches / total entries in file1)
        score = (matches / total_entries_in_file1) * 100

    print("\n--- CSV Comparison Report ---")
    print(f"File 1 (Reference): {file1_path}")
    print(f"File 2 (Compared): {file2_path}")
    print(f"Compared column: '{value_column}' using key: '{id_column}'\n")

    print(f"Total entries in File 1: {total_entries_in_file1}")
    print(f"Total entries in File 2: {len(file2_ids)}")
    print(f"Common entries (based on '{id_column}'): {len(common_ids)}")
    
    print(f"\n--- Comparison Results (based on File 1 as reference) ---")
    print(f"Matching '{value_column}' values (for common IDs): {matches}")
    print(f"Mismatched '{value_column}' values (for common IDs): {mismatches_value_diff}")
    print(f"Entries in File 1 but not in File 2: {mismatches_missing_in_file2}")
    print(f"Entries in File 2 but not in File 1: {len(ids_only_in_file2)}")
    
    if total_entries_in_file1 > 0 :
        print(f"\nScore (Matches / Total entries in File 1): {score:.2f}%")

    if mismatch_details:
        print("\n--- Mismatch Details (common IDs with different values) ---")
        for i, detail in enumerate(mismatch_details):
            if i < 10: # Print first 10 mismatches
                print(f"  ID: {detail['id']}, File1_Value: {detail[f'{value_column}_file1']}, File2_Value: {detail[f'{value_column}_file2']}")
            elif i == 10:
                print(f"  ... and {len(mismatch_details) - 10} more mismatches of this type.")
                break
            
    if ids_only_in_file1:
        print(f"\n--- IDs in File 1 but not in File 2 (first 10) ---")
        for i, row_id in enumerate(list(ids_only_in_file1)):
            if i < 10:
                print(f"  ID: {row_id} (File 1 '{value_column}': {data1[row_id]})")
            elif i == 10:
                print(f"  ... and {len(ids_only_in_file1) - 10} more missing IDs.")
                break
    print("\n--- End of Report ---")

if __name__ == "__main__":
    # Define the paths to your CSV files here
    file_path1 = '/home/randitha/Desktop/IT/Personal/DataStormV6/Submissions/Rothila-1-submission.csv'
    file_path2 = '/home/randitha/Desktop/IT/Personal/DataStormV6/Submissions/Expected_modified.csv'
    
    # You can change id_column and value_column if your CSVs have different headers
    # id_column_name = 'row_id'
    # value_column_name = 'target_column'
    # compare_csv_files(file_path1, file_path2, id_column=id_column_name, value_column=value_column_name)
    
    compare_csv_files(file_path1, file_path2)