def process_and_delete_excel(file_path, column_name, threshold):
    """
    Reads an Excel file, calculates the average of a specified column,
    and deletes the file if the average is below a threshold.

    Args:
        file_path (str): The full path to the Excel file.
        column_name (str): The name of the column to average.
        threshold (float): The threshold for the average.
    """
    try:
        df = pd.read_excel(file_path) # Read the Excel file into a pandas DataFrame

        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in '{file_path}'")
            return

        column_data = pd.to_numeric(df[column_name], errors='coerce')  # Convert column to numeric, handle non-numeric values as NaN

        column_average = column_data.mean()  # Calculate the average of the column

        if column_average < threshold:
            os.remove(file_path)  # Delete the file if the average is below the threshold
            print(f"File '{file_path}' deleted. Column '{column_name}' average was {column_average:.2f} (below {threshold})")
        else:
            print(f"File '{file_path}' kept. Column '{column_name}' average was {column_average:.2f} (above or equal to {threshold})")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")