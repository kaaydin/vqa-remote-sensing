import json
import pandas as pd
import os
from tqdm import tqdm


def json_to_dataframe(json_file_path, delimiter):
    """
    This function converts a JSON file to a pandas DataFrame.

    Args:
    json_file_path : str : the path to the JSON file.

    Returns:
    df : DataFrame : a pandas DataFrame created from the JSON file, or
    None : if an error occurs.
    """
    
    try:
        # Open the JSON file
        with open(json_file_path, 'r') as json_file:
            # Load the content of the file
            # Assuming the JSON structure is a flat dictionary-like structure
            # If the structure is different, this line may need adjustment
            json_data = json.load(json_file)[delimiter]
        
        # Convert the JSON data to a DataFrame
        # Note: Depending on the JSON structure, you might need a different approach
        df = pd.DataFrame(json_data)

        # Return the DataFrame
        return df
    
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error occurred while decoding JSON from file: {json_file_path}")
        return None
    except Exception as e:
        # Catch any other exceptions that occur
        print(f"An unexpected error occurred: {str(e)}")
        return None

def remove_nan_rows(df, delimiter):
    """
    Remove rows with NaN in the 'question' column from a DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The modified DataFrame with rows containing NaN in 'question' column removed.
    """
    # Validate if 'question' column exists in the DataFrame
    if delimiter in df.columns:
        # Remove rows where 'question' column is NaN
        df_clean = df.dropna(subset=[delimiter])
        return df_clean
    else:
        raise ValueError(f"No {delimiter} column found in the DataFrame")
    

def remove_columns(dataframe, columns_to_remove):
    """
    Remove specified columns from a pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The original DataFrame.
    columns_to_remove (list): A list of column names to remove.

    Returns:
    pd.DataFrame: A new DataFrame with specified columns removed.
    """

    # Check if all columns to remove are in the DataFrame
    for col in columns_to_remove:
        if col not in dataframe.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Drop the columns
    dataframe = dataframe.drop(columns=columns_to_remove)
    return dataframe

def merge_dataframes_on_column(df1, df2, common_column, how='inner'):
    """
    Merge two pandas DataFrames on a specific common column.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    common_column (str): The name of the common column to merge on.
    how (str): Type of merge to be performed ('left', 'right', 'outer', 'inner'), default is 'inner'.

    Returns:
    pd.DataFrame: A new DataFrame resulting from the merge of the two input DataFrames.
    """
    # Check if the common column exists in both DataFrames
    if common_column not in df1.columns or common_column not in df2.columns:
        raise ValueError(f"The common column '{common_column}' must exist in both DataFrames.")

    # Merge the DataFrames on the common_column
    result = pd.merge(df1, df2, on=common_column, how=how)
    return result

## Check length of each question 

def length_checker(df, tokenizer):
    lengths = []

    for row in tqdm(df.itertuples()):
        question = row.question
        
        if question[-1] == '?':
            question = question[:-1]
        else:
            question = question
        
        tokenized = tokenizer.encode_plus(question, add_special_tokens=True, return_attention_mask = False, return_token_type_ids=False)["input_ids"]
        length = len(tokenized)
        lengths.append(length)
    
    return lengths

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)