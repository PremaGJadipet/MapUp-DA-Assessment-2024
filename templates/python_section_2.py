import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    if 'id_start' not in df.columns or 'id_end' not in df.columns or 'distance' not in df.columns:
        raise ValueError("Dataframe must contain 'id_start', 'id_end', and 'distance' columns")
    
    
    ids = pd.concat([df['id_start'], df['id_end']]).unique()
    
    
    distance_matrix = pd.DataFrame(np.zeros((len(ids), len(ids))), index=ids, columns=ids)
    
   
    for index, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Symmetric matrix
    
    # Calculate cumulative distances using Floyd-Warshall algorithm
    for k in ids:
        for i in ids:
            for j in ids:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], 
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    return distance_matrix


    df = pd.read_csv('dataset.csv')


     distance_matrix = calculate_distance_matrix(df)
     print(distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Initialize empty list to store unrolled rows
    unrolled_rows = []
    
    # Unroll distance matrix
    for i, row_index in enumerate(df.index):
        for j, col_index in enumerate(df.columns):
            if row_index != col_index:  # Exclude same id_start to id_end combinations
                unrolled_rows.append({
                    'id_start': row_index,
                    'id_end': col_index,
                    'distance': df.loc[row_index, col_index]
                })
    
    # Convert unrolled rows to DataFrame
    unrolled_df = pd.DataFrame(unrolled_rows)
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
