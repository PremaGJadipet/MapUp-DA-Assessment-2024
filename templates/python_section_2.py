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


     df = calculate_distance_matrix(df)
     print(df)


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
    df = pd.DataFrame(unrolled_rows)
    
    return df


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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if reference_id not in df['id_start'].values:
        raise ValueError("Reference ID not found in 'id_start' column")
    
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    threshold = reference_avg_distance * 0.1
    
    filtered_df = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_df = filtered_df[(filtered_df['distance'] >= reference_avg_distance - threshold) & 
                              (filtered_df['distance'] <= reference_avg_distance + threshold)]
    
    return sorted(filtered_df['id_start'].tolist())


 


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    toll_rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, coefficient in toll_rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient
    

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
    weekday_discount_factors = {
        '00:00:00-10:00:00': 0.8,
        '10:00:00-18:00:00': 1.2,
        '18:00:00-23:59:59': 0.8
    }
    weekend_discount_factor = 0.7
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    result_rows = []
    
    for index, row in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for i in range(24):
                start_time = f'{i:02d}:00:00'
                end_time = f'{(i+1)%24:02d}:00:00'  # Wrap around to 00:00:00 for last interval
                
                if day in ['Saturday', 'Sunday']:
                    discount_factor = weekend_discount_factor
                else:
                    for time_interval, discount in weekday_discount_factors.items():
                        start, end = time_interval.split('-')
                        if start <= start_time < end:
                            discount_factor = discount
                            break
                
                moto = row['moto'].values[0] * discount_factor
                car = row['car'].values[0] * discount_factor
                rv = row['rv'].values[0] * discount_factor
                bus = row['bus'].values[0] * discount_factor
                truck = row['truck'].values[0] * discount_factor
                
                result_rows.append({
                    'id_start': index[0],
                    'id_end': index[1],
                    'distance': row['distance'].values[0],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': moto,
                    'car': car,
                    'rv': rv,
                    'bus': bus,
                    'truck': truck
                })
    
     df = pd.DataFrame(result_rows)
    
    return df

