from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    i = 0
    while i < len(lst):
        left = i
        right = min(i + n - 1, len(lst) - 1)
        
        while left < right:
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1
        i += n
    
    return lst



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = [string]
        else:
            result[length].append(string)
    
    return dict


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
      
        result = {}
        for key, value in nested.items():
            new_key = prefix + key if prefix else key
            
            if isinstance(value, dict):
                result.update(flatten(value, new_key + sep))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        result.update(flatten(item, new_key + f"[{i}]" + sep))
                    else:
                        result[new_key + f"[{i}]"] = item
            else:
                result[new_key] = value
                
        return result
    
    return dict()

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start, end):
        if start == end:
            permutations.add(tuple(nums))
        for i in range(start, end):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start+1, end)
            nums[start], nums[i] = nums[i], nums[start]
            
    permutations = set()
    backtrack(0, len(nums))
    return [list(p) for p in permutations]




def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [r"\d{2}-\d{2}-\d{4}", r"\d{2}/\d{2}/\d{4}", r"\d{4}\.\d{2}\.\d{2}"]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return dates

  

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coords = polyline.decode(polyline_str)
    latitudes, longitudes = zip(*coords)
    distances = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = radians(coords[i-1][0]), radians(coords[i-1][1])
        lat2, lon2 = radians(coords[i][0]), radians(coords[i][1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371 * c * 1000  # in meters
        distances.append(distance + distances[-1])
    
    df = pd.DataFrame({
        "latitude": latitudes,
        "longitude": longitudes,
        "distance": distances
    })
    return df

    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated = [[matrix[n-j-1][i] for j in range(n)] for i in range(n)]
    result = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i][:j] + rotated[i][j+1:])
            col_sum = sum([rotated[k][j] for k in range(n) if k != i])
            result[i][j] = row_sum + col_sum

    # Your code here
    return result

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    def is_complete(group):
        days = group["startDay"].unique()
        times = group["startTime"].unique()
        return len(days) == 7 and len(times) == 24
    
    result = df.groupby(["id", "id_2"]).apply(is_complete)
    return result

  

   
