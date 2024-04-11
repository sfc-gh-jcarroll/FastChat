def extract_diff(last_chunk, current_chunk):
    """
    Extracts the difference between the last chunk and the current chunk.

    Parameters:
    - last_chunk (str): The last received chunk of data.
    - current_chunk (str): The current chunk of data.

    Returns:
    - str: The new data present in the current chunk that was not in the last chunk.
    """
    # Find the index where the current chunk starts to differ from the last chunk
    min_len = min(len(last_chunk), len(current_chunk))
    diff_start_index = next((i for i in range(min_len) if last_chunk[i] != current_chunk[i]), min_len)

    # Extract and return the new data from the current chunk
    new_data = current_chunk[diff_start_index:]
    return new_data
