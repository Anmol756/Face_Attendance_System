def read_excel(file_path):
    import pandas as pd
    return pd.read_excel(file_path)

def write_excel(file_path, data_frame):
    import pandas as pd
    data_frame.to_excel(file_path, index=False)

def append_to_excel(file_path, data_frame):
    import pandas as pd
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        combined_df = pd.concat([existing_df, data_frame], ignore_index=True)
        combined_df.to_excel(file_path, index=False)
    else:
        data_frame.to_excel(file_path, index=False)