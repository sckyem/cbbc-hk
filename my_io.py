import requests
from zipfile import ZipFile
from default_modules import *
from pathlib import Path

def folder_path(*folders):
    if len(folders) == 1:
        return os.path.abspath(*folders)
    elif len(folders) > 1:
        return os.path.join(*folders)
    else:
        return os.getcwd()

def file_name(path):
    return os.path.basename(path)

def file_path(relative_path='', file_name=''):
    if relative_path:
        if file_name:            
            return folder_path(relative_path, file_name)
        else:
            if file_name:
                return os.path.join(folder_path(relative_path), file_name)
            else:
                return file_paths(relative_path)
    else:
        return os.path.abspath(__file__)

def file_paths(folder='', extension='', include_file_name=False):
    if folder:
        if not os.path.isabs(folder):
            folder = folder_path(folder)
    else:
        folder = folder_path()
    file_names = [i for i in os.listdir(folder) if extension and i.endswith(f"{extension}")]

    if include_file_name:
        return {i:os.path.join(folder, i) for i in file_names}
    else: 
        return [os.path.join(folder, i) for i in file_names]

def read_csv(*folders_and_file_name):
    try:
        file = path(*folders_and_file_name)
        if not str(file).endswith(".csv"):
            file += ".csv"
        df = pd.read_csv(file, index_col=0, parse_dates=[0])
        df.columns = strings_to_columns(df.columns)
        return df
    except:
        print(f"Cant read {file}")

def write_csv(dataframe, *folders_and_file_name):
    if dataframe is not None and not dataframe.empty:
        try:
            file = path(*folders_and_file_name)
            df = dataframe.copy()    
            if not str(file).endswith(".csv"):
                file += ".csv"
            df.columns = columns_to_strings(df.columns)
            if df.index.name is None:
                df.index.name = 'date' if isinstance(df.index, pd.DatetimeIndex) else 'index'
            df.reset_index().to_csv(file, index=False)
            print(f"Exported {file_name(file)} to {file}")
            return True
        except:
            False

def read_parquet(*folders_and_file_name):
    try:
        file = path(*folders_and_file_name)
        if not str(file).endswith('.parquet'): file += '.parquet'      
        print(f"load from {file}")
        return pd.read_parquet(file)
    except:
        print(f"Cant read {file}")

def write_parquet(dataframe, *folders_and_file_name):
    if dataframe is not None and not dataframe.empty:
        try:
            file_path = path(*folders_and_file_name)
            if not str(file_path).endswith('.parquet'): file_path += '.parquet'
            dataframe.to_parquet(file_path)
            return True
        except:
            False

def download_zip(url, download_folder='', unzip_folder=''):
    response = requests.get(url)
    if response.status_code == 200:
        download_file = path(download_folder, file_name(url))        
        with open(download_file, 'wb') as file:
            file.write(response.content)

        if unzip_folder:
            with ZipFile(download_file, 'r') as file:
                file.extractall(unzip_folder)
                print(f"Download and extract all to {unzip_folder}.")
                return True
        else:
            print(f"Download {file_name(url)} to {download_folder}.")
            return True
    else:
        print(f"Failed to download file from {url}.")
        return False

def path(*folders_and_file, is_abs_path=True, is_file_paths=False, extensions=[], is_file_name_as_key=False, is_make_dirs=False):    
    path = os.path.join(*folders_and_file)
    if is_abs_path:
        path = os.path.abspath(path)
    
    if os.path.isdir(path):
        if is_file_paths:
            files = os.listdir(path)
            if extensions:
                if isinstance(extensions, str):
                    extensions = [extensions]
                extensions = [e if str(e).startswith(".") else f".{e}" for e in extensions]
                files = [f for f in files for e in extensions if str(f).endswith(f"{e}")]
            if is_file_name_as_key:
                return {i:os.path.join(path, i) for i in files}
            else:
                return [os.path.join(path, i) for i in files]
    else:
        if not os.path.isfile(path) and is_make_dirs:
            folder, file = os.path.split(path)
            os.makedirs(folder, exist_ok=True) if "." in file else os.makedirs(path)
    return path
    
def read_data(root, file, folders=[], func=None, is_read_csv=False, is_read_parquet=False, is_read_mongodb=False, is_write_mongodb=False, is_write_csv=False, is_write_parquet=False, query={}, projection={}):
    if is_read_mongodb or is_write_mongodb:
        from my_mongodb import Mongodb
        document = Mongodb(root, file)
    file = path(root, *folders, file)    
    if is_read_mongodb:
        df = document.read(query, projection, is_dataframe=True)
    elif is_read_csv:
        df = read_csv(file)
    elif is_read_parquet:
        df = read_parquet(file)
    else:
        df = func()
    if df is not None:
        if is_write_mongodb:
            document.update(df)
        if is_write_csv:
            write_csv(df, file)
        if is_write_parquet:
            write_parquet(df, file)
    return df
