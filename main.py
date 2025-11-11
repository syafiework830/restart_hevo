import requests, base64, json, pandas as pd, os, datetime as dt, humanfriendly
from requests.models import Response
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def throw_error(function_name:str, error_raise:Exception, error_message:Optional[str] =None)-> None:
    print('-'*150)
    print(function_name.center(150).upper())
    if error_message:
        print("⚠️", error_message)
    print('-'*150)
    raise error_raise

def fx_separator(function_name:str, size:int =75)->None:
    print('\n')
    print("="*size)
    print(function_name.center(size).upper())
    print("="*size)

def step_separator(function_name:str, size:int =75)-> None:
    print('\n')
    print("-"*size)
    print(function_name.center(size).title())
    print("-"*size)  

def map_response_code(code:int|str)-> str:
    """Map code into a brief explanation to ease debugging.

    Args:
        code (int | str): The status code based on the responses received.

    Returns:
        str: Brief explanation about the status code.
    """
    if 100 <= code <= 199:
        return 'Informational responses'
    elif 200 <= code <= 299:
        return '✅ Successful responses'
    elif 300 <= code <= 399:
        return 'Redirection responses'
    elif 400 <= code <= 499:
        return 'Client error responses'
    elif 500 <= code <= 599:
        return 'Server error responses'
    else:
        return f'Code: {code} is undefined!'
#================================================================
# check if file exist in the given path
#================================================================

def check_path_existence(file_path:str)-> str:
    
    """Validate and resolve the given file path.
    
    This function dynamically checks whether the specified file exists
    in the provided path, the current working directory, or a 'src' subdirectory.
    It ensures flexible file lookup regardless of where the file is stored.
    
    Args:
        ``file_path`` (str): Path to the file, either relative or absolute.
        
    Returns:
        str: The resolved absolute path to the existing file.
        
    Raises:
        FileNotFoundError: If the file cannot be found in any of the expected locations.
    """
    
    step_separator('check_path_existence')
    
    if not os.path.exists(file_path):
        print(f"⚠️ File is not found at {file_path}")
        
        if 'src' not in os.getcwd():
            new_file_path = os.path.join(os.getcwd(), 'src', file_path)
            print(f"Looking for file in {os.path.dirname(new_file_path)}")
            
            if os.path.exists(new_file_path):
                print(f"✅ File is found at: {os.path.dirname(new_file_path)}")
                return new_file_path
            
            else:
                throw_error('check_path_existence in load_param_for_excel', 
                            FileNotFoundError, 
                            'No excel file found in the given path')
        
        elif 'src' in os.getcwd():
            new_file_path = os.path.join(os.getcwd(), '..\\', file_path)
            print(f"Looking for file in {os.path.dirname(new_file_path)}")
            
            if os.path.exists(new_file_path):
                print(f"✅ File is found at: {os.path.dirname(new_file_path)}")
                return new_file_path
            
            else:
                throw_error('check_path_existence in load_param_for_excel', 
                            FileNotFoundError, 
                            'No excel file found in the given path')
        
        else:
            throw_error('check_path_existence in load_param_for_excel', 
                        FileNotFoundError, 
                        'No excel file found in the given path')
    
    else:
        return file_path

# ==============================================================
# Handles credentials
# ==============================================================

def handle_keys(cred_file_path:str)-> str:
    '''
    Merge the given keys from a json file and convert them into Base64 format as required by Hevo's API.

    Args:
        ``cred_file_path`` (str): The path of the file that stores Hevo's credentials.

    Returns:
        Tupe[str, str]: Url that link to the API host and a Base64-encoded string representing the Hevo authentication credentials.
        
    Raises:
        ValueError: If one or more required fields are missing.
        FileNotFoundError: If the file not exist in the given path.
    '''
    fx_separator('handle_keys')
    
    try:
    
        access_key:str | None = os.getenv("access_key")
        secret_key:str | None = os.getenv("secret_key")
        api_host:str | None = os.getenv("api_host")
           
        if not all([access_key, secret_key, api_host]):
            raise ValueError("Missing one or more required fields: 'access_key', 'secret_key', or 'api_host'")
        
        print(f"📌 Encoding authentication key") 
          
        key:str = f"{access_key}:{secret_key}"
        key_base64:bytes = str(base64.b64encode(key.encode('ascii')))
        authentication_key:str = key_base64.strip().replace("=",'').removeprefix('b').replace("'",'')

        print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
        print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now()-start)}")

        return authentication_key, api_host
    
    except FileNotFoundError:
            throw_error('handle_keys', FileNotFoundError, f'File is not found at {cred_file_path}')
    
    except Exception:
            throw_error('handle_keys', Exception)

# ==============================================================
# Retrieves all pipelines
# ==============================================================

def get_all_pipelines(api_host:str,
                      auth_key:str
                      ) -> Response:

    """
    Retrieve all pipelines and their metadata from a given Hevo pipeline.

    Args:
        api_host (str): Base API host URL for your Hevo region (e.g., "https://us.hevodata.com").
        auth_key (str): Base64-encoded authentication key (AccessKey:SecretKey).

    Returns:
        list: A list of all pipelines and their metadata from the hosted API and credentials.
    """
    fx_separator('get_all_pipelines')
    
    try:
        start = dt.datetime.now()        
        url = f"{api_host}/api/public/v2.0/pipelines?"
        
        print(f"📌 Calling {url}")
        
        headers = {
            "accept": "application/json",
            "authorization": f"Basic {auth_key}"
        }

        print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
        print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now()-start)}")

        return requests.get(url, headers=headers)
    
    except Exception as e:
        throw_error('get_all_pipelines', e, 'error in getting all of the pipelines.')

# ==============================================================
# Retrieves all object in a pipeline
# ==============================================================

def get_all_objects(
    api_host: str,
    auth_key: str,
    pipeline_id: str | int,
) -> list:
    """
    Retrieve all objects and their metadata from a given Hevo pipeline,
    handling pagination automatically (non-recursive).

    Args:
        api_host (str): Base API host URL for your Hevo region (e.g., "https://us.hevodata.com").
        auth_key (str): Base64-encoded authentication key (AccessKey:SecretKey).
        pipeline_id (str | int): The ID of the pipeline to retrieve objects from.

    Returns:
        list: A list of all objects and their metadata from the specified pipeline.
    """
    step_separator(f'get_all_objects : {pipeline_id}')
    start = dt.datetime.now()

    collected: list = []
    next_token: str | None = None

    print(f"📌 Pipeline Id: {pipeline_id}")
    
    pagination_tracker = 1
    
    while True:
        url = f"{api_host}/api/public/v2.0/pipelines/{pipeline_id}/objects"
        if next_token:
            pagination_tracker += 1
            url += f"?starting_after={next_token}"
            print(f"\t📌 {pagination_tracker} Pagination detected... Fetching next page ({next_token})")
        else:
            print(f"📌 Calling initial URL: {url}")

        headers = {
            "accept": "application/json",
            "authorization": f"Basic {auth_key}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        collected.extend(data.get("data", []))
        next_token = data.get("pagination", {}).get("starting_after")

        if not next_token:
            break

    print(f"✅ Total objects retrieved: {len(collected)}")
    print(f"⏱️ : {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"⌛ : {humanfriendly.format_timespan(dt.datetime.now() - start)}")

    return collected
        
# ==============================================================
# Restart object
# ==============================================================

def restart_object(api_host:str, auth_key:str, pipeline_id: str|int, object_name:str, tracker:int = 0, total_object:int = 0)-> None:
    """
    Restart object in a particular pipeline and api host.
    
    Args:
        api_host (str): Base API host URL for your Hevo region (e.g., "https://us.hevodata.com").
        auth_key (str): Base64-encoded authentication key (AccessKey:SecretKey).
        pipeline_id (str | int): The ID of the pipeline to retrieve objects from.
        object_name (str): Object name to be restarted.
        
    Returns:
        None: Print the status code for the request
    """
    #fx_separator('restart_object')
    step_separator(f"Restart object: {tracker}/{total_object}")
    
    try:
        start = dt.datetime.now()        
        print(f"✅ Pipeline ID: {pipeline_id}")
        print(f"✅ Object Name: {object_name}")
        
        url = f"{api_host}/api/public/v2.0/pipelines/{pipeline_id}/objects/{object_name}/restart"
        headers = {"authorization": f"Basic {auth_key}"}
        response = requests.post(url, headers=headers)

        if response.status_code >= 200 and response.status_code <= 299:
            print(f"✅ Status: {response.status_code} 👉 {object_name} 👈 restarted successfully!")
        else:
            print(f"⚠️ Restarting {object_name} failed! Status: {response.status_code}")
            raise f"""
                    Restarting:
                    \t📌Object name: {object_name}
                    \t📌Pipeline Id: {pipeline_id}
                    \t📌Url: {url}
                    """

        print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
        print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now()-start)}")
        
        return response.status_code, start
        
    except Exception as e:
        throw_error("restart_object", e, f'error in restarting object! Object: {object_name}')
        
# ==============================================================
# Load from excel
# ==============================================================

def load_param_from_excel(excel_path:str) -> pd.DataFrame:
    """Load all of the object name and their pipeline ids from an Excel (.xslx) file.

    Args:
        ``excel_path`` (str): Directory path that stores the Excel file.
        
    Returns:
        ``Dataframe`` (pd.Dataframe): A dataframe of the table from the file input.
        
    Raises:
        ``TypeError`` : If input file received is not in .xslx or .csv file format.
    """
        
    start = dt.datetime.now()
    
    fx_separator('load_param_from_excel')
    print(f"📌 Load excel from {excel_path}")
    print(f"📌 File Name: {os.path.basename(excel_path)}")
    
    if excel_path.endswith('xlsx'):
        df = pd.read_excel(check_path_existence(excel_path))
    elif excel_path.endswith('csv'):
        df = pd.read_csv(check_path_existence(excel_path))
    else:
        raise TypeError("Input file only accepts Excel or CSV format file type only!")
    
    print('\n')
    step_separator('Excel df')
    print(df.head(5))
    print('\n')
    
    print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now()-start)}")
    return df

# ==============================================================
# Retrives all pipeline ids, and object names
# ==============================================================

def create_hevo_df(api_host:str, auth_key:str)-> pd.DataFrame:
    """
    Retrieve all pipeline objects from Hevo and return them as a DataFrame.

    This function fetches all available pipelines using the Hevo API, 
    then retrieves all objects (such as tables or entities) associated 
    with each pipeline. The results are compiled into a pandas DataFrame 
    for easy inspection and processing.

    Args:
        ``api_host`` (str): Base API host URL for your Hevo region 
            (e.g., "https://us.hevodata.com").
        ``auth_key`` (str): Base64-encoded authentication key in the format 
            "AccessKey:SecretKey", required for accessing the Hevo API.

    Returns:
        pd.DataFrame: A DataFrame containing two columns:
            - `pipeline_id` (int): The unique ID of the pipeline.
            - `object_name` (str): The name of each object within the pipeline.

    Raises:
        requests.RequestException: If any of the API calls fail.
        KeyError: If expected fields are missing from the API response.
    """
    start = dt.datetime.now()
    
    all_pipelines = get_all_pipelines(api_host, auth_key).json()

    pipeline_ids = [item.get('id') for item in all_pipelines['data']]

    hevo_list = {}

    for id in pipeline_ids:
        
        all_object = get_all_objects(api_host, auth_key, id)
        object_names = [item.get('name') for item in all_object]
        hevo_list[id] = object_names

    hevo_df = pd.DataFrame([
        {'pipeline_id': int(pid), 'object_name': obj}
        for pid, objects in hevo_list.items()
        for obj in objects
    ])
    
    print('\n')
    step_separator('hevo_df')
    print(hevo_df.head(5))
    print('\n')

    print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now()-start)}")
    return hevo_df

# ==============================================================
# Map empty cell with values from Hevo
# ==============================================================

def is_empty_val(v):
    return pd.isna(v) or str(v).strip().lower() in ['nan', '', 'none', 'null', '0']

def map_null_cell(df_excel: pd.DataFrame, df_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing pipeline IDs or object names in an Excel DataFrame using Hevo reference data.
    If a pipeline_id exists without an object_name, all corresponding objects from Hevo
    are expanded into new rows in the Excel DataFrame.

    Args:
        df_excel (pd.DataFrame): Input DataFrame that may contain missing pipeline_id or object_name.
        df_truth (pd.DataFrame): Reference DataFrame from Hevo that contains complete mappings
                                 of pipeline_id to object_name.

    Returns:
        pd.DataFrame: The updated DataFrame with missing pipeline_id or object_name values filled
                      and pipeline_id rows expanded for all associated objects.
    """
    fx_separator("map_null_cell")
    start = dt.datetime.now()

    df_excel['pipeline_id'] = df_excel['pipeline_id'].astype('Int64', errors='ignore')
    df_excel['object_name'] = df_excel['object_name'].astype('string')

    # ✅ Build reference truth mappings
    valid_truth = df_truth.dropna(subset=["pipeline_id", "object_name"])
    truth_by_pipeline = (
        df_truth.dropna(subset=['pipeline_id', 'object_name'])
        .groupby('pipeline_id')['object_name']
        .apply(list)
        .to_dict()
    )
    truth_by_object = valid_truth.set_index(
        valid_truth["object_name"].astype(str).str.strip()
    )["pipeline_id"].to_dict()

    updated_rows = []

    for idx, row in df_excel.iterrows():
        pid = row.get("pipeline_id")
        obj = row.get("object_name")

        pid_is_empty = is_empty_val(pid)
        obj_is_empty = is_empty_val(obj)

        # Case A: pipeline exists but object missing → add ALL objects from truth
        if not pid_is_empty and obj_is_empty:
            if pid in truth_by_pipeline and truth_by_pipeline[pid]:
                for candidate in truth_by_pipeline[pid]:
                    if not is_empty_val(candidate):
                        new_row = row.copy()
                        new_row["object_name"] = candidate
                        updated_rows.append(new_row)
                        print(f"✅ Added object_name='{candidate}' for pipeline_id={pid}")
            else:
                print(f"⚠️ No truth mapping found for pipeline_id={pid}")
                updated_rows.append(row)

        # Case B: object exists but pipeline missing → fill pipeline from truth
        elif pid_is_empty and not obj_is_empty:
            obj_key = str(obj).strip()
            if obj_key in truth_by_object:
                new_pid = truth_by_object[obj_key]
                row["pipeline_id"] = new_pid
                updated_rows.append(row)
                print(f"✅ Filled pipeline_id={new_pid} for object_name='{obj}'")
            else:
                print(f"⚠️ No truth mapping found for object_name='{obj}'")
                updated_rows.append(row)

        # Case C: both present → keep row as is (validate consistency)
        elif not pid_is_empty and not obj_is_empty:
            if pid in truth_by_pipeline:
                truth_objs = [str(x).strip() for x in truth_by_pipeline[pid] if not is_empty_val(x)]
                if str(obj).strip() not in truth_objs:
                    print(f"⚠️ MISMATCH: pipeline_id={pid} has truth={truth_objs} but found '{obj}'")
            updated_rows.append(row)

        # Case D: both empty → skip
        else:
            print(f"⚠️ Row {idx}: both pipeline_id and object_name empty — skipped")

    # ✅ Create final expanded DataFrame
    result_df = pd.DataFrame(updated_rows).reset_index(drop=True)
    print(f"⏱️: {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"⌛: {humanfriendly.format_timespan(dt.datetime.now() - start)}")

    return result_df

# ==============================================================
# Main App Function to Restart the objects and pipelines
# ==============================================================

def app_restart_object_with_reports(config_file:str, excel_path:str, is_save_df:bool = False) -> None:
    """Restart object in Hevo pipelines.
    
    This function restarts all of the objects in each pipeline specified in the Excel file.
    This also fetches all objects and pipelines to map empty values from the list in Excel file.

    Args:
        ``config_file`` (str): File path that has the credentials stored.
        ``excel_path`` (str): File path of the Excel file that has the pipeline ids and object names.
        ``is_save`` (bool): Set True to save summary Excel file. Default is False.
        
    Returns:
        None
    """
    
    # Configure credentials
    auth_key,api_host  = handle_keys(config_file)

    # Load excel file and df
    excel_df = load_param_from_excel(excel_path)

    # Create hevo df as truth
    hevo_df = create_hevo_df(api_host, auth_key)

    # Map null values in Excel df to avoid interruption
    cleaned_excel_df = map_null_cell(excel_df, hevo_df)
    
    # Iterate each rows to restart
    fx_separator("restart_object")
    total_object = cleaned_excel_df.shape[1]
    tracker = 0
    
    summary_restart = {}
    
    for row in cleaned_excel_df.itertuples(index=False):
        if pd.notna(row.pipeline_id) and pd.notna(row.object_name):
            tracker += 1
            status_code, start = restart_object(api_host, auth_key, int(row.pipeline_id), str(row.object_name).strip(), tracker, total_object)
            status_response = map_response_code(status_code)
            summary_restart[tracker] = {"pipeline_id": row.pipeline_id, "object_name": row.object_name, "status_code": status_code, "restart_response": status_response, "restarted_at":start.strftime("%H:%M:%S")}

    # Convert dict of dicts into a list of lists
    rows = [v.values() for v in summary_restart.values()]
    headers = list(next(iter(summary_restart.values())).keys())  # extract column names

    #print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
    if is_save_df:
        df = pd.DataFrame.from_dict(summary_restart,orient='index')
        output_filename = f"{dt.datetime.now().strftime('%Y-%m-%d')}_restarted_object.xlsx"
        os.makedirs(output_filename) if not os.path.exists(output_filename) else None
        df.to_excel(output_filename)
        step_separator(f"📁 File successfully saved as: {output_filename} \n at: {os.path.dirname(output_filename) if os.getcwd() in output_filename else os.getcwd()}")
        
def app_restart_object(config_file:str, excel_path:str) -> dict:

    # Configure credentials
    auth_key,api_host  = handle_keys(config_file)

    # Load excel file and df
    excel_df = load_param_from_excel(excel_path)

    # Create hevo df as truth
    hevo_df = create_hevo_df(api_host, auth_key)

    # Map null values in Excel df to avoid interruption
    cleaned_excel_df = map_null_cell(excel_df, hevo_df)
    
    # Iterate each rows to restart
    fx_separator("restart_object")
    total_object = cleaned_excel_df.shape[1]
    tracker = 0
    
    summary_restart = {}
    
    for row in cleaned_excel_df.itertuples(index=False):
        if pd.notna(row.pipeline_id) and pd.notna(row.object_name):
            tracker += 1
            status_code, start = restart_object(api_host, auth_key, int(row.pipeline_id), str(row.object_name).strip(), tracker, total_object)
            status_response = map_response_code(status_code)
            summary_restart[tracker] = {"pipeline_id": row.pipeline_id, "object_name": row.object_name, "status_code": status_code, "restart_response": status_response, "restarted_at":start.strftime("%H:%M:%S")}
    
    return summary_restart

from flask import Flask, jsonify, request
from tempfile import NamedTemporaryFile

app = Flask(__name__)

@app.route('/restart_hevo_object', methods=['POST'])
def restart_hevo_object():
    print("="*75)
    print(f"Time Start: {dt.datetime.now().strftime('%H-%M')}")

    # Get JSON from Power Automate
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # Validate keys exist
    if 'config_file' not in data or 'param_file' not in data:
        return jsonify({"error": "Both 'config_file' and 'param_file' must be provided"}), 400

    # Decode config file
    try:
        config_bytes = base64.b64decode(data['config_file']['contentBytes'])
        config_filename = data['config_file'].get('fileName', 'config.json')
        with NamedTemporaryFile(delete=False, suffix=config_filename) as temp_config:
            temp_config.write(config_bytes)
            config_path = temp_config.name
    except Exception as e:
        return jsonify({"error": f"Failed to decode config_file: {str(e)}"}), 400

    # Decode param file
    try:
        param_bytes = base64.b64decode(data['param_file']['contentBytes'])
        param_filename = data['param_file'].get('fileName', 'param.xlsx')
        with NamedTemporaryFile(delete=False, suffix=param_filename) as temp_param:
            temp_param.write(param_bytes)
            param_path = temp_param.name
    except Exception as e:
        return jsonify({"error": f"Failed to decode param_file: {str(e)}"}), 400

    # Call your existing function
    try:
        result = app_restart_object(config_path, param_path)
    except Exception as e:
        return jsonify({"error": f"app_restart_object failed: {str(e)}"}), 500

    # Check for failures in results (assuming result is a dict of statuses)
    has_failures = any(
        not (200 <= v.get("status_code", 0) <= 299)
        for v in result.values()
    )

    return jsonify({
        "message": "Restart completed",
        "has_failures": has_failures,
        "results": result,
    }), 200


if __name__ == '__main__':  
    
    app.run(host="0.0.0.0", port=8000, debug=True)
    
    start = dt.datetime.now()
    #app_restart_object(configuration_file, object_name, True)
    duration = humanfriendly.format_timespan(dt.datetime.now()-start)
    print(f"{duration}")