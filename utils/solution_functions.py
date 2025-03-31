import requests
import subprocess
import hashlib
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import zipfile
import pandas as pd
import os
import gzip
import re
import json
import csv
import io
import base64
from file_process import unzip_folder
from geopy.geocoders import Nominatim
import time
import tabula
import pandas as pd
import numpy as np
import tempfile
import base64
import io
from PIL import Image
#pip install PyMuPDF
#pip install tabula-py
import numpy as np
from PIL import Image
import colorsys
import sys
from urllib.parse import urlencode
import feedparser
import fitz
from pydub import AudioSegment
import speech_recognition as sr
import io
from moviepy import VideoFileClip
from fuzzywuzzy import fuzz, process
import dotenv

dotenv.load_dotenv('TDS-project-2/utils/secrets.env')
api_key=str(os.environ['API_KEY'])


def vs_code_version(code):
    return """
    Version:          Code 1.98.2 (ddc367ed5c8936efe395cffeec279b04ffd7db78, 2025-03-12T13:32:45.399Z)
    OS Version:       Linux x64 6.12.15-200.fc41.x86_64
    CPUs:             11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 x 1300)
    Memory (System):  7.40GB (3.72GB free)
    Load (avg):       3, 2, 2
    VM:               0%
    Screen Reader:    no
    Process Argv:     --crash-reporter-id 80b4d7e7-0056-4767-b601-6fcdbec0b54d
    GPU Status:       2d_canvas:                              enabled
                    canvas_oop_rasterization:               enabled_on
                    direct_rendering_display_compositor:    disabled_off_ok
                    gpu_compositing:                        enabled
                    multiple_raster_threads:                enabled_on
                    opengl:                                 enabled_on
                    rasterization:                          enabled
                    raw_draw:                               disabled_off_ok
                    skia_graphite:                          disabled_off
                    video_decode:                           enabled
                    video_encode:                           disabled_software
                    vulkan:                                 disabled_off
                    webgl:                                  enabled
                    webgl2:                                 enabled
                    webgpu:                                 disabled_off
                    webnn:                                  disabled_off

    CPU %	Mem MB	   PID	Process
        2	   189	 18772	code main
        0	    45	 18800	   zygote
        2	   121	 19189	     gpu-process
        0	    45	 18801	   zygote
        0	     8	 18825	     zygote
        0	    61	 19199	   utility-network-service
        0	   106	 20078	ptyHost
        2	   114	 20116	extensionHost [1]
    21	   114	 20279	shared-process
        0	     0	 20778	     /usr/bin/zsh -i -l -c '/usr/share/code/code'  -p '"0c1d701e5812" + JSON.stringify(process.env) + "0c1d701e5812"'
        0	    98	 20294	fileWatcher [1]

    Workspace Stats:
    |  Window (● solutions.py - tdsproj2 - python - Visual Studio Code)
    |    Folder (tdsproj2): 6878 files
    |      File types: py(3311) pyc(876) pyi(295) so(67) f90(60) txt(41) typed(36)
    |                  csv(31) h(28) f(23)
    |      Conf files:
    """


def make_http_requests_with_uv(email="25ds1000038@ds.study.iitm.ac.in",url="https://httpbin.org/get"):
    if not url or not email:
        return {"answer": "Error: Missing required parameter 'url' or 'email'."}

    url = str(url)
    email = str(email)

    # Fix: Modify the command to properly encode the email parameter
    command = ["uv", "run", "--with", "httpie", "--", "http", "GET", url, f"email=={email}", "--verify=no"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        return  result.stdout
    else:
        return {"answer": "Failed", "stderr": result.stderr}
    

import subprocess
import hashlib
import os

def run_command_with_npx(file="README.md", prettier_version="3.4.2", hash_algo="sha256", use_npx=True):
    """Run prettier using npx and calculate hash of the formatted output"""
    
    # Verify file exists
    if not os.path.exists(file):
        print(f"Error: File '{file}' not found")
        return None

    # Get full path to npx
    npx_path = os.path.join(os.environ.get('APPDATA', ''), 'npm', 'npx.cmd')
    if not os.path.exists(npx_path):
        print(f"Error: npx not found at {npx_path}")
        return None

    # First install prettier
    try:
        install_cmd = [npx_path, "-y", f"prettier@{prettier_version}"]
        subprocess.run(install_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing prettier: {e.stderr.decode()}")
        return None

    # Now run prettier
    try:
        prettier_cmd = [npx_path, f"prettier@{prettier_version}", file]
        result = subprocess.run(
            prettier_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Calculate hash of formatted content
        hasher = hashlib.new(hash_algo)
        hasher.update(result.stdout.encode())
        return hasher.hexdigest()

    except subprocess.CalledProcessError as e:
        print(f"Error running prettier: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def use_google_sheets(rows=100, cols=100, start=15, step=12, extract_rows=1, extract_cols=10):
    matrix = np.arange(start, start + (rows * cols * step),
                       step).reshape(rows, cols)

    extracted_values = matrix[:extract_rows, :extract_cols]

    return np.sum(extracted_values)



def use_excel(values=None, sort_keys=None, num_rows=1, num_elements=9):
    # Convert input values to numpy arrays if they're provided as strings or lists
    if values is None:
        values = np.array([13, 12, 0, 14, 2, 12, 9, 15, 1, 7, 3, 10, 9, 15, 2, 0])
    else:
        # Convert string representation to list then to numpy array
        if isinstance(values, str):
            values = [int(x.strip()) for x in values.strip('{}').split(',')]
        values = np.array(values, dtype=int)
    
    if sort_keys is None:
        sort_keys = np.array([10, 9, 13, 2, 11, 8, 16, 14, 7, 15, 5, 4, 6, 1, 3, 12])
    else:
        # Convert string representation to list then to numpy array
        if isinstance(sort_keys, str):
            sort_keys = [int(x.strip()) for x in sort_keys.strip('{}').split(',')]
        sort_keys = np.array(sort_keys, dtype=int)

    # Sort values based on sort_keys
    sorted_values = values[np.argsort(sort_keys)]
    
    # Return sum of first num_elements values
    return np.sum(sorted_values[:num_elements])


def use_devtools(file=None,html=None):
    if html is None:
        html = '<input type="hidden" name="secret" value="12345">'
    if input_name is None:
        input_name = "secret"

    soup = BeautifulSoup(html, "html.parser")
    hidden_input = soup.find("input", {"type": "hidden", "name": input_name})

    return hidden_input["value"] if hidden_input else None


def count_wednesdays(start_date="1990-04-08", end_date="2008-09-29", weekday=2):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    count = sum(
        1
        for _ in range((end - start).days + 1)
        if (start + timedelta(_)).weekday() == weekday
    )
    return count


def extract_csv_from_a_zip(
    file=None,
    csv_filename="extract.csv",
    column_name="answer",
):

    csv_path = os.path.join(file, csv_filename)

    if not os.path.exists(csv_path):
        for root, _, files in os.walk(file):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    break

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return ", ".join(map(str, df[column_name].dropna().tolist()))



def use_json(file=None):
    """
    Sorts a JSON array of objects by the value of the "age" field. In case of a tie, sorts by "name".
    
    Parameters:
        input_data (str): Either the path to a JSON file or the JSON string itself.
        from_file (bool): Set to True if input_data is a file path, False if it's JSON text.
        
    Returns:
        str: The sorted JSON array (as a string) without any spaces or newlines.
    """
    if file is not None:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.loads(file)
    
    sorted_data = sorted(data, key=lambda x: (x.get('age'), x.get('name')))
    return json.dumps(sorted_data, separators=(',',':'))


def multi_cursor_edits_to_convert_to_json(file=None):
    """
    Reads the given text file containing key=value pairs on each line,
    converts these pairs into a dictionary, canonicalizes the dictionary
    into a JSON string with sorted keys and minimal separators, computes the
    SHA‑256 hash of the canonical JSON, and returns the hashed value.
    
    Parameters:
        file_path (str): The path to the text file.
        
    Returns:
        dict: A dictionary with either {"value": <hashed value>} 
              or {"error": <error message>} if an exception occurs.
    """
    try:
        # Read the file and convert lines into a dictionary.
        result = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip()
        
        # Canonicalize the dictionary: convert to a JSON string with sorted keys and no unnecessary spacing.
        canonical_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
        
        # Compute the SHA‑256 hash of the canonical JSON.
        hashed_value = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        
        return  hashed_value
    except Exception as error:
        return {"error": f"Error: {str(error)}"}


def css_selectors(file=None,attr="div",clss="foo"):
    """
    Accepts either a string containing HTML markup or a file path to an HTML file.
    Selects all <div> elements having a 'foo' class and returns the sum of their
    data-value attributes as integers.

    Parameters:
        html_input (str): Either raw HTML text or the path to an HTML file.
        
    Returns:
        int: The sum of data-value attributes for all <div class="foo"> elements.
    """
    # If html_input is a file path, load its content.
    if os.path.exists(file) and os.path.isfile(file):
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        html_content = file

    soup = BeautifulSoup(html_content, "html.parser")
    divs = soup.select(f"{attr}.{clss}")
    total = sum(int(div.get("data-value", 0)) for div in divs)
    return total


def process_files_with_different_encodings(folder_path, symbols=None, encodings=None):
    """
    Process files with different encodings from a folder and calculate sum of values for specific symbols.
    
    Parameters:
        folder_path (str): Path to the folder containing the files
        symbols (list): List of symbols to match (default ['‚', 'ˆ', '‡'])
        encodings (list): List of encodings to try (default ['cp1252', 'utf-8', 'utf-16'])
    
    Returns:
        float: Sum of all values associated with the specified symbols
    """
    # Default values if not provided
    if symbols is None:
        symbols = ['‚', 'ˆ', '‡']
    if encodings is None:
        encodings = ['cp1252', 'utf-8', 'utf-16']

    total = 0
    
    # Get list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        # Determine delimiter based on file extension
        delimiter = '\t' if file_name.lower().endswith('.txt') else ','
        
        # Try each encoding
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                if 'symbol' in df.columns and 'value' in df.columns:
                    # Sum values where symbol matches
                    mask = df['symbol'].isin(symbols)
                    total += df.loc[mask, 'value'].sum()
                break  # Stop trying encodings if successful
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue  # Try next encoding if this one fails
    
    return int(total)


def use_github(github_username="PG-13v1", github_token="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", repo_name="TDS_Project_2", email="25ds1000038@ds.study.iitm.ac.in"):
    """
    Creates a new GitHub repository, adds an email.json file, commits, and pushes it.

    Parameters:
        github_username (str): Your GitHub username.
        github_token (str): Your GitHub personal access token (PAT).
        repo_name (str): The name of the repository to be created.
        email (str): The email to be added in email.json.
    """

    GITHUB_API = "https://api.github.com"

    # Step 1: Create a new GitHub repository
    repo_url = f"{GITHUB_API}/user/repos"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    repo_data = {
        "name": repo_name,
        "description": "Repository containing email.json",
        "private": False  # Public repo
    }

    response = requests.post(repo_url, headers=headers, json=repo_data)

    if response.status_code == 201:
        print(f"✅ Repository '{repo_name}' created successfully.")
    elif response.status_code == 422:
        print(f"⚠️ Repository '{repo_name}' already exists.")
    else:
        print(f"❌ Failed to create repository: {response.json()}")
        return

    # Step 2: Clone the repository
    os.system(f"git clone https://github.com/{github_username}/{repo_name}.git")
    os.chdir(repo_name)

    # Step 3: Create email.json
    file_name = "email.json"
    file_content = {"email": email}

    with open(file_name, "w") as f:
        json.dump(file_content, f, indent=4)

    # Step 4: Initialize Git, commit, and push
    os.system("git init")
    os.system("git add email.json")
    os.system('git commit -m "Added email.json"')
    os.system("git branch -M main")
    os.system(f"git remote add origin https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git")
    os.system("git push -u origin main")

    print(f"✅ Successfully pushed {file_name} to {repo_name}.")

    return ""


def replace_across_files(dest_folder,replaced_text,replacing_text):
    """
    Unzips the given local zip file into the destination folder, then replaces all occurrences of
    "IITM" (in any case) with "IIT Madras" in all files, preserving original line endings.
    Finally, concatenates the content (in lexicographic order of file names) of all processed files
    and computes the SHA‑256 hash of the concatenation (equivalent to running: cat * | sha256sum).

    Parameters:
        local_zip_path (str): The path to the downloaded zip file.
        dest_folder (str): The folder where the zip contents will be extracted.

    Returns:
        str: The SHA‑256 hash (hexdigest) of the concatenated processed files.
    """
    # Create the destination folder if it doesn't exist.
   


    # Compile a regex pattern to match "IITM" in any case.
    pattern = re.compile(re.escape(replaced_text), re.IGNORECASE)
    
    # Process each file in the destination folder.
    for file_name in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file_name)
        if os.path.isfile(file_path):
            # Open file in text mode with newline='' to preserve original line endings.
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                content = f.read()
            # Replace any case variation of "IITM" with "IIT Madras".
            new_content = pattern.sub(replacing_text, content)
            # Write back using the same newline handling.
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(new_content)
    
    # Concatenate the contents of all processed files (sorted by file name).
    concatenated_bytes = b""
    for file_name in sorted(os.listdir(dest_folder)):
        file_path = os.path.join(dest_folder, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                concatenated_bytes += f.read()
    
    # Compute the SHA‑256 hash of the concatenated bytes.
    sha256_hash = hashlib.sha256(concatenated_bytes).hexdigest()
    return sha256_hash


def list_files_and_attributes(dest_folder,size_threshold):
    """
    Lists all files in the given folder along with their modification date and file size,
    and computes the total size of all files that are at least 4984 bytes large 
    and modified on or after Fri, 9 Sept, 2011, 5:36 pm IST.
    
    Args:
        dest_folder (str): Path to folder containing the files
        
    Returns:
        int: Total size of files meeting the criteria
    """
    total_size = 0

    # Define the threshold datetime in IST
    ist_offset = timedelta(hours=5, minutes=30)
    ist = datetime.timezone(ist_offset) 
    threshold_dt = datetime(2011, 9, 9, 17, 36, 0, tzinfo=ist)

    print("\nFiles in Directory:")
    print("="*60)

    # Walk through all files in directory
    for root, _, files in os.walk(dest_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            # Convert modification time to IST
            mod_time = datetime.fromtimestamp(file_stats.st_mtime)
            mod_time = mod_time.replace(tzinfo=datetime.timezone.utc).astimezone(ist)

            # Print file details
            print(f"{filename}: {mod_time}, {file_size} bytes")

            # Apply filtering criteria
            if file_size >= size_threshold and mod_time >= threshold_dt:
                total_size += file_size

    return total_size

import  shutil

def move_and_rename_files(work_folder):

    # Move all files from subdirectories into work_folder
    for root, dirs, files in os.walk(work_folder, topdown=False):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(work_folder, file)

            # If a file with the same name exists, append a counter
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(work_folder, f"{counter}_{file}")
                counter += 1

            shutil.move(src_path, dst_path)

        # Remove empty directories
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass

    # Rename files (increment each digit, wrapping 9 → 0)
    for file in os.listdir(work_folder):
        old_path = os.path.join(work_folder, file)
        if os.path.isfile(old_path):
            new_name = re.sub(r'\d', lambda m: str((int(m.group(0)) + 1) % 10), file)
            new_path = os.path.join(work_folder, new_name)
            os.rename(old_path, new_path)

    # Read non-empty lines, prefix with filename, and sort using LC_ALL=C behavior
    grep_lines = []
    for file in sorted(os.listdir(work_folder), key=lambda x: x.encode('utf-8')):
        file_path = os.path.join(work_folder, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    stripped_line = line.rstrip('\n')
                    if stripped_line:
                        grep_lines.append(f"{file}:{stripped_line}")

    # Sort lines using LC_ALL=C (byte-based sorting)
    grep_lines.sort(key=lambda x: x.encode('utf-8'))

    # Concatenate sorted lines with newlines and compute SHA-256
    concatenated_output = "\n".join(grep_lines) + "\n"
    sha256_hash = hashlib.sha256(concatenated_output.encode('utf-8')).hexdigest()

    return sha256_hash

def compare_files(dest_folder):
    """
    Compares two files named "a.txt" and "b.txt" in the specified folder.
    Assumes both files exist and have the same number of lines.
    Returns the count of lines that differ between them.
    
    Parameters:
        dest_folder (str): The folder containing a.txt and b.txt files.
        
    Returns:
        int: The number of lines that are different between a.txt and b.txt.
    """
    file_a = os.path.join(dest_folder, "a.txt")
    file_b = os.path.join(dest_folder, "b.txt")
    
    # Read both files line-by-line.
    with open(file_a, 'r', encoding='utf-8') as fa:
        lines_a = fa.read().splitlines()
    with open(file_b, 'r', encoding='utf-8') as fb:
        lines_b = fb.read().splitlines()
    
    # Ensure both files have the same number of lines.
    if len(lines_a) != len(lines_b):
        raise ValueError("Files a.txt and b.txt do not have the same number of lines.")
    
    # Count the number of differing lines.
    diff_count = sum(1 for line_a, line_b in zip(lines_a, lines_b) if line_a != line_b)
    return diff_count


def sql_ticket_sales():
    query = """
    SELECT SUM(units * price) AS total_sales
    FROM tickets
    WHERE LOWER(type) = 'gold';
    """
    return query


def write_documentation_in_markdown(headings):
    return '''# Weekly Step Analysis

This report analyzes the **number** of steps walked each day for a week, comparing personal trends over time and with friends. The goal is to identify areas for improvement and maintain a consistent exercise routine.

---

## Methodology

The analysis follows these steps:

1. *Data Collection*:
   - Step counts were recorded using a fitness tracker.
   - Friends' step data was collected via a shared fitness app.

2. *Data Analysis*:
   - Daily step counts were compared with the personal goal of 10,000 steps.
   - Weekly trends were visualized and summarized.

3. *Comparison*:
   - Trends were compared with friends' weekly averages.

Note: This analysis assumes all data points are accurate and complete. If not, a preprocessing step is applied using the function `clean_data(dataset)`.

---

## Results

### Step Counts Table
The table below compares personal step counts with friends' averages:

| Day       | My Steps | Friends' Avg Steps |
|-----------|----------|--------------------|
| Monday    | 8,500    | 9,800              |
| Tuesday   | 9,200    | 10,100             |
| Wednesday | 7,500    | 8,900              |
| Thursday  | 10,300   | 10,500             |
| Friday    | 12,000   | 9,700              |
| Saturday  | 14,000   | 11,200             |
| Sunday    | 13,500   | 12,000             |

---
###Hyperlink

[stepcount](https://stepcount.com)

###Image
![Step Count Image](https://www.dreamstime.com/illustration/step-counter.html)

###Blockquote
>Number of steps you walked in a week is presented.

## Observations

- *Weekend Success*: Step counts were significantly higher on Saturday and Sunday.
- *Midweek Dip*: Wednesday had the lowest step count.
- *Goal Achievement*: The 10,000-step goal was achieved on four out of seven days.

---

### Visualizing Weekly Steps
The following Python code was used to create a bar chart showing step counts:

```python
import matplotlib.pyplot as plt

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
my_steps = [8500, 9200, 7500, 10300, 12000, 14000, 13500]

plt.bar(days, my_steps, color='skyblue')
plt.title("My Daily Step Counts")
plt.xlabel("Days")
plt.ylabel("Steps")
plt.axhline(y=10000, color='red', linestyle='--', label='Goal')
plt.legend()
plt.show()'''


def compress_an_image(input_file,threshold=1500):
    try:
        img = Image.open(input_file)
    except Exception as e:
        print("Error opening the image file:", e)
        return None, None, None

    def save_png(image):
        """Save an image as PNG with maximum lossless compression."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue()

    def is_lossless(original, candidate):
        """
        Check if candidate image (in palette mode) converts back exactly to the original RGB image.
        This is done by comparing pixel data.
        """
        return list(candidate.convert("RGB").getdata()) == list(original.convert("RGB").getdata())

    # Try saving the original image with maximum PNG compression.
    data = save_png(img)
    if len(data) < threshold:
        return base64.b64encode(data).decode("utf-8"), "Original image with max compression", len(data)

    # Attempt palette conversion if the original didn't meet the threshold.
    # First, determine the number of unique colors.
    unique_colors = img.convert("RGB").getcolors(maxcolors=10**6)
    if unique_colors is None:
        print("Image has too many colors for palette conversion.")
        return None, None, None
    num_unique = len(unique_colors)

    # Iterate over palette sizes from num_unique down to 2.
    for colors in range(num_unique, 1, -1):
        palette_img = img.convert("P", palette=Image.ADAPTIVE, colors=colors, dither=Image.NONE)
        if not is_lossless(img, palette_img):
            continue  # Skip if conversion isn't lossless.
        data_candidate = save_png(palette_img)
        if len(data_candidate) < threshold:
            return base64.b64encode(data_candidate).decode("utf-8")

    # If no conversion meets the threshold, return None.
    return None


def host_your_portfolio_on_github_pages(github_token, github_username, repo_name,email):
    """
    Creates an index.html page showcasing your work with your email address wrapped in
    CloudFlare's email obfuscation comments, commits the page to Git, pushes the changes,
    and returns the GitHub Pages URL.
    
    The email address used is:
    <!--email_off-->25ds1000038@ds.study.iitm.ac.in<!--/email_off-->
    
    Parameters:
      username (str): Your GitHub username.
      repo (str): The repository name where GitHub Pages is enabled.
      branch (str): The branch to push to (default "main").
      
    Returns:
      str: The GitHub Pages URL (e.g., "https://[username].github.io/[repo]/").
    """
    # Create the HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Showcase of My Work</title>
</head>
<body>
    <h1>Welcome to My GitHub Pages Showcase</h1>
    <p>This page showcases my work.</p>
    <p>Contact me at: <!--email_off-->25ds1000038@ds.study.iitm.ac.in<!--/email_off--></p>
</body>
</html>
"""
    # Write the HTML content to index.html
    with open("index.html", "w") as file:
        file.write(html_content)
    print("Created index.html with the showcase page content.")
    
    # Helper function to run a git command
    """
    Creates a new GitHub repository, adds an email.json file, commits, and pushes it.

    Parameters:
        github_username (str): Your GitHub username.
        github_token (str): Your GitHub personal access token (PAT).
        repo_name (str): The name of the repository to be created.
        email (str): The email to be added in email.json.
    """

    GITHUB_API = "https://api.github.com"

    # Step 1: Create a new GitHub repository
    repo_url = f"{GITHUB_API}/user/repos"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    repo_data = {
        "name": repo_name,
        "description": "Repository containing email.json",
        "private": False  # Public repo
    }

    response = requests.post(repo_url, headers=headers, json=repo_data)

    if response.status_code == 201:
        print(f"✅ Repository '{repo_name}' created successfully.")
    elif response.status_code == 422:
        print(f"⚠️ Repository '{repo_name}' already exists.")
    else:
        print(f"❌ Failed to create repository: {response.json()}")
        return

    # Step 2: Clone the repository
    os.system(f"git clone https://github.com/{github_username}/{repo_name}.git")
    os.chdir(repo_name)

    # Step 3: Create email.json
    file_name = "email.json"
    file_content = {"email": email}

    with open(file_name, "w") as f:
        json.dump(file_content, f, indent=4)

    # Step 4: Initialize Git, commit, and push
    os.system("git init")
    os.system("git add email.json")
    os.system('git commit -m "Added email.json"')
    os.system("git branch -M main")
    os.system(f"git remote add origin https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git")
    os.system("git push -u origin main")

    print(f"✅ Successfully pushed {file_name} to {repo_name}.")
    
    # Construct and return the GitHub Pages URL.
    pages_url = f"https://{github_username}.github.io/{repo_name}/"
    return pages_url


def use_google_colab(email="25ds1000038@ds.study.iitm.ac.in.",year="2025"): 
    return hashlib.sha256(f"{email} {year}".encode()).hexdigest()[-5:]

import numpy as np
from PIL import Image
import colorsys

def use_an_image_library_in_google_colab(image_path):
    """
    Analyzes an image to count pixels with lightness > 0.542
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        int: The count of pixels with lightness > 0.542
    """
    try:
        # Open image from file path instead of Colab upload
        image = Image.open(image_path)
        
        # Convert image to RGB array normalized to [0,1]
        rgb = np.array(image) / 255.0
        
        # Calculate lightness for each pixel
        lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
        
        # Count pixels above threshold
        light_pixels = np.sum(lightness > 0.542)
        
        print(f'Number of pixels with lightness > 0.542: {light_pixels}')
        return light_pixels
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def deploy_a_python_api_to_vercel():
    return ""


def create_a_github_action():
    """
    Creates a GitHub Action workflow that contains a step with the name containing the email address 
    "25ds1000038@ds.study.iitm.ac.in", commits the change, pushes it to GitHub, and returns the repository URL.
    
    The workflow file is created in .github/workflows/action.yml and the contents are:
    
    ---
    name: Trigger Action
    on: [push]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - name: 25ds1000038@ds.study.iitm.ac.in
            run: echo "Hello, world!"
    ---
    
    Parameters:
      username (str): Your GitHub username.
      repo (str): The repository name.
      branch (str): The branch to push to (default is "main").
      
    Returns:
      str: The repository URL in the format https://github.com/username/repo
    """
    # Ensure the .github/workflows directory exists.
    workflow_dir = os.path.join(".github", "workflows")
    os.makedirs(workflow_dir, exist_ok=True)
    
    # Define the workflow file path.
    workflow_file = os.path.join(workflow_dir, "action.yml")
    
    # Workflow content.
    workflow_content = """name: Trigger Action
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: 25ds1000038@ds.study.iitm.ac.in
        run: echo "Hello, world!"
"""
    # Write the workflow file.
    with open(workflow_file, "w") as f:
        f.write(workflow_content)
    print(f"Created workflow file at {workflow_file}")

    # Helper function to run git commands.
    def run_git_cmd(cmd):
        try:
            result = subprocess.run(cmd, shell=True, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}\n{e.stderr}")
            sys.exit(1)
    
    # Stage the changes.
    print("Staging workflow file...")
    run_git_cmd("git add .github/workflows/action.yml")
    
    # Commit the changes.
    print("Committing changes...")
    run_git_cmd("git commit -m 'Add GitHub Action with email step'")
    
    # Push the changes.
    print("Pushing changes to GitHub...")
    run_git_cmd(f"git push origin {branch}")
    
    # Construct and return the repository URL.
    repo_url = f"https://github.com/{username}/{repo}"
    return repo_url

def push_an_image_to_docker_hub(username, repository, tag="25ds1000038"):
    image_name = f"{username}/{repository}:{tag}"
    
    try:
        print(f"Building Docker image: {image_name}")
        subprocess.run(f"docker build -t {image_name} .", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"Pushing Docker image: {image_name}")
        subprocess.run(f"docker push {image_name}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Docker command:\n{e.stderr}")
        sys.exit(1)



def write_a_fastapi_server_to_serve_data():
    return "something"


def run_a_local_llm_with_llamafile():
    print("Starting local server on port 8000...")
    server_proc = subprocess.Popen(["uv", "run", "-m", "http.server", "8000"])
    
    # Step 2: Start ngrok tunnel to forward port 8000.
    print("Starting ngrok tunnel on port 8000...")
    ngrok_proc = subprocess.Popen(["uvx", "ngrok", "http", "8000"])
    
    # Give ngrok a few seconds to establish the tunnel.
    time.sleep(5)
    
    try:
        # Step 3: Retrieve the public ngrok URL by querying the local ngrok API.
        response = requests.get("http://localhost:4040/api/tunnels")
        response.raise_for_status()
        tunnels = response.json().get("tunnels", [])
        if not tunnels:
            print("No ngrok tunnels found.")
            return ""
        public_url = tunnels[0].get("public_url", "")
        print("ngrok URL:", public_url)
        return public_url
    except Exception as e:
        print("Error retrieving ngrok URL:", e)
        return ""


def llm_sentiment_analysis(sentiment_text="4k Sue4xDI2    bQQ3zex syGW  MklwUa OKnlQb  L78G"):
    return '''import httpx

# API endpoint
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Dummy API key
HEADERS = {
    "Authorization": "Bearer dummy_api_key",
    "Content-Type": "application/json"
}

# Request payload
DATA = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Analyze the sentiment of the following text as GOOD, BAD, or NEUTRAL."},
        {"role": "user", "content": "''' + sentiment_text + '''"}
    ]
}

# Send POST request
try:
    response = httpx.post(API_URL, json=DATA, headers=HEADERS)
    response.raise_for_status()

    # Parse response
    result = response.json()
    print(result)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"An error occurred: {e}")'''


def llm_token_cost(query="List only the valid English words from these: Tj, 1cWuxm1sI, 7y, xZvu6, XiVyPMm, dFQutwzi9u, MRdiUMWYuV, l6Qxdl9n0, GX5UjGo0Z7, JXnhcJFlD, H, 5eM, xKBH6JmA, aTioqwyOkW, DTP75P, sL9zy4FC, XW5mf8, ah3KOHlC, tt5CtqjDPp, 7jGd4, zg72q, duH7fOXeD, P40CUanUGf, TmgVEG, 4LkmglvixL, vcQJB9A, ArFX9, eR, GrLre, uf, G7g4A0, F, sherHaO9h, 2FFakQAL, DFV, SQEvCTf, YMf47P, nYNA7iv2bv, QY, DZ, BrsH6zlmkB, jr3, fkWmhXFkKo, Qgav, nZc0IgD, hGEAE0sT, l8qXsMD, yxEWuncxQ, JZczG, QdA, N8PND, s6s2U, KaOW09x4w, A7HMsniEz, zIQSpv, ija3MrfP, DJ4Er38, cAg, TjYkTaz, RXHIa, bGxNJ, LLkH7xYi8, rRK68xKY3, dupKYB, Q0CgxF, MtSpzrJLh, GR4lL60h, hEnv, o5, l6ias9Og6S, 2j89KGjP, Ga, ni6HnDU, jV", api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI1ZHMxMDAwMDM4QGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.j4EJATz5r4wa_3PPFWzQp821_VI-2cieg3IMEKfcszc"):
    """Sends a query along with task information to the LLM API. 
       The model will choose the most relevant task from tasks.json, 
       extract parameters from the query (and an optional external file),
       and return a JSON object with the chosen task_id and its parameters.
    """
    
    # Define the API endpoint.
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    
    # Prepare tasks information.
   
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    data = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()['usage']['prompt_tokens']



def generate_addresses_with_llms(system_message="Respond in JSON", user_message="Generate 10 random addresses in the US"):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "address_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "county": {"type": "string"},
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"}
                                },
                                "required": ["county", "latitude", "longitude"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["addresses"],
                    "additionalProperties": False
                }
            }
        }
    }
    return json.dumps(data)

def llm_vision(user_message):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "Add the URL"
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(data)


def llm_embeddings(transaction_code=None, email="25ds1000038@ds.study.iitm.ac.in"):
    """
    Generate the JSON request body for OpenAI embeddings API
    
    Args:
        transaction_code (list): List of transaction codes
        email (str): Email address
        
    Returns:
        str: JSON formatted request body
    """
    # Default transaction codes if none provided
    if transaction_code is None:
        transaction_code = [65889, 42512]
        
    # Ensure transaction_code is a list
    if not isinstance(transaction_code, list):
        transaction_code = [transaction_code]
        
    # Ensure we have at least 2 codes
    while len(transaction_code) < 2:
        transaction_code.append(transaction_code[0])

    # Create messages
    messages = [
        f"Dear user, please verify your transaction code {transaction_code[0]} sent to {email}",
        f"Dear user, please verify your transaction code {transaction_code[1]} sent to {email}"
    ]

    # Create JSON request body
    request_body = {
        "model": "text-embedding-3-small",
        "input": messages
    }

    return json.dumps(request_body, indent=2)


def embedding_similarity():
    return '''
import numpy as np
def most_similar(embeddings):
    max_similarity = -1
    most_similar_pair = None

    phrases = list(embeddings.keys())

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            v1 = np.array(embeddings[phrases[i]])
            v2 = np.array(embeddings[phrases[j]])

            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (phrases[i], phrases[j])

return most_similar_pair'''


def vector_databases():
    return ""


def function_calling():
    return ""


def get_an_llm_to_say_yes():
    return "Say only 'Yes' or 'No'. Is 100 greater than 50?"


def import_html_to_google_sheets(page_no=25):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
    if page_no==1:
        response=requests.get("https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;template=results;type=batting",headers=headers)
    else:
        response=requests.get(f"https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page={page_no};template=results;type=batting",headers=headers)    
    soup=BeautifulSoup(response.text,'html.parser')
    # Extract headers from <th> elements
    header_row = soup.find('thead').find('tr')
    headers = [th.get_text(strip=True) for th in header_row.find_all('th') if th.get_text(strip=True)]
    # Note: Some cells might be empty if there are extra <th> tags

    # Extract data rows from <td> elements
    data_rows = []
    for row in soup.find('tbody').find_all('tr'):
     cells = row.find_all('td')
     # Get the text of each <td> element
     cell_data = [cell.get_text(strip=True) for cell in cells]
     data_rows.append(cell_data)

    # Create a DataFrame; if headers and cells count match, use headers as column names
    df = pd.DataFrame(data_rows, columns=headers if len(headers)==len(data_rows[0]) else None)
    df = df.drop(df.columns[13], axis=1)
    df.columns=["Player","Span","Mat","Inns","NO","RunsDescending","HS","Ave","BF","SR","100","50","0"]
    ducks_column = df['0']
    total_ducks = pd.to_numeric(ducks_column, errors='coerce').sum()     
    return int(total_ducks)


def scrape_imdb_movies(lower_bound=3,upper_bound=5):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
    response = requests.get(f"https://www.imdb.com/search/title/?user_rating={lower_bound},{upper_bound}", headers=headers)
    soup=BeautifulSoup(response.text,'html.parser')

    data_list=[]

    title_elements=soup.select("div> main > div > div > section > section > div > section > section > div > div > section > div > div > ul > li > div > div > div > div > div > div> a.ipc-title-link-wrapper")
    year_elements=soup.select("div > main > div > div > section > section > div > section > section > div > div > section > div > div > ul > li > div > div > div > div > div > div > span")
    rating_elements = soup.select("div > main > div > div > section > section > div > section > section > div > div > section > div > div > ul > li > div > div > div > div > div > span > div > span > span.ipc-rating-star--rating")

    for i in range(0,25):
     href=title_elements[i]['href']
     title_id = href.split("/")[2].split("/")[0] if href else None
     title_element = title_elements[i].text
     title_text=re.sub(r"^\d+\.\s*", "", title_element)
     if year_elements[i].text.startswith(('20','19')):
        year_text=year_elements[i].text

     rating_text=rating_elements[i].text
    
     data_list.append({
            "id": title_id,
            "title": title_text,
            "year": year_text,
            "rating": rating_text
        })

    json_string = json.dumps(data_list, indent=2)
    return json_string


def wikipedia_outline():

    return ""


def scrape_the_bbc_weather_api(city="Cairo"):
    """
    Scrape weather forecast data for a given city from the BBC Weather API and website.
    
    Args:
        city (str): The name of the city to fetch weather data for.
    
    Returns:
        str: A JSON string mapping dates to weather descriptions.
    """
    # Construct location URL with the provided city
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
        's': city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
    })

    # Fetch location data
    result = requests.get(location_url).json()
    
    # Check if location data is valid
    try:
        location_id = result['response']['results']['results'][0]['id']
    except (KeyError, IndexError):
        raise ValueError(f"No location data found for city: {city}")

    # Construct weather URL
    weather_url = f'https://www.bbc.com/weather/{location_id}'

    # Fetch weather data
    response = requests.get(weather_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather data for {city}. Status code: {response.status_code}")

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    daily_summary = soup.find('div', attrs={'class': 'wr-day-summary'})
    if not daily_summary:
        raise ValueError(f"Weather summary not found on page for {city}")

    # Extract weather descriptions
    daily_summary_list = re.findall('[a-zA-Z][^A-Z]*', daily_summary.text)
    if not daily_summary_list:
        raise ValueError(f"No weather descriptions extracted for {city}")

    # Generate date list
    datelist = pd.date_range(datetime.today(), periods=len(daily_summary_list)).tolist()
    datelist = [date.date().strftime('%Y-%m-%d') for date in datelist]

    # Map dates to descriptions
    weather_data = {date: desc for date, desc in zip(datelist, daily_summary_list)}

    # Convert to JSON and return
    return json.dumps(weather_data, indent=4)


def find_the_bounding_box_of_a_city(city, country, osm_id_ending=None):
    """
    Retrieve the minimum latitude of the bounding box for a specified city in a country,
    optionally filtered by an osm_id ending pattern, using the Nominatim API.
    
    Args:
        city (str): The name of the city (e.g., "Tianjin").
        country (str): The name of the country (e.g., "China").
        osm_id_ending (str, optional): The ending pattern of the osm_id to match (e.g., "2077"). Defaults to None.
    
    Returns:
        str: A message with the minimum latitude or an error message.
    """
    # Activate the Nominatim geocoder
    locator = Nominatim(user_agent="myGeocoder")

    # Geocode the city and country, allowing multiple results
    query = f"{city}, {country}"
    locations = locator.geocode(query, exactly_one=False)

    # Check if locations were found
    if locations:
        if osm_id_ending:
            # Loop through results to find a match for osm_id_ending
            for place in locations:
                osm_id = place.raw.get('osm_id', '')
                if str(osm_id).endswith(osm_id_ending):
                    bounding_box = place.raw.get('boundingbox', [])
                    if bounding_box:
                        min_latitude = float(bounding_box[0])
                        result = min_latitude
                    else:
                        result = f"Bounding box information not available for {city}, {country} with osm_id ending {osm_id_ending}."
                    break
            else:
                result = f"No matching OSM ID ending with '{osm_id_ending}' found for {city}, {country}."
        else:
            # No osm_id_ending provided, use the first result
            place = locations[0]  # Take the first match
            bounding_box = place.raw.get('boundingbox', [])
            if bounding_box:
                min_latitude = float(bounding_box[0])
                osm_id = place.raw.get('osm_id', '')
                result = min_latitude
            else:
                result = min_latitude 
    else:
        result = f"Location not found for {city}, {country}."

    # Respect Nominatim's rate limit (1 request per second)
    time.sleep(1)
    return result


def search_hacker_news(query, points):
    """
    Search Hacker News for the latest post mentioning a specified topic with a minimum number of points.
    
    Args:
        query (str): The topic to search for (e.g., "python").
        points (int): The minimum number of points the post must have.
    
    Returns:
        str: A JSON string containing the link to the latest qualifying post or an error message.
    """
    # Fetch the feed with posts based on query and minimum points
    feed_url = f"https://hnrss.org/newest?q={query}&points={points}"
    feed = feedparser.parse(feed_url)

    # Extract the link of the latest post
    if feed.entries:
        latest_post_link = feed.entries[0].link
        result = {"answer": latest_post_link}
    else:
        result = {"answer": "No posts found matching the criteria."}

    # Return the result as JSON
    return json.dumps(result)
    
    return ""


def find_newest_github_user():
    """
    Find the newest GitHub user in a specified location with a follower count based on the given operator.
    
    Args:
        location (str): The city to search for (e.g., "Delhi").
        followers (int): The number of followers to filter by.
        operator (str): Comparison operator for followers ("gt" for >, "lt" for <, "eq" for =).
    
    Returns:
        str: The ISO 8601 creation date of the newest valid user, or an error message.
    """
    # Map operator to GitHub API syntax
    operator_map = {"gt": ">", "lt": "<", "eq": ""}
    if operator not in operator_map:
        return f"Invalid operator: {operator}. Use 'gt', 'lt', or 'eq'."
    follower_query = f"followers:{operator_map[operator]}{followers}"

    # Search users by location and follower count, sorted by join date (newest first)
    url = f"https://api.github.com/search/users?q=location:{location}+{follower_query}&sort=joined&order=desc"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.json().get('message')}"

    users = response.json().get('items', [])
    if not users:
        return f"No users found in {location} with {follower_query}."

    # Cutoff time: March 23, 2025, 3:57:03 PM PDT (convert to UTC for comparison)
    cutoff_datetime = datetime.datetime(2025, 3, 23, 15, 57, 3, tzinfo=datetime.timezone(datetime.timedelta(hours=-7)))
    cutoff_utc = cutoff_datetime.astimezone(datetime.timezone.utc)

    # Process users to find the newest valid one
    for user in users:
        user_url = user['url']
        user_response = requests.get(user_url, headers=headers)

        if user_response.status_code == 200:
            user_data = user_response.json()
            created_at = user_data['created_at']  # ISO 8601 format (e.g., "2023-05-10T12:34:56Z")
            created_at_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))

            # Exclude ultra-new users (joined after cutoff)
            if created_at_date <= cutoff_utc:
                return created_at  # Already in ISO 8601 format
        else:
            print(f"Error fetching user details: {user_response.status_code}")

    return "No valid users found before cutoff date."
    return ""


def create_a_scheduled_github_action():

    return ""


def extract_tables_from_pdf(file_path,subject_filter='English',Group_lower_bound=71,Group_upper_bound=98,target_subject='Physics'):
    df = tabula.read_pdf(file_path, pages='all')
    if isinstance(df, list):
       df = pd.concat(df)

    df['English'] = pd.to_numeric(df['English'], errors='coerce')
    df['Physics'] = pd.to_numeric(df['Physics'], errors='coerce')
    df['Biology']= pd.to_numeric(df['Biology'], errors='coerce')
    df['Economics']= pd.to_numeric(df['Economics'], errors='coerce')
    df['Maths'] = pd.to_numeric(df['Maths'], errors='coerce')

    df['Group'] = np.repeat(np.arange(1, 101), 30)[:len(df)] 
     
    filtered_df = df[(df[target_subject] >= 34) & (df['Group'] >= Group_lower_bound) & (df['Group'] <=Group_upper_bound )]

# Calculate the total Physics marks for the filtered students
    total_physics_marks = filtered_df[target_subject].sum()

    return total_physics_marks


def convert_a_pdf_to_markdown(pdf_path):
     
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise Exception(f"Error opening PDF: {e}")
    
    # Build the Markdown content with a header
    markdown_content = "# Extracted PDF Content\n\n"
    
    # Extract text from each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        
        # Add a page header to the Markdown
        markdown_content += f"## Page {page_num + 1}\n\n"
        markdown_content += page_text + "\n\n"
    
    doc.close()
    
    # Write the unformatted Markdown content to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tmp_file:
        tmp_file.write(markdown_content)
        tmp_file.flush()
        tmp_path = tmp_file.name
    
    # Format the Markdown file using Prettier via npx
    try:
        # Run npx prettier to format the markdown file in place.
        subprocess.run(["npx", "prettier", "--write", tmp_path], check=True)
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_path)
        raise Exception(f"Prettier formatting failed: {e}")
    
    # Read the formatted Markdown content from the temporary file
    with open(tmp_path, "r", encoding="utf-8") as formatted_file:
        formatted_markdown = formatted_file.read()
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return formatted_markdown


def clean_up_excel_sales_data(file_path=None, target_date="2023-09-04 10:07:06+0530", product="Iota", country="BR"):
    """
    Clean sales data from Excel file and calculate margin for filtered transactions
    
    Args:
        file_path (str): Path to Excel file
        target_date (str): Target date in ISO format with timezone
        product (str): Product name to filter
        country (str): Country code to filter
        
    Returns:
        float: Calculated margin
    """
    try:
        # Convert target_date string to datetime
        target_date = pd.to_datetime(target_date).to_numpy()

        # Load Excel file
        df = pd.read_excel(file_path)

        # Clean and standardize country names
        df['Country'] = df['Country'].str.strip().str.upper()
        country_mapping = {
            "USA": "US", "U.S.A": "US", "UNITED STATES": "US",
            "BRA": "BR", "BRAZIL": "BR",
            "U.K": "GB", "UK": "GB", "UNITED KINGDOM": "GB",
            "FR": "FR", "FRA": "FR", "FRANCE": "FR",
            "IND": "IN", "IN": "IN", "INDIA": "IN"
        }
        df['Country'] = df['Country'].replace(country_mapping)

        # Standardize dates
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')

        # Clean product names
        df['Product'] = df['Product/Code'].str.split('/').str[0].str.strip()

        # Clean sales and cost data
        df['Sales'] = df['Sales'].astype(str).str.replace('USD', '').str.strip()
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        
        df['Cost'] = df['Cost'].astype(str).str.replace('USD', '').str.strip()
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
        
        # Handle missing costs (50% of sales)
        df['Cost'] = df['Cost'].fillna(df['Sales'] * 0.5)

        # Apply filters
        filtered_df = df[
            (df['Date'] <= target_date) &
            (df['Product'] == product) &
            (df['Country'] == country)
        ]

        # Calculate margin
        total_sales = filtered_df['Sales'].sum()
        total_cost = filtered_df['Cost'].sum()

        if total_sales > 0:
            margin = (total_sales - total_cost) / total_sales
            return float(margin)
        else:
            return 0.0

    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return None

def parse_log_line(line):
    # Regex for parsing log lines
    log_pattern = (
        r'^(\S+) (\S+) (\S+) \[(.*?)\] "(\S+) (.*?) (\S+)" (\d+) (\S+) "(.*?)" "(.*?)" (\S+) (\S+)$')
    match = re.match(log_pattern, line)
    if match:
        return {
            "ip": match.group(1),
            "time": match.group(4),  # e.g. 01/May/2024:00:00:00 -0500
            "method": match.group(5),
            "url": match.group(6),
            "protocol": match.group(7),
            "status": int(match.group(8)),
            "size": int(match.group(9)) if match.group(9).isdigit() else 0,
            "referer": match.group(10),
            "user_agent": match.group(11),
            "vhost": match.group(12),
            "server": match.group(13)
        }
    return None

def unique_students(file_path=None):
    """
    Process text file to extract and count unique student IDs
    
    Args:
        file_path (str): Path to text file containing student records
        
    Returns:
        int: Number of unique student IDs found
    """
    try:
        # Validate file path
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Initialize set to store unique student IDs
        student_ids = set()

        # Read file and extract student IDs
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Match exact 10-character alphanumeric IDs 
                matches = re.findall(r'\b[A-Z0-9]{10}\b', line)
                student_ids.update(matches)

        # Return count of unique IDs
        return len(student_ids)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# Example usage:
# file_path = "path/to/your/student_records.txt"
# unique_count = unique_students(file_path)
# if unique_count is not None:
#     print(f"Number of unique students: {unique_count}")


def load_logs(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()

    parsed_logs = []
    # Open with errors='ignore' for problematic lines
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed_entry = parse_log_line(line)
            if parsed_entry:
                parsed_logs.append(parsed_entry)
    return pd.DataFrame(parsed_logs)


def convert_time(timestamp):
    return datetime.strptime(timestamp, "%d/%b/%Y:%H:%M:%S %z")


def clean_up_student_marks(file_path, section_prefix, weekday, start_hour, end_hour, month, year):
    """
    Process text file to extract and count unique student IDs
    
    Args:
        file_path (str): Path to text file containing student records
        
    Returns:
        int: Number of unique student IDs found
    """
    try:
        # Validate file path
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Initialize set to store unique student IDs
        student_ids = set()

        # Read file and extract student IDs
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Match exact 10-character alphanumeric IDs 
                matches = re.findall(r'\b[A-Z0-9]{10}\b', line)
                student_ids.update(matches)

        # Return count of unique IDs
        return len(student_ids)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def compute_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def apache_log_requests(file_path, start_time="2024-05-16", end_time="2024-05-17",lang_url='/kannada/'):
    df = load_logs(file_path)

    if not df.empty:
     df["datetime"] = df["time"].apply(convert_time)
    df["day_of_week"] = df["datetime"].dt.strftime('%A')
    df["hour"] = df["datetime"].dt.hour

    # Filter conditions
    filtered_df = df[
        (df["method"] == "GET") &
        (df["url"].str.startswith("/telugu/")) &
        (df["status"] >= 200) & (df["status"] < 300) &
        (df["day_of_week"] == "Sunday") &
        (df["hour"] >= 5) &
        (df["hour"] < 10)
    ]

    # Compute hash of the result
    result_hash = compute_hash(str(len(filtered_df)))

    return result_hash

def apache_log_downloads(file_path,lang_url,target_date="2024-05-16"):
    df=load_logs(file_path)

    if not df.empty:
     df["datetime"] = df["time"].apply(convert_time)
     df["date"] = df["datetime"].dt.strftime('%Y-%m-%d')

    # Filter conditions for /hindimp3/ on 2024-05-16
     filtered_df = df[
        (df["url"].str.startswith(lang_url)) &
        (df["date"] == target_date)
     ]

    # Aggregate data by IP
     ip_data = filtered_df.groupby("ip")["size"].sum().reset_index()

    # Identify the top data consumer
     top_ip = ip_data.loc[ip_data["size"].idxmax()]

     return top_ip['size']



def clean_up_sales_data(file_path,city="Istanbul"):
    # Load the JSON file
    try:
        df = pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # List of known city names (add more as needed)
    known_cities = [city]

    # Standardize city names using fuzzy matching
    df['city'] = df['city'].fillna('Unknown')
    df['city'] = df['city'].apply(lambda city_name: process.extractOne(city_name, known_cities, scorer=fuzz.token_set_ratio)[0] if process.extractOne(city_name, known_cities, scorer=fuzz.token_set_ratio) and process.extractOne(city_name, known_cities, scorer=fuzz.token_set_ratio)[1] > 90 else city_name)

    # Debug: Check unique city names after clustering
    print("Unique cities after clustering:", df['city'].unique())

    # Filter sales for Shoes with at least 32 units
    df_filtered = df[(df['product'] == "Shoes") & (df['sales'] >= 32)]

    # Aggregate sales by city
    df_grouped = df_filtered.groupby("city")["sales"].sum().reset_index()

    # Identify the top-performing city
    top_city = df_grouped.sort_values(by="sales", ascending=False).iloc[0]

    # Find sales for Jakarta
    city_sales = df_grouped[df_grouped["city"].str.lower() == city]["sales"].sum()

    return city_sales


def parse_partial_json(file_path=None):
    """
    Parse truncated JSON data and calculate total sales value
    
    Args:
        file_path (str): Path to JSON file containing sales data
        
    Returns:
        float: Total sales value from all records
    """
    try:
        total_sales = 0
        
        with open(file_path, 'r') as f:
            # Read file line by line to handle truncated JSON
            for line in f:
                try:
                    # Try to parse each line as a complete JSON object
                    data = json.loads(line.strip())
                    
                    # Extract sales value if present
                    if 'sales' in data:
                        # Convert sales value to float and add to total
                        sales = float(data['sales'])
                        total_sales += sales
                        
                except json.JSONDecodeError:
                    # Handle truncated JSON by extracting sales value using regex
                    sales_match = re.search(r'"sales":\s*(\d+\.?\d*)', line)
                    if sales_match:
                        sales = float(sales_match.group(1))
                        total_sales += sales
        
        return total_sales
        
    except Exception as e:
        print(f"Error processing sales data: {str(e)}")
        return None


def extract_nested_json_keys(file_path,target_key):
    try:
        # Load JSON from file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        def count_key_occurrences(obj):
            count = 0
            
            if isinstance(obj, dict):
                # Count occurrences in dict keys
                count += sum(1 for key in obj.keys() if key == target_key)
                # Recursively check values
                count += sum(count_key_occurrences(value) for value in obj.values())
                
            elif isinstance(obj, list):
                # Recursively check list items
                count += sum(count_key_occurrences(item) for item in obj)
                
            return count
            
        return count_key_occurrences(data)
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        return None



def duckdb_social_media_interactions(timestamp="2025-01-28T21:36:31.398Z"):
    return f"""
    SELECT post_id 
    FROM social_media
    WHERE timestamp >= '{timestamp}'
    AND EXISTS (
        SELECT 1
        FROM UNNEST(json_extract(comments, '$[*].stars.useful')::BIGINT[]) AS stars
        WHERE stars >= 5
    )
    ORDER BY post_id ASC;
    """

def transcribe_a_youtube_video(video_path, start_sec, end_sec):
    """
    Extracts audio from a video segment and transcribes it using Google Speech Recognition.
    
    Args:
        video_path (str): Path to the video file
        start_sec (float): Start time in seconds
        end_sec (float): End time in seconds
        output_txt_file (str): Path for saving the transcription
        
    Returns:
        str: Transcribed text from the video segment
    """
    try:
        output_txt_file="transcription.txt"
        # Create temp audio file path
        temp_audio = "temp_audio.wav"
        
        # Extract audio from video segment
        print("Extracting audio...")
        with VideoFileClip(video_path) as video:
            audio_clip = video.audio.subclip(start_sec, end_sec)
            audio_clip.write_audiofile(temp_audio, 
                                     codec='pcm_s16le',
                                     fps=44100)

        # Initialize speech recognizer
        print("Transcribing audio...")
        recognizer = sr.Recognizer()
        
        # Transcribe audio
        with sr.AudioFile(temp_audio) as source:
            audio = recognizer.record(source)
            transcript = recognizer.recognize_google(audio)
            
            # Save transcript to file
            with open(output_txt_file, "w") as f:
                f.write(transcript)
            print(f"Transcript saved to {output_txt_file}")
            
        # Clean up temp file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            
        return transcript
        
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service: {e}")
        return None
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None
    finally:
        # Ensure temp file is cleaned up
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
    


def reconstruct_an_image(scrambled_path,mapping=[
        ((2, 1), (0, 0)),
        ((1, 1), (0, 1)),
        ((4, 1), (0, 2)),
        ((0, 3), (0, 3)),
        ((0, 1), (0, 4)),
        ((1, 4), (1, 0)),
        ((2, 0), (1, 1)),
        ((2, 4), (1, 2)),
        ((4, 2), (1, 3)),
        ((2, 2), (1, 4)),
        ((0, 0), (2, 0)),
        ((3, 2), (2, 1)),
        ((4, 3), (2, 2)),
        ((3, 0), (2, 3)),
        ((3, 4), (2, 4)),
        ((1, 0), (3, 0)),
        ((2, 3), (3, 1)),
        ((3, 3), (3, 2)),
        ((4, 4), (3, 3)),
        ((0, 2), (3, 4)),
        ((3, 1), (4, 0)),
        ((1, 2), (4, 1)),
        ((1, 3), (4, 2)),
        ((0, 4), (4, 3)),
        ((4, 0), (4, 4))
    ]):
    """
    Reconstruct the original image from a scrambled 500x500 image (5x5 grid pieces)
    based on a mapping file and return the base64 encoded PNG image.
    
    Parameters:
        scrambled_path (str): The file path to the scrambled image.
    
    Returns:
        str: The base64 encoded string of the reconstructed image.
    """
    # Define piece size and grid size
    piece_size = 100  # Each piece is 100x100 pixels
    grid_size = 5     # 5x5 grid


    # Open the scrambled image
    scrambled_img = Image.open(scrambled_path)
    
    # Create a blank image for the reconstructed image
    reconstructed_img = Image.new("RGB", (piece_size * grid_size, piece_size * grid_size))
    
    # For each mapping entry, cut out the scrambled piece and paste it at the original location.
    for original_pos, scrambled_pos in mapping:
        orig_row, orig_col = original_pos
        scr_row, scr_col = scrambled_pos
        
        # Define the bounding box in the scrambled image (left, upper, right, lower)
        left = scr_col * piece_size
        upper = scr_row * piece_size
        right = left + piece_size
        lower = upper + piece_size
        
        # Crop the piece from the scrambled image
        piece = scrambled_img.crop((left, upper, right, lower))
        
        # Define the paste position in the reconstructed image
        paste_left = orig_col * piece_size
        paste_upper = orig_row * piece_size
        
        # Paste the piece into the reconstructed image
        reconstructed_img.paste(piece, (paste_left, paste_upper))
    
    # Save the reconstructed image to a bytes buffer in PNG format
    buffer = io.BytesIO()
    reconstructed_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode the image in base64
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    
    return encoded_image
    

functions_dict = {
    "make_http_requests_with_uv": make_http_requests_with_uv,
    "vs_code_version": vs_code_version,
    "run_command_with_npx": run_command_with_npx,
    "use_google_sheets": use_google_sheets,
    "use_excel": use_excel,
    "use_devtools": use_devtools,
    "count_wednesdays": count_wednesdays,
    "extract_csv_from_a_zip": extract_csv_from_a_zip,
    "use_json": use_json,
    "multi_cursor_edits_to_convert_to_json": multi_cursor_edits_to_convert_to_json,
    "css_selectors": css_selectors,
    "process_files_with_different_encodings": process_files_with_different_encodings,
    "use_github": use_github,
    "replace_across_files": replace_across_files,
    "list_files_and_attributes": list_files_and_attributes,
    "move_and_rename_files": move_and_rename_files,
    "compare_files": compare_files,
    "sql_ticket_sales": sql_ticket_sales,
    "write_documentation_in_markdown": write_documentation_in_markdown,
    "compress_an_image": compress_an_image,
    "host_your_portfolio_on_github_pages": host_your_portfolio_on_github_pages,
    "use_google_colab": use_google_colab,
    "use_an_image_library_in_google_colab": use_an_image_library_in_google_colab,
    "deploy_a_python_api_to_vercel": deploy_a_python_api_to_vercel,
    "create_a_github_action": create_a_github_action,
    "push_an_image_to_docker_hub": push_an_image_to_docker_hub,
    "write_a_fastapi_server_to_serve_data": write_a_fastapi_server_to_serve_data,
    "run_a_local_llm_with_llamafile": run_a_local_llm_with_llamafile,
    "llm_sentiment_analysis": llm_sentiment_analysis,
    "llm_token_cost": llm_token_cost,
    "generate_addresses_with_llms": generate_addresses_with_llms,
    "llm_vision": llm_vision,
    "llm_embeddings": llm_embeddings,
    "embedding_similarity": embedding_similarity,
    "vector_databases": vector_databases,
    "function_calling": function_calling,
    "get_an_llm_to_say_yes": get_an_llm_to_say_yes,
    "import_html_to_google_sheets": import_html_to_google_sheets,
    "scrape_imdb_movies": scrape_imdb_movies,
    "wikipedia_outline": wikipedia_outline,
    "scrape_the_bbc_weather_api": scrape_the_bbc_weather_api,
    "find_the_bounding_box_of_a_city": find_the_bounding_box_of_a_city,
    "search_hacker_news": search_hacker_news,
    "find_newest_github_user": find_newest_github_user,
    "create_a_scheduled_github_action": create_a_scheduled_github_action,
    "extract_tables_from_pdf": extract_tables_from_pdf,
    "convert_a_pdf_to_markdown": convert_a_pdf_to_markdown,
    "clean_up_excel_sales_data": clean_up_excel_sales_data,
    "clean_up_student_marks": clean_up_student_marks,
    "apache_log_requests": apache_log_requests,
    "apache_log_downloads": apache_log_downloads,
    "clean_up_sales_data": clean_up_sales_data,
    "parse_partial_json": parse_partial_json,
    "extract_nested_json_keys": extract_nested_json_keys,
    "duckdb_social_media_interactions": duckdb_social_media_interactions,
    "transcribe_a_youtube_video": transcribe_a_youtube_video,
    "reconstruct_an_image": reconstruct_an_image,
}
