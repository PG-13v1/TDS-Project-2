from multiprocessing import process
import subprocess
from flask import Flask, request, jsonify
import os
from question_matching import find_similar_question
from file_process import unzip_folder
from function_definations_llm import function_definitions_objects_llm
from openai_api import extract_parameters
from solution_functions import functions_dict
from dotenv import load_dotenv

load_dotenv('secret.env')



tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)

app = Flask(__name__)


SECRET_PASSWORD = os.getenv("SECRET_PASSWORD")


@app.route("/api", methods=["POST"])
def process_file():
    question = request.form.get("question")
    file = request.files.get("file")  # Get the uploaded file (optional)
    file_names = []

    # Ensure tmp_dir is always assigned
    tmp_dir = "tmp_uploads"

    try:
        matched_function, matched_description, matched_files = find_similar_question(question)

        if '.zip' in file:
            temp_dir, file_names = unzip_folder(file)
            tmp_dir = temp_dir  # Update tmp_dir if a file is uploaded

        parameters = extract_parameters(
            str(question),
            function_definitions_llm=function_definitions_objects_llm[matched_function],
        )

        solution_function = functions_dict.get(
            str(matched_function), lambda parameters: "No matching function found"
        )

        if file is not None:
            if '.zip' in file:
                if parameters:
                    answer = solution_function(tmp_dir, **parameters)
                else:
                    answer = solution_function(tmp_dir)
            else:
                if parameters:
                    answer = solution_function(file, **parameters)
                else:
                    answer = solution_function(file)
        else:
            if parameters:
                answer = solution_function(**parameters)
            else:
                answer = solution_function()

        return jsonify({"answer": answer})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/redeploy', methods=['GET'])
def redeploy():
    password = request.args.get('password')
    print(password)
    print(SECRET_PASSWORD)
    if password != SECRET_PASSWORD:
        return "Unauthorized", 403

    subprocess.run(["../redeploy.sh"], shell=True)
    return "Redeployment triggered!", 200


if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1",port=8000)
