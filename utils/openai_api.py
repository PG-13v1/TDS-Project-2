import json
import httpx
import os
from question_matching import find_similar_question
from function_definations_llm import function_definitions_objects_llm
from solution_functions import *
from dotenv import load_dotenv

load_dotenv('TDS-project-2/utils/secrets.env')

# OpenAI API settings
openai_api_chat = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
openai_api_key = str(os.environ['API_KEY'])

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

def extract_parameters(prompt: str, function_definitions_llm):
    """Send a user query to OpenAI API and extract structured parameters."""
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                openai_api_chat,
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an intelligent assistant that extracts structured parameters from user queries."},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": function_definitions_llm.get("name", "default_function_name"),
                                **function_definitions_llm
                            }
                        }
                    ],
                    "tool_choice": "auto"
                },
            )
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and "tool_calls" in response_data["choices"][0]["message"]:
            extracted_data = response_data["choices"][0]["message"]["tool_calls"][0]["function"]
            return json.loads(extracted_data.get("arguments", "{}"))
        else:
            print("No parameters detected")
            return None
    except httpx.RequestError as e:
        print(f"An error occurred while making the request: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(
            f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None 
    
extract_parameters("extract the email from query", function_definitions_objects_llm["extract_email"])


