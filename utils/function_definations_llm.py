function_definitions_objects_llm = {
    "vs_code_version": {
        "name": "vs_code_version",
        "description": "running code -s to get diagnostics of vs_code ",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "extract command  from to given query"
                }
            },
            "required": ["code"]
        }
    },

    "make_http_requests_with_uv": {
        "name": "make_http_requests_with_uv",
        "description": "extract the http url and email from the given text for example 'uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL. Send a HTTPS request to with the URL encoded parameter country set to India and city set to Chennai. What is the JSON output of the command? (Paste only the JSON body, not the headers)' in this example country: India and city: Chennai are the query parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                "type": "string",
                "description": "email in text"
            },
            "url": {
                "type": "string",
                "description": "url in text"
            },
        },
            "required": ["email","url"]
        }
    },

    "run_command_with_npx": {
        "name": "run_command_with_npx",
        "description": "Run npx and prettier on an md file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The filename to verify using Prettier and SHA-256 checksum."
                }
            },
            "required": ["file_path"]
        }
    },

    "use_google_sheets": {
        "name": "use_google_sheets",
        "description": "solve the google sheets formula",
        "parameters": {
            "type": "object",
            "properties": {
                "rows": {
                "type": "integer",
                "description": "rows given in text"
            },
            "cols": {
                "type": "integer",
                "description": "columns given in text"
            },
            "start": {
                "type": "integer",
                "description": "starting point given in text"
            },
            "step": {
                "type": "integer",
                "description": "step given in text"
            },
            "extract_rows": {
                "type": "integer",
                "description": "extract rows given in text"
            },
            "extract_cols": {
                "type": "integer",
                "description": "extract columns given in text"
            },
        },
            "required": ["rows","cols","start","step","extract_rows","extract_cols"]
        }
    },
    
    "use_excel": {
        "name": "use_excel",
        "description": "solve the ms excel formula given ",
        "parameters": {
            "type": "object",
            "properties": {
                "values": {
                "type": "array",
                "items": {
                        "type": "number"
                },
                "description": "values given in the formula"
            },
            "sort_keys": {
                "type": "array",
                "items": {
                        "type": "number"
                    },
                "description": "sorted keys in text"
            },
            "num_rows": {
                "type": "integer",
                "description": "number of rows"
            },
            "num_elements": {
                "type": "integer",
                "description": "number of elements"
            },

        },
            "required": ["values","sort_keys","num_rows","num_elements"]
        }
    },

    "use_devtools": {
        "name": "use_devtools",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["file"]
        }
    },

    "count_wednesdays": {
    "name": "count_wednesdays",
    "description": "description",
    "parameters": {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format"
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format"
            },
            "weekday": {
                "type": "integer",
                "description": "Day of the week "
            }
        },
        "required": ["start_date", "end_date", "weekday"]
    }
},

    "extract_csv_from_a_zip": {
        "name": "extract_csv_from_a_zip",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "name of the column which you have to find the value of"
                }
                
            },
            "required": ["column"]
        }
    },

    "use_json": {
        "name": "use_json",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },

            },
            "required": ["input_data"]
        }
    },

    "multi_cursor_edits_to_convert_to_json": {
        "name": "multi_cursor_edits_to_convert_to_json",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": []
        }
    },

    "css_selectors": {
        "name": "css_selectors",
        "description": "select particular attributes with particular classes",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The html to extract the data from"
                }
            },
            "required": ["html_file"]
        }
    },
    "process_files_with_different_encodings": {
        "name": "process_files_with_different_encodings",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "array",
                    "description": "encoding like cp-1252 or utf-8"
                },
                "symbol": {
                    "type": "array",
                    "description": "symbols of what to calculate for"
                },


            },
            "required": ["input_data", "from_file"]
        }
    },

    "use_github": {
        "name": "use_github",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "github_username": {
                    "type": "string",
                    "description": "github_username"
                },
                "github_token": {
                    "type": "string",
                    "description": "github_token"
                },
                "repo_name": {
                    "type": "string",
                    "description": "repo_name"
                },
                "email": {
                    "type": "string",
                    "description": "email"
                }
            },
            "required": ["github_username", "github_token", "repo_name", "email"]
        }
    },

    "replace_across_files": {
        "name": "replace_across_files",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "replaced_text": {
                    "type": "string",
                    "description": "extract the name of the tthe string which will replace the text"
                },
                "replacing_text": {
                    "type": "string",
                    "description": "extract the name of the the string which is gonna replace the repaced text"
                }
            },
            "required": []
        }
    },

    "list_files_and_attributes": {
        "name": "list_files_and_attributes",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "size": {
                    "type": "integer",
                    "description": "size of the threhold for which the file size should be greater"
                }
            },
            "required": []
        }
    },

    "move_and_rename_files": {
        "name": "move_and_rename_files",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": []
        }
    },

    "compare_files": {
        "name": "compare_files",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": []
        }
    },

    "sql_ticket_sales": {
        "name": "sql_ticket_sales",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": []
        }
    },

    "write_documentation_in_markdown": {
        "name": "write_documentation_in_markdown",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "headings": {
                    "type": "string",
                    "description": "The headings"
                }
            },
            "required": ["heading"]
        }
    },

    "compress_an_image": {
        "name": "compress_an_image",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "integer",
                    "description": "The upper limit value up to which the image is to be compressed,the compressed value should be less than the threshold value"
                },
            },
            "required": ["threshold"]
        }
    },

    "host_your_portfolio_on_github_pages": {
        "name": "host_your_portfolio_on_github_pages",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "github_token": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "github_username": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "github_username": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "email": {
                    "type": "string",
                    "description": "extract the email from query"
                }
            },
            "required": ["text"]
        }
    },

    "use_google_colab": {
        "name": "use_google_colab",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "extract email from the text"
                }
            },
            "required": ["email"]
        }
    },

    "use_an_image_library_in_google_colab": {
        "name": "use_an_image_library_in_google_colab",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "year": {
                    "type": "string",
                    "description": "extract year from the text"
                }
            },
            "required": ["email", "year"]
        }
    },

    "deploy_a_python_api_to_vercel": {
        "name": "deploy_a_python_api_to_vercel",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "create_a_github_action": {
        "name": "create_a_github_action",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "github_token": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "github_username": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "github_username": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "email": {
                    "type": "string",
                    "description": "extract the email from query"
                }
            },
            "required": ["text"]
        }
    },

    "push_an_image_to_docker_hub": {
        "name": "push_an_image_to_docker_hub",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "username of user"
                },
                "repository": {
                    "type": "string",
                    "description": "repository name in docker"
                },
                "tag": {
                    "type": "string",
                    "description": "tag to be pushed in the image"
                }
            },
            "required": ["text"]
        }
    },

    "write_a_fastapi_server_to_serve_data": {
        "name": "write_a_fastapi_server_to_serve_data",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "username of user"
                },
            },
            "required": ["text"]
        }
    },

    "run_a_local_llm_with_llamafile": {
        "name": "run_a_local_llm_with_llamafile",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "llm_sentiment_analysis": {
        "name": "llm_sentiment_analysis",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment_text": {
                    "type": "string",
                    "description": "extract the text for which the sentiment is to be predicted if its bad good or neutral."
                }
            },
            "required": ["sentiment_text"]
        }
    },

    "llm_token_cost": {
        "name": "llm_token_cost",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "send the text to get the token cost"
                },
                 "api_key":{
                    "query": "string",
                    "description": "api_key"
                }
            },
            "required": ["query","api_key"]
        }
    },

    "generate_addresses_with_llms": {
        "name": "generate_addresses_with_llms",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "system_message": {
                    "type": "string",
                    "description": "messaGe going to be provided to the system or system prompt"
                },
                 "user_message":{
                    "query": "string",
                    "description": "message provided by the user or user prompt"
                }
            },
            "required": ["system_message","user_message"]
        }
    },

    "llm_vision": {
        "name": "llm_vision",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "user_message":{
                    "query": "string",
                    "description": "message provided by the user or user prompt"
                }
            },
            "required": ["user_message"]
        }
    },

    "llm_embeddings": {
        "name": "llm_embeddings",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_code": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "the  transaction codes provided in query multiple transaction codes would be present"
                },
                "email": {
                    "type": "string",
                    "description": "extract the email from query"
                }
            },
            "required": ["transaction_code","email"]
        }
    },

    "embedding_similarity": {
        "name": "embedding_similarity",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "vector_databases": {
        "name": "vector_databases",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "function_calling": {
        "name": "function_calling",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "get_an_llm_to_say_yes": {
        "name": "get_an_llm_to_say_yes",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": []
        }
    },

    "import_html_to_google_sheets": {
        "name": "import_html_to_google_sheets",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "page_no": {
                    "type": "string",
                    "description": "the webpage to scrape from multiple pages"
                }
            },
            "required": ["page_no"]
        }
    },

    "scrape_imdb_movies": {
        "name": "scrape_imdb_movies",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "lower_bound": {
                    "type": "string",
                    "description": "the lower bound of the user rating to ,ehich is minimum rating above which scraping is done"
                },
                "upper_bound": {
                    "type": "string",
                    "description": "the upper bound of the user rating to ,which is maximum rating above which scraping is done"
                },
            },
            "required": ["page_no"]
        }
    },

    "wikipedia_outline": {
        "name": "wikipedia_outline",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "scrape_the_bbc_weather_api": {
        "name": "scrape_the_bbc_weather_api",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city for which weather is to be determined"
                }
            },
            "required": ["city"]
        }
    },

    "find_the_bounding_box_of_a_city": {
    "name": "find_the_bounding_box_of_a_city",
    "description": "Find the bounding box coordinates for a specified city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "name of the city mentioned in the text"
            },
            "country": {
                "type": "string",
                "description": "name of the country mentioned in the text"
            },
            "osm_id": {
                "type": "string",
                "description": "osm_id of the city mentioned in the text else keep it None"
            }
        },
        "required": ["city", "country"]  # Only city and country are required, osm_id is optional
    }
},
    "search_hacker_news": {
        "name": "search_hacker_news",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to search for"
                },
                "points": {
                    "type": "string",
                    "description": "The topic to search for"
                }
            },
            "required": ["text"]
        }
    },

    "find_newest_github_user": {
        "name": "find_newest_github_user",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "create_a_scheduled_github_action": {
        "name": "create_a_scheduled_github_action",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "extract_tables_from_pdf": {
        "name": "extract_tables_from_pdf",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "subject_filter": {
                    "type": "string",
                    "description": "the name of the subject which will be used as a filtering subject for marks."
                },
                "Group_lower_bound": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "Group_upper_bound": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "target_subject": {
                    "type": "string",
                    "description": "he name of the subject for which the marks have to be calculated"
                }
                
            },
            "required": ['subject_filter','Group_lower_bound','Group_upper_bound','target_subject']
        }
    },

    "convert_a_pdf_to_markdown": {
        "name": "convert_a_pdf_to_markdown",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": []
        }
    },

    "clean_up_excel_sales_data": {
        "name": "clean_up_excel_sales_data",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {
                    "type": "Sales that occurred up to and including a specified date",
                    "description": "The text to extract the data from"
                },
                "product": {
                    "type": "string",
                    "description": "The text to extract the data from"
                },
                "country": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["target_date", "product", "country"]
        }
    },

    "clean_up_student_marks": {
        "type": "function",
        "function": {
            "name": "clean_up_student_marks",
            "description": "Analyzes logs to count the number of successful GET requests matching criteria such as URL prefix, weekday, time window, month, and year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the gzipped log file."
                    },
                    "section_prefix": {
                        "type": "string",
                        "description": "URL prefix to filter log entries (e.g., '/telugu/')."
                    },
                    "weekday": {
                        "type": "integer",
                        "description": "Day of the week as an integer (0=Monday, ..., 6=Sunday)."
                    },
                    "start_hour": {
                        "type": "integer",
                        "description": "Start hour (inclusive) in 24-hour format."
                    },
                    "end_hour": {
                        "type": "integer",
                        "description": "End hour (exclusive) in 24-hour format."
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month number (e.g., 5 for May)."
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year (e.g., 2024)."
                    }
                },
                "required": [
                    "file_path",
                    "section_prefix",
                    "weekday",
                    "start_hour",
                    "end_hour",
                    "month",
                    "year"
                ]
            }
        }
    },

    "apache_log_downloads": {
        "name": "apache_log_downloads",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "clean_up_sales_data": {
        "name": "clean_up_sales_data",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": ""
                }
            },
            "required": ["text"]
        }
    },

    "parse_partial_json": {
        "name": "parse_partial_json",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "extract_nested_json_keys": {
        "name": "extract_nested_json_keys",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "find the timestamp from the query"
                }
            },
            "required": ["timestamp"]
        }
    },

    "duckdb_social_media_interactions": {
        "name": "duckdb_social_media_interactions",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "transcribe_a_youtube_video": {
        "name": "transcribe_a_youtube_video",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "start_sec": {
                    "type": "integer",
                    "description": "start seconds of the video"
                },
                "end_sec": {
                    "type": "string",
                    "description": "ending seconds of the video"
                },
            },
            "required": ["start_sec", "end_sec"]
        }
    },

    "reconstruct_an_image": {
        "name": "reconstruct_an_image",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "string",
                    "description": "mapping according to which image will be reconstructed"
                }
            },
            "required": ["text"]
        }
    },
}
