function_definitions_objects_llm = {
    "vs_code_version": {
        "name": "vs_code_version",
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

    "make_http_requests_with_uv": {
        "name": "make_http_requests_with_uv",
        "description": "extract the http url and query parameters from the given text",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address to send the request to"
                },
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to"
                },
                "query_params": {
                    "type": "object",
                    "description": "The query parameters to send with the request"
                }
            },
            "required": ["email", "url"]
        }
    },

    "run_command_with_npx": {
        "name": "run_command_with_npx",
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

    "use_google_sheets": {
        "name": "use_google_sheets",
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

    "use_excel": {
        "name": "use_excel",
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
            "required": ["text"]
        }
    },

    "count_wednesdays": {
        "name": "count_wednesdays",
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

    "extract_csv_from_a_zip": {
        "name": "extract_csv_from_a_zip",
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

    "use_json": {
        "name": "use_json",
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

    "multi_cursor_edits_to_convert_to_json": {
        "name": "multi_cursor_edits_to_convert_to_json",
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

    "css_selectors": {
        "name": "css_selectors",
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

    "process_files_with_different_encodings": {
        "name": "process_files_with_different_encodings",
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

    "use_github": {
        "name": "use_github",
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

    "replace_across_files": {
        "name": "replace_across_files",
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

    "list_files_and_attributes": {
        "name": "list_files_and_attributes",
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

    "move_and_rename_files": {
        "name": "move_and_rename_files",
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

    "compare_files": {
        "name": "compare_files",
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
            "required": ["text"]
        }
    },

    "write_documentation_in_markdown": {
        "name": "write_documentation_in_markdown",
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

    "compress_an_image": {
        "name": "compress_an_image",
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

    "host_your_portfolio_on_github_pages": {
        "name": "host_your_portfolio_on_github_pages",
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

    "use_google_colab": {
        "name": "use_google_colab",
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

    "use_an_image_library_in_google_colab": {
        "name": "use_an_image_library_in_google_colab",
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "llm_token_cost": {
        "name": "llm_token_cost",
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

    "generate_addresses_with_llms": {
        "name": "generate_addresses_with_llms",
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

    "llm_vision": {
        "name": "llm_vision",
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

    "llm_embeddings": {
        "name": "llm_embeddings",
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "import_html_to_google_sheets": {
        "name": "import_html_to_google_sheets",
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

    "scrape_imdb_movies": {
        "name": "scrape_imdb_movies",
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "find_the_bounding_box_of_a_city": {
        "name": "find_the_bounding_box_of_a_city",
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

    "search_hacker_news": {
        "name": "search_hacker_news",
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "convert_a_pdf_to_markdown": {
        "name": "convert_a_pdf_to_markdown",
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

    "clean_up_excel_sales_data": {
        "name": "clean_up_excel_sales_data",
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

    "clean_up_student_marks": {
        "name": "clean_up_student_marks",
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

    "apache_log_requests": {
        "name": "apache_log_requests",
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
                    "description": "The text to extract the data from"
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
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
                "text": {
                    "type": "string",
                    "description": "The text to extract the data from"
                }
            },
            "required": ["text"]
        }
    },

    "reconstruct_an_image": {
        "name": "reconstruct_an_image",
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
}
