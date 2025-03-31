import os
import sys
from dotenv import load_dotenv

load_dotenv('TDS-project-2/utils/secrets.env')

#sys.path.append('C:/Users/Pratul/OneDrive/Desktop/TDS_Project_2/TDS-project-2')

print(type(os.environ['API_KEY']))