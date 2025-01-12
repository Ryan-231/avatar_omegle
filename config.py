import os
class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')  # Use environment variable, default to localhost
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')      # Use environment variable, default to root
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')  # Use environment variable, default to empty string
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'matching_app')  # Use environment variable, default to 'matching_app'

