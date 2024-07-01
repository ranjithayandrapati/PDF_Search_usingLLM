from chromadb import Client
from chromadb.config import Settings

# Define the settings
CHROMADB_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db",
    anonymized_telemetry=False,
)

# Print the settings
print("Chroma settings:", CHROMADB_SETTINGS)

# Print individual settings for clarity
print("chroma_db_impl:", CHROMADB_SETTINGS.chroma_db_impl)
print("persist_directory:", CHROMADB_SETTINGS.persist_directory)
print("anonymized_telemetry:", CHROMADB_SETTINGS.anonymized_telemetry)

# Attempt to create the client
try:
    client = Client(settings=CHROMADB_SETTINGS)
    print("Client created successfully")
except ValueError as ve:
    print("ValueError encountered:", ve)
except Exception as e:
    print("An unexpected error occurred:", e)
