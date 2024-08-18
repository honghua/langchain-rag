import argparse
from pathlib import Path

from datastore import DataStoreGenerator
from query_processor import QueryProcessor

def main():
    data_path = Path("data")
    DataStoreGenerator.generate_data_store(data_path, overwrite=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    processor = QueryProcessor()
    processor.process_query(args.query_text)

if __name__ == "__main__":
    main()