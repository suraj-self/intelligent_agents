# import argparse
from pathlib import Path
import database
from gemma_processor import GemmaProcessor

def main(image_path: str):
    """Main function to process an invoice image and save it to the database."""
    print("--- Starting Financial Agent ---")

    # 1. Ensure the database is set up
    database.setup_database()

    # 2. Initialize the AI processor
    try:
        processor = GemmaProcessor()
    except Exception:
        # Error is already printed in the constructor, just exit
        return

    # 3. Process the invoice image
    invoice_data = processor.get_invoice_data(image_path)

    # 4. Save the data if extraction was successful
    if invoice_data:
        print("\n--- Extracted Invoice Data ---")
        print(invoice_data)
        print("----------------------------\n")
        database.save_invoice(invoice_data)
    else:
        print("Could not process invoice. No data was saved.")

    print("\n--- Financial Agent Finished ---")

if __name__ == "__main__":
    main("/home/ss/intelligent_database/invoices/invoice_2.jpeg")
