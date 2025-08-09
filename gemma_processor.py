import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import time
import json

class GemmaProcessor:
    """A class to handle all interactions with the Gemma model."""
    def __init__(self, model_id="google/gemma-3n-e2b-it"):
        print("Initializing Gemma Processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Gemma model and processor loaded successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not load Gemma model. {e}")
            print("Please ensure you have accepted the license and are logged in via `huggingface-cli login`.")
            raise

    def get_invoice_data(self, image_path: str) -> dict | None:
        """Processes an invoice image and returns the extracted data as a dictionary."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None

        image_token = self.processor.tokenizer.image_token
        prompt_text = self._build_prompt(image_token)
        
        chat = [{"role": "user", "content": prompt_text}]
        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

        print("\nExtracting data from invoice with high precision...")
        start_time = time.time()
        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        elapsed_time = time.time() - start_time
        print(f"Data extracted in {elapsed_time:.2f} seconds.")

        response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer_string = response_text.split("model\n")[-1].strip()

        if answer_string.startswith("```json"):
            answer_string = answer_string[7:-3].strip()
        
        try:
            return json.loads(answer_string)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse model output as JSON. {e}")
            print("--- Model's Raw Response ---")
            print(answer_string)
            print("----------------------------")
            return None

    def _build_prompt(self, image_token: str) -> str:
        """Creates the detailed and restrictive prompt for the model."""
        return f"""{image_token}
            You are a meticulous data extraction expert. Your task is to analyze the provided invoice image and generate a single, perfectly accurate JSON object. You MUST follow these rules without deviation.

            ### Rule 1: Extract Main Details
            - `store_name`: The name of the store.
            - `store_location`: The city and primary location of the store.
            - `invoice_date`: The date of the bill in DD/MM/YYYY format.
            - `transaction_id`: The unique bill or invoice number.

            ### Rule 2: Extract Financials with Extreme Care
            - `total_amount`: Find the final grand total. This is usually labeled 'Amount Received From Customer' or a similar final total.

            ### Rule 3: Extract the Items List Flawlessly
            - You MUST ONLY extract items from the main list under the 'HSN' or 'Description' or 'Particulars' heading.
            - **NEGATIVE CONSTRAINT:** DO NOT include any lines from the 'GST Breakup-Details' section in the items list.
            - **NEGATIVE CONSTRAINT:** DO NOT invent or hallucinate items. If an item is not clearly visible in the 'Particulars' list, do not include it.
            - For each real item:
                - `name`: The description of the item. Be precise.
                - `quantity`: The value from the 'Qty/Kg' column.
                - `total_price`: The value from the 'Value' column for that specific item.
                - `unit_price`: You MUST calculate this by dividing the item's `total_price` by its `quantity`.

            ### Final Checklist Before Output
            Before you generate the JSON, mentally confirm the following:
            1.  Are all items taken ONLY from the 'HSN' or 'Description' or 'Particulars' list?
            2.  Are there any invented items in the list? (There should be none).

            Now, generate ONLY the single, valid JSON object and nothing else.
            """