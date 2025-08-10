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

        print("\nExtracting data from invoice with simplified universal prompt...")
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
        """Creates a robust and universal prompt for invoice data extraction."""
        return f"""{image_token}
            You are a smart detective. Your job is to find specific clues from this receipt and create a JSON report.

            ### Clue 1: Find the Store Details
            * `store_name`: Find the big name of the shop at the very top.
            * `store_location`: Find the city where the shop is.
            * `invoice_date`: Find the date the shopping happened.
            * `transaction_id`: Find the "Bill No" or "Invoice No.".

            ### Clue 2: Find the Final Price
            * `total_amount`: Find the final total price paid. Look for "Total Amount" or "Payable Amt".

            ### Clue 3: List the Shopping Items (Most Important!)
            * Look for the main list of all the things that were bought.
            * For each thing on the list, you only need two pieces of information:
                * `name`: The name of the item. Just the name, no extra codes or numbers.
                * `total_price`: The final price for that item, which is always the number on the far right of the list.

            ### Final Rule
            * If you can't find a clue, just write `null`.
            * Give me ONLY the JSON report and nothing else.
            """