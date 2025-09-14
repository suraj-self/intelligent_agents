
import re
import json
import torch
import sqlite3
import warnings
import pandas as pd
import os
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from prompts import get_planner_prompt
from utils import create_bar_chart, exec_pandas_plan, sanitize_sql, create_line_chart
# optional jsonschema
try:
    import jsonschema
except Exception:
    jsonschema = None

warnings.filterwarnings("ignore")

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, "plans.jsonl")

class FinancialAIAgentV2:
    """
    An autonomous financial AI agent that plans its actions, including data visualization.
    """

    def __init__(self, config_path="config.json"):
        # Load configuration from JSON file
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.db_path = config.get('db_path', 'agent_workspace.db')
            self.table_name = config.get('table_name', 'knowledge_base')
            self.model_id = config.get('model_id', 'google/gemma-3n-e2b-it')
        except FileNotFoundError:
            print(f"Error: {config_path} not found. Using default values.")
            self.db_path = 'agent_workspace.db'
            self.table_name = 'knowledge_base'
            self.model_id = 'google/gemma-3n-e2b-it'

        # Initialize database connection (read-only)
        uri = f"file:{self.db_path}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.current_year = datetime.now().year

        # Dynamically fetch column names from database schema
        try:
            columns_df = pd.read_sql_query(f"PRAGMA table_info({self.table_name})", self.conn)
            self.columns = columns_df['name'].tolist()
            if not self.columns:
                raise ValueError("No columns found in the table.")
        except Exception as e:
            print(f"Error fetching column names: {str(e)}")
            self.conn.close()
            raise

        # Dynamically fetch unique categories from the Category column if exists
        try:
            if 'Category' in self.columns:
                categories_df = pd.read_sql_query(f"SELECT DISTINCT Category FROM {self.table_name}", self.conn)
                self.categories = categories_df['Category'].dropna().tolist()
            else:
                self.categories = []
        except Exception as e:
            print(f"Error fetching categories: {str(e)}")
            self.conn.close()
            raise

        # Initialize Gemma model and processor
        print("Initializing Gemma Processor and Model...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Gemma model and processor loaded successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not load Gemma model. {e}")
            print("Please ensure you have accepted the license and are logged in via `huggingface-cli login`.")
            self.conn.close()
            raise

        # Load plan schema if available
        self.plan_schema = None
        if os.path.exists(SCHEMA_PATH):
            try:
                with open(SCHEMA_PATH, 'r') as f:
                    self.plan_schema = json.load(f)
            except Exception:
                self.plan_schema = None

        print("Agent initialized. Ask about your expenses (e.g., 'top 3 categories in May 2025').")

    def generate_text(self, prompt: str, max_new_tokens: int = 512, do_sample: bool = False) -> str:
        """Generates text using the Gemma model."""
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        response_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response_text.split("model\n")[-1].strip()

    def generate_plan(self, question: str) -> dict:
        """Generates a structured plan of action (JSON) based on the user's question."""
        print("Generating a plan of action...")
        prompt = get_planner_prompt(self.columns, self.categories, self.current_year)
        plan_prompt = f"{prompt}\n\nUser Query: \"{question}\""

        raw_plan = self.generate_text(plan_prompt, max_new_tokens=512, do_sample=False)

        # Extract JSON object from the model response
        json_match = re.search(r'\{.*\}', raw_plan, re.DOTALL)
        if not json_match:
            print("Error: The model did not return a valid JSON plan.")
            print("Raw plan:", raw_plan)
            return None
        try:
            plan = json.loads(json_match.group(0))
            print("Plan generated successfully.")
            print(plan)
            return plan
        except json.JSONDecodeError:
            print("Error: Failed to decode the JSON plan from the model's response.")
            print(f"Raw response was: {raw_plan}")
            return None

    def validate_plan(self, plan: dict) -> bool:
        """Validates the generated plan against the schema if available."""
        if self.plan_schema is None or jsonschema is None:
            # Fallback minimal checks
            required = ["intent","requires_graph","summary_text","sql_query","pandas_plan","confidence"]
            for k in required:
                if k not in plan:
                    print(f"Plan missing required key: {k}")
                    return False
            return True
        try:
            jsonschema.validate(instance=plan, schema=self.plan_schema)
            return True
        except Exception as e:
            print(f"Plan validation error: {e}")
            return False

    def fetch_data(self, sql_query: str) -> pd.DataFrame | str:
        """Executes a SQL query and returns a DataFrame or an error message."""
        try:
            # sanitize query
            safe_sql = sanitize_sql(sql_query)
            df = pd.read_sql_query(safe_sql, self.conn)
            return df
        except Exception as e:
            return f"Error executing SQL: {str(e)}"

    def execute_plan(self, plan: dict):
        """Executes an already validated plan: run SQL, run pandas_plan, and optionally plot."""
        # Log start
        log_entry = {"timestamp": time.time(), "question": plan.get("_orig_question"), "plan": plan, "status":"started"}
        append_log(log_entry)

        # check clarify
        if plan.get("clarify") and plan["clarify"].get("ask"):
            print("Clarification required:", plan["clarify"].get("question"))
            log_entry["status"]="clarify"
            append_log(log_entry)
            return

        # sanitize sql and fetch
        sql = plan.get("sql_query")
        try:
            df = self.fetch_data(sql)
            if isinstance(df, str):
                raise RuntimeError(df)
        except Exception as e:
            print("Failed to fetch data:", e)
            log_entry["status"]="fetch_error"
            log_entry["error"]=str(e)
            append_log(log_entry)
            return

        if df.empty:
            print("No transactions found for the specified category and period.")
            log_entry["status"]="no_data"
            append_log(log_entry)
            return

        # execute pandas_plan
        pandas_plan = plan.get("pandas_plan", [])
        try:
            result = exec_pandas_plan(df, pandas_plan, params=plan.get("graph_params", {}))
        except Exception as e:
            print("Failed to execute pandas_plan:", e)
            log_entry["status"]="plan_exec_error"
            log_entry["error"]=str(e)
            append_log(log_entry)
            return

        # present the result
        requires_graph = plan.get("requires_graph", False)
        if requires_graph:
            graph_type = plan.get("graph_type", "bar_chart")
            title = plan.get("summary_text", "Chart")
            try:
                if graph_type == "bar_chart":
                    # expect result as df with first col category and second col value
                    if isinstance(result, (pd.Series, pd.DataFrame)):
                        if isinstance(result, pd.Series):
                            plot_df = result.reset_index()
                            plot_df.columns = [ "Category", "Amount" ]
                        else:
                            plot_df = result.copy()
                        plot_path = create_bar_chart(plot_df, title, horizontal=False)
                        print(plan.get("summary_text"))
                        print("Displaying generated chart...", plot_path)
                        log_entry["status"]="success"
                        log_entry["plot"]=plot_path
                        append_log(log_entry)
                    else:
                        print("Graph expected, but result is not a table.")
                        print(result)
                        log_entry["status"]="unexpected_result_type"
                        append_log(log_entry)
                elif graph_type == "line_chart":
                    plot_path = create_line_chart(result, title)
                    print(plan.get("summary_text"))
                    print("Displaying generated chart...", plot_path)
                    log_entry["status"]="success"
                    log_entry["plot"]=plot_path
                    append_log(log_entry)
                else:
                    print("Graph type not yet supported, showing data instead:")
                    print(result)
                    log_entry["status"]="unsupported_graph"
                    append_log(log_entry)
            except Exception as e:
                print("Failed to generate chart:", e)
                log_entry["status"]="plot_error"
                log_entry["error"]=str(e)
                append_log(log_entry)
                return
        else:
            # textual or numeric result
            if isinstance(result, (int, float)):
                 print(f"₹{result:,.2f}")
            else:
                # pretty print small tables
                print(result.head(20) if hasattr(result,'head') else result)
            print(plan.get("summary_text"))
            log_entry["status"]="success"
            append_log(log_entry)
        print("="*58 + "\n")

    def process_question(self, question: str):
        """High-level method to process a user question end-to-end."""
        plan = self.generate_plan(question)
        if not plan:
            print("Could not generate a plan.")
            return
        plan["_orig_question"] = question
        valid = self.validate_plan(plan)
        if not valid:
            print("Plan validation failed. Aborting.")
            return
        # confidence threshold
        if plan.get("confidence",1.0) < 0.6:
            if plan.get("clarify") and plan["clarify"].get("ask"):
                print("Clarifying question:", plan["clarify"].get("question"))
                return
            else:
                print("Low confidence in plan. Please rephrase or be more specific.")
                return
        # execute
        self.execute_plan(plan)

    def __del__(self):
        """Closes the database connection when the object is destroyed."""
        try:
            self.conn.close()
        except Exception:
            pass

# Logging helper
def append_log(entry: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print("Failed to write log:", e)
