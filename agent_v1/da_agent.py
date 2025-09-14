import re
import json
import torch
import sqlite3
import warnings
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor

warnings.filterwarnings("ignore")

class FinancialAIAgent:
    """Autonomous financial AI agent to manage expenses using local Gemma 3N 2B model."""
    
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
        
        # Initialize database connection
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.current_year = datetime.now().year  # Current year for context (2025)
        
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
        
        # Dynamically fetch unique categories from the Category column
        try:
            categories_df = pd.read_sql_query(f"SELECT DISTINCT Category FROM {self.table_name}", self.conn)
            self.categories = categories_df['Category'].dropna().tolist()
            if not self.categories:
                print("Warning: No categories found in the database. Using empty list.")
                self.categories = []
        except Exception as e:
            print(f"Error fetching categories: {str(e)}")
            self.conn.close()
            raise
        
        # Initialize Gemma model and processor
        print("Initializing Gemma Processor...")
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

    def generate_text(self, prompt: str, max_new_tokens: int = 256, do_sample: bool = False) -> str:
        """Generates text using the Gemma model."""
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response_text.split("model\n")[-1].strip()

    def generate_sql_query(self, question: str) -> str:
        """Generates SQL query for the given question."""
        prompt = f"""
        You are a SQL expert. The database has a table named EXACTLY '{self.table_name}' with columns: {', '.join(self.columns)}.
        Categories are: {', '.join(self.categories)}.
        Generate ONLY the SQL SELECT query to fetch the MINIMUM required data efficiently to answer the question: '{question}'.
        Do not include any markdown, code fences (like ```sql), or explanations—just the raw SQL query.
        Ensure the query is safe, read-only, and uses proper formatting for dates (assume 'Date' is TEXT in YYYY-MM-DD format).
        Use exact column names from: {', '.join(self.columns)}.
        For date ranges, extract the month and year from the question (e.g., 'May 2025' means Date >= '2025-05-01' AND Date < '2025-06-01'). If no year is specified, use the current year ({self.current_year}). If the month is ambiguous, assume the most recent occurrence relative to today (August {self.current_year}).
        For queries asking for 'least spending category', select Category and Amount for all transactions in the specified period to allow Pandas to group and sum absolute amounts. Do NOT use MIN, GROUP BY, or ORDER BY in SQL for these queries, as aggregation is handled in Pandas.
        Example: For "least spending category in May 2025", use: SELECT Category, Amount FROM {self.table_name} WHERE Date >= '2025-05-01' AND Date < '2025-06-01'
        Example: For "spending in July 2025 for Fruits & Vegetables", use: SELECT Date, Amount FROM {self.table_name} WHERE Category = 'Fruits & Vegetables' AND Date >= '2025-07-01' AND Date < '2025-08-01'
        Example: For "spending in June for Groceries", use: SELECT Date, Amount FROM {self.table_name} WHERE Category = 'Groceries' AND Date >= '{self.current_year}-06-01' AND Date < '{self.current_year}-07-01'
        """
        raw_query = self.generate_text(prompt, max_new_tokens=256)
        # Strip any markdown or code fences
        cleaned_query = re.sub(r'```sql|```', '', raw_query).strip()
        return cleaned_query

    def fetch_data(self, sql_query: str) -> pd.DataFrame | str:
        """Executes SQL query and returns data as a Pandas DataFrame."""
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            return df
        except Exception as e:
            return f"Error executing SQL: {str(e)}"

    def generate_pandas_code(self, question: str, df_description: str) -> str:
        """Generates Pandas code to analyze the fetched data."""
        prompt = f"""
        You are an autonomous Pandas expert. Given a DataFrame 'df' with columns: {df_description}, analyze the user query: '{question}'.
        Observe the query, think step-by-step about its intent (e.g., sum, average, least spending category, top N), plan the Pandas code, and write ONLY the Python code snippet to compute and print the final answer.
        Rules:
        - Use ONLY columns in the DataFrame: {df_description}. Do NOT assume other columns like 'Date' unless explicitly listed.
        - Do NOT create a new DataFrame or load data.
        - Convert 'Date' to datetime using pd.to_datetime(df['Date']) ONLY if date operations are needed and 'Date' is in the DataFrame.
        - Handle data types: 'Date' as TEXT initially (if present), 'Amount' as float (negative values are debits), others as strings.
        - For spending, use absolute value of 'Amount' (e.g., df['Amount'].abs()).
        - Do NOT filter by date, month, year, or category—SQL already handles this. Date filtering in Pandas will cause errors.
        - Use proper syntax (e.g., wrap conditions in parentheses: (df['col'] == value)).
        - End with print(result) where result is the final value or summary.
        - Do NOT include markdown, code fences, or imports (e.g., import pandas as pd).
        - Avoid explicit if-else in reasoning; flow naturally from query to code.
        Observe: Break down the query into its core ask (e.g., least spending category, average, top 3).
        Think: Reason about required Pandas operations (e.g., groupby, sum, idxmin for least spending).
        Plan: Outline code steps (e.g., convert Amount to float, group by Category, find minimum).
        Write: Provide the complete code snippet.
        """
        raw_code = self.generate_text(prompt, max_new_tokens=300)
        # Strip any markdown or code fences
        cleaned_code = re.sub(r'```python|```', '', raw_code).strip()
        return cleaned_code

    def execute_pandas_code(self, code: str, df: pd.DataFrame):
        """Safely executes the generated Pandas code."""
        try:
            # Basic validation to catch common errors
            if '&' in code and not '(' in code:
                print("Warning: Potential syntax error in generated code (missing parentheses for logical operations).")
                return
            if 'Amount >= 0' in code or 'Amount > 0' in code:
                print("Warning: Generated code incorrectly filters out negative Amount values.")
                return
            local_vars = {'pd': pd, 'df': df}
            exec(code, globals(), local_vars)
        except Exception as e:
            print(f"Error executing Pandas code: {str(e)}")

    def process_question(self, question: str):
        """Processes a natural language question about expenses."""
        print(f"Processing question: {question}")
        
        # Debug: Check total rows for categories in 2025
        debug_query_2025 = f"SELECT COUNT(*) FROM {self.table_name} WHERE Category IN ({','.join([f"'{c}'" for c in self.categories])}) AND Date LIKE '2025%'"
        try:
            debug_count_2025 = pd.read_sql_query(debug_query_2025, self.conn).iloc[0, 0]
            print(f"Debug: Total rows for all categories in 2025: {debug_count_2025}")
        except Exception as e:
            print(f"Debug: Error checking database: {str(e)}")
        
        # Generate and execute SQL
        sql_query = self.generate_sql_query(question)
        print(f"Generated SQL: {sql_query}")
        
        df = self.fetch_data(sql_query)
        if isinstance(df, str):
            print(df)
            return
        print(f"Fetched {len(df)} rows.")
        print(f"Fetched {df.shape}")
        # Debug: Print first few rows of fetched data
        print("Fetched data sample:")
        print(df.head())
        
        if df.empty:
            print("No transactions found for the specified category and period.")
            return
        
        # Generate and execute Pandas code
        df_description = str(df.dtypes.to_dict())
        code = self.generate_pandas_code(question, df_description)
        print(f"Generated Pandas code:\n{code}")
        
        self.execute_pandas_code(code, df)

    def __del__(self):
        """Closes the database connection when the object is destroyed."""
        self.conn.close()

# Example usage
if __name__ == "__main__":
    agent = FinancialAIAgent()
    while True:
        question = input("Ask about your expenses (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        agent.process_question(question)