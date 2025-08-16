# 📊 Financial AI Agent

A Python-based AI agent for querying financial expense data using natural language, powered by the **Gemma 3N 2B** model from Hugging Face. The agent connects to a **SQLite database**, generates **SQL queries**, and processes results with **Pandas** to answer questions like:

> *"What is my spending in May 2025 for Fruits & Vegetables?"*

---

## 🚀 Features
- **Natural Language Queries**: Ask questions about expenses (e.g., *"Total spending on Groceries in June 2025"*).
- **Dynamic Schema**: Automatically detects database columns and categories.
- **Configurable**: Uses `config.json` for database path, table name, and model ID.
- **Local Execution**: Runs with Gemma 3N 2B, optimized for ~6GB VRAM GPUs.

---

## 📦 Requirements
- **Python**: `3.10` or later
- **Hardware**: ~6GB VRAM GPU (e.g., NVIDIA) or CPU (slower)
- **Dependencies**: Listed in `requirements.txt`
- **Hugging Face Account**: Required for accessing `google/gemma-3n-e2b-it`
- **SQLite Database**: Must contain at least:
  - `Date` (TEXT, YYYY-MM-DD)
  - `Amount` (FLOAT, negative for debits)
  - `Category` (TEXT)

---

## ⚙️ Installation

**1. Clone the Repository:**
```bash
git clone https://github.com/<your-username>/financial-ai-agent.git
cd financial-ai-agent
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Log in to Hugging Face (required for Gemma model access):**
```bash
huggingface-cli login
```
Follow the prompts to enter your Hugging Face token.

**4. Prepare SQLite Database:**
```sql
CREATE TABLE knowledge_base (
    Date TEXT,
    Transaction_Details TEXT,
    Type TEXT,
    Amount FLOAT,
    Category TEXT
);

INSERT INTO knowledge_base VALUES 
('2025-05-01', 'Grocery Store', 'Debit', -105.0, 'Fruits & Vegetables');
```

**5. Configure Settings (`config.json`):**
```json
{
    "db_path": "agent_workspace.db",
    "table_name": "knowledge_base",
    "model_id": "google/gemma-3n-e2b-it"
}
```

---

## ▶️ Usage

**Run the Script:**
```bash
python ai_agent.py
```

**Enter Queries (examples):**
```text
What is my spending in May 2025 for Fruits & Vegetables?
Total spending on Groceries in June 2025?
Average daily spending on Transport in July 2025?
```
Type `exit` to quit.

**Output:**
- Generates SQL query
- Fetches data
- Processes with Pandas
- Returns result (e.g., `868.0` for total spending)
- Displays debug info for transparency

---

## 💻 Example
```bash
$ python ai_agent.py
Initializing Gemma Processor...
Gemma model and processor loaded successfully.
Ask about your expenses (or 'exit' to quit): what is my spending in may 2025 for Fruits & Vegetables?
Processing question: what is my spending in may 2025 for Fruits & Vegetables?
Debug: Total rows for all categories in 2025: 110
Generated SQL: SELECT Date, Amount FROM knowledge_base WHERE Category = 'Fruits & Vegetables' AND Date >= '2025-05-01' AND Date < '2025-06-01'
Fetched 36 rows.
Fetched (36, 2)
Fetched data sample:
         Date  Amount
0  2025-05-01  -105.0
1  2025-05-03  -187.0
2  2025-05-04  -153.0
3  2025-05-04   -58.0
4  2025-05-05  -365.0
Generated Pandas code:
df['Amount'] = df['Amount'].astype(float)
total = df['Amount'].abs().sum()
print(total)
868.0
```

---

## 📝 Notes
- **Database**: Ensure SQLite DB path matches `config.json`.
- **GPU/CPU**: Optimized for ~6GB VRAM GPUs. For CPU-only:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **Hugging Face**: Requires access to `google/gemma-3n-e2b-it`.
- **Customization**: Modify `config.json` for different DB/table/model.

---

## 🤝 Contributing
- Submit **issues** or **pull requests** to improve the agent.
- Suggestions for features (e.g., visualization, confidence scores) are welcome!

---