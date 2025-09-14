
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import uuid
from typing import List, Dict, Any
import numpy as np
import json

PLOT_DIR = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def safe_filename(prefix="plot"):
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{uid}.png"

def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def create_bar_chart(df: pd.DataFrame, title: str, x_label: str = "Category", y_label: str = "Amount", horizontal=False, annotate=True):
    """
    Generates and saves a styled bar chart from a Pandas DataFrame.
    Returns the path to the saved PNG file.
    """
    if isinstance(df, pd.Series):
        df = df.reset_index()
        df.columns = [x_label, y_label]
    if df.empty:
        raise ValueError("Cannot generate chart: No data available.")

    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        ax.barh(df.iloc[:,0].astype(str), df.iloc[:,1])
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        ax.bar(df.iloc[:,0].astype(str), df.iloc[:,1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    ax.set_title(title, fontsize=14)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if annotate:
        for i, v in enumerate(df.iloc[:,1]):
            if horizontal:
                ax.text(v, i, f"₹{v:,.2f}", va='center', fontsize=9)
            else:
                ax.text(i, v, f"₹{v:,.2f}", ha='center', va='bottom', fontsize=9)
    filename = safe_filename("bar")
    return save_plot(fig, filename)

def create_line_chart(df: pd.DataFrame, title: str, x_label="Date", y_label="Amount"):
    if df.empty:
        raise ValueError("No data to plot.")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.iloc[:,0], df.iloc[:,1], marker='o')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    filename = safe_filename("line")
    return save_plot(fig, filename)

def create_histogram(df: pd.DataFrame, column, title="Histogram"):
    if df.empty:
        raise ValueError("No data to plot.")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df[column].dropna(), bins=20)
    ax.set_title(title)
    filename = safe_filename("hist")
    return save_plot(fig, filename)

# --------------------------
# Pandas Plan Executor
# --------------------------
def exec_pandas_plan(df: pd.DataFrame, plan: List[str], params: Dict[str, Any]=None):
    """
    Executes a deterministic pandas_plan (list of steps) on df.
    Allowed steps and simple syntax:
      - "abs_amount"
      - "groupby_sum:Category"  -> groupby column and sum Amount
      - "sort_desc"             -> sort by first numeric column desc
      - "nlargest:3"            -> take top 3 by numeric column
      - "mean"                  -> compute mean of Amount
      - "sum"                   -> compute sum of Amount
      - "count"                 -> count rows
      - "resample_monthly"      -> expects Date column, resample monthly sum
      - "filter_eq:Column:Value"-> filter df[df[Column]==Value]
    Returns a result object (number, DataFrame, Series).
    """
    if params is None:
        params = {}
    working = df.copy()
    result = None
    for step in plan:
        if step == "abs_amount":
            if "Amount" in working.columns:
                working["Amount"] = working["Amount"].abs()
            else:
                raise ValueError("Amount column missing for abs_amount")
            result = working
        elif step.startswith("filter_eq:"):
            parts = step.split(":",2)
            if len(parts) == 3:
                col, val = parts[1], parts[2]
                working = working[working[col]==val]
                result = working
            else:
                raise ValueError(f"Invalid filter_eq step: {step}")
        elif step.startswith("groupby_sum:"):
            parts = step.split(":",1)
            col = parts[1]
            if "Amount" not in working.columns:
                raise ValueError("Amount column required for groupby_sum")
            grouped = working.groupby(col)["Amount"].sum().reset_index()
            grouped = grouped.sort_values(by="Amount", ascending=False)
            result = grouped
            working = result
        elif step == "sort_desc":
            numeric_cols = working.select_dtypes(include='number').columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns to sort")
            working = working.sort_values(by=numeric_cols[0], ascending=False)
            result = working
        elif step.startswith("nlargest:"):
            n = int(step.split(":")[1])
            numeric_cols = working.select_dtypes(include='number').columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns for nlargest")
            result = working.nlargest(n, numeric_cols[0])
            working = result
        elif step == "mean":
            if "Amount" not in working.columns:
                raise ValueError("Amount column required for mean")
            result = working["Amount"].mean()
        elif step == "sum":
            if "Amount" not in working.columns:
                raise ValueError("Amount column required for sum")
            result = working["Amount"].sum()
        elif step == "count":
            result = len(working)
        elif step == "resample_month":
            # expects Date column in datetime
            working = working.copy()
            working['Date'] = pd.to_datetime(working['Date'])
            working = working.set_index('Date').resample('M').sum().reset_index()
            result = working
        else:
            raise ValueError(f"Unknown plan step: {step}")
    return result

# --------------------------
# SQL sanitizer
# --------------------------
import re
def sanitize_sql(sql: str, max_limit=5000) -> str:
    """
    Basic sanitizer: allow only SELECT, single statement, add LIMIT if missing.
    """
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string")
    lowered = sql.lower()
    if ";" in sql:
        raise ValueError("Multiple statements or semicolon not allowed")
    if "update " in lowered or "delete " in lowered or "insert " in lowered or "drop " in lowered or "alter " in lowered or "attach " in lowered:
        raise ValueError("Only read-only SELECT queries are allowed")
    # ensure starts with select
    if not lowered.strip().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")
    # enforce limit
    if "limit" not in lowered:
        sql = f"{sql.rstrip()} LIMIT {max_limit}"
    return sql
