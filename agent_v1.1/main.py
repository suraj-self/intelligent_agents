from agent import FinancialAIAgentV2

def main():
    """
    Initializes and runs the Financial AI Agent V1.1.
    """
    print("--- Welcome to your Autonomous Financial AI Agent (v1.1) ---")
    print("I can now generate charts for comparisons!")
    
    try:
        agent = FinancialAIAgentV2(config_path="config.json")
    except Exception as e:
        print(f"Failed to initialize the agent: {e}")
        return

    while True:
        question = input("Ask about your expenses or 'exit' to quit: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        if not question:
            continue
        
        agent.process_question(question)

if __name__ == "__main__":
    main()
