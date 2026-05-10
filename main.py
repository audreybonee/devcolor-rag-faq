import time
import sys
from dotenv import load_dotenv
import os

from rag import RAGSystem

def main():
    # Load environment variables from .env
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your_api_key_here":
        print("Error: Please set your OPENAI_API_KEY in the .env file.")
        sys.exit(1)
    
    print("Initializing RAG System... Please wait.")
    rag = RAGSystem("devcolorfaq.txt")
    
    print("\n" + "="*50)
    print("Welcome to the /dev/color FAQ Assistant.")
    print("Ask me anything about /dev/color.")
    print("Type 'exit', 'quit', or 'q' to quit.")
    print("="*50)

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower().strip() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        start_time = time.time() 
        response = rag.query(user_input)
        elapsed = time.time() - start_time
        
        print(f"\nAssistant: {response}")
        print(f"(Response time: {elapsed:.2f}s)")

if __name__ == "__main__":
    main()