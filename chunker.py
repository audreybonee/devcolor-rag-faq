import re

class Chunker:
    """
    Parses the FAQ document and chunks it by Q&A pairs.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.chunks = []
        self._parse()

    def _parse(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: Could not find corpus file at {self.filepath}")
            return

        content = re.sub(r'\\s*', '', content)    

        # Split by pattern "Number. **" or "Number. "
        # The regex looks for start of a new question
        # e.g. "1. **What is..."
        pattern = r'(\d+\.\s+\*\*.*?\*\*)'
        
        # We split the text keeping the split token
        parts = re.split(pattern, content)
        
        # parts[0] is everything before the first question (might be empty)
        # parts[1], parts[3], ... are the question lines "1. **What...**"
        # parts[2], parts[4], ... are the answers
        
        current_id = 1
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                question_match = parts[i].strip()
                answer = parts[i+1].strip()
                
                # Extract clean question without "1. **" and "**"
                clean_question = re.sub(r'^\d+\.\s+\*\*', '', question_match)
                clean_question = re.sub(r'\*\*$', '', clean_question).strip()
                
                full_text = f"Question: {clean_question}\nAnswer: {answer}"
                
                self.chunks.append({
                    "id": current_id,
                    "question": clean_question,
                    "answer": answer,
                    "full_text": full_text
                })
                current_id += 1

    def get_chunks(self):
        return self.chunks

if __name__ == "__main__":
    # Test the chunker
    c = Chunker("devcolorfaq.txt")
    chunks = c.get_chunks()
    print(f"Found {len(chunks)} chunks.")
    if chunks:
        print(chunks[0])
