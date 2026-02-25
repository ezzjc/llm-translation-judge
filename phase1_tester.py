import google.generativeai as genai

# 1. Authenticate (Replace with your actual API key)
genai.configure(api_key="")

# 2. Choose the model 
model = genai.GenerativeModel('gemini-2.0-flash')

def get_phase1_score(reference, system_output):
    """Takes a reference and system translation and outputs a 0-1 score."""
    
    # 3. Construct the Prompt logic
    prompt = f"""
    You are an expert translation evaluator.
    Compare the System Translation against the Reference Translation.
    Rate the System Translation on a continuous scale from 0.0 (terrible) to 1.0 (perfect).
    
    Reference Translation: {reference}
    System Translation: {system_output}
    
    Output strictly a single number between 0.0 and 1.0. Do not include any other text.
    """
    
    # 4. Call the API
    response = model.generate_content(prompt)
    
    # 5. Extract the output
    try:
        # Strip out any whitespace/newlines and convert to a float
        score = float(response.text.strip())
        return score
    except ValueError:
        print(f"Error parsing score from response: {response.text}")
        return None

# --- Test the Code ---
ref = "The quick brown fox jumps over the lazy dog."
sys_trans = "A fast brown fox leaps above the lazy human."

score = get_phase1_score(ref, sys_trans)
print(f"Phase I Score: {score}")

