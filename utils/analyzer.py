import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def perform_gap_analysis(summaries, topic):
    # Consolidate the Topic Maps from Agent 1
    context_text = "\n\n".join([f"SOURCE: {s['title']}\n{s['content']}" for s in summaries])

    prompt = f"""
    Based on the following research summaries about '{topic}', perform a deep gap analysis.
    
    RESEARCH SUMMARIES:
    {context_text}
    
    INSTRUCTIONS:
    1. Identify the topics covered by the research summaries.
    2. Identify 5 topics which are relevant but not covered in the research summaries.
    3. Write a 600 word article on the topic only highlighting topics identified in step 2.

    NOTE: THE ARTICLE WRITTEN BY YOU SHOULD BE THE ONLY THING GIVEN AS OUTPUT. NO OTHER MESSAGE OR EXPRESSION.
    """

    # Gemini 3 flash specific configuration
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8
        )
    )
    
    # Return the full text of the report

    return response.text



