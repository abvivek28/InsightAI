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
    1. Identify 'The Over-Reported Consensus': What is everyone saying?
    2. Identify 3 'Blind Spots': What critical technical, ethical, or human angles are completely ignored?
    3. Write a 600-word investigative report titled 'The Missing Angle of {topic}'.
    """

    # Gemini 3 Pro specific configuration
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1.0,  # Recommended for Gemini 3 reasoning
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH, # Maximize investigative reasoning
                include_thoughts=True # This allows you to see the logic in the debug logs
            )
        )
    )
    
    # Return the full text of the report
    return response.text