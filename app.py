import streamlit as st
import os
import re
import time
from tavily import TavilyClient
from utils.analyzer import perform_gap_analysis
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()
st.set_page_config(page_title="InsightAI | Analysis", layout="wide", initial_sidebar_state="collapsed")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Apply InsightAI styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Reset Streamlit defaults */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 900px;
    }
    
    /* Apply Inter font globally */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background */
    .stApp {
        background-color: #F4F4F9;
    }
    
    /* Text gradient */
    .text-gradient {
        background: linear-gradient(135deg, #7C3AED, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7C3AED, #2563EB, #10b981);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7C3AED !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    /* Loading card */
    .loading-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 1.5rem;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.1);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.5rem;
    }
    
    .badge-purple {
        background: rgba(124, 58, 237, 0.15);
        color: #7C3AED;
    }
    
    .badge-yellow {
        background: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
    }
    
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }
    
    /* Floating animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulsing {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- AGENT FUNCTIONS ---
@retry(wait=wait_exponential(multiplier=2, min=15, max=60), stop=stop_after_attempt(3))
def summarize_source(title, raw_text, topic):
    content_safe = raw_text[:4000] 
    
    prompt = f"""
    You are a Content Extraction Agent. Create a 'Topic Map' of this article.
    
    ARTICLE: {title}
    FOCUS: {topic}
    CONTENT: {content_safe}
    
    INSTRUCTIONS:
    1. List Main Topics as Headings.
    2. Provide 2 sentences of detail per heading.
    3. Keep the total output brief (max 400 words).
    """
    
    response = client.models.generate_content(
        model="gemini-3.0-flash-preview", 
        contents=prompt
    )
    return response.text

def clean_web_text(text):
    if not text: return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- GET TOPIC FROM URL ---
query_params = st.query_params
topic = query_params.get("topic", "")

if not topic:
    # No topic provided - show message
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; min-height: 60vh; display: flex; flex-direction: column; justify-content: center;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>
            <span class='text-gradient'>InsightAI</span> Intelligence
        </h1>
        <p style='color: #64748b; font-size: 1.25rem; margin-bottom: 2rem;'>
            Please use the landing page to submit a topic for analysis.
        </p>
        <p style='color: #94a3b8; font-size: 1rem;'>
            ← Go back to <a href="https://insightai123.vercel.app" style='color: #7C3AED; text-decoration: underline;'>the landing page</a> to get started
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # --- HEADER WITH TOPIC ---
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
            <span class='text-gradient'>InsightAI</span> Analysis
        </h1>
        <p style='color: #64748b; font-size: 1.125rem; margin-bottom: 0.5rem;'>
            Analyzing: <strong style='color: #475569;'>{topic}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- LOADING INTERFACE ---
    loading_container = st.empty()
    
    with loading_container.container():
        st.markdown("""
        <div class='loading-card'>
            <div style='font-size: 4rem; margin-bottom: 1.5rem;' class='floating'>🔍</div>
            <h2 style='color: #475569; margin-bottom: 1rem;'>Deep Analysis in Progress</h2>
            <p style='color: #64748b; margin-bottom: 2rem;'>Scanning global archives and detecting information gaps...</p>
        </div>
        """, unsafe_allow_html=True)
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown("<div style='text-align: center;'><span class='status-badge badge-purple pulsing'>🛡️ Aggregating</span></div>", unsafe_allow_html=True)
        with status_col2:
            st.markdown("<div style='text-align: center;'><span class='status-badge badge-yellow'>🎯 Analyzing</span></div>", unsafe_allow_html=True)
        with status_col3:
            st.markdown("<div style='text-align: center;'><span class='status-badge badge-green'>⚡ Synthesizing</span></div>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
    
    # --- ANALYSIS LOGIC ---
    try:
        # Step 1: Fetch sources
        progress_bar.progress(10)
        search_result = tavily.search(query=topic, max_results=5, include_raw_content=True)
        raw_articles = search_result.get('results', [])
        
        if not raw_articles:
            loading_container.empty()
            st.error("❌ No articles found. Please try a different topic.")
        else:
            # Step 2: Summarize sources
            valid_summaries = []
            progress_increment = 60 / len(raw_articles)
            current_progress = 10
            
            for i, a in enumerate(raw_articles):
                clean_txt = clean_web_text(a.get('raw_content') or a.get('content'))
                summary = summarize_source(a['title'], clean_txt, topic)
                
                valid_summaries.append({
                    "title": a['title'],
                    "url": a['url'],
                    "content": summary
                })
                
                current_progress += progress_increment
                progress_bar.progress(int(current_progress))
                
                if i < len(raw_articles) - 1:
                    time.sleep(12)
            
            # Step 3: Generate final report
            progress_bar.progress(75)
            
            context_text = "\n\n".join([f"SOURCE: {s['title']}\n{s['content']}" for s in valid_summaries])
            
            final_analyzer_prompt = f"""
ROLE: Lead Investigative Journalist & Information Architect.
TASK: Perform a Meta-Analysis on the following pre-summarized research data regarding '{topic}'.

RESEARCH DATA:
{context_text}

INSTRUCTIONS:
1. Identify 'The Over-Reported Consensus': What is everyone saying?
2. Identify 3 'Blind Spots': What critical technical, ethical, or social angle is completely ignored?
3. Write a 600-word investigative report titled 'Beyond the Headlines: The {topic} Paradox'.
NOTE: If the passed content is empty or None then you have full autonomy to create an article on the topic as long as the topic is sensible. If not just return a message saying "Sorry, could not create article".

The final answer you give should only be the report generated.
"""
            
            report = perform_gap_analysis(valid_summaries, topic)
            progress_bar.progress(100)
            
            # Clear loading interface
            time.sleep(0.5)
            loading_container.empty()
            
            # --- DISPLAY FINAL REPORT ---
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(124, 58, 237, 0.05), rgba(37, 99, 235, 0.05)); 
                        padding: 3rem; border-radius: 1.5rem; border-left: 5px solid #7C3AED; margin-top: 2rem;'>
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <span class='status-badge badge-green'>✓ Analysis Complete</span>
                </div>
                <h2 style='color: #7C3AED; margin-bottom: 1.5rem; text-align: center;'>📊 Intelligence Report</h2>
                {report}
            </div>
            """, unsafe_allow_html=True)
            
            # Footer
            st.markdown("""
            <div style='text-align: center; padding: 3rem 0 1rem 0; color: #94a3b8; font-size: 0.875rem; margin-top: 3rem;'>
                <div style='margin-bottom: 0.5rem; font-weight: 800; font-size: 1rem;' class='text-gradient'>InsightAI</div>
                <p>&copy; 2026 InsightAI Intelligence. All rights reserved.</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        loading_container.empty()
        st.markdown(f"""
        <div style='background: rgba(239, 68, 68, 0.1); padding: 2rem; border-radius: 1rem; border-left: 4px solid #ef4444; text-align: center;'>
            <h3 style='color: #ef4444; margin-bottom: 1rem;'>❌ Analysis Failed</h3>
            <p style='color: #64748b;'>{str(e)}</p>
        </div>

        """, unsafe_allow_html=True)




