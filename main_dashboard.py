import streamlit as st
import os
import tempfile
import re
import base64
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Career Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_pdf_text(uploaded_file):
    """Saves uploaded file temporarily, extracts text, and cleans up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text = "\n\n".join(doc.page_content for doc in documents)
    os.unlink(tmp_file_path)
    return text

def display_pdf(uploaded_file):
    """Displays PDF in an iframe on the frontend."""
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" style="border-radius:10px; border:1px solid #ccc;"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def extract_score(response_text):
    """Extracts ATS numerical score from the LLM output."""
    match = re.search(r'\*\*Score:\*\*\s*(\d+)', response_text)
    return int(match.group(1)) if match else 0

def create_pie_chart(score):
    """Creates a donut chart for ATS Score."""
    df = pd.DataFrame({"Category": ["Match", "Gap"], "Value": [score, 100 - score]})
    fig = px.pie(
        df, values='Value', names='Category', color='Category',
        color_discrete_map={'Match': '#00CC96', 'Gap': '#EF553B'}, hole=0.45
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16)
    fig.update_layout(
        title_text="JD Match Analysis", title_x=0.25,
        margin=dict(t=40, b=0, l=0, r=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 3. MAIN APPLICATION SETUP
# ==========================================
def main():
    load_dotenv()
    
    # Check API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()

    # Shared LLM Instance
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("🚀 AI Career Suite")
    st.sidebar.markdown("Navigate between services below:")
    
    app_mode = st.sidebar.radio(
        "Choose a Tool:",
        ["Resume Evaluator", "ATS Resume Scorer", "Cover Letter Generator", "AI Career Coach"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Upload your resume and let AI empower your job search workflow.")

    # ==========================================
    # TOOL 1: RESUME EVALUATOR
    # ==========================================
    if app_mode == "Resume Evaluator":
        st.title("📄 AI Resume Evaluator")
        st.markdown("Get a general critique, strengths, weaknesses, and structural analysis of your resume.")

        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="eval_upload")

        if st.button("Evaluate Resume ✨", type="primary"):
            if not uploaded_file:
                st.warning("⚠️ Please upload a PDF file first.")
            else:
                with st.spinner("Analyzing your resume formatting, clarity, and impact..."):
                    try:
                        context = get_pdf_text(uploaded_file)
                        prompt = PromptTemplate(
                            input_variables=["context", "question"],
                            template="""You are an advanced resume evaluation assistant. Analyze the provided resume and score it out of 100.
                            Resume: {context}
                            
                            Structure your response exactly as follows:
                            1. **Score**: [Provide score out of 100]
                            2. **Strengths**: [At least 3 strengths]
                            3. **Weaknesses**: [At least 3 weaknesses/areas to improve]
                            4. **Skills Mentioned**: [List explicitly mentioned skills]
                            5. **Recommended Skills**: [Suggest missing relevant skills]
                            6. **Next Career Paths**: [Suggest next logical career steps]
                            
                            User Question: {question}"""
                        )
                        formatted_prompt = prompt.format(context=context, question="Please evaluate this resume")
                        
                        st.subheader("📊 Evaluation Results")
                        with st.container(border=True):
                            st.write_stream((chunk.content for chunk in llm.stream(formatted_prompt)))
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


    # ==========================================
    # TOOL 2: ATS RESUME SCORER
    # ==========================================
    elif app_mode == "ATS Resume Scorer":
        st.title("🎯 AI ATS Resume Scorer")
        st.markdown("Compare your resume against a specific Job Description to find keyword gaps and get an ATS match score.")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"], key="ats_upload")
        with col2:
            job_description = st.text_area("2. Paste Job Description", height=150)

        if st.button("🚀 Analyze ATS Compatibility", type="primary"):
            if not uploaded_file or not job_description.strip():
                st.warning("⚠️ Please provide both a Resume and a Job Description.")
            else:
                with st.spinner("🤖 Analyzing resume against JD... Please wait."):
                    try:
                        context = get_pdf_text(uploaded_file)
                        template = f"""Act as an expert Applicant Tracking System (ATS). Evaluate this resume against the job description.
                        Job Description: {job_description}
                        Candidate's Resume: {context}

                        STRICT RULE: Do not hallucinate experience.
                        Provide your evaluation EXACTLY in this structure:
                        **Score:** [Number out of 100]
                        **Overall Match:** [Percentage]
                        **Keywords Matched:** [List]
                        **Missing Keywords:** [List]
                        **Readability Score:** [Score/100]
                        **ATS Compatibility Score:** [Score/100]

                        **Format Analysis:** [2 lines assessing layout/parsability]
                        **Skill Gap Analysis:** [Brief analysis of core skills vs actual skills]
                        **Overall Improvement Suggestions:** [2-3 actionable tips]
                        **Industry Specific Feedback:** [Tailored feedback/certifications needed]"""

                        response = llm.invoke(template)
                        result_text = response.content
                        score = extract_score(result_text)

                        st.markdown("---")
                        st.header("📊 ATS Results")
                        res_col1, res_col2 = st.columns([1, 2])
                        with res_col1:
                            st.plotly_chart(create_pie_chart(score), use_container_width=True)
                            st.metric(label="Overall ATS Score", value=f"{score}/100")
                        with res_col2:
                            with st.container(border=True):
                                st.markdown(result_text)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


    # ==========================================
    # TOOL 3: COVER LETTER GENERATOR
    # ==========================================
    elif app_mode == "Cover Letter Generator":
        st.title("✍️ Cover Letter Generator")
        st.markdown("Instantly draft a personalized cover letter tailored to your target job.")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"], key="cl_upload")
        with col2:
            job_description = st.text_area("2. Paste Job Description", height=150, placeholder="Company: Tech Inc...\nRole: Engineer...")

        if st.button("✨ Generate Custom Cover Letter", type="primary"):
            if not uploaded_file or not job_description.strip():
                st.warning("⚠️ Please provide both a Resume and a Job Description.")
            else:
                with st.spinner("Drafting your perfect cover letter..."):
                    try:
                        context = get_pdf_text(uploaded_file)
                        template = f"""Write a professional, compelling cover letter tailored to the job description provided.
                        Emphasize the candidate's most relevant experience that directly matches the role.
                        Structure: Header, Opening (role + why strong fit), 1-2 Body paragraphs (evidence), Closing (call to action).
                        
                        Job Description: {job_description}
                        Resume: {context}
                        
                        DO NOT invent facts not present in the resume."""

                        st.subheader("✉️ Your Cover Letter")
                        with st.container(border=True):
                            st.write_stream((chunk.content for chunk in llm.stream(template)))
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


    # ==========================================
    # TOOL 4: AI CAREER COACH
    # ==========================================
    elif app_mode == "AI Career Coach":
        st.title("💬 AI Career Coach")
        st.markdown("Chat with an expert AI coach to get interview prep, bullet-point rewrites, and career strategy.")

        # Initialize Session States
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "sys_msg" not in st.session_state:
            st.session_state.sys_msg = None

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Your Document")
            uploaded_file = st.file_uploader("Upload resume to set context", type=["pdf"], key="coach_upload")
            
            if uploaded_file:
                display_pdf(uploaded_file)
                # Parse new file if it's uploaded
                if "pdf_name" not in st.session_state or st.session_state.pdf_name != uploaded_file.name:
                    with st.spinner("Coach is reviewing your resume..."):
                        context = get_pdf_text(uploaded_file)
                        sys_content = f"""You are an elite Career Coach. 
                        Tone: Empathetic, encouraging, radically candid.
                        Responsibilities: ATS Optimization (XYZ formula), Interview Prep (STAR method), Career Strategy.
                        Rule: Do not hallucinate experience. Point out flaws before rewriting.
                        Candidate Resume: {context}"""
                        
                        st.session_state.sys_msg = SystemMessage(content=sys_content)
                        st.session_state.chat_history = []
                        st.session_state.pdf_name = uploaded_file.name

        with col2:
            st.subheader("🤖 Chat Area")
            if not uploaded_file:
                st.info("👈 Upload your resume on the left to activate your personal Career Coach.")
            else:
                # Display history
                chat_container = st.container(height=500, border=True)
                with chat_container:
                    for msg in st.session_state.chat_history:
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        with st.chat_message(role):
                            st.markdown(msg.content)

                # Input bar
                if prompt := st.chat_input("Ask: 'Can you rewrite my 2nd bullet point under my latest job?'"):
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            messages = [st.session_state.sys_msg] + st.session_state.chat_history
                            response = st.write_stream((chunk.content for chunk in llm.stream(messages)))
                            
                    st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()