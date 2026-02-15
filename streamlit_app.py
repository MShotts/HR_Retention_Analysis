
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Home"  # Browser tab title
    # page_icon="🏠",
    # layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #336699, #2F4858);
    }
    </style>
    """, unsafe_allow_html=True)

# Add clickable link in sidebar
st.sidebar.markdown(
    """
    <a href="https://hrretentionmshotts.streamlit.app/" target="_blank">
        <button style="
            background-color: transparent;
            color: white;
            padding: 0px 0px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            width: 100%;
        ">
            🔗 Example: Predicting Attrition with HR Data
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# st.badge("Work in progress")

st.markdown('<span style="font-size:48px; color:#9EE493">**Matthew Shotts**</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:22px; color:#DAF7DC">Delivering Insights | Improving Processes | Empowering Staff</span>', unsafe_allow_html=True)

st.link_button("Linkedin Profile", 'https://www.linkedin.com/in/mattshotts/')

df_years_exp=pd.DataFrame({
    "Domain": ["Project Mgmt","People Mgmt","SAS/SQL","DataViz/PBI","Cloud DW","Agile"],
    "Years_Experience": [16,15,11,5,4,4]
})

years_exp_chart = alt.Chart(df_years_exp).mark_bar(
    color='#86BBD8'
).encode(
    x=alt.X('Years_Experience',title='Years of Experience',axis=alt.Axis(tickCount=5)),
    y=alt.Y('Domain',sort='-x',title=None)
).configure(
    background='transparent'
).configure_axis(
    labelColor='#FFFFFF',
    titleColor='#FFFFFF'
).properties(
    # width='container',
    width=510,
    height=250
)
st.altair_chart(years_exp_chart, use_container_width=False)

st.markdown('<span style="font-size:22px; color:#DAF7DC">Drove enterprise-level analytics strategy and execution across healthcare, retail, and assessment organizations:</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:14px; color:#FFFFFF">Delivering cost savings (six-figure annual vendor fee reductions), improved decision-making, and scalable data products used by hundreds of users weekly.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:22px; color:#DAF7DC">Built and led high-performing analytics and visualization teams:</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:14px; color:#FFFFFF">Recruiting and developing talent, defining roles and operating models, and stabilizing organizations through change including re-organizations that reduced team size by 50% yet had no disruption to service or user experience.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:22px; color:#DAF7DC">Designed and operationalized modern data platforms and semantic layers:</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:14px; color:#FFFFFF">Standing up a Power BI ecosystem (Center of Excellence, standards, governance, and support) and contributing to the launch of new healthcare and assessment products.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:22px; color:#DAF7DC">Partnered cross-functionally with executives, internal stakeholders, and external vendors:</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:14px; color:#FFFFFF">Resolving complex data and process issues, validating third-party solutions, and representing analytics functions to senior leadership and international clients.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:22px; color:#DAF7DC">Delivered sustained efficiency and quality improvements:</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:14px; color:#FFFFFF">Modernizing systems and statistical models, introducing agile delivery practices, and reducing critical workflows by up to 3-4 weeks while maintaining accuracy, compliance, and stakeholder trust.</span>', unsafe_allow_html=True)

# Use the following in terminal (lower side ribbon) to run the app
# streamlit run C:\Users\DrShotts\PycharmProjects\Streamlit\streamlit_app.py
