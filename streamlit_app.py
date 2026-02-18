
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sets the background color
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #336699, #2F4858);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<span style="font-size:48px; color:#9EE493">**Matthew Shotts**</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:22px; color:#DAF7DC">Analysis Case Study Determining Main Drivers for Employee Attrition</span>', unsafe_allow_html=True)

# st.markdown('<span style="font-size:16px; color:#FFFFFF">The following analysis is of a Kaggle dataset concerning Employee Attrition & Performance.</span>', unsafe_allow_html=True)
st.markdown(
    '<span style="font-size:16px; color:#FFFFFF">'
    'The following analysis is of a Kaggle dataset concerning Employee Attrition & Performance. This dataset can be found at: '
    '<a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset" '
    'style="color:#DAF7DC; text-decoration:none;">HR Attrition Kaggle Dataset</a>'
    '</span>',
    unsafe_allow_html=True
)

st.markdown('<span style="font-size:16px; color:#FFFFFF">First, let\'s see what types of data is included, whether there are null values, and the uniqueness of the fields.</span>', unsafe_allow_html=True)

HR_df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the format and other characteristics of the data
info_df = pd.DataFrame({
    'Column': HR_df.columns,
    'Type': HR_df.dtypes.values,
    'Unique': HR_df.nunique().values,
    'Null': HR_df.isnull().sum().values,
    'Non-Null': HR_df.count().values
})
st.dataframe(info_df, use_container_width=True)

st.markdown('<span style="font-size:16px; color:#FFFFFF">Right away, we can see signs that the data quality is good.</span>', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:16px; color:#FFFFFF;">
    <ul>
        <li>There are no nulls in the data.</li>
        <li>For EmployeeNumber, the counts in the Non-Null and Unique columns match which indicates that this key is unique and there is no need to create a surrogate.</li>
        <li>The data types are comprised of integers and objects. This is mixed news since we will likely need to create nominal/ordinal fields in order to include them in our analysis.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Create a line to break up the text
st.markdown('<hr style="border: 1px solid #DAF7DC;">', unsafe_allow_html=True)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Next up, let\'s see the statistical properties of the data.</span>', unsafe_allow_html=True)
describe_stats=HR_df.describe().T.round(2)
selected_describe_stats=describe_stats[['mean','std','min','max']]

st.dataframe(selected_describe_stats, use_container_width=True, hide_index=False)

st.write("")

st.markdown("""
<div style="font-size:16px; color:#FFFFFF;">
    <ul>
        <li>EmployeeCount and StandardHours have no variability (std=0) and thus no predictive value. We will remove them from consideration.</li>
        <li>We can see that there are survey results, possibly from a Likert scale, such as EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, etc based on the min and max and std ~1.</li>
        <li>The description of the Kaggle dataset explains that ratings go from low to high (i.e. 1=Low vs 4=Very High). There is no reverse scoring but it should be noted that the text descriptions do vary (Very High, Best, Outstanding).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Create a line to break up the text
st.markdown('<hr style="border: 1px solid #DAF7DC;">', unsafe_allow_html=True)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Let\'s see how bad the attrition problem is by checking how much of the workforce has left.</span>', unsafe_allow_html=True)

yes_count = (HR_df['Attrition'] == 'Yes').sum()
# no_count = (HR_df['Attrition'] == 'No').sum()
yes_monthly_rate = HR_df[HR_df['Attrition'] == 'Yes']['MonthlyRate'].sum()
# yes_monthly_rate = (HR_df['Attrition'] == 'Yes').sum(HR_df['MonthlyRate'])
total_count = len(HR_df)
attrition_rate = round(((yes_count / total_count) * 100),1)
# remained_rate = (no_count / total_count) * 100

attr_col1, attr_col2, attr_col3 = st.columns(3)
with attr_col1:
    st.markdown(f"""
        <div style="text-align: center;">
            <p style="color: #FFFFFF; font-size: 16px; margin-bottom: 0px;">
                Total Employees
            </p>
            <h1 style="color: #FFFFFF; font-size: 18px; margin: 0; margin-top:0;">
                {total_count}
            </h1>
        </div>
        """, unsafe_allow_html=True)

with attr_col2:
    st.markdown(f"""
        <div style="text-align: center;">
            <p style="color: #FFFFFF; font-size: 16px; margin-bottom: 0px;">
                Left Company
            </p>
            <h1 style="color: #FFFFFF; font-size: 18px; margin: 0; margin-top:0;">
                {yes_count}
            </h1>
        </div>
        """, unsafe_allow_html=True)

with attr_col3:
    st.markdown(f"""
        <div style="text-align: center;">
            <p style="color: #FFFFFF; font-size: 16px; margin-bottom: 0px;">
                Percent Left Company
            </p>
            <h1 style="color: #FFFFFF; font-size: 18px; margin: 0; margin-top:0;">
                {attrition_rate}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<span style="font-size:16px; color:#DAF7DC">Use the slider below to select the typical number of months needed to train staff in your org and see the resulting cost.</span>', unsafe_allow_html=True)

import streamlit as st

# Customize the look of the slider
st.markdown("""
    <style>
    /* Change slider color */
    div[data-baseweb="slider"] > div > div > div {
        background-color: FFFFFF !important;  /* Track color */
    }
    # div[data-baseweb="slider"] > div > div {
    #     background-color: #FFFFFF !important; /* Unfilled track color */
    # }
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #9EE493;  /* Thumb color */
    }
        div[data-baseweb="slider"] div {
        color: #FFFFFF !important;  /* Number color */
    }
    div[data-baseweb="slider"] > div > div > div > div {
        border: 3px solid #9EE493 !important; /*Border color around thumb */
        box-shadow: 0 0 5px rgba(231, 76, 60, 0.5);
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="tick-bar"] div {
        color: #FFFFFF !important;            /* Min/max number color */
    }
    </style>
    """, unsafe_allow_html=True)

train_replacement_months=st.slider("",1,12)
cost_to_retrain=yes_monthly_rate*train_replacement_months
st.markdown(f'<span style="font-size:16px; color:#FFFFFF">'
            f'It would cost **${cost_to_retrain:,}** to train the incoming hires assuming similar salaries to exiting staff members.'
            f'</span>', unsafe_allow_html=True)

# Create a line to break up the text
st.markdown('<hr style="border: 1px solid #DAF7DC;">', unsafe_allow_html=True)
st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Next, let\'s see how correlated these fields are. For example, I would expect fields like Age and TotalWorkingYears to be highly correlated.</span>', unsafe_allow_html=True)

# Correlation function
numeric_cols = HR_df.select_dtypes(include=['number']).columns.tolist()
# Removing the two fields that have no variability
# Should revise this to automatically capture any fields with std=0 and place into the array
fields_to_remove = {"EmployeeCount","StandardHours"}
filtered_numeric_cols=[x for x in numeric_cols if x not in fields_to_remove]
if len(filtered_numeric_cols) < 2:
    st.error("Need at least 2 numeric columns")
else:
    # Calculate correlation
    corr = HR_df[filtered_numeric_cols].corr()
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xticklabels(ax.get_xticklabels(), color="white")
    ax.set_xticklabels(ax.get_xticklabels(), color="white")
    ax.set_yticklabels(ax.get_yticklabels(), color="white")
    ax.collections[0].colorbar.remove()
    # plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Stay tuned. More coming soon!</span>', unsafe_allow_html=True)


# streamlit run C:\Users\DrShotts\PycharmProjects\Kaggle_HR_Retention\streamlit_app.py
