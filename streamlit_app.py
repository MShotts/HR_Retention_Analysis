
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

st.markdown('<span style="font-size:16px; color:#FFFFFF">The following analysis is of a Kaggle dataset concerning Employee Attrition & Performance.</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:16px; color:#FFFFFF">This dataset can be found at: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:16px; color:#FFFFFF">First, let\'s see what types of data is included, whether there are null values, and the uniqueness of the fields.</span>', unsafe_allow_html=True)

HR_df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the format and other characteristics of the data
info_df = pd.DataFrame({
    'Column': HR_df.columns,
    'Type': HR_df.dtypes.values,
    'Non-Null': HR_df.count().values,
    'Null': HR_df.isnull().sum().values,
    'Unique': HR_df.nunique().values
})
st.dataframe(info_df, use_container_width=True)

st.markdown('<span style="font-size:16px; color:#FFFFFF">Right away, we can see signs that the data quality is good</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:16px; color:#FFFFFF">There are no nulls in the data</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:16px; color:#FFFFFF">For EmployeeNumber, the counts in the Non-Null and Unique columns match which indicates that this key is unique and there is no need to create a surrogate.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:16px; color:#FFFFFF">The data types are comprised of integers and objects. This is mixed news since we will likely need to create nominal fields in order to include them in our analysis</span>', unsafe_allow_html=True)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Next up, let\'s see the statistical properties of the data</span>', unsafe_allow_html=True)
st.dataframe(HR_df.describe().T.round(2), use_container_width=True, hide_index=False)

st.markdown('<span style="font-size:16px; color:#FFFFFF">EmployeeCount and StandardHours have no variability (std=0) and thus no predictive value. We will remove them from consideration in later analyses.</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:16px; color:#FFFFFF">We can see that there are survey results, possibly from a Likert scale, such as EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, etc based on the min, 25%, 50%, 75%, and max increasing incrementally from 1-5 or 1-4.</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:16px; color:#FFFFFF">The description of the Kaggle dataset explains that ratings go from low to high (i.e. 1=Low vs 4=Very High). There is no reverse scoring but it should be noted that the text descriptions do vary (Very High, Best, Outstanding)</span>', unsafe_allow_html=True)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Next, let\'s see how correlated these fields are. For example, I would expect fields like Age and TotalWorkingYears to be highly correlated.</span>', unsafe_allow_html=True)

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


# streamlit run C:\Users\DrShotts\PycharmProjects\Kaggle_HR_Retention\streamlit_app.py
