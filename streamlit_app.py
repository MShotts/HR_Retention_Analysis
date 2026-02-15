
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

st.write("The following analysis is of a Kaggle dataset concerning Employee Attrition & Performance.")
st.write("This dataset can be found at: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")

st.write("First, let's see what types of data is included, whether there are null values, and the uniqueness of the fields.")

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

st.write("Right away, we can see signs that the data quality is good.")
st.write("There are no nulls in the data.")
st.write("For EmployeeNumber, the counts in the Non-Null and Unique columns match which indicates that this key is unique and there is no need to create a surrogate.")

st.write("The data types are comprised of integers and objects. This is mixed news since we'll likely need to create nominal fields in order to include them in our analysis.")

st.write("")
st.write("Next up, let's see the statistical properties of the data.")
st.dataframe(HR_df.describe().T.round(2), use_container_width=True, hide_index=False)

st.write("EmployeeCount and StandardHours have no variability (std=0) and thus no predictive value. We'll remove them from consideration in later analyses.")
st.write("We can see that there are survey results, possibly from a Likert scale, such as EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, etc based on the min, 25%, 50%, 75%, and max increasing incrementally from 1-5 or 1-4.")
st.write("The description of the Kaggle dataset explains that ratings go from low to high (i.e. 1=Low vs 4=Very High). There is no reverse scoring but it should be noted that the text descriptions do vary (Very High, Best, Outstanding)")

st.write("")
st.write("Next, let's see how correlated these fields are. For example, I'd expect fields like Age and TotalWorkingYears to be highly correlated.")

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


# st.write("### Columns Detected as Numeric")
# numeric_cols = HR_df.select_dtypes(include="number").columns.tolist()
# st.write(numeric_cols)




# streamlit run C:\Users\DrShotts\PycharmProjects\Kaggle_HR_Retention\streamlit_app.py
