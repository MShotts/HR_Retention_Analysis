
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
        <li>The data types are comprised of integers and objects. This is mixed news since we\'ll need to create one-hot encoded versions of the object fields in order to include them in the analysis.</li>
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


# Customize the look of the slider - still unable to get rid of the red
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
            f'It would cost **${cost_to_retrain:,}** to train the incoming hires assuming similar salaries to exiting staff members. Please note these numbers should be viewed with caution. I compared hourly rate to monthly rate by multiplying 40 hours per week * 4 weeks and the values were generally lower than the monthly rate.  It\'s possible there are additional considerations driving this difference like sales commissions or exempt employees but the provided data is insufficient to explain.'
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

# Create a line to break up the text
st.markdown('<hr style="border: 1px solid #DAF7DC;">', unsafe_allow_html=True)

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">We\'re going to prepare the data for analysis by doing one-hot encoding. This process converts, for example, the gender field into two columns. Gender has responses of \'Male\' and \'Female\' so the encoding creates a column of Gender_Female and Gender_Male with a boolean to define the employee\'s gender.  This is needed 1) because models need numeric data and 2) we don\'t want to mistakenly indicate a hierarchy where none exists like we would if we created a numeric gender field and converted each gender to a number.</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size:16px; color:#FFFFFF">Additionally, we need to address the Age field. Reducing cardinality (unique values) will strengthen our prediction so we\'re going to group ages together. We\'ll use quartiles to determine reasonable cuts.</span>', unsafe_allow_html=True)
# Determine good age buckets based on quartiles
quantiles = HR_df['Age'].quantile([0.25, 0.5, 0.75])

st.markdown('<span style="font-size:16px; color:#DAF7DC">Age Quartiles</span>', unsafe_allow_html=True)
st.markdown(f"<p style='color:FFFFFF;'>25th Percentile: <b>{quantiles[0.25]}</b></p>", unsafe_allow_html=True)
st.markdown(f"<p style='color:FFFFFF;'>50th Percentile (Median): <b>{quantiles[0.50]}</b></p>", unsafe_allow_html=True)
st.markdown(f"<p style='color:FFFFFF;'>75th Percentile: <b>{quantiles[0.75]}</b></p>", unsafe_allow_html=True)

# Create the age buckets
HR_df['Age_Bin'] = pd.cut(HR_df['Age'],
                          bins=[18, 30, 36, 43, 100],
                          labels=['18-30', '31-36', '37-43', '44+'])
# One-hot encoding
HR_df_encoded=pd.get_dummies(HR_df)
# Uncomment the below to show the encoding

# Your target column (adjust name if different)
X = HR_df_encoded.drop(columns=['Attrition_Yes','Attrition_No','StandardHours','EmployeeCount','Over18_Y','Age'])
y = HR_df['Attrition'].map({'Yes': 1, 'No': 0})

st.write("")
st.markdown('<span style="font-size:16px; color:#FFFFFF">Let\'s do a quick review after encoding to ensure things look good (they do).</span>', unsafe_allow_html=True)
st.write(y.head())
st.write(X.head())

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale your features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, never fit on test data

# Train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

st.markdown('<span style="font-size:16px; color:#DAF7DC">Model Performance</span>', unsafe_allow_html=True)
st.text(classification_report(y_test, y_pred))
# The following formats it with color but the presentation looks awful
# classification_report = classification_report(y_test, y_pred)
# st.markdown(f"""
#     <pre style='color:FFFFFF; font-size:16px;'>{classification_report}</pre>
# """, unsafe_allow_html=True)

st.write("")
# Display results with markdown
roc_score = round(roc_auc_score(y_test, y_proba), 3)
st.markdown(f"""
    <p style='font-size:16px; color:#FFFFFF'>ROC-AUC Score</p>
    <p style='font-size:16px; color:#FFFFFF; font-weight:bold;'>{roc_score}</p>
""", unsafe_allow_html=True)

# Interpret the coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

st.write("")
st.markdown('<span style="font-size:18px; color:#DAF7DC">Attrition Drivers</span>', unsafe_allow_html=True)
# Changing the bar plot to horizontal for readability
coef_df_sorted = coef_df.sort_values('Coefficient', ascending=True)  # ascending=True so highest is at top

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'])

for index, value in enumerate(coef_df_sorted['Coefficient']):
    if abs(value) >= 0.4:
        offset = 0.002 if value >= 0 else -0.002
        ha = 'left' if value >= 0 else 'right'
        ax.text(value + offset, index, f'{value:.3f}', va='center', ha=ha)

ax.set_xlabel('Coefficient')
ax.set_title('Attrition Drivers')
st.pyplot(fig)

st.markdown('<span style="font-size:18px; color:#FFFFFF">As of now, Job Level is in the lead as the top predictor of attrition.  Stay tuned as more is coming soon!</span>', unsafe_allow_html=True)

# streamlit run C:\Users\DrShotts\PycharmProjects\Kaggle_HR_Retention\streamlit_app.py
