import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.set_page_config(layout="wide", page_title="Employee Salary Prediction App", page_icon="💼")

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    /* Main content area background */
    .main {
        background-color: #f8f9fa; /* Light grey background */
        padding: 1rem 2rem;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #ffffff; /* White sidebar background */
        padding: 1.5rem;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .stSidebar .stFileUploader {
        margin-bottom: 20px;
    }
    .stSidebar .stSelectbox {
        margin-top: 20px;
    }

    /* Header styling */
    .header-title {
        font-size: 2.8em; /* Slightly larger */
        color: #2c3e50; /* Darker blue/grey */
        text-align: center;
        margin-bottom: 1.5rem;
        padding-top: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        font-weight: 800; /* Extra bold */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Subheadings */
    h2 {
        color: #2c3e50;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #34495e;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Card styling for metrics and info sections */
    .info-card, .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* More prominent shadow */
        border: 1px solid #dcdcdc; /* Slightly darker border */
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover, .metric-card:hover {
        transform: translateY(-5px); /* More pronounced lift */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15); /* Even larger shadow on hover */
    }
    .metric-label {
        font-size: 0.95em;
        color: #6c757d;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: 700;
        color: #34495e;
    }

    /* Button styling */
    .stButton>button {
        background-color: #007bff; /* Primary blue button */
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 10px rgba(0, 123, 255, 0.2);
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
    }
    .stDataFrame thead th {
        background-color: #f0f2f6 !important;
        color: #555 !important;
        font-weight: 600 !important;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f5f5f5;
    }

    /* Plotly chart container styling */
    .stPlotlyChart {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        margin-top: 1.5rem;
    }

    /* Matplotlib plot container styling */
    .stImage {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        margin-top: 1.5rem;
    }

    /* General text styling */
    p {
        line-height: 1.6;
        margin-bottom: 1em;
    }
    ul {
        margin-bottom: 1em;
        padding-left: 20px;
    }
    li {
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Define Page Navigation ---
PAGES = {
    "Home": "home_page",
    "Salary Prediction": "salary_prediction_page",
    "Data Exploration": "data_exploration_page",
    "Model Analytics": "model_analytics_page",
    "About": "about_page"
}

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'uploaded_file_object' not in st.session_state:
    st.session_state.uploaded_file_object = None

# --- Sidebar Navigation ---
st.sidebar.markdown("## Navigation")
st.sidebar.write("Available Pages:")
for page_name in PAGES.keys():
    st.sidebar.markdown(f"- {page_name}")

# Dropdown for navigation
selected_page_name = st.sidebar.selectbox(
    "Go to Page:",
    list(PAGES.keys()),
    index=list(PAGES.keys()).index(st.session_state.current_page)
)

if selected_page_name != st.session_state.current_page:
    st.session_state.current_page = selected_page_name
    st.rerun()

DEPARTMENT_COLUMN = 'workclass'
SALARY_COLUMN = 'hours-per-week'
NAME_COLUMN = 'age'

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = [DEPARTMENT_COLUMN, SALARY_COLUMN]
            if not all(col in df.columns for col in required_columns):
                st.error(f"Error: The CSV file must contain '{DEPARTMENT_COLUMN}' and '{SALARY_COLUMN}' columns.")
                return pd.DataFrame()
            else:
                df[SALARY_COLUMN] = pd.to_numeric(df[SALARY_COLUMN], errors='coerce')
                df.dropna(subset=[SALARY_COLUMN], inplace=True)
                if df.empty:
                    st.error(f"Error: No valid data found in '{SALARY_COLUMN}' after processing.")
                    return pd.DataFrame()
                return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}. Please ensure it's a valid CSV file.")
            return pd.DataFrame()
    return pd.DataFrame()
st.sidebar.markdown("---")
st.sidebar.markdown("## Data Upload")
uploaded_file_from_widget = st.sidebar.file_uploader("adult 3.csv", type="csv", key="csv_uploader")

if uploaded_file_from_widget is not None:
    st.session_state.uploaded_file_object = uploaded_file_from_widget
elif st.session_state.uploaded_file_object is None:
    pass

df_global = load_data(st.session_state.uploaded_file_object)


# --- Page Functions ---

def home_page():
    st.markdown('<h1 class="header-title">💼Employee Salary Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9em;'>Version 1.2 - Persistent CSV Upload</p>", unsafe_allow_html=True)


    st.markdown("""
    <div class="info-card">
        <h2>Overview</h2>
        <p>Welcome to the "Employee Salary Prediction App"! This interactive web application is designed to help you understand, analyze, and even predict employee salaries based on various demographic and employment factors. Whether you're an HR professional, a data analyst, or just curious, this tool provides insights into salary trends and influences.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Technology Stack Used</h2>
        <ul>
            <li><strong>Streamlit:</strong> For building interactive web applications purely in Python.</li>
            <li><strong>Pandas:</strong> For robust data manipulation and analysis.</li>
            <li><strong>Plotly & Matplotlib:</strong> For creating rich, interactive, and static data visualizations.</li>
            <li><strong>NumPy:</strong> For numerical operations and array manipulation.</li>
            <li><strong>Custom CSS:</strong> For enhanced visual appeal and a responsive user interface.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Key Features</h2>
        <ul>
            <li><strong>Multi-Page Navigation:</strong> Seamlessly switch between different sections of the application.</li>
            <li><strong>Data Exploration:</strong> Dive deep into salary distributions and departmental insights.</li>
            <li><strong>Interactive Prediction:</strong> Input employee details to get a predicted salary (illustrative).</li>
            <li><strong>Model Analytics:</strong> Understand the theoretical performance and interpretability of a typical prediction model (illustrative).</li>
            <li><strong>Comprehensive Feature Set:</strong> Utilize various employee attributes for analysis and prediction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Model Performance (Illustrative)</h2>
        <p>While this application uses a simplified prediction logic for demonstration, a real-world model would typically aim for high accuracy and robustness. Key metrics like R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) would be used to evaluate performance. Feature importance analysis would reveal which factors most influence salary.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Comprehensive Feature Set</h2>
        <p>The system leverages a rich set of features from the dataset, including age, workclass, education, marital status, occupation, relationship, race, and gender, to provide a holistic view of factors influencing salary (represented by hours-per-week in this dataset).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col = st.columns(5)[2]
    with col:
        if st.button("Explore the Salary Predictor"):
            st.session_state.current_page = "Salary Prediction"
            st.rerun()

def salary_prediction_page():
    st.markdown('<h1 class="header-title">💰 Salary Prediction</h1>', unsafe_allow_html=True)
    st.markdown("<h2>Enter Employee Information</h2>", unsafe_allow_html=True)

    if df_global.empty:
        st.warning("Please upload a CSV file on the sidebar to enable salary prediction.")
        return

    workclass_options = sorted(df_global.get('workclass', pd.Series([])).unique().tolist())
    education_options = sorted(df_global.get('education', pd.Series([])).unique().tolist())
    marital_status_options = sorted(df_global.get('marital-status', pd.Series([])).unique().tolist())
    occupation_options = sorted(df_global.get('occupation', pd.Series([])).unique().tolist())
    relationship_options = sorted(df_global.get('relationship', pd.Series([])).unique().tolist())
    race_options = sorted(df_global.get('race', pd.Series([])).unique().tolist())
    gender_options = sorted(df_global.get('gender', pd.Series([])).unique().tolist())
    native_country_options = sorted(df_global.get('native-country', pd.Series([])).unique().tolist())

    if not workclass_options: workclass_options = ['Unknown']
    if not education_options: education_options = ['Unknown']
    if not marital_status_options: marital_status_options = ['Unknown']
    if not occupation_options: occupation_options = ['Unknown']
    if not relationship_options: relationship_options = ['Unknown']
    if not race_options: race_options = ['Unknown']
    if not gender_options: gender_options = ['Unknown']
    if not native_country_options: native_country_options = ['Unknown']

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=17, max_value=90, value=35)
            workclass = st.selectbox("Workclass", workclass_options)
            education = st.selectbox("Education", education_options)
            education_num = st.slider("Years of Education (Educational-Num)", min_value=1, max_value=16, value=10)
            marital_status = st.selectbox("Marital Status", marital_status_options)

        with col2:
            occupation = st.selectbox("Occupation", occupation_options)
            relationship = st.selectbox("Relationship", relationship_options)
            race = st.selectbox("Race", race_options)
            gender = st.selectbox("Gender", gender_options)
            capital_gain = st.slider("Capital Gain", min_value=0, max_value=100000, value=0, step=1000)
            capital_loss = st.slider("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
            native_country = st.selectbox("Native Country", native_country_options)


        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        predicted_salary = 40.0
        if age > 40:
            predicted_salary += 5
        if education_num > 12:
            predicted_salary += 10
        if workclass == 'Private':
            predicted_salary += 2
        elif workclass == 'Self-emp-not-inc':
            predicted_salary += 5
        if occupation in ['Prof-specialty', 'Exec-managerial']:
            predicted_salary += 15
        elif occupation in ['Sales', 'Tech-support']:
            predicted_salary += 8
        if gender == 'Female':
            predicted_salary -= 3
        if capital_gain > 0:
            predicted_salary += 5
        if capital_loss > 0:
            predicted_salary -= 2

        predicted_salary = max(10.0, predicted_salary)
        predicted_salary = min(99.0, predicted_salary)

        st.markdown(f"""
        <div class="info-card" style="background-color: #e6ffe6; border-color: #00cc00;">
            <h3 style="color: #008000;">Predicted Salary:</h3>
            <p style="font-size: 2.5em; font-weight: bold; color: #008000;">{predicted_salary:,.2f} hours/week</p>
            <p style="font-size: 0.9em; color: #008000;">(Note: This is an illustrative prediction based on simplified rules. A real model would be more complex and accurate.)</p>
        </div>
        """, unsafe_allow_html=True)

def data_exploration_page():
    st.markdown('<h1 class="header-title">📊 Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown("<h2>Salary Analysis</h2>", unsafe_allow_html=True)

    if df_global.empty:
        st.warning("Please upload a CSV file on the sidebar to explore data.")
        return

    st.markdown("<h3>Salary Distribution by Department (Workclass)</h3>", unsafe_allow_html=True)

    # Matplotlib Plot
    st.write("#### Matplotlib: Salary Distribution (Hours/Week) by Workclass (Department)")
    fig_mpl, ax_mpl = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=DEPARTMENT_COLUMN, y=SALARY_COLUMN, data=df_global, ax=ax_mpl, palette='viridis')
    ax_mpl.set_title(f'Salary ({SALARY_COLUMN.replace("-", " ").title()}) Distribution by {DEPARTMENT_COLUMN.replace("-", " ").title()}', fontsize=16)
    ax_mpl.set_xlabel(DEPARTMENT_COLUMN.replace("-", " ").title(), fontsize=12)
    ax_mpl.set_ylabel(SALARY_COLUMN.replace("-", " ").title(), fontsize=12)
    ax_mpl.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_mpl)
    plt.close(fig_mpl)

    st.markdown("---")
    st.write("#### Plotly: Interactive Salary Distribution (Hours/Week) by Workclass (Department)")
    fig_plotly = px.box(df_global, x=DEPARTMENT_COLUMN, y=SALARY_COLUMN,
                        title=f'Interactive Salary ({SALARY_COLUMN.replace("-", " ").title()}) Distribution by {DEPARTMENT_COLUMN.replace("-", " ").title()}',
                        labels={SALARY_COLUMN: f'{SALARY_COLUMN.replace("-", " ").title()} (Hours/Week)',
                                DEPARTMENT_COLUMN: DEPARTMENT_COLUMN.replace("-", " ").title()},
                        color=DEPARTMENT_COLUMN,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_plotly.update_layout(xaxis_title_text=DEPARTMENT_COLUMN.replace("-", " ").title(),
                            yaxis_title_text=f'{SALARY_COLUMN.replace("-", " ").title()} (Hours/Week)',
                            hovermode="x unified",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=12, color="#333"),
                            title_font_size=20,
                            margin=dict(l=0, r=0, t=40, b=0))
    fig_plotly.update_yaxes(gridcolor='#e0e0e0', showgrid=True)
    st.plotly_chart(fig_plotly, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>Overall Salary Statistics</h3>", unsafe_allow_html=True)
    if not df_global.empty:
        col1, col2, col3, col4 = st.columns(4)
        min_salary = df_global[SALARY_COLUMN].min()
        max_salary = df_global[SALARY_COLUMN].max()
        avg_salary = df_global[SALARY_COLUMN].mean()
        median_salary = df_global[SALARY_COLUMN].median()

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Min {SALARY_COLUMN.replace('-', ' ').title()}</div>
                <div class="metric-value">{min_salary:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max {SALARY_COLUMN.replace('-', ' ').title()}</div>
                <div class="metric-value">{max_salary:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg {SALARY_COLUMN.replace('-', ' ').title()}</div>
                <div class="metric-value">{avg_salary:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Median {SALARY_COLUMN.replace('-', ' ').title()}</div>
                <div class="metric-value">{median_salary:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

def model_analytics_page():
    st.markdown('<h1 class="header-title">📈 Model Analytics</h1>', unsafe_allow_html=True)
    st.markdown("<h2>Model Performance Analytics</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Trained Model Metrics (Illustrative)</h3>
        <p>In a real-world scenario, after training a machine learning model for salary prediction, we would evaluate its performance using various metrics. Here are some common ones with illustrative values:</p>
        <ul>
            <li><strong>R-squared ($R^2$):</strong> Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. A higher $R^2$ (closer to 1) indicates a better fit. <br> <em>Illustrative Value:</em> <strong>0.85</strong></li>
            <li><strong>Mean Absolute Error (MAE):</strong> The average of the absolute differences between predictions and actual values. It gives a clear idea of the prediction error in the original units. <br> <em>Illustrative Value:</em> <strong>5.2 hours/week</strong></li>
            <li><strong>Root Mean Squared Error (RMSE):</strong> The square root of the average of the squared differences between predictions and actual values. It penalizes larger errors more. <br> <em>Illustrative Value:</em> <strong>6.8 hours/week</strong></li>
        </ul>
        <p>These metrics help in understanding how well the model generalizes to unseen data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Cross Validation (Illustrative)</h3>
        <p>Cross-validation is a technique to evaluate how well a model generalizes to an independent dataset. It involves splitting the data into multiple folds, training the model on a subset of folds, and testing on the remaining fold. This process is repeated multiple times.</p>
        <p><em>Example: 5-Fold Cross-Validation R-squared Scores:</em></p>
        <ul>
            <li>Fold 1: 0.83</li>
            <li>Fold 2: 0.86</li>
            <li>Fold 3: 0.84</li>
            <li>Fold 4: 0.87</li>
            <li>Fold 5: 0.85</li>
        </ul>
        <p>Consistent scores across folds indicate a robust model that is not overly dependent on a specific data split.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Feature Importance (Illustrative)</h3>
        <p>Feature importance helps us understand which input variables have the most significant impact on the predicted salary. This is crucial for interpretability and identifying key drivers.</p>
        <p><em>Illustrative Top Features:</em></p>
        <ul>
            <li><strong>Educational-Num:</strong> High importance</li>
            <li><strong>Occupation:</strong> High importance</li>
            <li><strong>Age:</strong> Medium importance</li>
            <li><strong>Capital-Gain:</strong> Medium importance</li>
            <li><strong>Workclass:</strong> Medium importance</li>
            <li><strong>Marital-Status:</strong> Low importance</li>
        </ul>
        <p>Visualizations like bar charts are typically used to show feature importance, with longer bars indicating higher importance.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Prediction Distribution (Illustrative)</h3>", unsafe_allow_html=True)
    np.random.seed(42)
    actual_salaries = np.random.normal(loc=40, scale=10, size=200)
    predicted_salaries = actual_salaries * 0.9 + np.random.normal(loc=5, scale=3, size=200)
    predicted_salaries = np.clip(predicted_salaries, 10, 90)
    prediction_df = pd.DataFrame({'Actual Salary (Hours/Week)': actual_salaries, 'Predicted Salary (Hours/Week)': predicted_salaries})

    fig_pred_dist = px.scatter(prediction_df, x='Actual Salary (Hours/Week)', y='Predicted Salary (Hours/Week)',
                               title='Actual vs. Predicted Salary (Illustrative)',
                               labels={'Actual Salary (Hours/Week)': 'Actual Salary (Hours/Week)',
                                       'Predicted Salary (Hours/Week)': 'Predicted Salary (Hours/Week)'},
                               opacity=0.6, trendline="ols")
    fig_pred_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", size=12, color="#333"))
    fig_pred_dist.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                                selector=dict(mode='markers'))
    st.plotly_chart(fig_pred_dist, use_container_width=True)

    st.markdown("<h3>Residual Plot (Illustrative)</h3>", unsafe_allow_html=True)
    residuals = predicted_salaries - actual_salaries
    residual_df = pd.DataFrame({'Predicted Salary (Hours/Week)': predicted_salaries, 'Residuals': residuals})

    fig_res_plot = px.scatter(residual_df, x='Predicted Salary (Hours/Week)', y='Residuals',
                              title='Residual Plot (Illustrative)',
                              labels={'Predicted Salary (Hours/Week)': 'Predicted Salary (Hours/Week)', 'Residuals': 'Prediction Error'},
                              opacity=0.6)
    fig_res_plot.add_hline(y=0, line_dash="dash", line_color="red")
    fig_res_plot.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", size=12, color="#333"))
    st.plotly_chart(fig_res_plot, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <h3>Model Comparison (Illustrative)</h3>
        <p>In practice, multiple models (e.g., Linear Regression, Random Forest, Gradient Boosting) would be trained and compared to find the best performer. Comparison is often done using metrics like R-squared, MAE, and RMSE on a validation set.</p>
        <p><em>Example Comparison:</em></p>
        <ul>
            <li><strong>Linear Regression:</strong> R2=0.75, MAE=7.0</li>
            <li><strong>Random Forest:</strong> R2=0.85, MAE=5.2</li>
            <li><strong>Gradient Boosting:</strong> R2=0.88, MAE=4.8</li>
        </ul>
        <p>This helps in selecting the most suitable model for deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>SHAP Explainability (Feature Impact) (Illustrative)</h3>
        <p>SHAP (SHapley Additive exPlanations) values help explain the output of any machine learning model. They show how much each feature contributes to the prediction for a specific instance, or globally how features impact the model's output on average.</p>
        <p><em>Illustrative SHAP Summary:</em></p>
        <ul>
            <li>Positive SHAP for 'Educational-Num' means higher education tends to increase predicted salary.</li>
            <li>Negative SHAP for 'Capital-Loss' means higher capital loss tends to decrease predicted salary.</li>
            <li>The magnitude of SHAP values indicates the strength of impact.</li>
        </ul>
        <p>This provides transparency into the "black box" of complex models.</p>
    </div>
    """, unsafe_allow_html=True)

def about_page():
    st.markdown('<h1 class="header-title">ℹ️ About</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Purpose and Vision</h2>
        <p>The "Employee Salary Prediction System" was developed as an interactive demonstration of data analysis, visualization, and the principles behind machine learning for salary prediction. Our vision is to provide a user-friendly platform where individuals can explore the factors influencing salaries and understand how predictive models can offer insights into compensation trends.</p>
        <p>While the prediction model used here is simplified for illustrative purposes, it showcases the potential of data-driven approaches in human resources and economic analysis. We aim to make complex data science concepts accessible and engaging.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Dataset Used</h2>
        <p>This application utilizes a modified version of the "Adult Income Dataset" (from the UCI Machine Learning Repository). This dataset contains various demographic and employment-related attributes, with 'workclass' serving as a proxy for department and 'hours-per-week' used as a numerical representation for salary in this demonstration.</p>
        <p><strong>Key features in the dataset include:</strong> Age, Workclass, Education, Educational-Num, Marital-Status, Occupation, Relationship, Race, Gender, Capital-Gain, Capital-Loss, Hours-per-Week, and Native-Country.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h2>Development Team & Contact</h2>
        <p>This project was developed by an AI assistant to demonstrate capabilities in building interactive data applications.</p>
        <p>For any inquiries or feedback, please feel free to reach out. While this is a demo, we value your thoughts on how such tools can be improved or applied in real-world scenarios.</p>
        <p><strong>Technologies:</strong> Streamlit, Pandas, Plotly, Matplotlib, NumPy, Custom CSS.</p>
    </div>
    """, unsafe_allow_html=True)

def main_app():
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Salary Prediction":
        salary_prediction_page()
    elif st.session_state.current_page == "Data Exploration":
        data_exploration_page()
    elif st.session_state.current_page == "Model Analytics":
        model_analytics_page()
    elif st.session_state.current_page == "About":
        about_page()

if __name__ == '__main__':
    main_app()
