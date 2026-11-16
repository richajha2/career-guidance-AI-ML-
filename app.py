
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FutureCareer",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# CUSTOM UI DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.stApp h1 {
    color: #172554 !important;
    font-weight: 800 !important;
}

.stApp h2 {
    color: #243B65 !important;
    font-weight: 750 !important;
}

.stApp h3 {
    color: #243B65 !important;
    font-weight: 700 !important;
}

.stApp p {
    color: #334155 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] {
    background-color: #172554 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #172554 !important;
    color: #172554 !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 800 !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li * {
    background-color: #FFFFFF !important;
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
    font-weight: 700 !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #E0E7FF !important;
}

.stApp label {
    font-weight: 700 !important;
    color: #243B65 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #4F46E5, #6366F1);
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 750 !important;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #3730A3 !important;
    color: #FFFFFF !important;
    border: none;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    padding: 12px;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #172554 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    font-weight: 600 !important;
}

div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"] {
    background-color: #FFFFFF;
    border-radius: 10px;
}

div[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 12px;
}

div[data-testid="stCaptionContainer"] {
    font-weight: 700 !important;
}

hr {
    border-color: #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD COMPANY TECHNOLOGY DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "DATASETS_company_domains_final.xlsx",
        sheet_name="Company_Domain_Tech"
    )

    df["Company"] = (
        df["Company"]
        .astype("string")
        .str.strip()
    )

    df["Technology Domain"] = (
        df["Technology Domain"]
        .astype("string")
        .str.split(",")
    )

    df = df.explode("Technology Domain")

    df["Technology Domain"] = (
        df["Technology Domain"]
        .astype("string")
        .str.strip()
    )

    df = df.dropna(
        subset=["Company", "Technology Domain"]
    )

    df = df[
        (df["Company"] != "") &
        (df["Technology Domain"] != "") &
        (df["Technology Domain"].str.lower() != "nan")
    ]

    df = df.drop_duplicates(
        subset=["Company", "Technology Domain"]
    )

    return df


# ==========================================
# LOAD LSTM PREDICTIONS
# ==========================================

@st.cache_data
def load_predictions():

    prediction_df = pd.read_csv(
        "prediction_2025_2029.csv"
    )

    if "Year" in prediction_df.columns:

        prediction_df = prediction_df.set_index("Year")

    elif prediction_df.columns[0].lower().startswith("unnamed"):

        prediction_df = prediction_df.set_index(
            prediction_df.columns[0]
        )

    else:

        prediction_df = prediction_df.set_index(
            prediction_df.columns[0]
        )

    prediction_df.index.name = "Year"
    prediction_df.index = prediction_df.index.astype(str)

    prediction_df.columns = (
        prediction_df.columns.astype(str).str.strip()
    )

    prediction_df = prediction_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    prediction_df = prediction_df.fillna(0)

    return prediction_df


# ==========================================
# LOAD COLLEGE-WISE DATA
# ==========================================

college_sheets = [
    "DTU",
    "IGDTUW",
    "MAIT",
    "BPIT",
    "IIIT_Delhi"
]


@st.cache_data
def load_college_data():

    college_domains = {}

    for college in college_sheets:

        college_df = pd.read_excel(
            "DATASETS_company_domains_final.xlsx",
            sheet_name=college
        )

        if "Technology Domain" not in college_df.columns:

            raise ValueError(
                f"Technology Domain column missing in {college}"
            )

        domains = (
            college_df["Technology Domain"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )

        domains = domains[
            (domains != "") &
            (domains.str.lower() != "nan")
        ]

        college_domains[college] = domains.tolist()

    return college_domains


# ==========================================
# LOAD DATA SAFELY
# ==========================================

try:

    df = load_data()
    data_loaded = True

except Exception as e:

    df = pd.DataFrame()
    data_loaded = False

    st.error(f"Dataset loading error: {e}")


# ==========================================
# MAIN HEADER
# ==========================================

st.title("🎓 FutureCareer")

st.subheader(
    "AI/ML-Based Technology Demand Analysis and Prediction"
)

st.write(
    "Discover relevant companies and technologies "
    "based on your academic background and career interests."
)

st.divider()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🚀 FutureCareer")

st.sidebar.caption(
    "AI/ML-Based Technology Demand Platform"
)

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.selectbox(
    "Select a Section",
    [
        "Home",
        "Technology Demand Analysis",
        "Five-Year Prediction",
        "College-Wise Comparison",
        "Career Guidance",
        "About Project"
    ]
)


# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.header("Welcome to FutureCareer")

    st.write(
        """
        Explore technology domains, discover relevant companies
        and understand future technology demand.

        Select Career Guidance from the navigation menu
        to find companies according to your interests.
        """
    )

    st.info(
        "Start exploring your career opportunities!"
    )

    st.divider()

    st.subheader("Explore Project Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader("📊")

            st.markdown("**Technology Analysis**")

            st.caption(
                "Explore technology demand across five colleges."
            )

    with col2:

        with st.container(border=True):

            st.subheader("🔮")

            st.markdown("**Five-Year Prediction**")

            st.caption(
                "Explore LSTM-based technology forecasts for 2025–2029."
            )

    with col3:

        with st.container(border=True):

            st.subheader("🎯")

            st.markdown("**Career Guidance**")

            st.caption(
                "Discover companies based on your technology interests."
            )

    st.divider()

    st.subheader("Dataset Coverage")

    college_cols = st.columns(5)

    college_names = [
        "DTU",
        "IGDTUW",
        "MAIT",
        "BPIT",
        "IIIT Delhi"
    ]

    for col, college in zip(college_cols, college_names):

        with col:

            with st.container(border=True):

                st.markdown(f"**{college}**")

    st.info(
        "The platform uses placement-related company and "
        "technology data collected from five colleges."
    )


# ==========================================
# TECHNOLOGY DEMAND ANALYSIS
# ==========================================

elif page == "Technology Demand Analysis":

    st.header("Technology Demand Analysis")

    st.write(
        """
        This section focuses on technology demand
        identified from placement-related data
        collected across five colleges.
        """
    )

    st.info(
        "Detailed technology demand analysis and visualizations "
        "are presented in the research analysis and model."
    )


# ==========================================
# FIVE-YEAR PREDICTION
# ==========================================

elif page == "Five-Year Prediction":

    st.header("Five-Year Technology Demand Prediction")

    st.write(
        """
        This section presents the five-year technology demand
        forecast generated using the LSTM model.
        """
    )

    try:

        prediction_df = load_predictions()

        if prediction_df.empty or len(prediction_df.columns) == 0:

            st.warning("No prediction data is available.")

        else:

            st.success(
                "LSTM predictions loaded successfully!"
            )

            st.subheader(
                "Technology Demand Forecast (2025–2029)"
            )

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

            st.divider()

            st.subheader(
                "Explore Technology Predictions"
            )

            selected_technology = st.selectbox(
                "Select Technology Domain",
                prediction_df.columns.tolist()
            )

            graph_df = pd.DataFrame({

                "Year": prediction_df.index.astype(str),

                "Predicted Demand": prediction_df[
                    selected_technology
                ].values

            })

            st.subheader(
                f"{selected_technology} - Future Demand"
            )

            fig = px.line(
                graph_df,
                x="Year",
                y="Predicted Demand",
                markers=True,
                title=f"{selected_technology} - Future Demand"
            )

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Predicted Demand",
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(
                    color="#243B65"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                "Note: These values represent exploratory "
                "LSTM-based technology demand forecasts "
                "for 2025–2029."
            )

    except Exception as e:

        st.error(
            f"Prediction loading error: {e}"
        )

        st.warning(
            "Please ensure prediction_2025_2029.csv "
            "is present in the project folder."
        )


# ==========================================
# COLLEGE-WISE COMPARISON
# ==========================================

elif page == "College-Wise Comparison":

    st.header("College-Wise Technology Comparison")

    st.write(
        """
        Compare technology domains across five colleges
        using placement-related company data.
        """
    )

    try:

        college_domains = load_college_data()

        college_sets = {
            college: set(domains)
            for college, domains in college_domains.items()
        }

        # College-Wise Summary
        st.subheader(
            "College-Wise Technology Summary"
        )

        summary = []

        for college in college_sheets:

            domains = college_domains[college]

            other_domains = set.union(
                *[
                    college_sets[other]
                    for other in college_sheets
                    if other != college
                ]
            )

            unique_domains = (
                college_sets[college] - other_domains
            )

            summary.append({

                "College": college.replace("_", " "),

                "Total Technology Mentions": len(domains),

                "Distinct Technology Domains": len(
                    college_sets[college]
                ),

                "Unique Technology Domains": len(
                    unique_domains
                )

            })

        summary_df = pd.DataFrame(summary)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Common Technologies
        st.subheader(
            "Common Technologies Across All Colleges"
        )

        common_domains = set.intersection(
            *[
                college_sets[college]
                for college in college_sheets
            ]
        )

        st.metric(
            "Common Technology Domains",
            len(common_domains)
        )

        if common_domains:

            for domain in sorted(common_domains):
                st.write(f"• {domain}")

        else:

            st.info(
                "No common technology domains found."
            )

        st.divider()

        # College-Wise Unique Technologies
        st.subheader(
            "College-Wise Unique Technologies"
        )

        selected_college = st.selectbox(
            "Select College",
            college_sheets
        )

        other_domains = set.union(
            *[
                college_sets[college]
                for college in college_sheets
                if college != selected_college
            ]
        )

        unique_domains = (
            college_sets[selected_college] -
            other_domains
        )

        st.write(
            f"Unique technologies in "
            f"{selected_college.replace('_', ' ')}:"
        )

        st.metric(
            "Unique Technology Domains",
            len(unique_domains)
        )

        if unique_domains:

            for domain in sorted(unique_domains):
                st.write(f"• {domain}")

        else:

            st.info(
                "No unique technology domains found."
            )

        st.divider()

        # Technology Domain Comparison Chart
        st.subheader(
            "Technology Domain Comparison"
        )

        chart_df = summary_df[
            [
                "College",
                "Distinct Technology Domains"
            ]
        ]

        fig = px.bar(
            chart_df,
            x="College",
            y="Distinct Technology Domains",
            title="Distinct Technology Domains Across Colleges",
            text="Distinct Technology Domains"
        )

        fig.update_layout(
            xaxis_title="College",
            yaxis_title="Number of Technology Domains",
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#243B65"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Note: Technology mentions represent entries "
            "found in the respective college sheets. "
            "These counts do not represent actual placement numbers."
        )

    except Exception as e:

        st.error(
            f"College comparison error: {e}"
        )


# ==========================================
# CAREER GUIDANCE WITH LSTM PREDICTION
# ==========================================

elif page == "Career Guidance":

    st.header("🎯 Find Companies Based on Your Interest")

    st.write(
        "Select your academic branch and one or more areas "
        "of interest to view future technology demand "
        "and discover relevant companies."
    )

    st.divider()

    if data_loaded:

        branch_options = [
            "Information Technology (IT)",
            "Computer Science Engineering (CSE)"
        ]

        selected_branch = st.selectbox(
            "Select Your Branch",
            branch_options
        )

        domains = sorted(
            df["Technology Domain"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_interests = st.multiselect(
            "Select Your Areas of Interest",
            domains,
            placeholder="Choose one or more technology domains"
        )

        if st.button(
            "🔍 Find Relevant Companies",
            type="primary",
            use_container_width=True
        ):

            if not selected_interests:

                st.warning(
                    "Please select at least one area of interest."
                )

            else:

                st.success(
                    f"Selected Branch: {selected_branch}"
                )

                st.write(
                    "**Selected Interests:** "
                    + ", ".join(selected_interests)
                )

                # ------------------------------------------
                # TECHNOLOGY NAME NORMALIZATION
                # ------------------------------------------

                def normalize_technology_name(name):

                    return "".join(
                        character.lower()
                        for character in str(name)
                        if character.isalnum()
                    )

                # ------------------------------------------
                # TECHNOLOGY ALIASES
                # ------------------------------------------

                technology_aliases = {

                    # Software Development
                    "softwaredeveloper": "Software Development",
                    "softwareengineer": "Software Development",
                    "softwaredevelopment": "Software Development",
                    "softwaredevelopersde": "Software Development",
                    "sde": "Software Development",

                    # API Development
                    "apidevelopmentintegration": "API",
                    "apidevelopment": "API",
                    "apiintegration": "API",
                    "apis": "API",

                    # Data Science
                    "datascientist": "Data Science",
                    "datascience": "Data Science",

                    # Data Analytics
                    "dataanalyst": "Data Analyst",
                    "dataanalytics": "Data Analytics",

                    # Business Analytics
                    "businessanalyst": "Business Analyst",
                    "businessintelligence": "Business Intelligence",

                    # Artificial Intelligence
                    "artificialintelligence": "AI/ML",
                    "machinelearning": "AI/ML",
                    "aiml": "AI/ML",

                    # Generative AI
                    "generativeai": "Generative AI",

                    # Cloud
                    "cloudcomputing": "Cloud",
                    "cloudtechnology": "Cloud",

                    # Web Development
                    "webdeveloper": "Web Development",
                    "webdevelopment": "Web Development",

                    # Full Stack
                    "fullstackdeveloper": "Full Stack",
                    "fullstackdevelopment": "Full Stack",

                    # Mobile Development
                    "mobileappdevelopment": "Mobile Development",
                    "mobiledeveloper": "Mobile Development",

                    # Cybersecurity
                    "cybersecurity": "Cybersecurity",

                    # Other Technologies
                    "devops": "DevOps",
                    "blockchain": "Blockchain",
                    "uiux": "UI/UX",
                    "iot": "IoT",
                    "robotics": "Robotics",
                    "nlp": "NLP",
                    "computervision": "Computer Vision",
                    "bigdata": "Big Data",
                    "database": "Database",
                    "microservices": "Microservices",
                    "systemdesign": "System Design",
                    "qa": "QA/Testing",
                    "qatesting": "QA/Testing"
                }

                # ------------------------------------------
                # 1. LSTM PREDICTION
                # ------------------------------------------

                st.divider()

                st.subheader(
                    "🔮 LSTM-Based Future Technology Demand"
                )

                try:

                    prediction_df = load_predictions()

                    prediction_column_lookup = {
                        normalize_technology_name(column): column
                        for column in prediction_df.columns
                    }

                    matched_predictions = 0

                    for interest in selected_interests:

                        normalized_interest = (
                            normalize_technology_name(interest)
                        )

                        prediction_column = technology_aliases.get(
                            normalized_interest
                        )

                        if prediction_column is None:

                            prediction_column = (
                                prediction_column_lookup.get(
                                    normalized_interest
                                )
                            )

                        if (
                            prediction_column is None
                            or prediction_column not in prediction_df.columns
                        ):

                            st.warning(
                                f"LSTM forecast is not available for "
                                f"'{interest}' because no matching "
                                f"technology column was found in "
                                f"prediction_2025_2029.csv."
                            )

                            continue

                        matched_predictions += 1

                        forecast_values = prediction_df[
                            prediction_column
                        ]

                        forecast_chart = pd.DataFrame({

                            "Year": prediction_df.index.astype(str),

                            "Predicted Demand": forecast_values.values

                        })

                        with st.container(border=True):

                            st.markdown(
                                f"### 📈 {interest}"
                            )

                            st.caption(
                                f"Matched LSTM prediction column: "
                                f"{prediction_column}"
                            )

                            st.dataframe(
                                forecast_chart,
                                use_container_width=True,
                                hide_index=True
                            )

                            fig = px.line(
                                forecast_chart,
                                x="Year",
                                y="Predicted Demand",
                                markers=True,
                                title=(
                                    f"{interest} - "
                                    "LSTM Technology Demand Forecast"
                                )
                            )

                            fig.update_layout(
                                xaxis_title="Year",
                                yaxis_title="Predicted Demand",
                                template="plotly_white",
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                font=dict(
                                    color="#243B65"
                                )
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                    if matched_predictions == 0:

                        st.info(
                            "No selected interest had a matching "
                            "LSTM prediction column. Company "
                            "recommendations are still shown below "
                            "using the Excel technology dataset."
                        )

                    st.caption(
                        "Forecasts are exploratory LSTM model outputs "
                        "based on historical technology data. They do "
                        "not represent actual job vacancies or "
                        "guaranteed hiring."
                    )

                except Exception as e:

                    st.error(
                        f"Could not load LSTM predictions: {e}"
                    )

                # ------------------------------------------
                # 2. COMPANY RECOMMENDATIONS
                # ------------------------------------------

                st.divider()

                st.subheader(
                    "🏢 Recommended Companies"
                )

                result = df[
                    df["Technology Domain"].isin(
                        selected_interests
                    )
                ]

                companies = (
                    result[
                        ["Company", "Technology Domain"]
                    ]
                    .drop_duplicates()
                    .sort_values("Company")
                )

                if not companies.empty:

                    total_companies = companies[
                        "Company"
                    ].nunique()

                    st.success(
                        f"Found {total_companies} relevant companies!"
                    )

                    for company, group in companies.groupby(
                        "Company",
                        sort=True
                    ):

                        with st.container(border=True):

                            st.subheader(
                                f"🏢 {company}"
                            )

                            st.write(
                                "**Associated Technologies:**"
                            )

                            technologies = sorted(
                                group[
                                    "Technology Domain"
                                ].unique().tolist()
                            )

                            for technology in technologies:

                                st.write(
                                    f"• {technology}"
                                )

                else:

                    st.warning(
                        "No matching companies found for "
                        "your selected interests."
                    )

    else:

        st.error(
            "Company technology dataset could not be loaded."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About FutureCareer")

    st.write(
        """
        **AI/ML-Based Five-Year Technology Demand Analysis
        and Prediction for Career Guidance**

        This project analyzes technology domains and company
        requirements across five colleges.
        """
    )

    st.divider()

    st.subheader("Dataset Coverage")

    col1, col2, col3, col4, col5 = st.columns(5)

    colleges = [
        ("DTU", col1),
        ("IGDTUW", col2),
        ("MAIT", col3),
        ("BPIT", col4),
        ("IIIT Delhi", col5)
    ]

    for name, col in colleges:

        with col:

            with st.container(border=True):

                st.markdown(f"**{name}**")

    st.divider()

    st.subheader("Technologies Used")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.write("🐍 Python")
        st.write("🐼 Pandas and NumPy")

    with tech_col2:

        st.write("🤖 Machine Learning")
        st.write("🧠 LSTM")

    with tech_col3:

        st.write("🎨 Streamlit")
        st.write("📊 Plotly")

    st.divider()

    st.info(
        "This project combines historical technology-domain "
        "analysis, LSTM-based forecasting and interest-based "
        "company discovery to support student career exploration."
    )
