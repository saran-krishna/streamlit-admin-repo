import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import openai
from typing import Dict, List
from dotenv import load_dotenv
import sys
import os

# Add the parent directory to the path to import shared_config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from shared_config import TEAM_LIST

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Multi-Company Employee Wellness Analytics Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .risk-high { border-left-color: #e53e3e !important; }
    .risk-medium { border-left-color: #dd6b20 !important; }
    .risk-low { border-left-color: #38a169 !important; }
    
    /* Clean selectbox styling - white background with dark text */
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Dropdown menu items */
    .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Ensure all content has good contrast */
    .stMarkdown, .stMarkdown div, .stMarkdown p, .stMarkdown span {
        color: #262730 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #38a169 !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

class AdminDashboard:
    def __init__(self):
        # API Configuration - Railway Backend
        self.api_base_url = os.getenv("RAILWAY_API_URL", "https://web-production-fa83.up.railway.app")
        self.admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        
        # Initialize OpenAI client with API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            st.warning("⚠️ OpenAI API key not configured. Report generation will not work.")
        
    def authenticate(self):
        """Company selection and password authentication"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'selected_company' not in st.session_state:
            st.session_state.selected_company = None
        
        if not st.session_state.authenticated:
            st.markdown('<div class="main-header"><h1>🏢 Multi-Company Admin Portal</h1></div>', unsafe_allow_html=True)
            
            # Company selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Step 1: Select Company")
                company_name = st.text_input(
                    "Company Name:", 
                    placeholder="Enter company name (e.g., 'Apple', 'Google', 'Microsoft')",
                    help="This will create company-specific database isolation"
                )
                
                if company_name:
                    # Validate company name format
                    import re
                    if re.match(r'^[a-zA-Z0-9\s\-_]+$', company_name) and 2 <= len(company_name) <= 50:
                        st.success(f"✅ Company: {company_name}")
                        st.session_state.selected_company = company_name
                    else:
                        st.error("❌ Invalid company name. Use only letters, numbers, spaces, hyphens, and underscores (2-50 characters)")
            
            with col2:
                st.subheader("Step 2: Admin Authentication")
                password = st.text_input("Admin Password:", type="password")
                
                if st.button("Login to Dashboard", type="primary"):
                    if not st.session_state.selected_company:
                        st.error("Please enter a valid company name first!")
                    elif password == self.admin_password:
                        st.session_state.authenticated = True
                        st.success(f"Authentication successful for {st.session_state.selected_company}!")
                        st.rerun()
                    else:
                        st.error("Invalid password!")
            
            # Display info about multi-company setup
            st.markdown("---")
            st.info("""
            **🔒 Multi-Company Setup:**
            - Each company gets its own isolated database
            - Company data is completely separated
            - URL format: `yoursite.com/COMPANY_NAME/chat/token`
            - Generate tokens specific to your company
            """)
            
            return False
        return True
    
    def logout(self):
        """Logout function"""
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.selected_company = None
            st.rerun()
    
    def get_analytics_data(self, team=None, days=30):
        """Fetch analytics data from API"""
        try:
            params = {
                'company_name': st.session_state.selected_company,
                'days': days
            }
            if team and team != 'All Teams':
                params['team'] = team
            
            response = requests.get(f"{self.api_base_url}/api/admin/analytics", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            return None
    
    def generate_token(self, email, name, team):
        """Generate new token for employee"""
        try:
            data = {
                "company_name": st.session_state.selected_company,
                "employee_email": email,
                "employee_name": name,
                "team": team
            }
            
            response = requests.post(f"{self.api_base_url}/api/admin/generate-token", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to generate token: {response.status_code}")
                return None
            response = requests.post(f"{self.api_base_url}/api/admin/generate-token", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to generate token: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error generating token: {e}")
            return None

    def get_tokens(self, limit=100):
        """Get all tokens from API"""
        try:
            params = {
                "company_name": st.session_state.selected_company,
                "limit": limit
            }
            response = requests.get(f"{self.api_base_url}/api/admin/tokens", params=params)
            if response.status_code == 200:
                return response.json().get("tokens", [])
            else:
                st.error(f"Failed to fetch tokens: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Error fetching tokens: {e}")
            return []
    
    def create_sample_data(self, team_filter=None):
        """Create sample data for demonstration when API is not available"""
        teams = TEAM_LIST.copy()
        if team_filter and team_filter != 'All Teams':
            teams = [team_filter]
        
        sample_data = []
        for i in range(50):
            team = np.random.choice(teams)
            sample_data.append({
                'team': team,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'sentiment_score': {
                    'overall_sentiment': np.random.choice(['positive', 'neutral', 'negative'], p=[0.3, 0.5, 0.2]),
                    'vader_compound': np.random.uniform(-1, 1)
                },
                'stress_level': {
                    'stress_level': np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2]),
                    'stress_score': np.random.uniform(0, 1)
                },
                'retention_risk': {
                    'retention_risk_level': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1]),
                    'retention_risk_score': np.random.uniform(0, 1)
                },
                'wellness_indicators': {
                    'wellness_status': np.random.choice(['good', 'neutral', 'needs_attention'], p=[0.4, 0.4, 0.2]),
                    'wellness_score': np.random.uniform(-1, 1)
                },
                'conversation_length': np.random.randint(10, 100),
                'key_topics': np.random.choice([
                    ['work_pressure'], ['team_dynamics'], ['career_growth'], 
                    ['work_life_balance'], ['management_issues'], ['compensation']
                ])
            })
        
        return {
            'analytics': sample_data,
            'token_stats': {
                'total_tokens': 100,
                'used_tokens': 45,
                'expired_tokens': 15,
                'active_tokens': 40
            },
            'teams': teams
        }
    
    def render_token_management(self):
        """Render token generation interface"""
        st.markdown(f'<div class="main-header"><h2>🎫 Token Management - {st.session_state.selected_company}</h2></div>', unsafe_allow_html=True)
        
        # Get Vercel frontend URL from environment
        frontend_url = os.getenv("VERCEL_FRONTEND_URL", "https://employee-wellness-frontend-i8su.vercel.app")
        company_slug = st.session_state.selected_company.lower().replace(' ', '-').replace('_', '-')
        
        # Display company info
        st.info(f"🏢 **Company:** {st.session_state.selected_company} | 🔗 **Chat URL Format:** `{frontend_url}/{company_slug}/chat/{{token}}`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generate New Token")
            with st.form("generate_token"):
                email = st.text_input("Employee Email", placeholder="employee@company.com")
                name = st.text_input("Employee Name", placeholder="John Doe")
                team = st.selectbox("Team", TEAM_LIST + ['Other'])
                
                if st.form_submit_button("Generate Token", type="primary"):
                    if email and name and team:
                        result = self.generate_token(email, name, team)
                        if result:
                            st.success("Token generated successfully!")
                            st.code(f"Chat URL: {result['chat_url']}")
                            st.info(f"Token expires in {result['expires_in_days']} days")
                    else:
                        st.error("Please fill in all fields")
        
        with col2:
            st.subheader("Bulk Token Generation")
            st.info("Upload a CSV file with columns: email, name, team")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if all(col in df.columns for col in ['email', 'name', 'team']):
                        if st.button("Generate Tokens for All"):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for idx, row in df.iterrows():
                                result = self.generate_token(row['email'], row['name'], row['team'])
                                if result:
                                    results.append({
                                        'email': row['email'],
                                        'name': row['name'],
                                        'team': row['team'],
                                        'chat_url': result['chat_url'],
                                        'token': result['token']
                                    })
                                progress_bar.progress((idx + 1) / len(df))
                            
                            if results:
                                results_df = pd.DataFrame(results)
                                st.success(f"Generated {len(results)} tokens successfully!")
                                st.dataframe(results_df[['email', 'name', 'team', 'chat_url']])
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "Download Results",
                                    csv,
                                    "generated_tokens.csv",
                                    "text/csv"
                                )
                    else:
                        st.error("CSV must contain columns: email, name, team")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        # Token List Section
        st.markdown("---")
        st.subheader("📋 Existing Tokens")
        
        # Fetch and display tokens
        tokens = self.get_tokens(limit=50)
        
        if tokens:
            # Create filters
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.selectbox("Filter by Status", 
                                           ["All", "active", "used", "expired"])
            with col2:
                team_filter = st.selectbox("Filter by Team", 
                                         ["All"] + list(set([token['team'] for token in tokens])))
            with col3:
                limit = st.number_input("Limit Results", min_value=10, max_value=200, value=50)
            
            # Apply filters
            filtered_tokens = tokens
            if status_filter != "All":
                filtered_tokens = [t for t in filtered_tokens if t['status'] == status_filter]
            if team_filter != "All":
                filtered_tokens = [t for t in filtered_tokens if t['team'] == team_filter]
            
            filtered_tokens = filtered_tokens[:limit]
            
            # Display tokens in a table
            if filtered_tokens:
                token_df = pd.DataFrame(filtered_tokens)
                
                # Format dates
                for date_col in ['created_at', 'expires_at', 'used_at']:
                    if date_col in token_df.columns:
                        token_df[date_col] = pd.to_datetime(token_df[date_col]).dt.strftime('%Y-%m-%d %H:%M')
                
                # Select columns to display
                display_cols = ['employee_name', 'employee_email', 'team', 'status', 'created_at', 'expires_at']
                if 'used_at' in token_df.columns:
                    display_cols.append('used_at')
                
                st.dataframe(
                    token_df[display_cols],
                    use_container_width=True,
                    column_config={
                        "status": st.column_config.TextColumn(
                            "Status",
                            help="Token status",
                            width="small"
                        ),
                        "created_at": st.column_config.TextColumn(
                            "Created",
                            width="medium"
                        ),
                        "expires_at": st.column_config.TextColumn(
                            "Expires",
                            width="medium"
                        )
                    }
                )
                
                # Token statistics
                st.markdown("### Token Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tokens", len(tokens))
                with col2:
                    active_count = len([t for t in tokens if t['status'] == 'active'])
                    st.metric("Active Tokens", active_count)
                with col3:
                    used_count = len([t for t in tokens if t['status'] == 'used'])
                    st.metric("Used Tokens", used_count)
                with col4:
                    expired_count = len([t for t in tokens if t['status'] == 'expired'])
                    st.metric("Expired Tokens", expired_count)
            else:
                st.info("No tokens match the selected filters.")
        else:
            st.info("No tokens found. Generate some tokens to see them here.")
    
    def render_overview_metrics(self, data):
        """Render overview metrics cards"""
        if not data or 'analytics' not in data:
            return
        
        analytics = data['analytics']
        token_stats = data.get('token_stats', {})
        
        # Calculate key metrics
        total_sessions = len(analytics)
        
        if total_sessions > 0:
            # Stress levels - with error handling
            stress_levels = []
            for item in analytics:
                try:
                    stress_levels.append(item['stress_level']['level'])
                except (KeyError, TypeError):
                    stress_levels.append('unknown')
            high_stress = stress_levels.count('high')
            medium_stress = stress_levels.count('medium')
            elevated_stress = high_stress + medium_stress  # Total concerning stress
            
            # Retention risk - with error handling
            retention_risks = []
            for item in analytics:
                try:
                    retention_risks.append(item['retention_risk']['risk'])
                except (KeyError, TypeError):
                    retention_risks.append('unknown')
            high_risk = retention_risks.count('high')
            medium_risk = retention_risks.count('medium')
            elevated_risk = high_risk + medium_risk  # Total concerning risk
            
            # Sentiment - with error handling
            sentiments = []
            for item in analytics:
                try:
                    sentiments.append(item['sentiment_score']['emotional_state'])
                except (KeyError, TypeError):
                    sentiments.append('unknown')
            negative_sentiment = sentiments.count('negative') + sentiments.count('concerning')
            neutral_sentiment = sentiments.count('neutral')
            
            # Wellness - with error handling
            wellness_mental_health = []
            for item in analytics:
                try:
                    wellness_mental_health.append(item['wellness_indicators']['mental_health'])
                except (KeyError, TypeError):
                    wellness_mental_health.append('unknown')
            needs_attention = wellness_mental_health.count('concerning') + wellness_mental_health.count('poor')
            fair_wellness = wellness_mental_health.count('fair')
        else:
            elevated_stress = elevated_risk = negative_sentiment = needs_attention = 0
            neutral_sentiment = fair_wellness = 0
        
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="Total Sessions",
                value=total_sessions,
                delta=f"+{total_sessions - token_stats.get('used_tokens', 0)} new"
            )
        
        with col2:
            st.metric(
                label="Active Tokens",
                value=token_stats.get('active_tokens', 0),
                delta=f"{token_stats.get('expired_tokens', 0)} expired"
            )
        
        with col3:
            stress_pct = (elevated_stress / max(total_sessions, 1)) * 100
            st.metric(
                label="Elevated Stress",
                value=f"{elevated_stress} ({stress_pct:.1f}%)",
                delta=f"{'🔴' if stress_pct > 70 else '🟡' if stress_pct > 40 else '🟢'} {high_stress} high"
            )
        
        with col4:
            risk_pct = (elevated_risk / max(total_sessions, 1)) * 100
            st.metric(
                label="Flight Risk",
                value=f"{elevated_risk} ({risk_pct:.1f}%)",
                delta=f"{'🔴' if risk_pct > 60 else '🟡' if risk_pct > 30 else '🟢'} {high_risk} high"
            )
        
        with col5:
            # Show neutral + negative sentiment as a more realistic metric
            concerning_sentiment = negative_sentiment + neutral_sentiment
            sent_pct = (concerning_sentiment / max(total_sessions, 1)) * 100
            st.metric(
                label="Needs Attention",
                value=f"{concerning_sentiment} ({sent_pct:.1f}%)",
                delta=f"{'🔴' if negative_sentiment > 5 else '🟡' if neutral_sentiment > total_sessions*0.5 else '🟢'} sentiment"
            )
        
        with col6:
            # Include fair + poor wellness as needing attention
            wellness_concern = needs_attention + fair_wellness
            wellness_pct = (wellness_concern / max(total_sessions, 1)) * 100
            st.metric(
                label="Wellness Alert",
                value=f"{wellness_concern} ({wellness_pct:.1f}%)",
                delta=f"{'🔴' if needs_attention > 2 else '🟡' if fair_wellness > total_sessions*0.3 else '🟢'} health"
            )
        
        with col6:
            neg_pct = (negative_sentiment / max(total_sessions, 1)) * 100
            st.metric(
                label="Negative Sentiment",
                value=f"{negative_sentiment} ({neg_pct:.1f}%)",
                delta=f"{'🔴' if neg_pct > 40 else '🟡' if neg_pct > 25 else '🟢'}"
            )
        
        with col6:
            attention_pct = (needs_attention / max(total_sessions, 1)) * 100
            st.metric(
                label="Needs Attention",
                value=f"{needs_attention} ({attention_pct:.1f}%)",
                delta=f"{'🔴' if attention_pct > 30 else '🟡' if attention_pct > 15 else '🟢'}"
            )
    
    def render_team_analysis(self, data):
        """Render team-level analysis charts"""
        if not data or 'analytics' not in data:
            return
        
        analytics = data['analytics']
        if not analytics:
            st.warning("No data available for selected filters")
            return
        
        # Prepare data for analysis
        df = pd.DataFrame(analytics)
        
        # Team-wise metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Stress Levels by Team")
            
            # Safe extraction of stress levels with error handling
            def safe_extract_stress_levels(group):
                levels = []
                for item in group:
                    try:
                        levels.append(item['level'])
                    except (KeyError, TypeError):
                        levels.append('unknown')
                return pd.Series(levels).value_counts()
            
            team_stress = df.groupby('team')['stress_level'].apply(safe_extract_stress_levels).unstack(fill_value=0)
            
            # Ensure all expected columns exist
            expected_stress_cols = ['low', 'medium', 'high', 'unknown']
            for col in expected_stress_cols:
                if col not in team_stress.columns:
                    team_stress[col] = 0
            
            # Only plot columns that exist in the data
            available_cols = [col for col in expected_stress_cols if col in team_stress.columns and team_stress[col].sum() > 0]
            
            if available_cols and len(team_stress) > 0:
                fig = px.bar(
                    team_stress[available_cols].reset_index(),
                    x='team',
                    y=available_cols,
                    title="Stress Distribution by Team",
                    color_discrete_map={'low': '#38a169', 'medium': '#dd6b20', 'high': '#e53e3e', 'unknown': '#718096'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stress level data available for visualization")
        
        with col2:
            st.subheader("⚠️ Retention Risk by Team")
            
            # Safe extraction of retention risk with error handling
            def safe_extract_retention_risk(group):
                risks = []
                for item in group:
                    try:
                        risks.append(item['risk'])
                    except (KeyError, TypeError):
                        risks.append('unknown')
                return pd.Series(risks).value_counts()
            
            team_risk = df.groupby('team')['retention_risk'].apply(safe_extract_retention_risk).unstack(fill_value=0)
            
            # Ensure all expected columns exist
            expected_risk_cols = ['low', 'medium', 'high', 'unknown']
            for col in expected_risk_cols:
                if col not in team_risk.columns:
                    team_risk[col] = 0
            
            # Only plot columns that exist in the data
            available_risk_cols = [col for col in expected_risk_cols if col in team_risk.columns and team_risk[col].sum() > 0]
            
            if available_risk_cols and len(team_risk) > 0:
                fig = px.bar(
                    team_risk[available_risk_cols].reset_index(),
                    x='team',
                    y=available_risk_cols,
                    title="Retention Risk Distribution by Team",
                    color_discrete_map={'low': '#38a169', 'medium': '#dd6b20', 'high': '#e53e3e', 'unknown': '#718096'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No retention risk data available for visualization")
        
        # Sentiment timeline - with error handling
        st.subheader("📈 Sentiment Trends Over Time")
        
        try:
            df['date'] = pd.to_datetime([item['timestamp'] for item in analytics])
            # Safe extraction of sentiment scores
            sentiment_scores = []
            for item in analytics:
                try:
                    sentiment_scores.append(item['sentiment_score']['vader_compound'])
                except (KeyError, TypeError):
                    sentiment_scores.append(0.0)
            df['sentiment_score'] = sentiment_scores
            
            daily_sentiment = df.groupby([df['date'].dt.date, 'team'])['sentiment_score'].mean().reset_index()
            
            if len(daily_sentiment) > 0:
                fig = px.line(
                    daily_sentiment,
                    x='date',
                    y='sentiment_score',
                    color='team',
                    title="Daily Average Sentiment by Team",
                    labels={'sentiment_score': 'Sentiment Score (-1 to 1)', 'date': 'Date'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available for timeline visualization")
        except Exception as e:
            st.error(f"Error creating sentiment timeline: {str(e)}")
    
    def render_detailed_insights(self, data):
        """Render detailed insights and predictions"""
        if not data or 'analytics' not in data:
            return
        
        analytics = data['analytics']
        if not analytics:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Topics Analysis")
            
            # Extract and count topics
            all_topics = []
            for item in analytics:
                all_topics.extend(item.get('key_topics', []))
            
            topic_counts = pd.Series(all_topics).value_counts()
            
            if not topic_counts.empty:
                fig = px.pie(
                    values=topic_counts.values,
                    names=topic_counts.index,
                    title="Most Discussed Topics"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No topic data available")
        
        with col2:
            st.subheader("🔮 Predictive Risk Analysis")
            
            # Calculate risk scores by team - with error handling
            team_risk_scores = {}
            for item in analytics:
                team = item.get('team', 'Unknown')
                try:
                    # Use confidence as a proxy for risk score (0-1 scale)
                    risk_score = item.get('retention_risk', {}).get('confidence', 0.5)
                    stress_score = item.get('stress_level', {}).get('confidence', 0.5)
                    # Calculate wellness score from mental health status
                    mental_health = item.get('wellness_indicators', {}).get('mental_health', 'fair')
                    wellness_score = {'good': 0.8, 'fair': 0.5, 'concerning': 0.3, 'poor': 0.1}.get(mental_health, 0.5)
                    
                    # Combined risk metric
                    combined_risk = (risk_score * 0.4 + stress_score * 0.3 + (1 - wellness_score) * 0.3)
                    
                    if team not in team_risk_scores:
                        team_risk_scores[team] = []
                    team_risk_scores[team].append(combined_risk)
                except (KeyError, TypeError, AttributeError):
                    # Skip items with incomplete data
                    continue
            
            # Average risk by team
            avg_risk = {team: np.mean(scores) for team, scores in team_risk_scores.items() if scores}
            
            risk_df = pd.DataFrame(list(avg_risk.items()), columns=['Team', 'Risk Score'])
            risk_df['Risk Level'] = pd.cut(
                risk_df['Risk Score'],
                bins=[-1, 0.3, 0.6, 1],
                labels=['Low', 'Medium', 'High']
            )
            
            fig = px.bar(
                risk_df,
                x='Team',
                y='Risk Score',
                color='Risk Level',
                title="Team Risk Assessment",
                color_discrete_map={'Low': '#38a169', 'Medium': '#dd6b20', 'High': '#e53e3e'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed team table
        st.subheader("📋 Detailed Team Metrics")
        
        team_summary = []
        teams = list(set(item.get('team', 'Unknown') for item in analytics))
        
        for team in teams:
            team_data = [item for item in analytics if item.get('team') == team]
            if team_data:
                # Count high stress with error handling
                stress_high = sum(1 for item in team_data 
                                if item.get('stress_level', {}).get('level') == 'high')
                
                # Count high risk with error handling
                risk_high = sum(1 for item in team_data 
                              if item.get('retention_risk', {}).get('risk') == 'high')
                
                # Calculate average sentiment with error handling
                sentiment_scores = []
                for item in team_data:
                    try:
                        sentiment_scores.append(item['sentiment_score']['vader_compound'])
                    except (KeyError, TypeError):
                        sentiment_scores.append(0.0)
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                team_summary.append({
                    'Team': team,
                    'Total Sessions': len(team_data),
                    'High Stress Count': stress_high,
                    'High Risk Count': risk_high,
                    'Avg Sentiment': round(avg_sentiment, 3),
                    'Risk Level': 'High' if (stress_high + risk_high) / len(team_data) > 0.3 else 'Medium' if (stress_high + risk_high) / len(team_data) > 0.15 else 'Low'
                })
        
        if team_summary:
            summary_df = pd.DataFrame(team_summary)
            st.dataframe(summary_df, use_container_width=True)
    
    def generate_team_report(self, analytics_data: List[Dict], team_filter: str = None) -> str:
        """Generate comprehensive team wellness report using GPT analysis"""
        
        try:
            # Check if OpenAI client is available
            if not self.openai_client:
                return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file to enable report generation."
            
            # Filter data by team if specified
            if team_filter:
                filtered_data = [item for item in analytics_data if item.get('team') == team_filter]
                report_title = f"{team_filter} Team Wellness Report"
            else:
                filtered_data = analytics_data
                report_title = "Overall Wellness Report"
            
            if not filtered_data:
                return f"No data available for {team_filter if team_filter else 'the selected timeframe'}."
            
            # Prepare data summary for GPT
            total_sessions = len(filtered_data)
            
            # Aggregate metrics
            stress_levels = []
            retention_risks = []
            sentiments = []
            wellness_indicators = []
            key_topics = []
            
            for item in filtered_data:
                try:
                    stress_levels.append(item['stress_level']['level'])
                    retention_risks.append(item['retention_risk']['risk'])
                    sentiments.append(item['sentiment_score']['emotional_state'])
                    wellness_indicators.append(item['wellness_indicators']['mental_health'])
                    key_topics.extend(item.get('key_topics', []))
                except (KeyError, TypeError):
                    continue
            
            # Calculate percentages
            stress_high = stress_levels.count('high') / max(len(stress_levels), 1) * 100
            stress_medium = stress_levels.count('medium') / max(len(stress_levels), 1) * 100
            risk_high = retention_risks.count('high') / max(len(retention_risks), 1) * 100
            risk_medium = retention_risks.count('medium') / max(len(retention_risks), 1) * 100
            sentiment_negative = (sentiments.count('negative') + sentiments.count('concerning')) / max(len(sentiments), 1) * 100
            wellness_poor = (wellness_indicators.count('concerning') + wellness_indicators.count('poor')) / max(len(wellness_indicators), 1) * 100
            
            # Top concerns
            topic_counts = {}
            for topic in key_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create prompt for GPT
            prompt = f"""
            As a workplace wellness expert, analyze the following employee wellness data and provide a comprehensive report with actionable insights.

            **Data Summary:**
            - Total Sessions: {total_sessions}
            - High Stress: {stress_high:.1f}%
            - Medium Stress: {stress_medium:.1f}%
            - High Retention Risk: {risk_high:.1f}%
            - Medium Retention Risk: {risk_medium:.1f}%
            - Negative Sentiment: {sentiment_negative:.1f}%
            - Wellness Concerns: {wellness_poor:.1f}%
            - Top Issues: {', '.join([f"{topic} ({count} mentions)" for topic, count in top_topics])}

            Please provide a detailed report covering:
            1. **Executive Summary** - Overall team wellness status
            2. **Key Findings** - Major pain points and concerns
            3. **Risk Assessment** - Areas of immediate concern
            4. **Recommendations** - Specific actionable steps
            5. **Support Strategies** - How to improve team wellness

            Focus on practical insights and specific recommendations that management can implement.
            """
            
            # Generate report using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a workplace wellness expert and data analyst. Provide comprehensive, actionable insights based on employee wellness data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            report_content = response.choices[0].message.content
            
            # Format the final report
            final_report = f"""
# {report_title}
**Generated on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Data Period:** Last 30 days
**Sessions Analyzed:** {total_sessions}

{report_content}

---
*This report was generated using AI analysis of anonymous employee wellness conversations. All data is aggregated at the team level to protect individual privacy.*
            """
            
            return final_report
            
        except Exception as e:
            return f"Error generating report: {str(e)}. Please ensure OpenAI API key is configured."

    def render_report_generation_tab(self, data, selected_team):
        """Render dedicated AI report generation tab"""
        st.markdown("""
        ### 🤖 AI-Powered Wellness Reports
        Generate comprehensive team wellness reports using advanced AI analysis of employee conversations and analytics data.
        """)
        
        if not data or 'analytics' not in data or not data['analytics']:
            st.warning("⚠️ No analytics data available for report generation. Please ensure employees have used the wellness chatbot.")
            return
        
        analytics = data['analytics']
        teams = list(set(item.get('team', 'Unknown') for item in analytics))
        total_sessions = len(analytics)
        
        # Report configuration section
        st.subheader("📝 Report Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Team selection for report
            report_team_options = ['All Teams'] + teams
            report_team = st.selectbox(
                "Select Team for Report",
                report_team_options,
                index=0 if selected_team == 'All Teams' else report_team_options.index(selected_team) if selected_team in report_team_options else 0,
                help="Choose which team to generate the report for"
            )
        
        with col2:
            # Report type selection
            report_type = st.selectbox(
                "Report Type",
                ["Comprehensive Analysis", "Executive Summary", "Risk Assessment", "Action Plan"],
                help="Select the type of report to generate"
            )
        
        with col3:
            # Time range for report
            report_days = st.selectbox(
                "Data Time Range",
                [7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"Last {x} days",
                help="Select the time range for data analysis"
            )
        
        # Data preview section
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Filter data based on selections
        filtered_analytics = analytics
        if report_team != 'All Teams':
            filtered_analytics = [item for item in analytics if item.get('team') == report_team]
        
        if filtered_analytics:
            preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
            
            with preview_col1:
                st.metric("Sessions", len(filtered_analytics))
            
            with preview_col2:
                # Count high stress sessions
                high_stress = sum(1 for item in filtered_analytics 
                                if item.get('stress_level', {}).get('level') == 'high')
                st.metric("High Stress", high_stress)
            
            with preview_col3:
                # Count high risk sessions
                high_risk = sum(1 for item in filtered_analytics 
                              if item.get('retention_risk', {}).get('risk') == 'high')
                st.metric("High Risk", high_risk)
            
            with preview_col4:
                # Count negative sentiment
                negative_sentiment = sum(1 for item in filtered_analytics 
                                       if item.get('sentiment_score', {}).get('emotional_state') in ['negative', 'concerning'])
                st.metric("Negative Sentiment", negative_sentiment)
        else:
            st.warning(f"No data available for {report_team}")
            return
        
        # Report generation section
        st.markdown("---")
        st.subheader("🎯 Generate Report")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"""
            **Report will include:**
            - Executive summary of team wellness status
            - Key findings and pain points
            - Risk assessment and early warning indicators
            - Specific actionable recommendations
            - Support strategies for improvement
            
            **Data Analysis:** {len(filtered_analytics)} sessions from {report_team}
            """)
        
        with col2:
            if st.button("🚀 Generate Report", type="primary", use_container_width=True):
                with st.spinner(f"🤖 AI is analyzing {len(filtered_analytics)} sessions and generating your {report_type.lower()}..."):
                    try:
                        # Customize the report based on type
                        team_filter = report_team if report_team != 'All Teams' else None
                        
                        if report_type == "Executive Summary":
                            # Generate a shorter, executive-focused report
                            report = self.generate_executive_summary(filtered_analytics, team_filter)
                        elif report_type == "Risk Assessment":
                            # Generate a risk-focused report
                            report = self.generate_risk_assessment_report(filtered_analytics, team_filter)
                        elif report_type == "Action Plan":
                            # Generate an action-focused report
                            report = self.generate_action_plan_report(filtered_analytics, team_filter)
                        else:
                            # Default comprehensive report
                            report = self.generate_team_report(filtered_analytics, team_filter)
                        
                        # Display the report
                        st.markdown("---")
                        st.subheader(f"📄 {report_type} - {report_team}")
                        
                        # Report container with better styling
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; color: #262730; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; line-height: 1.6;">
                            {report.replace('#', '###').replace('**', '<strong>').replace('**', '</strong>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("📥 Download Options")
                        
                        download_col1, download_col2, download_col3 = st.columns(3)
                        
                        with download_col1:
                            st.download_button(
                                label="📄 Download as Text",
                                data=report,
                                file_name=f"{report_type.lower().replace(' ', '_')}_{report_team.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with download_col2:
                            # Convert to markdown format
                            markdown_report = report
                            st.download_button(
                                label="📝 Download as Markdown",
                                data=markdown_report,
                                file_name=f"{report_type.lower().replace(' ', '_')}_{report_team.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        
                        with download_col3:
                            # Share button (placeholder)
                            if st.button("📧 Email Report", use_container_width=True):
                                st.info("Email sharing feature coming soon!")
                        
                        # Success message
                        st.success(f"✅ {report_type} generated successfully! The report analyzed {len(filtered_analytics)} employee sessions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
        
        # Historical reports section (placeholder)
        st.markdown("---")
        st.subheader("📚 Report History")
        st.info("Report history and comparison features coming soon!")

    def generate_executive_summary(self, analytics_data: List[Dict], team_filter: str = None) -> str:
        """Generate an executive summary report"""
        # Use the same logic as generate_team_report but with a different prompt
        return self.generate_custom_report(analytics_data, team_filter, "executive_summary")
    
    def generate_risk_assessment_report(self, analytics_data: List[Dict], team_filter: str = None) -> str:
        """Generate a risk assessment focused report"""
        return self.generate_custom_report(analytics_data, team_filter, "risk_assessment")
    
    def generate_action_plan_report(self, analytics_data: List[Dict], team_filter: str = None) -> str:
        """Generate an action plan focused report"""
        return self.generate_custom_report(analytics_data, team_filter, "action_plan")
    
    def generate_custom_report(self, analytics_data: List[Dict], team_filter: str = None, report_type: str = "comprehensive") -> str:
        """Generate custom report based on type"""
        
        try:
            # Check if OpenAI client is available
            if not self.openai_client:
                return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file to enable report generation."
            
            # Filter data by team if specified
            if team_filter:
                filtered_data = [item for item in analytics_data if item.get('team') == team_filter]
                report_title = f"{team_filter} Team"
            else:
                filtered_data = analytics_data
                report_title = "Overall Team"
            
            if not filtered_data:
                return f"No data available for {team_filter if team_filter else 'the selected timeframe'}."
            
            # Prepare data summary for GPT
            total_sessions = len(filtered_data)
            
            # Aggregate metrics
            stress_levels = []
            retention_risks = []
            sentiments = []
            wellness_indicators = []
            key_topics = []
            
            for item in filtered_data:
                try:
                    stress_levels.append(item['stress_level']['level'])
                    retention_risks.append(item['retention_risk']['risk'])
                    sentiments.append(item['sentiment_score']['emotional_state'])
                    wellness_indicators.append(item['wellness_indicators']['mental_health'])
                    key_topics.extend(item.get('key_topics', []))
                except (KeyError, TypeError):
                    continue
            
            # Calculate percentages
            stress_high = stress_levels.count('high') / max(len(stress_levels), 1) * 100
            stress_medium = stress_levels.count('medium') / max(len(stress_levels), 1) * 100
            risk_high = retention_risks.count('high') / max(len(retention_risks), 1) * 100
            risk_medium = retention_risks.count('medium') / max(len(retention_risks), 1) * 100
            sentiment_negative = (sentiments.count('negative') + sentiments.count('concerning')) / max(len(sentiments), 1) * 100
            wellness_poor = (wellness_indicators.count('concerning') + wellness_indicators.count('poor')) / max(len(wellness_indicators), 1) * 100
            
            # Top concerns
            topic_counts = {}
            for topic in key_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create different prompts based on report type
            if report_type == "executive_summary":
                prompt = f"""
                As a workplace wellness executive consultant, provide a concise EXECUTIVE SUMMARY for leadership based on this employee wellness data:

                **Data Summary:**
                - Total Sessions: {total_sessions}
                - High Stress: {stress_high:.1f}%
                - High Retention Risk: {risk_high:.1f}%
                - Negative Sentiment: {sentiment_negative:.1f}%
                - Top Issues: {', '.join([f"{topic} ({count} mentions)" for topic, count in top_topics[:3]])}

                Provide a brief executive summary (max 400 words) covering:
                1. **Overall Status** - One sentence wellness assessment
                2. **Key Metrics** - 3-4 critical numbers leaders need to know
                3. **Immediate Actions** - Top 2-3 priorities for leadership
                4. **Business Impact** - Potential effects on productivity/retention
                """
            
            elif report_type == "risk_assessment":
                prompt = f"""
                As a workplace risk analyst, provide a detailed RISK ASSESSMENT based on this employee wellness data:

                **Data Summary:**
                - Total Sessions: {total_sessions}
                - High Stress: {stress_high:.1f}%
                - Medium Stress: {stress_medium:.1f}%
                - High Retention Risk: {risk_high:.1f}%
                - Medium Retention Risk: {risk_medium:.1f}%
                - Negative Sentiment: {sentiment_negative:.1f}%
                - Wellness Concerns: {wellness_poor:.1f}%
                - Top Issues: {', '.join([f"{topic} ({count} mentions)" for topic, count in top_topics])}

                Provide a detailed risk assessment covering:
                1. **Risk Matrix** - Categorize risks by severity and likelihood
                2. **Early Warning Indicators** - Signs of escalating problems
                3. **Retention Risk Analysis** - Flight risk assessment
                4. **Stress Impact Assessment** - Potential burnout indicators
                5. **Mitigation Strategies** - Specific risk reduction actions
                """
            
            elif report_type == "action_plan":
                prompt = f"""
                As a workplace wellness strategist, create a detailed ACTION PLAN based on this employee wellness data:

                **Data Summary:**
                - Total Sessions: {total_sessions}
                - High Stress: {stress_high:.1f}%
                - High Retention Risk: {risk_high:.1f}%
                - Negative Sentiment: {sentiment_negative:.1f}%
                - Top Issues: {', '.join([f"{topic} ({count} mentions)" for topic, count in top_topics])}

                Provide a comprehensive action plan with:
                1. **Immediate Actions (0-30 days)** - Urgent interventions needed
                2. **Short-term Initiatives (1-3 months)** - Programs to implement
                3. **Long-term Strategy (3-12 months)** - Systematic improvements
                4. **Resource Requirements** - Budget, personnel, tools needed
                5. **Success Metrics** - How to measure improvement
                6. **Timeline & Milestones** - Specific deadlines and checkpoints
                """
            
            else:  # comprehensive
                prompt = f"""
                As a workplace wellness expert, analyze the following employee wellness data and provide a comprehensive report with actionable insights.

                **Data Summary:**
                - Total Sessions: {total_sessions}
                - High Stress: {stress_high:.1f}%
                - Medium Stress: {stress_medium:.1f}%
                - High Retention Risk: {risk_high:.1f}%
                - Medium Retention Risk: {risk_medium:.1f}%
                - Negative Sentiment: {sentiment_negative:.1f}%
                - Wellness Concerns: {wellness_poor:.1f}%
                - Top Issues: {', '.join([f"{topic} ({count} mentions)" for topic, count in top_topics])}

                Please provide a detailed report covering:
                1. **Executive Summary** - Overall team wellness status
                2. **Key Findings** - Major pain points and concerns
                3. **Risk Assessment** - Areas of immediate concern
                4. **Recommendations** - Specific actionable steps
                5. **Support Strategies** - How to improve team wellness

                Focus on practical insights and specific recommendations that management can implement.
                """
            
            # Generate report using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a workplace wellness expert and data analyst. Provide comprehensive, actionable insights based on employee wellness data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            report_content = response.choices[0].message.content
            
            # Format the final report
            report_type_title = {
                "executive_summary": "Executive Summary",
                "risk_assessment": "Risk Assessment Report", 
                "action_plan": "Action Plan",
                "comprehensive": "Comprehensive Wellness Report"
            }.get(report_type, "Wellness Report")
            
            final_report = f"""
# {report_title} - {report_type_title}
**Generated on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Data Period:** Last 30 days
**Sessions Analyzed:** {total_sessions}

{report_content}

---
*This report was generated using AI analysis of anonymous employee wellness conversations. All data is aggregated at the team level to protect individual privacy.*
            """
            
            return final_report
            
        except Exception as e:
            return f"Error generating report: {str(e)}. Please ensure OpenAI API key is configured."

    def run(self):
        """Main dashboard function"""
        # Authentication first
        if not self.authenticate():
            return
        
        # Company-specific header
        st.markdown(f'<div class="main-header"><h1>📊 Employee Wellness Analytics - {st.session_state.selected_company}</h1></div>', unsafe_allow_html=True)
        
        # Display company info
        st.success(f"🏢 **Active Company:** {st.session_state.selected_company} | 🔒 **Isolated Database:** ✅ Secure")
        
        # Logout button in sidebar
        self.logout()
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters & Controls")
        st.sidebar.markdown(f"**Company:** {st.session_state.selected_company}")
        st.sidebar.markdown("---")
        
        # Time range filter
        days = st.sidebar.selectbox(
            "Time Range",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
        
        # Get data
        data = self.get_analytics_data(days=days)
        
        # Use sample data if API is not available
        if not data:
            st.sidebar.warning("API not available. Using sample data.")
            data = self.create_sample_data()
        
        # Team filter
        teams = ['All Teams'] + data.get('teams', [])
        selected_team = st.sidebar.selectbox("Team Filter", teams)
        
        # Apply team filter to data
        if selected_team != 'All Teams':
            filtered_analytics = [item for item in data['analytics'] if item['team'] == selected_team]
            data['analytics'] = filtered_analytics
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Team Analysis", "🔮 Insights & Predictions", "📋 AI Reports", "🎫 Token Management"])
        
        with tab1:
            st.header("Overview Metrics")
            self.render_overview_metrics(data)
            
            # Quick actions
            st.markdown("---")
            st.subheader("Quick Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📧 Send Wellness Survey", type="primary"):
                    st.success("Wellness survey deployment feature coming soon!")
            
            with col2:
                if st.button("📈 Quick Report", help="Generate a quick AI-powered wellness report"):
                    with st.spinner("Generating comprehensive team wellness report using AI analysis..."):
                        team_filter = selected_team if selected_team != 'All Teams' else None
                        report = self.generate_team_report(data.get('analytics', []), team_filter)
                        
                        st.subheader("🔍 AI-Powered Team Wellness Report")
                        
                        # Report container with better styling for readability
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; color: #262730; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; line-height: 1.6;">
                            {report.replace('#', '###').replace('**', '<strong>').replace('**', '</strong>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Option to download the report
                        st.download_button(
                            label="📄 Download Report as Text",
                            data=report,
                            file_name=f"team_wellness_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
            
            with col3:
                if st.button("⚠️ Alert High Risk Teams"):
                    st.success("Alert system feature coming soon!")
        
        with tab2:
            st.header("Team-Level Analysis")
            self.render_team_analysis(data)
        
        with tab3:
            st.header("Insights & Predictions")
            self.render_detailed_insights(data)
        
        with tab4:
            st.header("AI-Powered Team Wellness Reports")
            self.render_report_generation_tab(data, selected_team)
        
        with tab5:
            self.render_token_management()

# Run the dashboard
if __name__ == "__main__":
    dashboard = AdminDashboard()
    dashboard.run()
