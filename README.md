# Employee Wellness Analytics Dashboard

> **🔐 SECURE ADMIN INTERFACE** - No sensitive data included in this repository

Anonymous employee wellness analytics dashboard built with Streamlit for multi-company workplace wellness monitoring.

## 🌐 Live Dashboard
**Streamlit Cloud URL**: [To be updated after deployment]

## 🏗️ Architecture
- **Admin Dashboard**: This Streamlit Cloud deployment
- **Backend API**: Separate Railway deployment
- **Employee Frontend**: Separate Vercel deployment
- **Database**: Firebase Firestore (credentials in Streamlit secrets)

## 📊 Features
- **Multi-Company Support**: Switch between different companies
- **Token Management**: Generate secure access tokens for employees
- **Real-time Analytics**: Stress levels, sentiment analysis, retention risks
- **Team Insights**: Anonymous aggregated data by teams
- **Predictive Analytics**: AI-powered workplace wellness predictions

## 🔧 Deployment
This repository is configured for Streamlit Cloud deployment:

1. **Repository**: Public GitHub repo (no secrets included)
2. **Configuration**: Environment variables via Streamlit secrets
3. **Dependencies**: Listed in `requirements.txt`
4. **Main File**: `dashboard.py`

## 🔐 Security
- **No API Keys**: All credentials stored in Streamlit Cloud secrets
- **No Database Credentials**: Firebase config via environment variables
- **Anonymous Data Only**: No personal employee information stored
- **Token-based Access**: Secure employee authentication system

## 📋 Required Secrets (Streamlit Cloud)
```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
PROJECT_ID = "your-firebase-project-id"
RAILWAY_API_URL = "your-railway-backend-url"
VERCEL_FRONTEND_URL = "your-vercel-frontend-url"
FIREBASE_CREDENTIALS = '''{"type":"service_account",...}'''
```

## 🚀 Quick Start
1. Deploy to Streamlit Cloud
2. Configure secrets in Streamlit dashboard
3. Access admin interface
4. Generate employee tokens
5. Monitor wellness analytics

## 📈 Usage
1. **Select Company**: Choose from configured companies
2. **Generate Tokens**: Create secure access links for employees
3. **Monitor Analytics**: View real-time wellness insights
4. **Team Management**: Track team-level wellness trends
5. **Export Reports**: Download wellness analytics reports

## 🔗 Integration
- **Employee Access**: Tokens generate links to frontend chat interface
- **Data Flow**: Analytics from chat sessions → API → Dashboard
- **Real-time Updates**: Live monitoring of employee wellness metrics

## 🛠️ Development
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

**Note**: This is a public repository for deployment purposes. All sensitive configuration is handled through Streamlit Cloud secrets management.
