"""
Shared configuration for the Multi-Company Employee Wellness Analytics System
This file contains constants and configurations used across both backend and frontend
"""

# Team Configuration (Default teams - companies can customize via CSV)
TEAM_LIST = [
    "Design",
    "Engineering", 
    "Marketing",
    "Operations",
    "Sales"
]

# Analysis Configuration
STRESS_LEVELS = ["low", "medium", "high"]
RETENTION_RISK_LEVELS = ["low", "medium", "high"]
SENTIMENT_STATES = ["positive", "neutral", "negative"]

# Token Configuration
TOKEN_EXPIRY_DAYS = 7
TOKEN_LENGTH = 32

# API Configuration
API_BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"

# Multi-Company Configuration
COMPANY_URL_FORMAT = "{base_url}/{company_name}/chat/{token}"
SUPPORTED_COMPANY_NAME_PATTERN = r'^[a-zA-Z0-9\s\-_]+$'
MIN_COMPANY_NAME_LENGTH = 2
MAX_COMPANY_NAME_LENGTH = 50
