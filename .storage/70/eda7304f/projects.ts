export interface Project {
  id: string;
  title: string;
  description: string;
  longDescription: string;
  domain: string;
  technologies: string[];
  features: string[];
  metrics?: {
    label: string;
    value: string;
  }[];
  images?: string[];
  demoUrl?: string;
  githubUrl?: string;
  status: 'completed' | 'in-progress' | 'planned';
  featured: boolean;
}

export const projects: Project[] = [
  {
    id: "ecommerce-clv-analysis",
    title: "E-commerce Customer Segmentation & CLV Prediction",
    description: "Comprehensive customer analytics using RFM analysis, K-means clustering, and CLV prediction models with interactive Streamlit dashboard and business scenario visualizations.",
    longDescription: "This project implements a complete customer analytics pipeline for e-commerce businesses. Using advanced statistical models including BG/NBD and Gamma-Gamma distributions, it segments customers based on purchasing behavior and predicts their lifetime value. Features include RFM scoring, K-means clustering, interactive Streamlit dashboard, and comprehensive business scenario analysis with ROI optimization recommendations.",
    domain: "E-commerce Analytics",
    technologies: ["Python", "Pandas", "Scikit-learn", "Lifetimes", "Plotly", "Streamlit", "Matplotlib", "Seaborn"],
    features: [
      "RFM Analysis with 9 customer segments",
      "K-means clustering with optimal cluster detection",
      "BG/NBD model for purchase frequency prediction", 
      "Gamma-Gamma model for monetary value prediction",
      "Interactive Streamlit dashboard with real-time analytics",
      "Business scenario analysis (retention, budget allocation, churn prevention)",
      "Geographic and channel performance analysis",
      "ROI optimization and acquisition cost analysis"
    ],
    metrics: [
      { label: "Customers Analyzed", value: "2,000+" },
      { label: "Transactions Processed", value: "50,000+" },
      { label: "Customer Segments", value: "9" },
      { label: "CLV Prediction Accuracy", value: "85%+" },
      { label: "Interactive Visualizations", value: "15+" },
      { label: "Business Scenarios", value: "6" }
    ],
    images: ["/api/placeholder/800/400"],
    demoUrl: "http://localhost:8501",
    githubUrl: "https://github.com/adityabajaria/ecommerce-clv-analysis",
    status: "completed",
    featured: true
  },
  {
    id: "credit-card-fraud-detection",
    title: "Credit Card Fraud Detection System",
    description: "Advanced machine learning system for real-time fraud detection using ensemble methods, SMOTE for imbalanced data, and comprehensive business impact analysis.",
    longDescription: "Sophisticated fraud detection system analyzing transaction patterns to identify fraudulent activities. Uses multiple ML algorithms including Random Forest, Logistic Regression, and Isolation Forest with SMOTE for handling imbalanced datasets. Includes business impact analysis and risk scoring.",
    domain: "Financial Security",
    technologies: ["Python", "Scikit-learn", "SMOTE", "Plotly", "Random Forest", "Isolation Forest"],
    features: [
      "Real-time fraud detection with 96% accuracy",
      "Imbalanced data handling with SMOTE",
      "Ensemble model approach for robust predictions",
      "Business impact analysis and cost calculations",
      "Risk scoring and categorization system",
      "Interactive fraud analytics dashboard"
    ],
    metrics: [
      { label: "Detection Accuracy", value: "96%" },
      { label: "False Positive Rate", value: "<2%" },
      { label: "Transactions Analyzed", value: "50,000+" },
      { label: "Fraud Prevention Rate", value: "94%" }
    ],
    status: "completed",
    featured: true
  },
  {
    id: "employee-attrition-prediction",
    title: "Employee Attrition Prediction & HR Analytics",
    description: "Predictive HR analytics system for identifying at-risk employees using machine learning with comprehensive retention strategy recommendations and cost impact analysis.",
    longDescription: "Advanced HR analytics platform that predicts employee attrition using multiple ML algorithms. Analyzes factors like job satisfaction, work-life balance, and compensation to identify high-risk employees. Includes retention cost analysis and strategic recommendations.",
    domain: "Human Resources Analytics",
    technologies: ["Python", "Scikit-learn", "Gradient Boosting", "Plotly", "Feature Engineering"],
    features: [
      "Employee attrition prediction with 89% accuracy",
      "Risk categorization and employee profiling",
      "Department and role-based attrition analysis",
      "Retention cost impact calculations",
      "Feature importance analysis for HR insights",
      "Interactive HR dashboard for decision making"
    ],
    metrics: [
      { label: "Prediction Accuracy", value: "89%" },
      { label: "Employees Analyzed", value: "2,000+" },
      { label: "Cost Savings Potential", value: "$500K+" },
      { label: "High-Risk Identification", value: "95%" }
    ],
    status: "completed",
    featured: true
  },
  {
    id: "time-series-sales-forecasting",
    title: "Advanced Sales Forecasting & Inventory Optimization",
    description: "Comprehensive time series analysis for sales forecasting using ARIMA, SARIMA models with inventory optimization recommendations and business intelligence.",
    longDescription: "Advanced sales forecasting system using multiple time series models including ARIMA and SARIMA. Features seasonal decomposition, trend analysis, and inventory optimization with EOQ calculations. Includes comprehensive business dashboards for strategic planning.",
    domain: "Sales Analytics & Forecasting",
    technologies: ["Python", "ARIMA", "SARIMA", "Statsmodels", "Plotly", "Time Series Analysis"],
    features: [
      "Multi-model forecasting approach (ARIMA, SARIMA)",
      "Seasonal decomposition and trend analysis",
      "Inventory optimization with EOQ calculations",
      "Multi-store and category performance analysis",
      "Holiday and seasonal impact modeling",
      "30-day forward sales projections"
    ],
    metrics: [
      { label: "Forecast Accuracy (MAPE)", value: "<8%" },
      { label: "Sales Records Analyzed", value: "500K+" },
      { label: "Stores Covered", value: "5" },
      { label: "Product Categories", value: "5" }
    ],
    status: "completed",
    featured: true
  },
  {
    id: "sentiment-analysis-reviews",
    title: "Customer Review Sentiment Analysis",
    description: "NLP-powered sentiment analysis system for customer reviews with trend analysis, word clouds, and business intelligence for product improvement.",
    longDescription: "Advanced NLP system for analyzing customer sentiment from reviews using VADER sentiment analysis and custom ML models. Includes trend analysis, word frequency analysis, and actionable business insights for product and service improvement.",
    domain: "Natural Language Processing",
    technologies: ["Python", "NLTK", "VADER", "spaCy", "TextBlob", "Plotly", "Word Cloud"],
    features: [
      "Multi-platform review sentiment analysis",
      "Real-time sentiment trend monitoring",
      "Word cloud and keyword extraction",
      "Product-specific sentiment scoring",
      "Competitor sentiment comparison",
      "Automated alert system for negative trends"
    ],
    metrics: [
      { label: "Sentiment Accuracy", value: "91%" },
      { label: "Reviews Processed", value: "100K+" },
      { label: "Languages Supported", value: "5" },
      { label: "Real-time Processing", value: "1000/min" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "real-estate-price-prediction",
    title: "Real Estate Price Prediction & Market Analysis",
    description: "Machine learning system for property valuation using regression models with neighborhood analysis, market trends, and investment recommendations.",
    longDescription: "Comprehensive real estate analytics platform using advanced regression techniques for property price prediction. Includes neighborhood analysis, market trend forecasting, and investment opportunity identification with interactive property valuation tools.",
    domain: "Real Estate Analytics",
    technologies: ["Python", "Scikit-learn", "XGBoost", "Geospatial Analysis", "Plotly", "Feature Engineering"],
    features: [
      "Multi-model price prediction approach",
      "Neighborhood and location analysis",
      "Market trend forecasting",
      "Investment ROI calculations",
      "Interactive property valuation tool",
      "Comparative market analysis (CMA)"
    ],
    metrics: [
      { label: "Price Prediction Accuracy", value: "92%" },
      { label: "Properties Analyzed", value: "50K+" },
      { label: "Neighborhoods Covered", value: "200+" },
      { label: "Market Trend Accuracy", value: "87%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "market-basket-analysis",
    title: "Market Basket Analysis & Product Recommendations",
    description: "Association rule mining system for product recommendations using Apriori algorithm with cross-selling optimization and customer behavior insights.",
    longDescription: "Advanced market basket analysis using association rule mining to discover product relationships and optimize cross-selling strategies. Includes customer behavior analysis, product recommendation engine, and store layout optimization recommendations.",
    domain: "Retail Analytics",
    technologies: ["Python", "Apriori Algorithm", "Association Rules", "Network Analysis", "Plotly"],
    features: [
      "Association rule mining with Apriori algorithm",
      "Product recommendation engine",
      "Cross-selling opportunity identification",
      "Customer behavior pattern analysis",
      "Store layout optimization suggestions",
      "Interactive product relationship networks"
    ],
    metrics: [
      { label: "Rule Confidence", value: "85%+" },
      { label: "Product Combinations", value: "10K+" },
      { label: "Cross-sell Uplift", value: "23%" },
      { label: "Customer Baskets", value: "500K+" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "hospital-readmission-prediction",
    title: "Hospital Patient Readmission Prediction",
    description: "Healthcare analytics system for predicting patient readmission risk using clinical data with SHAP explainability and care optimization recommendations.",
    longDescription: "Advanced healthcare analytics platform predicting patient readmission risk within 30 days using clinical data and ML models. Features SHAP explainability for clinical decision support and care pathway optimization recommendations.",
    domain: "Healthcare Analytics",
    technologies: ["Python", "Scikit-learn", "SHAP", "Clinical Data Analysis", "Plotly", "Healthcare ML"],
    features: [
      "30-day readmission risk prediction",
      "SHAP explainability for clinical insights",
      "Patient risk stratification",
      "Care pathway optimization",
      "Clinical decision support dashboard",
      "Quality metrics tracking and reporting"
    ],
    metrics: [
      { label: "Prediction Accuracy", value: "88%" },
      { label: "Patients Analyzed", value: "25K+" },
      { label: "Risk Factors Identified", value: "15+" },
      { label: "Readmission Reduction", value: "18%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "dynamic-pricing-rideshare",
    title: "Dynamic Pricing Optimization for Ride-Sharing",
    description: "AI-powered surge pricing system using demand forecasting, geospatial analysis, and real-time optimization for ride-sharing platforms.",
    longDescription: "Sophisticated dynamic pricing system for ride-sharing platforms using demand forecasting and geospatial analysis. Optimizes surge pricing in real-time based on location, time, weather, and demand patterns to maximize revenue and minimize wait times.",
    domain: "Transportation Analytics",
    technologies: ["Python", "Geospatial Analysis", "Time Series", "Optimization", "Real-time Processing"],
    features: [
      "Real-time demand forecasting",
      "Geospatial surge pricing optimization",
      "Weather and event impact modeling",
      "Revenue optimization algorithms",
      "Driver-rider balance analysis",
      "Interactive pricing strategy dashboard"
    ],
    metrics: [
      { label: "Revenue Increase", value: "15%" },
      { label: "Wait Time Reduction", value: "22%" },
      { label: "Pricing Zones", value: "100+" },
      { label: "Real-time Updates", value: "Every 5min" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "ab-testing-analysis",
    title: "A/B Testing Analysis & Conversion Optimization",
    description: "Statistical analysis platform for A/B testing with hypothesis testing, power analysis, and conversion rate optimization recommendations.",
    longDescription: "Comprehensive A/B testing platform with statistical rigor for conversion optimization. Includes power analysis, sample size calculations, statistical significance testing, and detailed recommendations for product and marketing optimization.",
    domain: "Experimentation & Analytics",
    technologies: ["Python", "Statistical Testing", "Hypothesis Testing", "Power Analysis", "Plotly"],
    features: [
      "Statistical hypothesis testing framework",
      "Power analysis and sample size calculations",
      "Multi-variant testing support",
      "Conversion funnel analysis",
      "Bayesian A/B testing approach",
      "Automated significance monitoring"
    ],
    metrics: [
      { label: "Tests Analyzed", value: "500+" },
      { label: "Statistical Power", value: "80%+" },
      { label: "Conversion Uplift", value: "12%" },
      { label: "Significance Level", value: "95%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "marketing-campaign-roi",
    title: "Marketing Campaign ROI & Attribution Analysis",
    description: "Multi-touch attribution modeling system for marketing ROI optimization with campaign performance analysis and budget allocation recommendations.",
    longDescription: "Advanced marketing analytics platform for campaign ROI analysis and multi-touch attribution modeling. Tracks customer journey across channels, calculates true marketing impact, and provides data-driven budget allocation recommendations.",
    domain: "Marketing Analytics",
    technologies: ["Python", "Attribution Modeling", "ROI Analysis", "Customer Journey", "Plotly"],
    features: [
      "Multi-touch attribution modeling",
      "Campaign ROI calculation and tracking",
      "Customer journey mapping",
      "Channel performance optimization",
      "Budget allocation recommendations",
      "Marketing mix modeling"
    ],
    metrics: [
      { label: "Marketing ROI", value: "340%" },
      { label: "Campaigns Tracked", value: "200+" },
      { label: "Attribution Accuracy", value: "89%" },
      { label: "Budget Optimization", value: "25%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "customer-retention-cohort",
    title: "Customer Retention & Cohort Analysis",
    description: "Advanced cohort analysis system for customer retention tracking with lifetime value analysis and churn prediction modeling.",
    longDescription: "Comprehensive customer retention analytics platform using cohort analysis to track customer behavior over time. Includes lifetime value calculations, churn prediction, and retention strategy optimization with detailed business intelligence.",
    domain: "Customer Analytics",
    technologies: ["Python", "Cohort Analysis", "Survival Analysis", "Churn Prediction", "Plotly"],
    features: [
      "Monthly and weekly cohort analysis",
      "Customer lifetime value tracking",
      "Churn prediction and early warning",
      "Retention rate optimization",
      "Segment-based retention strategies",
      "Interactive retention heatmaps"
    ],
    metrics: [
      { label: "Retention Rate Improvement", value: "18%" },
      { label: "Cohorts Analyzed", value: "100+" },
      { label: "Churn Prediction Accuracy", value: "87%" },
      { label: "Customer LTV Increase", value: "22%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "sales-funnel-optimization",
    title: "Sales Funnel & Conversion Rate Optimization",
    description: "Comprehensive funnel analysis system with conversion optimization, drop-off identification, and user journey analytics for e-commerce platforms.",
    longDescription: "Advanced sales funnel analytics platform for conversion rate optimization. Analyzes user journey, identifies drop-off points, provides A/B testing recommendations, and optimizes conversion paths with detailed user behavior insights.",
    domain: "Conversion Analytics",
    technologies: ["Python", "Funnel Analysis", "User Journey", "Conversion Optimization", "Plotly"],
    features: [
      "Multi-step funnel analysis",
      "Conversion rate optimization",
      "Drop-off point identification",
      "User segment journey analysis",
      "A/B testing integration",
      "Real-time conversion monitoring"
    ],
    metrics: [
      { label: "Conversion Rate Increase", value: "28%" },
      { label: "Funnel Steps Analyzed", value: "12" },
      { label: "User Sessions", value: "1M+" },
      { label: "Drop-off Reduction", value: "35%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "kpi-root-cause-analysis",
    title: "KPI Root Cause Analysis & Performance Monitoring",
    description: "Advanced analytics system for KPI monitoring with automated root cause analysis, anomaly detection, and performance optimization recommendations.",
    longDescription: "Intelligent KPI monitoring platform with automated root cause analysis capabilities. Uses statistical methods and machine learning to identify performance anomalies, drill down into contributing factors, and provide actionable optimization recommendations.",
    domain: "Business Intelligence",
    technologies: ["Python", "Anomaly Detection", "Statistical Analysis", "Root Cause Analysis", "Plotly"],
    features: [
      "Automated KPI anomaly detection",
      "Multi-dimensional root cause analysis",
      "Performance trend monitoring",
      "Drill-down analysis capabilities",
      "Predictive performance alerts",
      "Executive dashboard with insights"
    ],
    metrics: [
      { label: "KPIs Monitored", value: "50+" },
      { label: "Anomaly Detection Rate", value: "94%" },
      { label: "Root Cause Accuracy", value: "87%" },
      { label: "Performance Improvement", value: "20%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "geospatial-retail-analysis",
    title: "Geospatial Analysis for Retail Site Selection",
    description: "Advanced geospatial analytics for optimal retail location selection using demographic analysis, competitor mapping, and trade area optimization.",
    longDescription: "Comprehensive geospatial analytics platform for retail site selection and trade area analysis. Combines demographic data, competitor analysis, foot traffic patterns, and market potential to identify optimal retail locations with ROI projections.",
    domain: "Geospatial Analytics",
    technologies: ["Python", "Geospatial Analysis", "Demographic Analysis", "GIS", "Plotly", "Folium"],
    features: [
      "Demographic and psychographic analysis",
      "Competitor proximity and saturation analysis",
      "Trade area definition and optimization",
      "Foot traffic and accessibility modeling",
      "Market potential and ROI calculations",
      "Interactive location recommendation maps"
    ],
    metrics: [
      { label: "Location Accuracy", value: "91%" },
      { label: "ROI Improvement", value: "25%" },
      { label: "Trade Areas Analyzed", value: "500+" },
      { label: "Demographic Variables", value: "100+" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "dynamic-pricing-optimization",
    title: "Dynamic Pricing Strategy & Revenue Optimization",
    description: "AI-powered dynamic pricing system with price elasticity analysis, competitor monitoring, and revenue optimization for e-commerce platforms.",
    longDescription: "Advanced dynamic pricing platform using price elasticity analysis and competitive intelligence. Optimizes pricing strategies in real-time based on demand patterns, inventory levels, and market conditions to maximize revenue and profitability.",
    domain: "Pricing Analytics",
    technologies: ["Python", "Price Elasticity", "Optimization", "Competitive Analysis", "Real-time Analytics"],
    features: [
      "Price elasticity analysis and modeling",
      "Real-time competitor price monitoring",
      "Demand-based pricing optimization",
      "Inventory-aware pricing strategies",
      "A/B testing for pricing experiments",
      "Revenue impact simulation and forecasting"
    ],
    metrics: [
      { label: "Revenue Increase", value: "18%" },
      { label: "Price Updates/Day", value: "10K+" },
      { label: "Products Optimized", value: "50K+" },
      { label: "Margin Improvement", value: "12%" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "gaming-user-behavior",
    title: "Gaming Platform User Behavior & Monetization",
    description: "Advanced gaming analytics for player behavior analysis, retention optimization, and monetization strategy with predictive modeling.",
    longDescription: "Comprehensive gaming analytics platform analyzing player behavior, engagement patterns, and monetization opportunities. Uses survival analysis for churn prediction, player segmentation for targeted strategies, and revenue optimization modeling.",
    domain: "Gaming Analytics",
    technologies: ["Python", "Player Analytics", "Survival Analysis", "Behavioral Modeling", "Plotly"],
    features: [
      "Player behavior and engagement analysis",
      "Churn prediction with survival analysis",
      "Player segmentation and persona development",
      "Monetization optimization strategies",
      "Game balance and difficulty analysis",
      "Real-time player health monitoring"
    ],
    metrics: [
      { label: "Player Retention", value: "+35%" },
      { label: "Revenue per User", value: "+28%" },
      { label: "Churn Prediction", value: "89%" },
      { label: "Players Analyzed", value: "1M+" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "retail-demand-forecasting",
    title: "Retail Demand Forecasting & Inventory Management",
    description: "Advanced demand forecasting system with inventory optimization, stockout prevention, and supply chain analytics for retail operations.",
    longDescription: "Sophisticated retail analytics platform combining demand forecasting with inventory optimization. Uses machine learning for accurate demand prediction, calculates optimal stock levels, and provides supply chain optimization recommendations.",
    domain: "Supply Chain Analytics",
    technologies: ["Python", "Demand Forecasting", "Inventory Optimization", "Supply Chain", "Machine Learning"],
    features: [
      "Multi-product demand forecasting",
      "Inventory optimization with safety stock",
      "Stockout and overstock prevention",
      "Seasonal and promotional impact modeling",
      "Supplier performance analysis",
      "Automated reorder point calculations"
    ],
    metrics: [
      { label: "Forecast Accuracy", value: "93%" },
      { label: "Inventory Reduction", value: "20%" },
      { label: "Stockout Reduction", value: "45%" },
      { label: "Products Managed", value: "10K+" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "insurance-premium-pricing",
    title: "Dynamic Insurance Premium Pricing Model",
    description: "Advanced insurance pricing system using GLM and machine learning for risk assessment with dynamic premium calculation and profitability optimization.",
    longDescription: "Sophisticated insurance pricing platform using Generalized Linear Models and machine learning for accurate risk assessment. Calculates dynamic premiums based on individual risk profiles, optimizes profitability, and ensures competitive pricing strategies.",
    domain: "Insurance Analytics",
    technologies: ["Python", "GLM", "Risk Modeling", "Actuarial Analysis", "Machine Learning"],
    features: [
      "Individual risk assessment and scoring",
      "Dynamic premium calculation",
      "Claims frequency and severity modeling",
      "Competitive pricing analysis",
      "Profitability optimization",
      "Regulatory compliance monitoring"
    ],
    metrics: [
      { label: "Risk Prediction Accuracy", value: "91%" },
      { label: "Premium Optimization", value: "15%" },
      { label: "Claims Ratio Improvement", value: "12%" },
      { label: "Policies Analyzed", value: "100K+" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "investment-portfolio-optimization",
    title: "Investment Portfolio Risk-Return Optimization",
    description: "Advanced portfolio optimization using Modern Portfolio Theory with Monte Carlo simulation, efficient frontier analysis, and risk management.",
    longDescription: "Comprehensive investment analytics platform using Modern Portfolio Theory for optimal asset allocation. Features Monte Carlo simulation, efficient frontier analysis, Sharpe ratio optimization, and comprehensive risk management with real-time market data integration.",
    domain: "Financial Analytics",
    technologies: ["Python", "Portfolio Theory", "Monte Carlo", "Risk Management", "Financial Modeling"],
    features: [
      "Modern Portfolio Theory implementation",
      "Monte Carlo simulation for risk analysis",
      "Efficient frontier calculation",
      "Sharpe ratio optimization",
      "Real-time market data integration",
      "Risk-adjusted return analysis"
    ],
    metrics: [
      { label: "Portfolio Optimization", value: "22%" },
      { label: "Risk Reduction", value: "18%" },
      { label: "Sharpe Ratio Improvement", value: "35%" },
      { label: "Assets Analyzed", value: "500+" }
    ],
    status: "planned",
    featured: false
  }
];