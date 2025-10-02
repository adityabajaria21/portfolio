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
    description: "Comprehensive customer analytics using RFM analysis, K-means clustering, and CLV prediction models to optimize marketing strategies and customer retention.",
    longDescription: "This project implements a complete customer analytics pipeline for e-commerce businesses. Using advanced statistical models including BG/NBD and Gamma-Gamma distributions, it segments customers based on purchasing behavior and predicts their lifetime value. The analysis includes RFM scoring, K-means clustering, interactive dashboards, and business scenario modeling for actionable insights.",
    domain: "E-commerce Analytics",
    technologies: ["Python", "Pandas", "Scikit-learn", "Lifetimes", "Plotly", "Dash", "Matplotlib", "Seaborn"],
    features: [
      "RFM Analysis with 9 customer segments",
      "K-means clustering with optimal cluster detection", 
      "BG/NBD model for purchase frequency prediction",
      "Gamma-Gamma model for monetary value prediction",
      "Interactive Plotly Dash dashboard",
      "Customer Lifetime Value predictions",
      "Business strategy recommendations",
      "Retention campaign analysis",
      "Revenue optimization scenarios",
      "Marketing budget allocation",
      "Customer journey analysis"
    ],
    metrics: [
      { label: "Customers Analyzed", value: "2,000+" },
      { label: "Transactions Processed", value: "50,000+" },
      { label: "Customer Segments", value: "9" },
      { label: "CLV Prediction Accuracy", value: "85%+" },
      { label: "Revenue Optimization", value: "15-20%" },
      { label: "Retention Improvement", value: "25%" }
    ],
    images: ["/api/placeholder/800/400"],
    githubUrl: "https://github.com/adityabajaria/ecommerce-clv-analysis",
    demoUrl: "http://localhost:8050",
    status: "completed",
    featured: true
  },
  // Placeholder projects for future development
  {
    id: "fraud-detection-ml",
    title: "Real-time Fraud Detection System",
    description: "Advanced machine learning system for detecting fraudulent transactions in real-time with ensemble methods and anomaly detection.",
    longDescription: "Coming soon - A sophisticated fraud detection system using ensemble learning and real-time processing.",
    domain: "Financial Security",
    technologies: ["Python", "Apache Spark", "Kafka", "XGBoost", "Isolation Forest"],
    features: ["Real-time processing", "Ensemble models", "Anomaly detection"],
    metrics: [
      { label: "Detection Accuracy", value: "TBD" },
      { label: "Processing Speed", value: "TBD" }
    ],
    status: "planned",
    featured: false
  },
  {
    id: "recommendation-engine",
    title: "AI-Powered Recommendation Engine",
    description: "Collaborative filtering and content-based recommendation system for e-commerce platforms.",
    longDescription: "Coming soon - Advanced recommendation system using collaborative filtering and deep learning.",
    domain: "Recommendation Systems",
    technologies: ["Python", "TensorFlow", "Collaborative Filtering", "Deep Learning"],
    features: ["Collaborative filtering", "Content-based recommendations", "Hybrid approach"],
    status: "planned",
    featured: false
  },
  {
    id: "predictive-maintenance",
    title: "IoT Predictive Maintenance",
    description: "Machine learning solution for predicting equipment failures using IoT sensor data.",
    longDescription: "Coming soon - Predictive maintenance system using time series analysis and IoT data.",
    domain: "Industrial IoT",
    technologies: ["Python", "Time Series", "IoT", "TensorFlow"],
    features: ["Sensor data processing", "Failure prediction", "Maintenance scheduling"],
    status: "planned",
    featured: false
  },
  {
    id: "sentiment-analysis",
    title: "Social Media Sentiment Analysis",
    description: "NLP-powered sentiment analysis for brand monitoring across social media platforms.",
    longDescription: "Coming soon - Real-time sentiment analysis using advanced NLP techniques.",
    domain: "Natural Language Processing",
    technologies: ["Python", "NLTK", "Transformers", "Social Media APIs"],
    features: ["Real-time monitoring", "Multi-platform analysis", "Sentiment scoring"],
    status: "planned",
    featured: false
  },
  {
    id: "supply-chain-optimization",
    title: "Supply Chain Optimization",
    description: "AI-powered supply chain optimization using operations research and machine learning.",
    longDescription: "Coming soon - Comprehensive supply chain optimization system.",
    domain: "Operations Research",
    technologies: ["Python", "OR-Tools", "Optimization", "Machine Learning"],
    features: ["Route optimization", "Inventory management", "Cost reduction"],
    status: "planned",
    featured: false
  },
  {
    id: "healthcare-analytics",
    title: "Healthcare Analytics Platform",
    description: "Patient outcome prediction and resource optimization for healthcare providers.",
    longDescription: "Coming soon - Healthcare analytics for improved patient outcomes.",
    domain: "Healthcare Technology",
    technologies: ["Python", "Healthcare Data", "Predictive Modeling"],
    features: ["Outcome prediction", "Resource optimization", "Risk assessment"],
    status: "planned",
    featured: false
  },
  {
    id: "financial-risk-modeling",
    title: "Financial Risk Assessment",
    description: "Advanced risk modeling for credit scoring and portfolio optimization.",
    longDescription: "Coming soon - Comprehensive financial risk assessment platform.",
    domain: "Financial Technology",
    technologies: ["Python", "Risk Modeling", "Credit Scoring", "Portfolio Optimization"],
    features: ["Credit scoring", "Risk assessment", "Portfolio optimization"],
    status: "planned",
    featured: false
  },
  {
    id: "computer-vision-quality",
    title: "Computer Vision Quality Control",
    description: "Automated quality control system using computer vision and deep learning.",
    longDescription: "Coming soon - AI-powered quality control for manufacturing.",
    domain: "Computer Vision",
    technologies: ["Python", "OpenCV", "Deep Learning", "Image Processing"],
    features: ["Defect detection", "Quality scoring", "Automated inspection"],
    status: "planned",
    featured: false
  },
  {
    id: "time-series-forecasting",
    title: "Advanced Time Series Forecasting",
    description: "Multi-variate time series forecasting for business planning and inventory management.",
    longDescription: "Coming soon - Advanced forecasting using state-of-the-art time series models.",
    domain: "Time Series Analysis",
    technologies: ["Python", "Prophet", "LSTM", "ARIMA", "Forecasting"],
    features: ["Multi-variate forecasting", "Seasonal decomposition", "Trend analysis"],
    status: "planned",
    featured: false
  },
  {
    id: "ab-testing-platform",
    title: "A/B Testing Analytics Platform",
    description: "Statistical analysis platform for A/B testing with advanced experiment design.",
    longDescription: "Coming soon - Comprehensive A/B testing platform with statistical rigor.",
    domain: "Experimentation",
    technologies: ["Python", "Statistical Analysis", "Hypothesis Testing", "Bayesian Methods"],
    features: ["Experiment design", "Statistical testing", "Results analysis"],
    status: "planned",
    featured: false
  },
  {
    id: "churn-prediction-telecom",
    title: "Telecom Churn Prediction",
    description: "Customer churn prediction model for telecommunications industry with retention strategies.",
    longDescription: "Coming soon - Advanced churn prediction for telecom companies.",
    domain: "Telecommunications",
    technologies: ["Python", "Machine Learning", "Feature Engineering", "Ensemble Methods"],
    features: ["Churn prediction", "Feature importance", "Retention strategies"],
    status: "planned",
    featured: false
  },
  {
    id: "price-optimization-dynamic",
    title: "Dynamic Price Optimization",
    description: "Real-time price optimization using demand forecasting and competitor analysis.",
    longDescription: "Coming soon - Dynamic pricing system for e-commerce platforms.",
    domain: "Pricing Analytics",
    technologies: ["Python", "Price Elasticity", "Optimization", "Real-time Analytics"],
    features: ["Dynamic pricing", "Demand forecasting", "Competitor analysis"],
    status: "planned",
    featured: false
  },
  {
    id: "inventory-optimization",
    title: "Intelligent Inventory Management",
    description: "AI-powered inventory optimization with demand forecasting and automated reordering.",
    longDescription: "Coming soon - Smart inventory management system.",
    domain: "Inventory Management",
    technologies: ["Python", "Demand Forecasting", "Optimization", "Automation"],
    features: ["Demand forecasting", "Stock optimization", "Automated reordering"],
    status: "planned",
    featured: false
  },
  {
    id: "marketing-attribution",
    title: "Multi-Touch Attribution Model",
    description: "Advanced marketing attribution modeling to optimize marketing spend across channels.",
    longDescription: "Coming soon - Comprehensive marketing attribution analysis.",
    domain: "Marketing Analytics",
    technologies: ["Python", "Attribution Modeling", "Marketing Mix", "ROI Analysis"],
    features: ["Multi-touch attribution", "Channel optimization", "ROI measurement"],
    status: "planned",
    featured: false
  },
  {
    id: "web-analytics-advanced",
    title: "Advanced Web Analytics Platform",
    description: "Comprehensive web analytics with user behavior analysis and conversion optimization.",
    longDescription: "Coming soon - Advanced web analytics beyond traditional metrics.",
    domain: "Web Analytics",
    technologies: ["Python", "Web Analytics", "User Behavior", "Conversion Optimization"],
    features: ["Behavior analysis", "Conversion funnels", "User segmentation"],
    status: "planned",
    featured: false
  },
  {
    id: "energy-consumption-optimization",
    title: "Energy Consumption Optimization",
    description: "Smart energy management system using IoT data and machine learning for cost reduction.",
    longDescription: "Coming soon - AI-powered energy optimization for smart buildings.",
    domain: "Energy Management",
    technologies: ["Python", "IoT", "Time Series", "Optimization"],
    features: ["Energy forecasting", "Cost optimization", "Smart scheduling"],
    status: "planned",
    featured: false
  },
  {
    id: "credit-scoring-alternative",
    title: "Alternative Credit Scoring",
    description: "Non-traditional credit scoring using alternative data sources and machine learning.",
    longDescription: "Coming soon - Credit scoring for underbanked populations.",
    domain: "Alternative Finance",
    technologies: ["Python", "Alternative Data", "Machine Learning", "Credit Modeling"],
    features: ["Alternative data analysis", "Risk assessment", "Financial inclusion"],
    status: "planned",
    featured: false
  },
  {
    id: "retail-analytics-suite",
    title: "Comprehensive Retail Analytics",
    description: "End-to-end retail analytics platform covering sales, inventory, and customer insights.",
    longDescription: "Coming soon - Complete retail analytics solution.",
    domain: "Retail Analytics",
    technologies: ["Python", "Retail Analytics", "Dashboard", "Business Intelligence"],
    features: ["Sales analysis", "Inventory tracking", "Customer insights"],
    status: "planned",
    featured: false
  },
  {
    id: "sports-analytics-performance",
    title: "Sports Performance Analytics",
    description: "Advanced sports analytics for player performance analysis and game strategy optimization.",
    longDescription: "Coming soon - Data-driven sports performance analysis.",
    domain: "Sports Analytics",
    technologies: ["Python", "Sports Data", "Performance Metrics", "Strategy Analysis"],
    features: ["Performance tracking", "Strategy optimization", "Predictive modeling"],
    status: "planned",
    featured: false
  }
];