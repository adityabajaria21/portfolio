export interface Project {
  id: string;
  title: string;
  domain: string;
  industry: string;
  description: string;
  executiveSummary: string;
  technologies: string[];
  dataSource: string;
  githubUrl: string;
  insights: string[];
  keyMetrics: {
    label: string;
    value: string;
    change?: string;
  }[];
  isDataScience: boolean;
}

export const projects: Project[] = [
  {
    id: "ecommerce-segmentation",
    title: "E-commerce Customer Segmentation & Lifetime Value",
    domain: "Marketing Analytics",
    industry: "E-commerce",
    description: "Advanced customer segmentation using RFM analysis and CLV prediction for a major online retailer",
    executiveSummary: "Identified 5 distinct customer segments, leading to 23% increase in targeted campaign ROI. High-value customers represent 15% of base but generate 45% of revenue.",
    technologies: ["Python", "SQL", "Pandas", "Scikit-learn", "Plotly", "PostgreSQL"],
    dataSource: "UCI Online Retail Dataset + Synthetic Enhancement",
    githubUrl: "https://github.com/adityabajaria/ecommerce-customer-analytics",
    insights: [
      "Champions segment (15%) generates 45% of total revenue",
      "At-risk customers show 67% retention rate with targeted campaigns",
      "Customer acquisition cost varies 3x across segments"
    ],
    keyMetrics: [
      { label: "Customer Segments", value: "5", change: "+25%" },
      { label: "Campaign ROI", value: "23%", change: "+23%" },
      { label: "Retention Rate", value: "67%", change: "+12%" }
    ],
    isDataScience: true
  },
  {
    id: "social-media-campaigns",
    title: "Multi-Channel Marketing Campaign Performance",
    domain: "Marketing Analytics", 
    industry: "Digital Marketing",
    description: "Comprehensive analysis of social media campaigns across Facebook, Instagram, and LinkedIn with attribution modeling",
    executiveSummary: "Optimized marketing spend allocation resulting in 34% improvement in ROAS. Instagram shows highest engagement but LinkedIn drives quality leads.",
    technologies: ["Python", "SQL", "Excel", "Tableau", "Google Analytics API"],
    dataSource: "Facebook Marketing API + Google Analytics",
    githubUrl: "https://github.com/adityabajaria/social-media-analytics",
    insights: [
      "LinkedIn generates 3x higher quality leads despite lower volume",
      "Instagram engagement peaks at 2-4 PM on weekdays",
      "Cross-channel attribution reveals 40% multi-touch journeys"
    ],
    keyMetrics: [
      { label: "ROAS Improvement", value: "34%", change: "+34%" },
      { label: "Lead Quality Score", value: "8.2/10", change: "+1.8" },
      { label: "Multi-touch Attribution", value: "40%", change: "+15%" }
    ],
    isDataScience: false
  },
  {
    id: "portfolio-optimization",
    title: "Investment Portfolio Risk-Return Optimization",
    domain: "Financial Analysis",
    industry: "Investment Management",
    description: "Modern Portfolio Theory implementation with Monte Carlo simulations for optimal asset allocation",
    executiveSummary: "Developed efficient frontier analysis reducing portfolio volatility by 18% while maintaining 12% annual returns. Risk-adjusted returns improved by 28%.",
    technologies: ["Python", "NumPy", "Pandas", "Scipy", "Monte Carlo", "SQL"],
    dataSource: "Yahoo Finance API + FRED Economic Data",
    githubUrl: "https://github.com/adityabajaria/portfolio-optimization",
    insights: [
      "Optimal allocation: 40% stocks, 35% bonds, 25% alternatives",
      "Monte Carlo shows 85% probability of positive returns",
      "Rebalancing quarterly reduces drawdown by 12%"
    ],
    keyMetrics: [
      { label: "Volatility Reduction", value: "18%", change: "-18%" },
      { label: "Sharpe Ratio", value: "1.34", change: "+0.28" },
      { label: "Max Drawdown", value: "8.2%", change: "-12%" }
    ],
    isDataScience: true
  },
  {
    id: "credit-risk-assessment",
    title: "Credit Risk Scoring & Default Prediction",
    domain: "Financial Analysis",
    industry: "Banking",
    description: "Machine learning model for credit risk assessment with feature importance analysis and regulatory compliance",
    executiveSummary: "Built predictive model achieving 89% accuracy in default prediction. Reduced bad debt provisions by £2.3M annually while maintaining lending volume.",
    technologies: ["Python", "XGBoost", "SQL", "SHAP", "Pandas", "Scikit-learn"],
    dataSource: "Lending Club Dataset + Credit Bureau Data",
    githubUrl: "https://github.com/adityabajaria/credit-risk-modeling",
    insights: [
      "Debt-to-income ratio is strongest predictor (importance: 0.23)",
      "Employment length shows non-linear relationship with default",
      "Geographic clustering reveals regional risk patterns"
    ],
    keyMetrics: [
      { label: "Model Accuracy", value: "89%", change: "+14%" },
      { label: "Bad Debt Reduction", value: "£2.3M", change: "-15%" },
      { label: "Approval Rate", value: "73%", change: "+3%" }
    ],
    isDataScience: true
  },
  {
    id: "insurance-fraud-detection",
    title: "Insurance Claims Fraud Detection System",
    domain: "Insurance Analytics",
    industry: "Property & Casualty Insurance",
    description: "Anomaly detection and classification system for identifying fraudulent insurance claims using ensemble methods",
    executiveSummary: "Implemented fraud detection system catching 94% of fraudulent claims while reducing false positives by 67%. Saved £4.8M in fraudulent payouts.",
    technologies: ["Python", "Isolation Forest", "Random Forest", "SQL", "Plotly", "SMOTE"],
    dataSource: "Insurance Claims Dataset + External Fraud Indicators",
    githubUrl: "https://github.com/adityabajaria/insurance-fraud-detection",
    insights: [
      "Claim amount and timing patterns strongest fraud indicators",
      "Network analysis reveals organized fraud rings",
      "Seasonal patterns in fraudulent claim submissions"
    ],
    keyMetrics: [
      { label: "Fraud Detection Rate", value: "94%", change: "+29%" },
      { label: "False Positive Reduction", value: "67%", change: "-67%" },
      { label: "Annual Savings", value: "£4.8M", change: "+£1.2M" }
    ],
    isDataScience: true
  },
  {
    id: "insurance-pricing-analysis",
    title: "Dynamic Insurance Premium Pricing Model",
    domain: "Insurance Analytics",
    industry: "Auto Insurance",
    description: "Advanced pricing model incorporating telematics data and external risk factors for personalized premium calculation",
    executiveSummary: "Developed dynamic pricing model improving profit margins by 22% while maintaining competitive rates. Customer satisfaction increased by 15%.",
    technologies: ["Python", "GLM", "SQL", "Excel", "Tableau", "Statistical Modeling"],
    dataSource: "Telematics Data + Claims History + External APIs",
    githubUrl: "https://github.com/adityabajaria/insurance-pricing-model",
    insights: [
      "Driving behavior data reduces pricing uncertainty by 35%",
      "Weather patterns significantly impact claim frequency",
      "Usage-based pricing preferred by 78% of customers"
    ],
    keyMetrics: [
      { label: "Profit Margin", value: "22%", change: "+22%" },
      { label: "Customer Satisfaction", value: "4.6/5", change: "+0.7" },
      { label: "Pricing Accuracy", value: "87%", change: "+18%" }
    ],
    isDataScience: false
  },
  {
    id: "retail-demand-forecasting",
    title: "Retail Demand Forecasting & Inventory Optimization",
    domain: "E-commerce Analytics",
    industry: "Retail",
    description: "Time series forecasting model for demand prediction with inventory optimization and supply chain analytics",
    executiveSummary: "Reduced inventory costs by 28% while improving stock availability to 96%. Demand forecasting accuracy improved to 91% for key product categories.",
    technologies: ["Python", "ARIMA", "Prophet", "SQL", "Pandas", "Plotly"],
    dataSource: "Retail Sales Data + External Economic Indicators",
    githubUrl: "https://github.com/adityabajaria/retail-demand-forecasting",
    insights: [
      "Seasonal patterns vary significantly across product categories",
      "Economic indicators improve forecast accuracy by 12%",
      "Promotional activities create 23% demand spike"
    ],
    keyMetrics: [
      { label: "Inventory Cost Reduction", value: "28%", change: "-28%" },
      { label: "Stock Availability", value: "96%", change: "+8%" },
      { label: "Forecast Accuracy", value: "91%", change: "+16%" }
    ],
    isDataScience: true
  },
  {
    id: "customer-churn-analysis",
    title: "E-commerce Customer Churn Prediction & Retention",
    domain: "E-commerce Analytics",
    industry: "Subscription Commerce",
    description: "Comprehensive churn analysis with predictive modeling and retention strategy optimization",
    executiveSummary: "Reduced customer churn by 31% through targeted retention campaigns. Identified key churn indicators 60 days before actual churn occurs.",
    technologies: ["Python", "Logistic Regression", "SQL", "Pandas", "Seaborn", "A/B Testing"],
    dataSource: "Customer Transaction Data + Behavioral Analytics",
    githubUrl: "https://github.com/adityabajaria/customer-churn-analysis",
    insights: [
      "Support ticket frequency predicts churn with 84% accuracy",
      "Customers with decreasing engagement churn within 45 days",
      "Personalized offers reduce churn probability by 43%"
    ],
    keyMetrics: [
      { label: "Churn Reduction", value: "31%", change: "-31%" },
      { label: "Early Warning", value: "60 days", change: "+30 days" },
      { label: "Retention Rate", value: "87%", change: "+12%" }
    ],
    isDataScience: true
  },
  {
    id: "gaming-user-behavior",
    title: "Gaming Platform User Behavior & Monetization Analysis",
    domain: "Entertainment Analytics",
    industry: "Gaming",
    description: "Player behavior analysis, retention modeling, and monetization optimization for mobile gaming platform",
    executiveSummary: "Increased player lifetime value by 45% through behavioral segmentation. Optimized in-game purchase timing resulting in 38% revenue increase.",
    technologies: ["Python", "Cohort Analysis", "SQL", "Plotly", "Statistical Testing"],
    dataSource: "Gaming Analytics Platform + In-App Purchase Data",
    githubUrl: "https://github.com/adityabajaria/gaming-analytics",
    insights: [
      "Day 7 retention strongly correlates with lifetime value",
      "Optimal purchase prompts occur after achievement unlocks",
      "Social features increase retention by 67%"
    ],
    keyMetrics: [
      { label: "Player LTV", value: "45%", change: "+45%" },
      { label: "Revenue Increase", value: "38%", change: "+38%" },
      { label: "D7 Retention", value: "42%", change: "+18%" }
    ],
    isDataScience: false
  },
  {
    id: "streaming-content-analytics",
    title: "Streaming Platform Content Performance & Recommendation Engine",
    domain: "Entertainment Analytics", 
    industry: "Streaming Media",
    description: "Content performance analysis and collaborative filtering recommendation system for video streaming platform",
    executiveSummary: "Improved content recommendation accuracy by 52%, increasing user engagement time by 34%. Content ROI analysis saved £3.2M in licensing costs.",
    technologies: ["Python", "Collaborative Filtering", "SQL", "Pandas", "Surprise", "NLP"],
    dataSource: "Streaming Platform Data + Content Metadata",
    githubUrl: "https://github.com/adityabajaria/streaming-analytics",
    insights: [
      "Genre preferences vary significantly by demographics",
      "Binge-watching patterns predict subscription renewal",
      "Content diversity increases platform stickiness by 28%"
    ],
    keyMetrics: [
      { label: "Recommendation Accuracy", value: "52%", change: "+52%" },
      { label: "Engagement Time", value: "34%", change: "+34%" },
      { label: "Cost Savings", value: "£3.2M", change: "+£3.2M" }
    ],
    isDataScience: true
  }
];