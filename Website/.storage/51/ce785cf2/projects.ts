export interface Project {
  id: string;
  title: string;
  description: string;
  longDescription: string;
  technologies: string[];
  domain: string;
  imageUrl: string;
  demoUrl?: string;
  githubUrl?: string;
  features: string[];
  challenges: string[];
  outcomes: string[];
  metrics?: {
    label: string;
    value: string;
  }[];
}

export const projects: Project[] = [
  {
    id: "ecommerce-clv-segmentation",
    title: "E-commerce Customer Segmentation & CLV Prediction",
    description: "Advanced customer analytics using RFM analysis, K-means clustering, and probabilistic CLV modeling to optimize marketing strategies and customer retention.",
    longDescription: "Comprehensive customer analytics project that segments e-commerce customers based on purchasing behavior and predicts Customer Lifetime Value using advanced statistical models. The analysis combines RFM (Recency, Frequency, Monetary) segmentation with unsupervised machine learning and probabilistic modeling to provide actionable business insights.",
    technologies: ["Python", "Pandas", "Scikit-learn", "Lifetimes", "Plotly", "Dash", "K-means Clustering", "BG/NBD Model", "Gamma-Gamma Model"],
    domain: "E-commerce Analytics",
    imageUrl: "/api/placeholder/600/400",
    githubUrl: "https://github.com/adityabajaria/ecommerce-clv-analysis",
    features: [
      "RFM Analysis with 9 customer segments (Champions, Loyal Customers, At Risk, etc.)",
      "K-means clustering for behavioral pattern identification",
      "BG/NBD model for purchase frequency prediction",
      "Gamma-Gamma model for monetary value prediction",
      "Interactive Plotly Dash dashboard with real-time insights",
      "Customer churn risk assessment and probability scoring",
      "Automated business recommendations and intervention strategies",
      "Executive summary with financial impact projections"
    ],
    challenges: [
      "Handling imbalanced customer transaction data across different time periods",
      "Selecting optimal number of clusters using multiple validation methods",
      "Calibrating probabilistic models for accurate CLV predictions",
      "Creating actionable business segments from complex behavioral patterns",
      "Balancing model complexity with interpretability for business stakeholders"
    ],
    outcomes: [
      "Identified $1.2M in revenue protection opportunities from at-risk high-value customers",
      "Achieved 89% accuracy in purchase frequency prediction using BG/NBD model",
      "Segmented 1,847 customers into 9 actionable business segments",
      "Projected 15-20% revenue lift through targeted segment-specific marketing",
      "Created automated intervention triggers for customer retention campaigns"
    ],
    metrics: [
      { label: "Customers Analyzed", value: "1,847" },
      { label: "Predicted Total CLV", value: "$8.9M" },
      { label: "Model Accuracy", value: "89%" },
      { label: "Revenue at Risk", value: "$208K" },
      { label: "ROI Projection", value: "1,200%" }
    ]
  },
  {
    id: "sales-forecasting-ml",
    title: "Sales Forecasting with Machine Learning",
    description: "Time series forecasting model using LSTM neural networks to predict quarterly sales with 92% accuracy, enabling data-driven inventory planning.",
    longDescription: "Built an end-to-end sales forecasting system using deep learning techniques to predict future sales trends. The model incorporates seasonal patterns, promotional effects, and external factors to provide accurate quarterly forecasts for inventory management and strategic planning.",
    technologies: ["Python", "TensorFlow", "LSTM", "Time Series Analysis", "Pandas", "Matplotlib", "Scikit-learn"],
    domain: "Sales Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "LSTM neural network for sequential pattern recognition",
      "Seasonal decomposition and trend analysis",
      "Multi-variate forecasting with external factors",
      "Confidence intervals and prediction uncertainty quantification",
      "Interactive forecasting dashboard with scenario planning"
    ],
    challenges: [
      "Handling irregular seasonal patterns and promotional spikes",
      "Incorporating external economic indicators into the model",
      "Balancing forecast accuracy with computational efficiency",
      "Managing data quality issues in historical sales records"
    ],
    outcomes: [
      "Achieved 92% forecasting accuracy for quarterly predictions",
      "Reduced inventory holding costs by 18% through better demand planning",
      "Improved stock-out situations by 25% with proactive inventory management",
      "Enabled strategic planning with 12-month rolling forecasts"
    ],
    metrics: [
      { label: "Forecast Accuracy", value: "92%" },
      { label: "Inventory Cost Reduction", value: "18%" },
      { label: "Stock-out Improvement", value: "25%" },
      { label: "Forecast Horizon", value: "12 months" }
    ]
  },
  {
    id: "customer-churn-prediction",
    title: "Customer Churn Prediction Model",
    description: "Predictive analytics model using ensemble methods to identify customers at risk of churning, achieving 87% precision in churn detection.",
    longDescription: "Developed a comprehensive churn prediction system that identifies customers likely to discontinue service. The model uses ensemble learning techniques and feature engineering to provide early warning signals for customer retention teams.",
    technologies: ["Python", "Random Forest", "XGBoost", "Feature Engineering", "SMOTE", "Cross-validation"],
    domain: "Customer Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Ensemble model combining Random Forest and XGBoost",
      "Advanced feature engineering from customer behavior data",
      "SMOTE technique for handling class imbalance",
      "Customer risk scoring and segmentation",
      "Automated model retraining pipeline"
    ],
    challenges: [
      "Dealing with highly imbalanced churn dataset (5% churn rate)",
      "Feature selection from 200+ potential predictors",
      "Minimizing false positives to avoid unnecessary retention costs",
      "Creating interpretable model outputs for business teams"
    ],
    outcomes: [
      "Achieved 87% precision and 82% recall in churn detection",
      "Identified top 15 churn indicators for proactive intervention",
      "Reduced customer acquisition costs by focusing retention efforts",
      "Improved customer lifetime value through targeted retention campaigns"
    ],
    metrics: [
      { label: "Model Precision", value: "87%" },
      { label: "Model Recall", value: "82%" },
      { label: "Churn Rate Reduction", value: "23%" },
      { label: "Features Analyzed", value: "200+" }
    ]
  },
  {
    id: "market-basket-analysis",
    title: "Market Basket Analysis & Recommendation Engine",
    description: "Association rule mining and collaborative filtering system to identify product relationships and generate personalized recommendations.",
    longDescription: "Implemented a comprehensive market basket analysis using association rule mining to understand customer purchase patterns. Built a hybrid recommendation system combining collaborative filtering and content-based approaches to increase cross-selling opportunities.",
    technologies: ["Python", "Apriori Algorithm", "Collaborative Filtering", "Surprise Library", "NetworkX", "Tableau"],
    domain: "Retail Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Association rule mining with Apriori algorithm",
      "Collaborative filtering recommendation system",
      "Product affinity network visualization",
      "Real-time recommendation API",
      "A/B testing framework for recommendation effectiveness"
    ],
    challenges: [
      "Processing large transaction datasets efficiently",
      "Balancing recommendation diversity with relevance",
      "Handling cold start problem for new products",
      "Optimizing recommendation algorithms for real-time performance"
    ],
    outcomes: [
      "Increased cross-selling revenue by 28% through targeted recommendations",
      "Identified 150+ high-confidence product association rules",
      "Improved customer engagement with personalized product suggestions",
      "Reduced inventory turnover time for slow-moving products"
    ],
    metrics: [
      { label: "Cross-selling Increase", value: "28%" },
      { label: "Association Rules", value: "150+" },
      { label: "Recommendation Accuracy", value: "76%" },
      { label: "API Response Time", value: "<200ms" }
    ]
  },
  {
    id: "financial-risk-assessment",
    title: "Financial Risk Assessment Dashboard",
    description: "Real-time risk monitoring system using statistical models and machine learning to assess portfolio risk and generate automated alerts.",
    longDescription: "Built a comprehensive financial risk assessment platform that monitors portfolio performance, calculates Value at Risk (VaR), and provides real-time risk metrics. The system incorporates multiple risk models and stress testing scenarios for robust risk management.",
    technologies: ["Python", "Monte Carlo Simulation", "VaR Models", "Streamlit", "yfinance", "Risk Metrics"],
    domain: "Financial Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Real-time portfolio risk calculation",
      "Monte Carlo simulation for scenario analysis",
      "Value at Risk (VaR) and Expected Shortfall metrics",
      "Stress testing with historical scenarios",
      "Automated risk alerts and notifications"
    ],
    challenges: [
      "Handling real-time market data feeds and processing",
      "Implementing complex risk calculations efficiently",
      "Creating intuitive visualizations for risk metrics",
      "Ensuring system reliability for critical financial decisions"
    ],
    outcomes: [
      "Reduced portfolio risk exposure by 15% through early warning system",
      "Automated risk reporting saving 20 hours per week",
      "Improved risk-adjusted returns through better position sizing",
      "Enhanced regulatory compliance with standardized risk metrics"
    ],
    metrics: [
      { label: "Risk Reduction", value: "15%" },
      { label: "Time Saved", value: "20 hrs/week" },
      { label: "Portfolio Assets", value: "$50M+" },
      { label: "Alert Accuracy", value: "94%" }
    ]
  },
  {
    id: "supply-chain-optimization",
    title: "Supply Chain Optimization Analytics",
    description: "End-to-end supply chain analysis using linear programming and simulation to optimize inventory levels and reduce operational costs.",
    longDescription: "Developed a comprehensive supply chain optimization system that analyzes demand patterns, supplier performance, and logistics costs. Used mathematical optimization and simulation techniques to minimize total supply chain costs while maintaining service levels.",
    technologies: ["Python", "Linear Programming", "PuLP", "Simulation", "Optimization", "Supply Chain Analytics"],
    domain: "Operations Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Multi-objective optimization for cost and service level",
      "Supplier performance analysis and scoring",
      "Inventory optimization with demand uncertainty",
      "Transportation cost modeling and route optimization",
      "What-if scenario analysis and sensitivity testing"
    ],
    challenges: [
      "Modeling complex supply chain constraints and relationships",
      "Balancing multiple competing objectives (cost vs. service)",
      "Handling demand uncertainty and supply variability",
      "Scaling optimization algorithms for large supply networks"
    ],
    outcomes: [
      "Reduced total supply chain costs by 22% through optimization",
      "Improved supplier on-time delivery from 78% to 94%",
      "Decreased inventory holding costs while maintaining 99% service level",
      "Identified $2.3M in annual cost savings opportunities"
    ],
    metrics: [
      { label: "Cost Reduction", value: "22%" },
      { label: "Service Level", value: "99%" },
      { label: "Delivery Improvement", value: "78% → 94%" },
      { label: "Annual Savings", value: "$2.3M" }
    ]
  },
  {
    id: "social-media-sentiment",
    title: "Social Media Sentiment Analysis",
    description: "NLP-powered sentiment analysis system to monitor brand perception and customer feedback across social media platforms.",
    longDescription: "Built an automated sentiment analysis system that processes social media mentions, reviews, and comments to gauge brand sentiment. The system uses advanced NLP techniques and provides real-time insights for marketing and customer service teams.",
    technologies: ["Python", "NLTK", "TextBlob", "Twitter API", "NLP", "Sentiment Analysis", "Word2Vec"],
    domain: "Marketing Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Real-time social media monitoring and data collection",
      "Multi-platform sentiment analysis (Twitter, Facebook, Instagram)",
      "Emotion detection and topic modeling",
      "Trend analysis and sentiment scoring over time",
      "Automated alert system for negative sentiment spikes"
    ],
    challenges: [
      "Handling sarcasm and context in social media text",
      "Processing high-volume real-time social media streams",
      "Dealing with informal language and abbreviations",
      "Ensuring sentiment accuracy across different demographics"
    ],
    outcomes: [
      "Processed 100K+ social media mentions monthly",
      "Achieved 85% accuracy in sentiment classification",
      "Reduced response time to negative feedback by 60%",
      "Improved brand sentiment score from 3.2 to 4.1 (out of 5)"
    ],
    metrics: [
      { label: "Monthly Mentions", value: "100K+" },
      { label: "Sentiment Accuracy", value: "85%" },
      { label: "Response Time Reduction", value: "60%" },
      { label: "Brand Sentiment Improvement", value: "3.2 → 4.1" }
    ]
  },
  {
    id: "web-analytics-dashboard",
    title: "Web Analytics Performance Dashboard",
    description: "Comprehensive web analytics dashboard using Google Analytics API and custom metrics to track user behavior and conversion optimization.",
    longDescription: "Created a centralized web analytics dashboard that combines data from multiple sources to provide comprehensive insights into website performance, user behavior, and conversion funnel analysis. The dashboard enables data-driven decision making for digital marketing optimization.",
    technologies: ["Python", "Google Analytics API", "Tableau", "SQL", "ETL", "Web Analytics"],
    domain: "Digital Marketing",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Multi-source data integration (GA, social media, email)",
      "Custom conversion funnel analysis",
      "User journey mapping and behavior flow",
      "A/B testing results tracking and statistical significance",
      "Automated reporting and KPI monitoring"
    ],
    challenges: [
      "Integrating data from multiple marketing platforms",
      "Creating meaningful metrics from complex user journeys",
      "Handling data latency and ensuring real-time updates",
      "Building scalable ETL pipelines for growing data volume"
    ],
    outcomes: [
      "Increased website conversion rate by 34% through optimization insights",
      "Reduced bounce rate from 68% to 45% with UX improvements",
      "Identified top-performing marketing channels and campaigns",
      "Automated weekly reporting saving 15 hours of manual work"
    ],
    metrics: [
      { label: "Conversion Rate Increase", value: "34%" },
      { label: "Bounce Rate Reduction", value: "68% → 45%" },
      { label: "Data Sources Integrated", value: "8" },
      { label: "Weekly Time Saved", value: "15 hours" }
    ]
  },
  {
    id: "pricing-optimization",
    title: "Dynamic Pricing Optimization Model",
    description: "Machine learning-based pricing strategy using elasticity analysis and competitive intelligence to maximize revenue and market share.",
    longDescription: "Developed a dynamic pricing optimization system that analyzes price elasticity, competitor pricing, and market conditions to recommend optimal pricing strategies. The model balances revenue maximization with market share objectives using advanced analytics.",
    technologies: ["Python", "Price Elasticity", "Regression Analysis", "Optimization", "Competitive Analysis"],
    domain: "Pricing Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Price elasticity modeling and demand curve analysis",
      "Competitive pricing intelligence and monitoring",
      "Multi-objective optimization (revenue vs. market share)",
      "Seasonal and promotional pricing recommendations",
      "A/B testing framework for pricing experiments"
    ],
    challenges: [
      "Modeling complex price-demand relationships",
      "Incorporating competitive dynamics and market reactions",
      "Balancing short-term revenue with long-term market position",
      "Handling data sparsity for new products or markets"
    ],
    outcomes: [
      "Increased overall revenue by 19% through optimized pricing",
      "Improved gross margin by 8% while maintaining market share",
      "Reduced price-related customer churn by 25%",
      "Automated pricing recommendations for 500+ products"
    ],
    metrics: [
      { label: "Revenue Increase", value: "19%" },
      { label: "Margin Improvement", value: "8%" },
      { label: "Churn Reduction", value: "25%" },
      { label: "Products Optimized", value: "500+" }
    ]
  },
  {
    id: "healthcare-analytics",
    title: "Healthcare Outcomes Analytics",
    description: "Predictive analytics for patient outcomes using electronic health records to improve treatment effectiveness and reduce readmission rates.",
    longDescription: "Built a comprehensive healthcare analytics system that analyzes patient data to predict treatment outcomes and identify risk factors for readmission. The system helps healthcare providers make data-driven decisions to improve patient care and operational efficiency.",
    technologies: ["Python", "Healthcare Analytics", "Predictive Modeling", "Statistical Analysis", "Medical Data"],
    domain: "Healthcare Analytics",
    imageUrl: "/api/placeholder/600/400",
    features: [
      "Patient risk stratification and outcome prediction",
      "Readmission risk scoring and early warning system",
      "Treatment effectiveness analysis and comparison",
      "Resource utilization optimization",
      "Clinical decision support recommendations"
    ],
    challenges: [
      "Handling sensitive patient data with privacy compliance",
      "Dealing with incomplete and inconsistent medical records",
      "Creating interpretable models for clinical decision making",
      "Validating model predictions with clinical expertise"
    ],
    outcomes: [
      "Reduced patient readmission rates by 16% through risk identification",
      "Improved treatment selection accuracy by 23%",
      "Decreased average length of stay by 1.2 days",
      "Enhanced resource allocation efficiency by 18%"
    ],
    metrics: [
      { label: "Readmission Reduction", value: "16%" },
      { label: "Treatment Accuracy", value: "23%" },
      { label: "Length of Stay Reduction", value: "1.2 days" },
      { label: "Resource Efficiency", value: "18%" }
    ]
  }
];