export interface Project {
  id: string;
  title: string;
  description: string;
  domain: string;
  technologies: string[];
  status: 'completed' | 'in-progress' | 'planned';
  impact: string;
  keyFeatures: string[];
  challenges: string[];
  results: string[];
  duration: string;
  teamSize: number;
  problemStatement?: string;
  objective?: string;
  outcomesAndImpact?: string[];
  githubLinks?: {
    report?: string;
    code?: string;
    dataset?: string;
    dataset2?: string;
  };
}

export const projects: Project[] = [
  {
    id: 'academic-project-1',
    title: 'E-commerce Customer Review Prediction Model',
    description: 'Machine learning solution to predict customers likely to leave positive post-purchase reviews for Nile, a leading South American e-commerce marketplace.',
    domain: 'Academic Projects',
    technologies: ['Python', 'Machine Learning', 'Excel', 'Data Visualization'],
    status: 'completed' as const,
    impact: 'Improved review volume and quality while reducing campaign waste through targeted review prompts',
    keyFeatures: [
      '🔍 Business Goal and Framing - Defined the objective with Nile, predict which customers are likely to leave positive post purchase reviews so review prompts and incentives can be focused where ROI is highest. This aligns with the brief to build a prediction model for review outcomes and to present a commercial pitch to management.',
      '🧹 Data Engineering across Marketplace Tables - Integrated the provided CSV extracts, orders, reviews, deliveries, payments, products, sellers, customers, and category translation, into a clean analytical base table at order and customer level. Applied sensible joins, de duplication, type fixes, missing value handling, and derived keys consistent with the assignment guidance on data preparation.',
      '🧪 Feature Development for Review Propensity - Built predictive features that reflect customer behavior and experience, for example delivery duration and on time flags, payment method, order value and basket size, product and category attributes via the translation mapping, customer tenure and repeat purchase signals, and simple seasonality indicators. These features map directly to the problem space described in the brief.',
      '📈 Exploratory Analysis for Signal Discovery - Profiled review distributions by star rating, looked at correlations with delivery performance and order value, and checked data leakage risks between review timestamps and order events, so the modeling set reflects only information available at decision time. Findings and rationale are documented in the accompanying report.',
      '🤖 Modeling, Framing, and Validation - Cast the task as a classification problem focused on positive reviews, for example 4 to 5 stars versus the rest, consistent with the brief, and trained classical machine learning models with appropriate evaluation. Chose business relevant metrics, precision, recall, F1, ROC AUC, and calibration where appropriate, as required by the assignment.',
      '🎯 Thresholding and Targeting Strategy - Converted predicted probabilities into an actionable targeting list using threshold based rules, so marketing can focus prompts and incentives on high likelihood customers. Included a simple scenario table that links different thresholds to expected volume, precision, and operational cost. The approach is written up for stakeholders in the report.',
      '🧭 Interpretability and Stakeholder Insights - Explained what drives positive reviews in plain language, for example faster delivery, certain payment behaviors, and specific category patterns. Framed insights as levers, where to prioritise logistics SLAs, where to adjust incentive strategy, how to time prompts. Kept explanations manager friendly for the pitch.',
      '✅ CRISP DM and Project Governance - Documented the end to end approach using CRISP DM, business understanding, data understanding, preparation, modeling, evaluation, and communication, which the brief expects, including choices made and trade offs.',
      '🗣️ Executive Pitch and Technical Deliverables - Prepared a concise management pitch focused on commercial value, and a technical report that details the data preparation, features, model selection, and evaluation, in line with the submission requirements.'
    ],
    challenges: [
      'Analyzing complex e-commerce data patterns',
      'Building accurate prediction models',
      'Presenting technical solution to business stakeholders'
    ],
    results: [
      'Successfully built prediction model for customer review outcomes',
      'Delivered comprehensive business case and technical approach',
      'Competitive pitch presentation to Nile management board'
    ],
    duration: 'Academic Project',
    teamSize: 1,
    problemStatement: 'A leading South American e-commerce marketplace, Nile, engaged our team to design and deploy a machine learning solution that predicts which customers are likely to leave positive post-purchase reviews. The goal is to focus review prompts and incentives on high likelihood customers, improve review volume and quality, and reduce campaign waste. The work formed part of a competitive pitch to win a major contract to deliver this predictive capability, using the platform\'s historical orders, deliveries, payments, and review data.',
    objective: 'Build a prediction model for customer review outcomes, with emphasis on identifying likely 4 to 5 star reviewers, and present the business case and technical approach to Nile\'s management board as part of the pitch.',
    outcomesAndImpact: [
      'Built a review propensity model that identifies likely 4 to 5 star reviewers, enabling targeted prompts and incentives rather than blanket campaigns.',
      'Translated scores into an actionable targeting plan, with probability thresholds that balance scale, hit rate, and budget, so marketing can pick the optimal cut off for each campaign.',
      'Reduced incentive waste by focusing on high likelihood customers, improving cost per positive review compared with untargeted outreach. The targeting table in the report shows trade offs by threshold choice.',
      'Surfaced clear operational levers for CX and logistics, faster and more reliable deliveries correlate with higher positive review rates, which informs SLA priorities and post purchase messaging.',
      'Provided category and customer insights that guide merchandising and CRM, certain product groups and repeat purchase signals lift review positivity, which supports smarter campaign segmentation.',
      'Established business ready evaluation, precision, recall, F1, ROC AUC, plus simple calibration checks, so leaders understand accuracy and risk at decision time.',
      'Delivered stakeholder friendly artefacts, an executive pitch for management and a technical report for analytics teams, aligned with the assignment\'s CRISP DM and presentation requirements.',
      'Created a reproducible path to deployment, a clean analytical base table, feature logic, and model selection criteria, which shortens time to integrate with CRM triggers and marketing automation.',
      'Linked model outputs to commercial outcomes, higher review volume and quality, stronger buyer trust, and improved marketplace reputation, which supports conversion and seller performance.'
    ],
    githubLinks: {
      report: 'https://github.com/adityabajaria21/Masters_Projects/blob/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/Promoter%20Targeting%20Model%20for%20E-commerce%20Report.pdf',
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/Promoter%20Targeting%20Model%20for%20E-commerce.ipynb',
      dataset: 'https://github.com/adityabajaria21/Masters_Projects/tree/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/brazilian-ecommerce%20dataset'
    }
  },
  {
    id: 'academic-project-2',
    title: 'Statistical Analysis for Public Health & Retail',
    description: 'Dual statistical analysis project examining cardiovascular disease factors in England and customer satisfaction drivers in furniture retail using classical regression and hypothesis testing.',
    domain: 'Academic Projects',
    technologies: ['R', 'Excel', 'Exploratory Data Analysis', 'Classical Regression', 'NHST and Estimation', 'Hypothesis Tests', 'ANOVA', 'Multicollinearity Diagnostics', 'Data Visualization'],
    status: 'completed' as const,
    impact: 'Provided evidence-based insights for public health resource allocation and retail customer experience optimization',
    keyFeatures: [
      '🔧 Tools and Workflow - Analysis authored in R with R Markdown, knitted to HTML via pandoc. Libraries include tidyverse, ggplot2, patchwork, broom, corrplot, car, e1071.',
      '📊 End-to-End Statistical Workflow - Data dictionaries and input checks, missing value handling, correlation analysis, outlier and skewness checks, multicollinearity diagnostics, linear regression, NHST, estimation with confidence intervals, and ANOVA.',
      '📈 Clear Visual Storytelling - Correlation matrices, scatter plots with linear trend and confidence bands, box plots by segments, side by side figures for quick comparison.',
      '🔬 Reproducible Modeling - Task 1: multiple linear regression of CVD on poverty, overweight, smoking, and wellbeing, with hypothesis tests and ANOVA to judge variable importance. Task 2: multiple linear regression of customer satisfaction on staff satisfaction, delivery time, new range, and SES category, plus correlation diagnostics and visuals.'
    ],
    challenges: [
      'Handling multicollinearity in regression models',
      'Interpreting interaction effects across socioeconomic segments',
      'Translating statistical findings into actionable business recommendations'
    ],
    results: [
      'Identified key cardiovascular disease risk factors for public health targeting',
      'Quantified customer satisfaction drivers for retail optimization',
      'Delivered statistical evidence for policy and business decision making'
    ],
    duration: 'Academic Project',
    teamSize: 1,
    problemStatement: 'Task 1: Cardiovascular Disease in England - Public health leaders need to understand which local factors are most associated with Cardiovascular Disease prevalence across English local authorities, so they can focus prevention budgets where impact is highest. The dataset includes CVD prevalence by area, plus rates of overweight, smoking, poverty, average wellbeing, and population counts. Task 2: Customer Satisfaction in Furniture Retail - A national furniture retailer wants to improve store level customer satisfaction by acting on the most influential operational and workforce drivers. The data covers each store\'s customer satisfaction score, staff job satisfaction score, delivery times for large and custom items, whether a new product range is stocked, and the store\'s socio economic segment.',
    objective: 'Task 1: Quantify the relationship between CVD prevalence and the key risk factors, identify the most actionable levers by effect size, and present an executive ready visual that shows the effect of poverty on CVD, with plain English conclusions for policy and funding decisions. Task 2: Measure the effect of delivery times, job satisfaction, product range status, and store segment on customer satisfaction, then test whether the impact of delivery time varies across low, medium, and high SES stores.',
    outcomesAndImpact: [
      'Task 1, CVD in England, public health value - Clear levers identified, higher overweight and smoking levels are linked with higher CVD prevalence, poverty_log shows a negative coefficient in this dataset, and all are statistically significant, which helps target prevention messages and community programs where they matter most.',
      'Prioritised resource allocation, the correlation and regression outputs provide an evidence base for directing budgets to areas with the strongest risk profiles, improving the expected return on prevention spend.',
      'Decision ready narrative, ANOVA and confidence intervals are presented in plain English, so health leaders can explain why interventions focus on smoking and weight management, and how expected prevalence bands support planning.',
      'Task 2, Furniture retail, customer experience value - Operational focus areas surfaced, faster delivery is associated with higher customer satisfaction, and higher staff satisfaction is associated with happier customers, which supports investment in delivery SLAs, workforce morale, and coaching.',
      'Targeted store actions, SES based views and the regression results help tailor goals by store type, for example tighter delivery targets in segments where satisfaction is most sensitive, and manager scorecards that track staff satisfaction alongside delivery metrics.',
      'Immediate marketing and CX use, the findings translate into practical steps, prioritise delivery improvements for low scoring stores, keep staff engagement programs visible, and use the plots to brief regional managers on where to intervene first.'
    ],
    githubLinks: {
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/3eae74ec033c0bc49f940d2a6a13c5c157a15879/Business%20Statistics/5646778.html',
      dataset: 'https://github.com/adityabajaria21/Masters_Projects/blob/3eae74ec033c0bc49f940d2a6a13c5c157a15879/Business%20Statistics/Cardio_Vascular_Disease.txt',
      dataset2: 'https://github.com/adityabajaria21/Masters_Projects/blob/3eae74ec033c0bc49f940d2a6a13c5c157a15879/Business%20Statistics/cust_satisfaction.txt'
    }
  },
  {
    id: 'academic-project-3',
    title: 'Project 3',
    description: 'Academic project placeholder - details to be provided',
    domain: 'Academic Projects',
    technologies: ['To be updated'],
    status: 'completed' as const,
    impact: 'To be provided',
    keyFeatures: ['Feature details to be added'],
    challenges: ['Challenge details to be added'],
    results: ['Results to be provided'],
    duration: 'TBD',
    teamSize: 1
  }
];