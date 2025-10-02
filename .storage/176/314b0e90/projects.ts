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
  outcomesAndImpact?: string;
  githubLinks?: {
    report?: string;
    code?: string;
    dataset?: string;
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
    outcomesAndImpact: 'Built a review propensity model that identifies likely 4 to 5 star reviewers, enabling targeted prompts and incentives rather than blanket campaigns. Translated scores into an actionable targeting plan, with probability thresholds that balance scale, hit rate, and budget, so marketing can pick the optimal cut off for each campaign. Reduced incentive waste by focusing on high likelihood customers, improving cost per positive review compared with untargeted outreach. The targeting table in the report shows trade offs by threshold choice. Surfaced clear operational levers for CX and logistics, faster and more reliable deliveries correlate with higher positive review rates, which informs SLA priorities and post purchase messaging. Provided category and customer insights that guide merchandising and CRM, certain product groups and repeat purchase signals lift review positivity, which supports smarter campaign segmentation. Established business ready evaluation, precision, recall, F1, ROC AUC, plus simple calibration checks, so leaders understand accuracy and risk at decision time. Delivered stakeholder friendly artefacts, an executive pitch for management and a technical report for analytics teams, aligned with the assignment\'s CRISP DM and presentation requirements. Created a reproducible path to deployment, a clean analytical base table, feature logic, and model selection criteria, which shortens time to integrate with CRM triggers and marketing automation. Linked model outputs to commercial outcomes, higher review volume and quality, stronger buyer trust, and improved marketplace reputation, which supports conversion and seller performance.',
    githubLinks: {
      report: 'https://github.com/adityabajaria21/Masters_Projects/blob/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/Promoter%20Targeting%20Model%20for%20E-commerce%20Report.pdf',
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/Promoter%20Targeting%20Model%20for%20E-commerce.ipynb',
      dataset: 'https://github.com/adityabajaria21/Masters_Projects/tree/964e568018a0c288fa0249f1f378f1748197fb3b/Analytics%20in%20Practise/brazilian-ecommerce%20dataset'
    }
  },
  {
    id: 'academic-project-2',
    title: 'Project 2',
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