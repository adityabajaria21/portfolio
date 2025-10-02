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
    title: 'Consumer Lending A/B Test of Loan Review Assistance',
    description: 'Randomized experiment analyzing the effectiveness of a new loan approval model versus the current system, measuring decision quality improvements and officer performance metrics.',
    domain: 'Academic Projects',
    technologies: ['R', 'A/B Testing', 'Advanced Statistics', 'Confusion Matrix', 'Welch Two Sample T-Tests', 'Confidence Intervals', 'Effect Size - Cohen\'s d', 'Power Analysis', 'Excel', 'Data Visualization'],
    status: 'completed' as const,
    impact: 'Reduced loan approval errors by 32-45% and improved officer confidence, providing clear evidence for model deployment decisions',
    keyFeatures: [
      '📊 Data Analysis and Visualisation - Profiled experiment activity and balance, day by day counts for control and treatment, verified where officers bypassed final model assisted decisions. Ran EDA to understand metric distributions and relationships before testing, then reported results for whole data versus a filtered set where model use was confirmed, to avoid diluting the treatment effect.',
      '🔍 Data Preparation and Quality Assurance - Duplicate scans, NA checks, structure audits, outlier reviews, and consistency checks on agreement versus conflict totals. Removed incomplete experiments where officers did not complete final decisions with the model, yielding 100 control and 280 treatment officer day experiments, then aggregated to the loan officer level for analysis.',
      '📈 Metrics Engineered and Evaluated - Type I error rate, rejecting good loans, Type II error rate, approving bad loans. Agreement ratio with the model after exposure, decision revision ratio toward model advice, officer final confidence score after exposure.',
      '🔬 Statistical Methods and Inference - Welch two sample t tests to compare control and treatment on each metric. Confidence intervals around mean differences to express uncertainty. Effect sizes using Cohen\'s d with interpretation to show practical significance. Power analysis and sample size guidance using pwr.t.test for next phase design.',
      '⚡ Experiment Evaluation Highlights - Type II error, filtered set with confirmed model use, fell from 0.393 to 0.267, 31.9 percent reduction, p = 0.005, d = −1.67. Type I error fell from 0.502 to 0.277, 44.9 percent reduction, p = 0.001, d = −1.96. Agreement ratio rose from 0.749 to 0.872, p = 0.0053, d = −1.60, large effect. Officer final confidence rose from 595.1 to 730.6, p = 4.69e−09, d = −0.63, medium effect. Decision revisions toward model advice increased, mean 0.055 to 0.120, p = 0.0054, d = −0.77, medium effect.'
    ],
    challenges: [
      'Managing experiment contamination and ensuring clean treatment effects',
      'Balancing statistical rigor with business timeline constraints',
      'Interpreting complex interaction effects between officer behavior and model recommendations'
    ],
    results: [
      'Demonstrated significant reduction in Type I and Type II loan approval errors',
      'Quantified improved officer confidence and model alignment metrics',
      'Provided statistical evidence and power analysis for full deployment recommendation'
    ],
    duration: 'Academic Project',
    teamSize: 1,
    problemStatement: 'A consumer lender needs to improve the quality of loan approval decisions. Loan officers currently use an older computer model when reviewing applications, which coincides with high error rates, approving bad loans and rejecting good loans. A new model has been built by another team. The business ran a randomized experiment, control uses the current model, treatment uses the new model, to determine whether the new assistance reduces costly decision errors and improves officer performance.',
    objective: 'Analyze experimental data to evaluate decision quality under control versus treatment, using the recorded metrics, Type I and II errors before and after seeing model predictions, agreement and conflict with the model, decision revisions, confidence, and completion counts, then recommend whether to continue, stop, or redesign the experiment, and outline the expected financial impact if the new model is deployed.',
    outcomesAndImpact: [
      'Fewer costly mistakes in loan approvals - Lower Type II error rate means fewer bad loans get approved, which reduces expected credit losses. Lower Type I error rate means fewer profitable loans are rejected, which preserves revenue and customer growth.',
      'Higher officer trust and alignment with the model - Higher agreement with model predictions and higher reported confidence indicate better adoption and smoother decision workflows, which supports consistent policy execution at scale.',
      'Operational clarity for rollout - The analysis provides threshold ready metrics and effect sizes that let leaders choose between speed of rollout and statistical confidence. Power guidance and enforcement steps reduce risk of an underpowered or biased test.',
      'Ready to integrate with risk controls - The cleaned, aggregated dataset and measurement framework, error rates, agreement, revisions, confidence, are suitable for ongoing monitoring, and can be wired into QA dashboards for model risk management and training needs.'
    ],
    githubLinks: {
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/7c979626d95f7c11511f1ce64d59eefa97ec1b61/Advanced%20Data%20Analysis/Loan%20A_B%20Testing.pdf',
      dataset: 'https://github.com/adityabajaria21/Masters_Projects/blob/7c979626d95f7c11511f1ce64d59eefa97ec1b61/Advanced%20Data%20Analysis/ADAproject_2025_data.txt'
    }
  },
  {
    id: 'academic-project-4',
    title: 'Implementation, Synthetic Data, and Business Insights',
    description: 'End-to-end data management solution implementing normalized database schema, generating realistic synthetic datasets, and delivering SQL-driven business intelligence reports.',
    domain: 'Academic Projects',
    technologies: ['Python', 'SQL', 'Relational Schema Design', 'Data Visualization', 'matplotlib', 'seaborn'],
    status: 'completed' as const,
    impact: 'Created single source of truth for business operations with 500+ records per entity, enabling faster decision-making and consistent reporting',
    keyFeatures: [
      '🗄️ Data Product and Schema - ER model and normalized SQLite schema with primary and foreign keys.',
      '🎲 Synthetic Data in Python - pandas, NumPy, Faker, seeded generation with realistic business rules.',
      '🔄 Load and QA Pipeline - sqlite3 loaders, foreign keys enabled, duplicate and NA checks, structure audits.',
      '📊 SQL Analytics - KPI queries for reviews, delivery performance, refunds and revenue, product damage.',
      '📈 Visualisation - concise matplotlib and seaborn charts tied to SQL outputs.'
    ],
    challenges: [
      'Designing normalized schema while maintaining query performance',
      'Generating realistic synthetic data that preserves business relationships',
      'Building efficient SQL queries for complex business intelligence requirements'
    ],
    results: [
      'Successfully implemented normalized SQLite database with enforced constraints',
      'Generated 500+ realistic records per focus entity with proper relationships',
      'Delivered comprehensive business intelligence reports with clear visualizations'
    ],
    duration: 'Academic Project',
    teamSize: 1,
    problemStatement: 'Decision makers need timely, trustworthy insights generated from consistent data. The team must implement the schema, populate it with realistic synthetic data at useful scale, and deliver SQL driven reports that answer the original business questions. The result should allow managers to track performance, identify issues, and support decisions without manual data wrangling.',
    objective: 'Implement the schema in SQLite using SQL DDL with all constraints applied. Generate synthetic yet realistic datasets with sufficient variety and at least 500 records per focus entity, then load the database. Build SQL queries that produce the agreed business reports, add clear visualisations, and summarise insights in a short report.',
    outcomesAndImpact: [
      'Single source of truth for core operations - A clean, normalized SQLite database with enforced relationships enables trustworthy reporting across sales, deliveries, refunds, reviews, and damage, so leaders are not reconciling spreadsheets or debating definitions.',
      'Faster decisions on service quality and carriers - Delivery KPIs highlight where delays are frequent or long, so operations can set SLA targets by carrier, adjust allocations, and track improvements with consistent metrics.',
      'Clear view of refund leakage and margin impact - Financial analysis links refunds to categories and products, helping finance and product teams focus quality fixes where refund costs and refund rates are highest, improving contribution margin.',
      'Customer experience signals that drive action - Review analysis connects ratings to delivery conditions, product types, and locations, guiding post purchase messaging, courier choices for sensitive SKUs, and targeted CX interventions.',
      'Supply and quality control visibility - Distributor supply and product damage views surface upstream issues, batch delays, and common damage types, so procurement and warehouse teams can intervene earlier and reduce repeat incidents.',
      'Reproducible, scalable foundation - The Python generator and loading pipeline let the team regenerate realistic datasets quickly, test schema changes safely, and keep the BI layer stable as requirements evolve, which shortens time to insight and de-risks future integrations.'
    ],
    githubLinks: {
      report: 'https://github.com/adityabajaria21/Masters_Projects/blob/1a635a81b6579614fc932cf057fc1e5945ce6bdf/Data%20Management/DM_Group_28_Report.pdf',
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/1a635a81b6579614fc932cf057fc1e5945ce6bdf/Data%20Management/Data%20Product%20and%20Database%20Design.py'
    }
  },
  {
    id: 'academic-project-5',
    title: 'Advanced Retrieval-Augmented Generation (RAG)',
    description: 'Production-ready RAG system combining semantic and keyword search with reranking to deliver accurate, source-backed responses for specialized medical knowledge queries.',
    domain: 'Academic Projects',
    technologies: ['Python', 'Jupyter Notebook', 'Vector Search', 'Embeddings', 'LLM', 'Keyword Search', 'Hybrid Retrieval', 'Reranking', 'Hugging Face Transformers', 'Generative AI'],
    status: 'completed' as const,
    impact: 'Achieved 88% accuracy on medical queries with reduced hallucination risk through hybrid retrieval and strict evidence guardrails',
    keyFeatures: [
      '🏥 Domain and Data Pipeline - Medical Q&A on cardiac anatomy, sourced from a specialist PDF, parsed with PyMuPDF, then chunked with a semantic splitter for retrieval.',
      '🔍 Embeddings and Storage - Dense embeddings with Sentence-Transformers all-mpnet-base-v2, indexed in FAISS for fast semantic search.',
      '🔄 Hybrid Retrieval and Reranking - Combined BM25 keyword search and FAISS vector search, then re-ranked candidates with BAAI/bge-reranker-base to surface the most relevant context.',
      '🤖 Generation and Prompting - Grounded generation with FLAN-T5-XL using a structured prompt, plus step-back prompting to rewrite ambiguous queries for better retrieval.',
      '🛡️ Guardrails and Answer Policy - Strict evidence checks, require multiple key terms present in retrieved context, otherwise decline to answer to reduce hallucinations.',
      '📊 Evaluation Against a Baseline - Compared advanced RAG to a plain LLM on depth, relevance, and accuracy, using a diverse test set including out-of-scope questions.'
    ],
    challenges: [
      'Balancing retrieval precision with computational efficiency in hybrid search',
      'Implementing effective guardrails to prevent hallucination without over-restricting responses',
      'Optimizing reranking models for domain-specific medical terminology and context'
    ],
    results: [
      'Achieved 88% accuracy on medical knowledge queries with comprehensive evaluation',
      'Demonstrated superior performance over baseline LLM in 20 of 26 test cases',
      'Successfully implemented production-ready pipeline with modular, extensible architecture'
    ],
    duration: 'Academic Project',
    teamSize: 1,
    problemStatement: 'Leaders need accurate answers that reflect specialised, current, and sometimes proprietary knowledge, which a standalone LLM cannot reliably provide. This project builds an advanced RAG system that grounds an LLM on a curated domain corpus, retrieves the right passages at the right time, and generates source backed responses that outperform a plain LLM on accuracy, relevance, and depth. The work covers a production minded pipeline, from data ingestion and storage through retrieval and re ranking to controlled generation, with clear benchmarks for quality, latency, and cost.',
    objective: 'Select a high value domain and justify why RAG is required. Source and preprocess the corpus, apply semantic or dynamic chunking, generate embeddings, and load a vector database. Implement baseline retrieval, for example cosine similarity, then add at least one recent advancement, for example hybrid keyword plus vector retrieval, reranking with a specialised model, HyDE style query expansion, iterative or multi hop retrieval, and generation strategies such as step back prompting or self refinement. Design a prompt template that integrates retrieved context with citations. Evaluate on a diverse query set, easy, ambiguous, domain specific, and report measurable lift on retrieval precision and nDCG, answer accuracy and citation faithfulness, along with latency and cost. Document failure cases and propose next steps, for example fine tuning components, alternative retrievers, or multi modal RAG, so the system is ready for stakeholder review and deployment planning.',
    outcomesAndImpact: [
      'Higher factual accuracy and depth - The RAG system answered 88 percent of test queries correctly and produced richer, textbook-like responses in 20 of 26 cases, compared with a plain LLM. This improves trust for knowledge intensive use cases.',
      'Better relevance to user intent - Hybrid retrieval and reranking increased contextual alignment, so answers matched medical terminology and query focus more consistently, which reduces follow up time for users.',
      'Reduced misinformation risk - Guardrails led the system to decline out-of-scope questions, for example generic anatomy outside the corpus, which lowers the risk of confident but wrong outputs in sensitive domains.',
      'Production minded design - The pipeline, parsing to chunking to embeddings to retrieval to generation to evaluation, is modular and extensible, so new documents or domains can be added without rework, and future enhancements, for example fine tuning or multimodal inputs, are straightforward.'
    ],
    githubLinks: {
      report: 'https://github.com/adityabajaria21/Masters_Projects/blob/382d31ad859d9c270e4b029daa77f43c8bccd292/Gen%20AI/Report%20Advanced%20Retrieval-Augmented%20Generation%20(RAG).pdf',
      code: 'https://github.com/adityabajaria21/Masters_Projects/blob/382d31ad859d9c270e4b029daa77f43c8bccd292/Gen%20AI/Advanced%20Retrieval-Augmented%20Generation%20(RAG).ipynb',
      dataset: 'https://github.com/adityabajaria21/Masters_Projects/blob/78847b51ba3729899f1071adb4979c8aaecd7daa/Gen%20AI/heart_anatomy.pdf'
    }
  }
];