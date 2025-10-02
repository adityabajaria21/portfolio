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
  githubLinks?: {
    report?: string;
    code?: string;
    dataset?: string;
  };
}

export const projects: Project[] = [
  {
    id: 'academic-project-1',
    title: 'E-commerce Review Prediction Model',
    description: 'Machine learning solution to predict customer review outcomes for targeted marketing campaigns',
    domain: 'Academic Projects',
    technologies: ['Python', 'Machine Learning', 'Excel', 'Visualization'],
    status: 'completed' as const,
    impact: 'Improved review volume and quality while reducing campaign waste',
    keyFeatures: ['Predictive modeling for 4-5 star reviews', 'Customer segmentation', 'Business case presentation'],
    challenges: ['Handling imbalanced review data', 'Feature engineering from e-commerce data', 'Model interpretability for business stakeholders'],
    results: ['Successfully identified high-likelihood positive reviewers', 'Delivered comprehensive business case', 'Competitive pitch presentation'],
    duration: '3 months',
    teamSize: 4,
    problemStatement: 'A leading South American e-commerce marketplace, Nile, engaged our team to design and deploy a machine learning solution that predicts which customers are likely to leave positive post-purchase reviews. The goal is to focus review prompts and incentives on high likelihood customers, improve review volume and quality, and reduce campaign waste. The work formed part of a competitive pitch to win a major contract to deliver this predictive capability, using the platform\'s historical orders, deliveries, payments, and review data.',
    objective: 'Build a prediction model for customer review outcomes, with emphasis on identifying likely 4 to 5 star reviewers, and present the business case and technical approach to Nile\'s management board as part of the pitch.',
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