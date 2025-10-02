export interface Project {
  id: string;
  title: string;
  description: string;
  domain: string;
  technologies: string[];
  status: 'completed' | 'in-progress' | 'planned';
  impact?: string;
  duration?: string;
  teamSize?: number;
  keyMetrics?: string[];
  challenges?: string[];
  outcomes?: string[];
  detailedDescription?: string;
  methodology?: string[];
  dataSource?: string;
  visualizations?: string[];
}

export const projects: Project[] = [
  {
    id: 'academic-project-1',
    title: 'Academic Project 1',
    description: 'Detailed academic research project focusing on advanced data analysis techniques and methodologies.',
    domain: 'Academic Projects',
    technologies: ['Python', 'R', 'Statistical Analysis', 'Machine Learning'],
    status: 'completed',
    impact: 'Research contribution to academic field',
    duration: '6 months',
    teamSize: 1,
    keyMetrics: ['Statistical significance', 'Model accuracy', 'Research impact'],
    challenges: ['Complex data preprocessing', 'Statistical validation', 'Academic rigor'],
    outcomes: ['Published research', 'Novel methodology', 'Academic recognition'],
    detailedDescription: 'This project will contain detailed information about the academic research conducted, including methodology, findings, and contributions to the field.',
    methodology: ['Literature review', 'Data collection', 'Statistical analysis', 'Validation'],
    dataSource: 'Academic datasets and primary research',
    visualizations: ['Statistical plots', 'Research diagrams', 'Result visualizations']
  },
  {
    id: 'academic-project-2',
    title: 'Academic Project 2',
    description: 'Comprehensive academic study exploring innovative approaches to data science and analytics.',
    domain: 'Academic Projects',
    technologies: ['Python', 'Deep Learning', 'Research Methods', 'Data Visualization'],
    status: 'completed',
    impact: 'Advancement in academic knowledge',
    duration: '8 months',
    teamSize: 1,
    keyMetrics: ['Research quality', 'Innovation index', 'Peer review scores'],
    challenges: ['Novel methodology development', 'Data complexity', 'Academic standards'],
    outcomes: ['Academic publication', 'Methodology framework', 'Knowledge contribution'],
    detailedDescription: 'This project will showcase advanced academic research with detailed methodology, analysis, and significant contributions to the academic community.',
    methodology: ['Experimental design', 'Data analysis', 'Peer review', 'Publication'],
    dataSource: 'Primary and secondary academic sources',
    visualizations: ['Research frameworks', 'Data analysis charts', 'Conceptual models']
  },
  {
    id: 'academic-project-3',
    title: 'Academic Project 3',
    description: 'Innovative academic research project demonstrating cutting-edge analytical techniques and scholarly contributions.',
    domain: 'Academic Projects',
    technologies: ['Advanced Analytics', 'Research Tools', 'Statistical Software', 'Academic Writing'],
    status: 'completed',
    impact: 'Significant academic contribution',
    duration: '10 months',
    teamSize: 1,
    keyMetrics: ['Citation potential', 'Methodology innovation', 'Academic impact'],
    challenges: ['Research complexity', 'Academic validation', 'Scholarly standards'],
    outcomes: ['Academic recognition', 'Research methodology', 'Scholarly publication'],
    detailedDescription: 'This project will present comprehensive academic research with detailed findings, methodology, and substantial contributions to academic knowledge.',
    methodology: ['Research design', 'Data collection', 'Analysis', 'Academic writing'],
    dataSource: 'Comprehensive academic research data',
    visualizations: ['Academic charts', 'Research models', 'Analytical diagrams']
  }
];