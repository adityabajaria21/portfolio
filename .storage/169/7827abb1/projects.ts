export interface Project {
  id: string;
  title: string;
  description: string;
  domain: string;
  technologies: string[];
  status: 'completed' | 'in-progress' | 'planned';
  impact?: string;
  metrics?: string[];
}

export const projects: Project[] = [
  {
    id: 'academic-project-1',
    title: 'Project 1',
    description: 'Academic project details will be provided later.',
    domain: 'Academic Projects',
    technologies: ['To be updated'],
    status: 'completed',
    impact: 'Details to be provided',
    metrics: ['Metrics to be updated']
  },
  {
    id: 'academic-project-2',
    title: 'Project 2',
    description: 'Academic project details will be provided later.',
    domain: 'Academic Projects',
    technologies: ['To be updated'],
    status: 'completed',
    impact: 'Details to be provided',
    metrics: ['Metrics to be updated']
  },
  {
    id: 'academic-project-3',
    title: 'Project 3',
    description: 'Academic project details will be provided later.',
    domain: 'Academic Projects',
    technologies: ['To be updated'],
    status: 'completed',
    impact: 'Details to be provided',
    metrics: ['Metrics to be updated']
  }
];