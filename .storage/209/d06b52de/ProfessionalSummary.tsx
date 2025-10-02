import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Award, Code, Database, BarChart3, Brain, Zap } from 'lucide-react';

export default function ProfessionalSummary() {
  const skills = [
    { category: 'Programming', items: ['Python', 'R', 'SQL', 'JavaScript', 'VBA'], icon: Code },
    { category: 'Data Analysis', items: ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch'], icon: Brain },
    { category: 'Visualization', items: ['Tableau', 'Power BI', 'Matplotlib', 'Seaborn', 'Plotly'], icon: BarChart3 },
    { category: 'Databases', items: ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Snowflake'], icon: Database },
    { category: 'Cloud & Tools', items: ['AWS', 'Azure', 'Docker', 'Git', 'Jupyter'], icon: Zap }
  ];

  const certifications = [
    {
      title: 'AWS Certified Solutions Architect',
      issuer: 'Amazon Web Services',
      year: '2023',
      status: 'Active'
    },
    {
      title: 'Microsoft Azure Data Scientist Associate',
      issuer: 'Microsoft',
      year: '2023',
      status: 'Active'
    },
    {
      title: 'Tableau Desktop Specialist',
      issuer: 'Tableau',
      year: '2022',
      status: 'Active'
    },
    {
      title: 'Google Analytics Individual Qualification',
      issuer: 'Google',
      year: '2022',
      status: 'Active'
    }
  ];

  const experience = [
    {
      title: 'Senior Data Analyst',
      company: 'TechCorp Solutions',
      period: '2022 - Present',
      achievements: [
        'Led cross-functional analytics initiatives resulting in 25% improvement in operational efficiency',
        'Developed automated reporting systems reducing manual work by 60%',
        'Built predictive models that increased customer retention by 18%'
      ]
    },
    {
      title: 'Data Analyst',
      company: 'DataFlow Industries',
      period: '2021 - 2022',
      achievements: [
        'Created comprehensive dashboards for C-level executives',
        'Implemented ETL pipelines processing 1M+ records daily',
        'Collaborated with product teams to optimize user experience through data insights'
      ]
    },
    {
      title: 'Junior Data Analyst',
      company: 'Analytics Plus',
      period: '2020 - 2021',
      achievements: [
        'Performed statistical analysis on customer behavior data',
        'Assisted in building machine learning models for demand forecasting',
        'Generated weekly reports for marketing and sales teams'
      ]
    }
  ];

  return (
    <div className="space-y-12">
      {/* Skills Section */}
      <div>
        <h3 className="text-2xl font-bold mb-6 text-center">Skills</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {skills.map((skillGroup, index) => {
            const IconComponent = skillGroup.icon;
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <IconComponent className="w-5 h-5 text-blue-600" />
                    {skillGroup.category}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {skillGroup.items.map((skill, skillIndex) => (
                      <Badge key={skillIndex} variant="secondary" className="text-xs">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Certifications Section */}
      <div>
        <h3 className="text-2xl font-bold mb-6 text-center">Certifications</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {certifications.map((cert, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-2">
                  <Award className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" />
                  <Badge variant="outline" className="text-xs">
                    {cert.status}
                  </Badge>
                </div>
                <h4 className="font-semibold text-lg mb-1">{cert.title}</h4>
                <p className="text-gray-600 mb-2">{cert.issuer}</p>
                <p className="text-sm text-gray-500">{cert.year}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Experience Section */}
      <div>
        <h3 className="text-2xl font-bold mb-6 text-center">Professional Experience</h3>
        <div className="space-y-6">
          {experience.map((job, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <div>
                    <h4 className="text-xl font-semibold text-blue-600">{job.title}</h4>
                    <p className="text-lg text-gray-700">{job.company}</p>
                  </div>
                  <Badge variant="outline" className="mt-2 md:mt-0 w-fit">
                    {job.period}
                  </Badge>
                </div>
                <ul className="space-y-2">
                  {job.achievements.map((achievement, achievementIndex) => (
                    <li key={achievementIndex} className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-gray-600">{achievement}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}