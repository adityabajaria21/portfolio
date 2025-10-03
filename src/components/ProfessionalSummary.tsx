import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Briefcase, GraduationCap, Award, Code, BarChart3, Database, Brain, TrendingUp } from 'lucide-react';

export default function ProfessionalSummary() {
  const experiences = [
    {
      title: "Research Assistant",
      company: "Warwick University Gillmore Centre",
      location: "Coventry, United Kingdom",
      period: "Jan 2025 – Apr 2025",
      type: "Internship",
      achievements: [
        "Researched AI and NLP applications in finance, focusing on voice-command systems for secure data retrieval. Developed a proof-of-concept tool in Python that converted voice queries into structured SQL commands for market analysis. Presented findings on emerging FinTech tools, using Tableau to visualize competitive landscapes and identify key differentiators."
      ]
    },
    {
      title: "Data Analyst",
      company: "Gracenote (Nielsen)",
      location: "India",
      period: "Feb 2022 – Aug 2024",
      achievements: [
        "Led a key initiative to automate reporting processes using Python and SQL (AWS Athena), reducing manual effort by 95% and cutting delivery times from 3 days to 6 hours. Delivered insights through Tableau and AWS QuickSight that accelerated business decisions by 45%, influenced client acquisition strategies, and solidified a role as the go-to analyst for new ETL tools like Talend Open Studio."
      ]
    },
    {
      title: "Analyst",
      company: "Nepa",
      location: "India", 
      period: "May 2021 – Nov 2021",
      achievements: [
        "Supported senior consultants by delivering brand health and customer satisfaction insights for major FMCG and telecom clients. Analyzed key marketing metrics using SQL, Excel, Power BI and SPSS, and standardized client-facing PowerPoint reporting templates to improve data consistency by 12%."
      ]
    },
    {
      title: "Associate Analyst",
      company: "Colgate-Palmolive",
      location: "India",
      period: "Jul 2020 – Apr 2021", 
      achievements: [
        "Modernized FMCG analytics by migrating legacy reports from R to Python, self-teaching the language to automate manual workflows and enhance financial reporting integrity. Subsequently designed the team's first consolidated dashboard using Google Looker and Tableau, helping stakeholders identify a 15% sales growth opportunity in a key market segment."
      ]
    }
  ];

  const education = [
    {
      degree: "MSc Business Analytics",
      school: "Warwick Business School",
      location: "Coventry, United Kingdom",
      period: "2024 – 2025",
      classification: "Merit (Expected)"
    },
    {
      degree: "B.Tech in Electronics Engineering", 
      school: "K. J. Somaiya College of Engineering",
      location: "India",
      period: "2017 – 2020",
      classification: "Merit"
    }
  ];

  const technicalSkills = {
    "Languages & Tools": ["SQL", "Python", "R", "AWS Athena", "PostgreSQL", "BigQuery", "AWS Glue", "JIRA", "Confluence", "Talend Open Studio", "Salesforce", "Excel", "Google Sheets"],
    "BI & Visualisation": [
      { name: "Tableau", tooltip: "Calculated Fields, Table Calculations, and Level of Detail (LOD) expressions" },
      { name: "Power BI", tooltip: "DAX, Power Query" },
      "AWS QuickSight", 
      "Looker Studio"
    ],
    "Analytics & Modelling": ["A/B Testing", "Regression", "Forecasting", "KPI Analysis", "Hypothesis Testing", "Predictive Modelling", "Statistical Modelling"],
    "BI & Data Engineering": ["ETL Pipelines", "Exploratory Data Analysis (EDA)", "Real-Time Dashboards", "Automated Reporting", "Data Quality Management"],
    "Business Acumen": ["Data Storytelling", "Executive Dashboards", "Cross-Functional Reporting", "Stakeholder Engagement", "Customer Behaviour Analytics", "Agile Framework"]
  };

  const certifications = [
    {
      name: "Applied Business Analytics",
      issuer: "Indian School of Business (ISB)",
      date: "Nov 2023",
      classification: "Distinction",
      modules: ["Regression", "Sentiment Analysis", "Cluster Analysis", "Network Analysis", "Recommendation Systems"]
    }
  ];

  const renderSkillBadge = (skill: any, skillIndex: number) => {
    if (typeof skill === 'object' && skill.tooltip) {
      return (
        <TooltipProvider key={skillIndex}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="text-sm cursor-help">
                {skill.name}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>{skill.tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    }
    return (
      <Badge key={skillIndex} variant="outline" className="text-sm">
        {skill}
      </Badge>
    );
  };

  return (
    <div className="space-y-8">
      {/* Professional Experience */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-2xl">
            <Briefcase className="w-6 h-6 text-blue-600" />
            Professional Experience
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {experiences.map((exp, index) => (
            <div key={index} className="border-l-2 border-blue-200 pl-6 pb-6 last:pb-0">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {exp.title}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {exp.period}
                </span>
              </div>
              <div className="flex items-center gap-2 mb-3">
                <p className="text-lg text-blue-600 font-medium">{exp.company}</p>
                <span className="text-gray-500">•</span>
                <p className="text-gray-600 dark:text-gray-300">{exp.location}</p>
                {exp.type && (
                  <>
                    <span className="text-gray-500">•</span>
                    <Badge variant="secondary" className="text-xs">{exp.type}</Badge>
                  </>
                )}
              </div>
              <div className="space-y-2">
                {exp.achievements.map((achievement, achIndex) => (
                  <p key={achIndex} className="text-gray-700 dark:text-gray-300 leading-relaxed">
                    {achievement}
                  </p>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Education */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-2xl">
            <GraduationCap className="w-6 h-6 text-green-600" />
            Education
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {education.map((edu, index) => (
            <div key={index} className="border-l-2 border-green-200 pl-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {edu.degree}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {edu.period}
                </span>
              </div>
              <p className="text-lg text-green-600 font-medium mb-1">{edu.school}</p>
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                <span>{edu.location}</span>
                <span>•</span>
                <Badge variant="outline" className="text-xs">{edu.classification}</Badge>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Skills */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-2xl">
            <Code className="w-6 h-6 text-orange-600" />
            Skills
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {Object.entries(technicalSkills).map(([category, skills], index) => {
            const icons = {
              "Languages & Tools": Database,
              "BI & Visualisation": BarChart3,
              "Analytics & Modelling": Brain,
              "BI & Data Engineering": TrendingUp,
              "Business Acumen": Briefcase
            };
            const Icon = icons[category as keyof typeof icons] || Code;
            
            return (
              <div key={index}>
                <h4 className="flex items-center gap-2 text-lg font-semibold text-gray-900 dark:text-white mb-3">
                  <Icon className="w-5 h-5 text-orange-600" />
                  {category}
                </h4>
                <div className="flex flex-wrap gap-2">
                  {skills.map((skill, skillIndex) => renderSkillBadge(skill, skillIndex))}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* Certifications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-2xl">
            <Award className="w-6 h-6 text-purple-600" />
            Certifications
          </CardTitle>
        </CardHeader>
        <CardContent>
          {certifications.map((cert, index) => (
            <div key={index} className="border-l-2 border-purple-200 pl-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {cert.name}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {cert.date}
                </span>
              </div>
              <div className="flex items-center gap-2 mb-3">
                <p className="text-lg text-purple-600 font-medium">{cert.issuer}</p>
                <span className="text-gray-500">•</span>
                <Badge variant="outline" className="text-xs">{cert.classification}</Badge>
              </div>
              <div className="flex flex-wrap gap-2">
                {cert.modules.map((module, modIndex) => (
                  <Badge key={modIndex} variant="secondary" className="text-xs">
                    {module}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}