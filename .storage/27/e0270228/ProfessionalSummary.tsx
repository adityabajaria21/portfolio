import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  User, 
  Briefcase, 
  Code, 
  GraduationCap,
  TrendingUp,
  Database,
  BarChart3,
  Brain
} from "lucide-react";

export default function ProfessionalSummary() {
  const experience = [
    {
      title: "Research Assistant",
      company: "Warwick University Gillmore Centre",
      period: "Jan 2025 – Apr 2025",
      location: "Coventry, UK",
      highlights: [
        "Researched and developed Python-based AI applications in finance, focusing on voice-command systems for secure data retrieval",
        "Conducted competitive analysis of emerging FinTech tools, evaluating efficacy in investment analysis and data summarisation",
        "Developed voice-to-data command tools for extracting actionable insights on market trends and historical performance"
      ]
    },
    {
      title: "Data Analyst",
      company: "Gracenote (Nielsen)",
      period: "Feb 2022 – Aug 2024",
      location: "India",
      highlights: [
        "Delivered targeted metadata insights that influenced OTT client acquisition strategies, securing 3 new contracts",
        "Streamlined recurring data analyses with Python and Tableau, accelerating business decisions by 45%",
        "Developed reusable data solutions adopted team-wide, improving analyst efficiency by 65%",
        "Mentored junior analysts in ETL tools and dashboard best practices"
      ]
    },
    {
      title: "Analyst",
      company: "Nepa",
      period: "May 2021 – Nov 2021",
      location: "India",
      highlights: [
        "Delivered critical customer satisfaction and brand insights for FMCG, retail, and telecom clients across Nordic markets",
        "Standardised reporting templates across accounts, improving data consistency by 12%",
        "Collaborated with senior consultants to communicate findings to executive audiences"
      ]
    },
    {
      title: "Associate Analyst",
      company: "Colgate-Palmolive",
      period: "Jul 2020 – Apr 2021",
      location: "India",
      highlights: [
        "Identified 15% sales growth opportunity in underinvested segments, informing strategic budget reallocation",
        "Enhanced financial reporting by replacing 4 manual workflows with scalable Python pipelines",
        "Designed 6 real-time dashboards in Tableau and Looker Studio, enabling KPI monitoring across regions",
        "Automated data quality checks, accelerating month-end reporting cycles by 30%"
      ]
    }
  ];

  const skills = {
    "Languages & Tools": [
      "SQL", "Python", "R", "AWS Athena", "PostgreSQL", "BigQuery", 
      "AWS Glue", "JIRA", "Confluence", "Talend Open Studio", "Salesforce"
    ],
    "BI & Visualisation": [
      "Tableau", "Power BI", "AWS QuickSight", "Looker Studio", "Power Query"
    ],
    "Analytics & Modelling": [
      "A/B Testing", "Regression", "Forecasting", "KPI Analysis", 
      "Hypothesis Testing", "Predictive Modelling", "Statistical Modelling"
    ],
    "Business Acumen": [
      "Data Storytelling", "Executive Dashboards", "Cross-Functional Reporting",
      "Stakeholder Engagement", "Customer Analytics", "Agile Framework"
    ]
  };

  const education = [
    {
      degree: "MSc Business Analytics",
      institution: "Warwick Business School",
      period: "2024 – 2025",
      location: "Coventry, UK",
      classification: "Merit (Expected)"
    },
    {
      degree: "B.Tech in Electronics Engineering",
      institution: "K. J. Somaiya College of Engineering",
      period: "2017 – 2020",
      location: "India",
      classification: "Merit"
    }
  ];

  const certifications = [
    {
      name: "Applied Business Analytics",
      institution: "Indian School of Business (ISB)",
      date: "Nov 2023",
      classification: "Distinction",
      modules: ["Regression", "Sentiment Analysis", "Cluster Analysis", "Network Analysis", "Recommendation Systems"]
    }
  ];

  return (
    <div className="space-y-8">
      {/* Professional Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="w-5 h-5 text-blue-600" />
            Professional Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Data Analyst with 3+ years of experience in SQL, Python, and BI tools, specialising in building ETL pipelines, data models, and automated dashboards on cloud platforms. Adept at enhancing data quality, building scalable analytics and translating insights into strategies that support commercial, marketing, and financial decision-making.
          </p>
        </CardContent>
      </Card>

      {/* Professional Experience */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-blue-600" />
            Professional Experience
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {experience.map((job, index) => (
            <div key={index} className="border-l-2 border-blue-200 pl-4 pb-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-2">
                <h3 className="font-semibold text-lg text-gray-900 dark:text-white">
                  {job.title}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {job.period}
                </span>
              </div>
              <p className="text-blue-600 dark:text-blue-400 font-medium mb-1">
                {job.company} • {job.location}
              </p>
              <ul className="space-y-2">
                {job.highlights.map((highlight, idx) => (
                  <li key={idx} className="text-gray-700 dark:text-gray-300 text-sm flex items-start">
                    <span className="text-blue-500 mr-2 mt-1">•</span>
                    {highlight}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Technical Skills */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="w-5 h-5 text-blue-600" />
            Technical Skills
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(skills).map(([category, skillList]) => (
            <div key={category}>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                {category === "Languages & Tools" && <Database className="w-4 h-4 text-blue-500" />}
                {category === "BI & Visualisation" && <BarChart3 className="w-4 h-4 text-blue-500" />}
                {category === "Analytics & Modelling" && <Brain className="w-4 h-4 text-blue-500" />}
                {category === "Business Acumen" && <TrendingUp className="w-4 h-4 text-blue-500" />}
                {category}
              </h4>
              <div className="flex flex-wrap gap-2">
                {skillList.map((skill) => (
                  <Badge key={skill} variant="secondary" className="text-xs">
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Education */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GraduationCap className="w-5 h-5 text-blue-600" />
            Education
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {education.map((edu, index) => (
            <div key={index} className="border-l-2 border-blue-200 pl-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-1">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  {edu.degree}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {edu.period}
                </span>
              </div>
              <p className="text-blue-600 dark:text-blue-400 font-medium">
                {edu.institution} • {edu.location}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Classification: {edu.classification}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Certifications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Badge className="w-5 h-5 text-blue-600" />
            Certifications
          </CardTitle>
        </CardHeader>
        <CardContent>
          {certifications.map((cert, index) => (
            <div key={index} className="border-l-2 border-blue-200 pl-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-1">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  {cert.name}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {cert.date}
                </span>
              </div>
              <p className="text-blue-600 dark:text-blue-400 font-medium mb-1">
                {cert.institution}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                Classification: {cert.classification}
              </p>
              <div className="flex flex-wrap gap-1">
                {cert.modules.map((module) => (
                  <Badge key={module} variant="outline" className="text-xs">
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