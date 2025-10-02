import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Database, BarChart3, Brain, Award, GraduationCap } from "lucide-react";

export default function ProfessionalSummary() {
  const skills = [
    "Python", "SQL", "Excel", "Tableau", "Power BI", "R", "Machine Learning", 
    "Statistical Analysis", "A/B Testing", "Data Visualization", "ETL", "PostgreSQL", 
    "MongoDB", "AWS", "Azure", "Pandas", "Scikit-learn", "TensorFlow"
  ];

  const experiences = [
    {
      title: "Senior Data Analyst",
      company: "FinTech Solutions Ltd",
      period: "2022 - Present",
      highlights: [
        "Led data-driven initiatives resulting in £2.5M annual cost savings",
        "Built predictive models improving customer retention by 34%",
        "Managed cross-functional analytics projects for C-level executives"
      ]
    },
    {
      title: "Data Analyst",
      company: "E-commerce Innovations",
      period: "2020 - 2022", 
      highlights: [
        "Developed customer segmentation models increasing marketing ROI by 45%",
        "Created automated reporting dashboards reducing manual work by 60%",
        "Conducted A/B tests optimizing conversion rates across product lines"
      ]
    }
  ];

  return (
    <div className="space-y-8">
      {/* Professional Summary */}
      <Card className="border-0 bg-gradient-to-br from-white to-blue-50 dark:from-gray-900 dark:to-blue-950 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl font-bold text-gray-900 dark:text-white">
            <TrendingUp className="w-6 h-6 mr-3 text-blue-600" />
            Professional Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            Experienced Data Analyst with 4+ years of expertise in transforming complex datasets into actionable business insights. 
            Specialized in statistical modeling, machine learning, and advanced analytics across finance, e-commerce, and insurance sectors. 
            Proven track record of delivering data-driven solutions that drive revenue growth, optimize operations, and enhance customer experience. 
            Passionate about leveraging cutting-edge analytics techniques to solve challenging business problems and create measurable impact.
          </p>
        </CardContent>
      </Card>

      {/* Experience */}
      <Card className="border-0 bg-gradient-to-br from-white to-purple-50 dark:from-gray-900 dark:to-purple-950 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl font-bold text-gray-900 dark:text-white">
            <BarChart3 className="w-6 h-6 mr-3 text-purple-600" />
            Professional Experience
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {experiences.map((exp, index) => (
              <div key={index} className="border-l-4 border-blue-500 pl-6">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">{exp.title}</h3>
                  <span className="text-sm text-blue-600 font-medium">{exp.period}</span>
                </div>
                <p className="text-lg text-gray-600 dark:text-gray-400 mb-3">{exp.company}</p>
                <ul className="space-y-2">
                  {exp.highlights.map((highlight, idx) => (
                    <li key={idx} className="text-gray-700 dark:text-gray-300 flex items-start">
                      <Award className="w-4 h-4 mr-2 mt-1 text-green-600 flex-shrink-0" />
                      {highlight}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Skills */}
      <Card className="border-0 bg-gradient-to-br from-white to-green-50 dark:from-gray-900 dark:to-green-950 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl font-bold text-gray-900 dark:text-white">
            <Brain className="w-6 h-6 mr-3 text-green-600" />
            Technical Skills
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {skills.map((skill) => (
              <Badge 
                key={skill} 
                variant="secondary" 
                className="px-3 py-1 text-sm bg-gradient-to-r from-blue-100 to-purple-100 text-blue-800 dark:from-blue-900 dark:to-purple-900 dark:text-blue-200 hover:scale-105 transition-transform"
              >
                {skill}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Education */}
      <Card className="border-0 bg-gradient-to-br from-white to-orange-50 dark:from-gray-900 dark:to-orange-950 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl font-bold text-gray-900 dark:text-white">
            <GraduationCap className="w-6 h-6 mr-3 text-orange-600" />
            Education
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="border-l-4 border-orange-500 pl-6">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Master of Science in Data Analytics</h3>
              <p className="text-lg text-gray-600 dark:text-gray-400">University of London • 2019-2020</p>
              <p className="text-gray-700 dark:text-gray-300 mt-2">
                Distinction • Specialized in Machine Learning, Statistical Modeling, and Business Intelligence
              </p>
            </div>
            <div className="border-l-4 border-orange-500 pl-6">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Bachelor of Engineering in Computer Science</h3>
              <p className="text-lg text-gray-600 dark:text-gray-400">Mumbai University • 2015-2019</p>
              <p className="text-gray-700 dark:text-gray-300 mt-2">
                First Class Honours • Focus on Database Systems and Algorithms
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}