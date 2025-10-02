import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowRight, Code, Database, TrendingUp, Users, Zap, Globe, Brain, Shield, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Index() {
  const [activeSection, setActiveSection] = useState('hero');

  const skills = [
    { name: 'Data Analytics', icon: BarChart3, level: 95 },
    { name: 'Machine Learning', icon: Brain, level: 90 },
    { name: 'Python/R', icon: Code, level: 92 },
    { name: 'SQL/Databases', icon: Database, level: 88 },
    { name: 'Business Intelligence', icon: TrendingUp, level: 85 },
    { name: 'Statistical Analysis', icon: Zap, level: 90 },
  ];

  const experience = [
    {
      title: 'Senior Data Scientist',
      company: 'Tech Corp',
      period: '2022 - Present',
      description: 'Leading ML initiatives, building predictive models, and driving data-driven decision making across the organization.'
    },
    {
      title: 'Data Analyst',
      company: 'Analytics Inc',
      period: '2020 - 2022',
      description: 'Developed dashboards, performed statistical analysis, and provided actionable insights to stakeholders.'
    },
    {
      title: 'Business Analyst',
      company: 'Consulting Group',
      period: '2018 - 2020',
      description: 'Analyzed business processes, identified optimization opportunities, and implemented data-driven solutions.'
    }
  ];

  const achievements = [
    { metric: '50+', label: 'Projects Completed' },
    { metric: '95%', label: 'Client Satisfaction' },
    { metric: '$2M+', label: 'Business Value Generated' },
    { metric: '5+', label: 'Years Experience' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-md border-b z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              DataAnalyst
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#hero" className="text-gray-600 hover:text-blue-600 transition-colors">Home</a>
              <a href="#about" className="text-gray-600 hover:text-blue-600 transition-colors">About</a>
              <a href="#skills" className="text-gray-600 hover:text-blue-600 transition-colors">Skills</a>
              <a href="#experience" className="text-gray-600 hover:text-blue-600 transition-colors">Experience</a>
              <Link to="/portfolio" className="text-gray-600 hover:text-blue-600 transition-colors">Portfolio</Link>
            </div>
            <Link to="/portfolio">
              <Button className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
                View Portfolio
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" className="pt-24 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h1 className="text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Data-Driven
              </span>
              <br />
              <span className="text-gray-800">Insights & Solutions</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Transforming complex data into actionable business intelligence. Specialized in machine learning, 
              predictive analytics, and strategic data visualization to drive organizational growth.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/portfolio">
                <Button size="lg" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
                  Explore Portfolio <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Button size="lg" variant="outline">
                Download Resume
              </Button>
            </div>
          </div>

          {/* Achievement Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
            {achievements.map((achievement, index) => (
              <Card key={index} className="text-center hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="text-3xl font-bold text-blue-600 mb-2">{achievement.metric}</div>
                  <div className="text-sm text-gray-600">{achievement.label}</div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">About Me</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Passionate data scientist with expertise in turning raw data into strategic business advantages
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-2xl font-semibold mb-6">Professional Background</h3>
              <p className="text-gray-600 mb-6">
                With over 5 years of experience in data science and analytics, I specialize in building 
                end-to-end machine learning solutions that solve real business problems. My expertise spans 
                across multiple industries including finance, healthcare, retail, and technology.
              </p>
              <p className="text-gray-600 mb-6">
                I'm passionate about making data accessible and actionable, whether it's through predictive 
                modeling, statistical analysis, or creating intuitive dashboards that empower decision-makers.
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Python</Badge>
                <Badge variant="secondary">R</Badge>
                <Badge variant="secondary">SQL</Badge>
                <Badge variant="secondary">Machine Learning</Badge>
                <Badge variant="secondary">Deep Learning</Badge>
                <Badge variant="secondary">Statistics</Badge>
                <Badge variant="secondary">Data Visualization</Badge>
                <Badge variant="secondary">Big Data</Badge>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Card className="p-6 text-center">
                <Shield className="h-12 w-12 text-blue-600 mx-auto mb-4" />
                <h4 className="font-semibold mb-2">Reliable</h4>
                <p className="text-sm text-gray-600">Consistent delivery of high-quality solutions</p>
              </Card>
              <Card className="p-6 text-center">
                <Globe className="h-12 w-12 text-green-600 mx-auto mb-4" />
                <h4 className="font-semibold mb-2">Scalable</h4>
                <p className="text-sm text-gray-600">Solutions that grow with your business</p>
              </Card>
              <Card className="p-6 text-center">
                <Users className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <h4 className="font-semibold mb-2">Collaborative</h4>
                <p className="text-sm text-gray-600">Working closely with stakeholders</p>
              </Card>
              <Card className="p-6 text-center">
                <Zap className="h-12 w-12 text-yellow-600 mx-auto mb-4" />
                <h4 className="font-semibold mb-2">Innovative</h4>
                <p className="text-sm text-gray-600">Cutting-edge approaches to data challenges</p>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section id="skills" className="py-20 px-6 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Technical Skills</h2>
            <p className="text-xl text-gray-600">
              Comprehensive expertise across the data science and analytics stack
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {skills.map((skill, index) => (
              <Card key={index} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-center mb-4">
                  <skill.icon className="h-8 w-8 text-blue-600 mr-4" />
                  <div className="flex-1">
                    <h3 className="font-semibold">{skill.name}</h3>
                    <div className="text-sm text-gray-600">{skill.level}% Proficiency</div>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${skill.level}%` }}
                  ></div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Experience Section */}
      <section id="experience" className="py-20 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Professional Experience</h2>
            <p className="text-xl text-gray-600">
              A track record of delivering impactful data solutions across various industries
            </p>
          </div>

          <div className="space-y-8">
            {experience.map((job, index) => (
              <Card key={index} className="p-8 hover:shadow-lg transition-shadow">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-semibold">{job.title}</h3>
                    <p className="text-blue-600 font-medium">{job.company}</p>
                  </div>
                  <Badge variant="outline" className="w-fit mt-2 md:mt-0">
                    {job.period}
                  </Badge>
                </div>
                <p className="text-gray-600">{job.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6">Ready to Transform Your Data?</h2>
          <p className="text-xl mb-8 opacity-90">
            Let's discuss how data-driven insights can accelerate your business growth
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/portfolio">
              <Button size="lg" variant="secondary">
                View My Work <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Button size="lg" variant="outline" className="text-white border-white hover:bg-white hover:text-blue-600">
              Get In Touch
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 bg-gray-900 text-white">
        <div className="max-w-6xl mx-auto text-center">
          <div className="text-2xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
            DataAnalyst
          </div>
          <p className="text-gray-400 mb-4">
            Transforming data into actionable insights • Building the future with analytics
          </p>
          <div className="flex justify-center space-x-6">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">LinkedIn</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">GitHub</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Email</a>
          </div>
        </div>
      </footer>
    </div>
  );
}