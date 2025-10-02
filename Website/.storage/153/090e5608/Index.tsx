import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowRight, Phone, Mail, Linkedin, MapPin, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ProjectCard from '@/components/ProjectCard';
import ContactSection from '@/components/ContactSection';
import ProfessionalSummary from '@/components/ProfessionalSummary';
import { projects } from '@/data/projects';

export default function Index() {
  const navigate = useNavigate();
  const [showAllProjects, setShowAllProjects] = useState(false);
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  const handleViewProject = (projectId: string) => {
    navigate(`/project/${projectId}`);
  };

  const handleResumeDownload = () => {
    window.open('/workspace/uploads/Aditya Bajaria CV (1).pdf', '_blank');
  };

  const domains = Array.from(new Set(projects.map(p => p.domain)));
  
  const filteredProjects = selectedDomain 
    ? projects.filter(p => p.domain === selectedDomain)
    : projects;
    
  const displayedProjects = showAllProjects ? filteredProjects : filteredProjects.slice(0, 6);

  const achievements = [
    { metric: '20+', label: 'Projects Completed' },
    { metric: '95%', label: 'Client Satisfaction' },
    { metric: '$2M+', label: 'Business Value Generated' },
    { metric: '5+', label: 'Years Experience' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md border-b z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold text-blue-600">
              Aditya Bajaria
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#hero" className="text-gray-600 hover:text-blue-600 transition-colors">Home</a>
              <a href="#about" className="text-gray-600 hover:text-blue-600 transition-colors">About</a>
              <a href="#projects" className="text-gray-600 hover:text-blue-600 transition-colors">Projects</a>
              <a href="#contact" className="text-gray-600 hover:text-blue-600 transition-colors">Contact</a>
            </div>
            <Button 
              onClick={handleResumeDownload}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              Download Resume
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" className="pt-24 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h1 className="text-6xl font-bold mb-6">
              <span className="text-blue-600">
                Data-Driven
              </span>
              <br />
              <span className="text-gray-800">Insights & Solutions</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-4xl mx-auto leading-relaxed">
              Strategic data analyst skilled in uncovering insights that drive growth, improve efficiency, and support smarter decision-making. Known for bridging the gap between data and business teams through advanced analytics, machine learning, and statistical modelling.
            </p>
            
            {/* Contact Info */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-8 text-gray-600">
              <a href="tel:+447587478594" className="flex items-center gap-2 hover:text-blue-600 transition-colors">
                <Phone className="w-4 h-4 text-pink-500" />
                <span>+44 7587478594</span>
              </a>
              <a href="mailto:adityabajaria21@gmail.com" className="flex items-center gap-2 hover:text-blue-600 transition-colors">
                <Mail className="w-4 h-4 text-purple-500" />
                <span>adityabajaria21@gmail.com</span>
              </a>
              <a href="https://www.linkedin.com/in/adityabajaria/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-blue-600 transition-colors">
                <Linkedin className="w-4 h-4 text-blue-500" />
                <span>LinkedIn Profile</span>
              </a>
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-red-500" />
                <span>London, United Kingdom</span>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <Button 
                size="lg" 
                className="bg-blue-600 hover:bg-blue-700 text-white"
                onClick={() => document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' })}
              >
                Explore Portfolio <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button size="lg" variant="outline" onClick={handleResumeDownload}>
                Download Resume
              </Button>
            </div>

            {/* Available Badge - moved below buttons */}
            <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full border border-green-200">
              <Sparkles className="w-4 h-4 mr-2" />
              <span className="text-sm font-medium">Available for New Opportunities</span>
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
              Experienced data professional with a passion for turning data into strategic business value
            </p>
          </div>
          <ProfessionalSummary />
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-20 px-6 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Featured Projects</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              A showcase of data analytics and machine learning projects across various industries
            </p>
            
            {/* Domain Filter Badges */}
            <div className="flex flex-wrap justify-center gap-2 mt-8">
              <Badge 
                variant={selectedDomain === null ? "default" : "secondary"}
                className={`px-4 py-2 text-sm cursor-pointer transition-all ${
                  selectedDomain === null 
                    ? "bg-blue-600 text-white hover:bg-blue-700" 
                    : "hover:bg-gray-200"
                }`}
                onClick={() => setSelectedDomain(null)}
              >
                All Projects
              </Badge>
              {domains.map((domain) => (
                <Badge 
                  key={domain} 
                  variant={selectedDomain === domain ? "default" : "secondary"}
                  className={`px-4 py-2 text-sm cursor-pointer transition-all ${
                    selectedDomain === domain 
                      ? "bg-blue-600 text-white hover:bg-blue-700" 
                      : "hover:bg-gray-200"
                  }`}
                  onClick={() => setSelectedDomain(domain)}
                >
                  {domain}
                </Badge>
              ))}
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {displayedProjects.map((project) => (
              <ProjectCard 
                key={project.id} 
                project={project} 
                onViewProject={handleViewProject}
              />
            ))}
          </div>
          
          {!showAllProjects && filteredProjects.length > 6 && (
            <div className="text-center">
              <Button 
                onClick={() => setShowAllProjects(true)}
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
              >
                View All Projects ({filteredProjects.length - 6} more)
              </Button>
            </div>
          )}
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 px-6 bg-white">
        <div className="max-w-4xl mx-auto">
          <ContactSection />
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 bg-gray-900 text-white">
        <div className="max-w-6xl mx-auto text-center">
          <div className="text-2xl font-bold mb-4 text-blue-400">
            Aditya Bajaria
          </div>
          <p className="text-gray-400 mb-4">
            Transforming data into actionable insights • Building the future with analytics
          </p>
          <div className="flex justify-center space-x-6">
            <a href="https://www.linkedin.com/in/adityabajaria/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors">LinkedIn</a>
            <a href="mailto:adityabajaria21@gmail.com" className="text-gray-400 hover:text-white transition-colors">Email</a>
            <a href="tel:+447587478594" className="text-gray-400 hover:text-white transition-colors">Phone</a>
          </div>
        </div>
      </footer>
    </div>
  );
}