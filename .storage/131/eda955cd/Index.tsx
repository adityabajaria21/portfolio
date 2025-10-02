import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, Sparkles, Database, Brain, Award, Mail, Phone, Linkedin, Download, MapPin } from 'lucide-react';
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
    // Link to the actual CV file
    window.open('/workspace/uploads/Aditya Bajaria CV (1).pdf', '_blank');
  };

  const domains = Array.from(new Set(projects.map(p => p.domain)));
  
  const filteredProjects = selectedDomain 
    ? projects.filter(p => p.domain === selectedDomain)
    : projects;
    
  const displayedProjects = showAllProjects ? filteredProjects : filteredProjects.slice(0, 6);
  
  const stats = [
    { label: "Projects Completed", value: "10+", icon: Database },
    { label: "ML Models Built", value: "15+", icon: Brain },
    { label: "Years Experience", value: "3+", icon: Award }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-purple-600/20"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmZmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PGNpcmNsZSBjeD0iMzAiIGN5PSIzMCIgcj0iMiIvPjwvZz48L2c+PC9zdmc+')] opacity-40"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 py-20">
          <div className="text-center mb-16">
            {/* Animated Badge */}
            <div className="inline-flex items-center px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 mb-8">
              <Sparkles className="w-4 h-4 mr-2 text-yellow-400" />
              <span className="text-white text-sm font-medium">Available for New Opportunities</span>
            </div>
            
            {/* Main Title */}
            <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 leading-tight">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Aditya Bajaria
              </span>
            </h1>
            
            <div className="text-2xl md:text-3xl text-blue-200 font-light mb-8">
              Data Analyst
            </div>

            {/* Contact Information */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-8 text-white">
              <a href="tel:+447587478594" className="flex items-center gap-2 hover:text-blue-300 transition-colors">
                <Phone className="w-4 h-4" />
                <span>+44 7587478594</span>
              </a>
              <a href="mailto:adityabajaria21@gmail.com" className="flex items-center gap-2 hover:text-blue-300 transition-colors">
                <Mail className="w-4 h-4" />
                <span>adityabajaria21@gmail.com</span>
              </a>
              <a href="https://www.linkedin.com/in/adityabajaria/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-blue-300 transition-colors">
                <Linkedin className="w-4 h-4" />
                <span>LinkedIn Profile</span>
              </a>
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-blue-300" />
                <span>London, United Kingdom</span>
              </div>
            </div>
            
            <div className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed mb-12">
              <p>Strategic data analyst skilled in uncovering insights that drive growth, improve efficiency, and support smarter decision-making. Known for bridging the gap between data and business teams through advanced analytics, machine learning, and statistical modeling.</p>
            </div>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <Button 
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-4 rounded-xl shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-105"
                onClick={() => document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' })}
              >
                <Database className="w-5 h-5 mr-2" />
                Explore My Projects
              </Button>
              <Button 
                onClick={handleResumeDownload}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white border-0 font-semibold px-8 py-4 rounded-xl shadow-2xl transition-all duration-300 hover:scale-105"
              >
                <Download className="w-5 h-5 mr-2" />
                Download Resume
              </Button>
            </div>
            
            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
              {stats.map((stat, index) => (
                <Card key={index} className="bg-white/10 backdrop-blur-sm border-white/20 hover:bg-white/15 transition-all duration-300">
                  <CardContent className="p-6 text-center">
                    <stat.icon className="w-8 h-8 mx-auto mb-3 text-blue-400" />
                    <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                    <div className="text-sm text-gray-300">{stat.label}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
          
          {/* Scroll Indicator */}
          <div className="text-center">
            <ChevronDown className="w-8 h-8 text-white/60 animate-bounce mx-auto" />
          </div>
        </div>
      </section>

      {/* Professional Summary Section */}
      <section className="py-20 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              About Me
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Experienced data professional with a passion for turning data into strategic business value
            </p>
          </div>
          <ProfessionalSummary />
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-20 bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Featured Projects
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              A showcase of data analytics and machine learning projects across various industries
            </p>
            
            {/* Domain Filter Badges */}
            <div className="flex flex-wrap justify-center gap-2 mt-8">
              <Badge 
                variant={selectedDomain === null ? "default" : "secondary"}
                className={`px-4 py-2 text-sm cursor-pointer transition-all ${
                  selectedDomain === null 
                    ? "bg-blue-600 text-white hover:bg-blue-700" 
                    : "hover:bg-gray-200 dark:hover:bg-gray-700"
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
                      : "hover:bg-gray-200 dark:hover:bg-gray-700"
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
          
          {displayedProjects.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400 text-lg">
                No projects found for the selected domain.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 bg-white dark:bg-gray-900">
        <div className="max-w-4xl mx-auto px-4">
          <ContactSection />
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="mb-8">
            <h3 className="text-2xl font-bold mb-2">Aditya Bajaria</h3>
            <p className="text-gray-400">Data Analyst</p>
          </div>
        </div>
      </footer>
    </div>
  );
}