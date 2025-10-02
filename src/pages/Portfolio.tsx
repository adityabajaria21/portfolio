import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { projects } from '@/data/projects';
import { ExternalLink, Github, Play, TrendingUp, Users, DollarSign, BarChart3 } from 'lucide-react';

export default function Portfolio() {
  const [selectedDomain, setSelectedDomain] = useState('all');
  
  const completedProjects = projects.filter(p => p.status === 'completed');
  const inProgressProjects = projects.filter(p => p.status === 'in-progress');
  const plannedProjects = projects.filter(p => p.status === 'planned');
  
  const domains = [...new Set(projects.map(p => p.domain))];
  
  const filteredProjects = selectedDomain === 'all' 
    ? projects 
    : projects.filter(p => p.domain === selectedDomain);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'in-progress': return 'bg-yellow-500';
      case 'planned': return 'bg-gray-400';
      default: return 'bg-gray-400';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'in-progress': return 'In Progress';
      case 'planned': return 'Planned';
      default: return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            Data Analytics Portfolio
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Comprehensive collection of 20 advanced data analytics projects
          </p>
          
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card>
              <CardContent className="p-6 text-center">
                <div className="flex items-center justify-center mb-2">
                  <TrendingUp className="h-8 w-8 text-green-500" />
                </div>
                <div className="text-3xl font-bold text-green-600">{completedProjects.length}</div>
                <div className="text-sm text-gray-600">Completed Projects</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6 text-center">
                <div className="flex items-center justify-center mb-2">
                  <BarChart3 className="h-8 w-8 text-yellow-500" />
                </div>
                <div className="text-3xl font-bold text-yellow-600">{inProgressProjects.length}</div>
                <div className="text-sm text-gray-600">In Progress</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6 text-center">
                <div className="flex items-center justify-center mb-2">
                  <Users className="h-8 w-8 text-blue-500" />
                </div>
                <div className="text-3xl font-bold text-blue-600">{domains.length}</div>
                <div className="text-sm text-gray-600">Analytics Domains</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6 text-center">
                <div className="flex items-center justify-center mb-2">
                  <DollarSign className="h-8 w-8 text-purple-500" />
                </div>
                <div className="text-3xl font-bold text-purple-600">$2M+</div>
                <div className="text-sm text-gray-600">Business Impact</div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2 justify-center">
            <Button
              variant={selectedDomain === 'all' ? 'default' : 'outline'}
              onClick={() => setSelectedDomain('all')}
              className="mb-2"
            >
              All Projects ({projects.length})
            </Button>
            {domains.map(domain => (
              <Button
                key={domain}
                variant={selectedDomain === domain ? 'default' : 'outline'}
                onClick={() => setSelectedDomain(domain)}
                className="mb-2"
              >
                {domain} ({projects.filter(p => p.domain === domain).length})
              </Button>
            ))}
          </div>
        </div>

        {/* Projects Tabs */}
        <Tabs defaultValue="completed" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="completed">Completed ({completedProjects.length})</TabsTrigger>
            <TabsTrigger value="in-progress">In Progress ({inProgressProjects.length})</TabsTrigger>
            <TabsTrigger value="planned">Planned ({plannedProjects.length})</TabsTrigger>
          </TabsList>
          
          <TabsContent value="completed" className="mt-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {completedProjects
                .filter(project => selectedDomain === 'all' || project.domain === selectedDomain)
                .map((project) => (
                <Card key={project.id} className="hover:shadow-lg transition-shadow duration-300">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg mb-2">{project.title}</CardTitle>
                        <CardDescription className="text-sm">{project.description}</CardDescription>
                      </div>
                      <Badge className={`${getStatusColor(project.status)} text-white ml-2`}>
                        {getStatusText(project.status)}
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-3">
                      <Badge variant="secondary" className="text-xs">{project.domain}</Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    {/* Technologies */}
                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Technologies:</h4>
                      <div className="flex flex-wrap gap-1">
                        {project.technologies.slice(0, 4).map((tech) => (
                          <Badge key={tech} variant="outline" className="text-xs">
                            {tech}
                          </Badge>
                        ))}
                        {project.technologies.length > 4 && (
                          <Badge variant="outline" className="text-xs">
                            +{project.technologies.length - 4} more
                          </Badge>
                        )}
                      </div>
                    </div>

                    {/* Key Features */}
                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Key Features:</h4>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {project.features.slice(0, 3).map((feature, index) => (
                          <li key={index} className="flex items-start">
                            <span className="w-1 h-1 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Metrics */}
                    {project.metrics && (
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold mb-2">Key Metrics:</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {project.metrics.slice(0, 4).map((metric, index) => (
                            <div key={index} className="text-center p-2 bg-gray-50 rounded">
                              <div className="text-sm font-bold text-blue-600">{metric.value}</div>
                              <div className="text-xs text-gray-600">{metric.label}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-2 mt-4">
                      {project.demoUrl && (
                        <Button size="sm" className="flex-1">
                          <Play className="h-4 w-4 mr-1" />
                          Demo
                        </Button>
                      )}
                      {project.githubUrl && (
                        <Button size="sm" variant="outline" className="flex-1">
                          <Github className="h-4 w-4 mr-1" />
                          Code
                        </Button>
                      )}
                      <Button size="sm" variant="outline">
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="in-progress" className="mt-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {inProgressProjects
                .filter(project => selectedDomain === 'all' || project.domain === selectedDomain)
                .map((project) => (
                <Card key={project.id} className="hover:shadow-lg transition-shadow duration-300 border-yellow-200">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg mb-2">{project.title}</CardTitle>
                        <CardDescription className="text-sm">{project.description}</CardDescription>
                      </div>
                      <Badge className={`${getStatusColor(project.status)} text-white ml-2`}>
                        {getStatusText(project.status)}
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-3">
                      <Badge variant="secondary" className="text-xs">{project.domain}</Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Technologies:</h4>
                      <div className="flex flex-wrap gap-1">
                        {project.technologies.slice(0, 4).map((tech) => (
                          <Badge key={tech} variant="outline" className="text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Progress:</h4>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-yellow-500 h-2 rounded-full" style={{width: '75%'}}></div>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Implementation in progress</p>
                    </div>

                    <Button size="sm" variant="outline" className="w-full">
                      View Progress
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="planned" className="mt-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {plannedProjects
                .filter(project => selectedDomain === 'all' || project.domain === selectedDomain)
                .map((project) => (
                <Card key={project.id} className="hover:shadow-lg transition-shadow duration-300 border-gray-200 opacity-75">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg mb-2">{project.title}</CardTitle>
                        <CardDescription className="text-sm">{project.description}</CardDescription>
                      </div>
                      <Badge className={`${getStatusColor(project.status)} text-white ml-2`}>
                        {getStatusText(project.status)}
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-3">
                      <Badge variant="secondary" className="text-xs">{project.domain}</Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Planned Technologies:</h4>
                      <div className="flex flex-wrap gap-1">
                        {project.technologies.slice(0, 4).map((tech) => (
                          <Badge key={tech} variant="outline" className="text-xs opacity-60">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="mb-4">
                      <h4 className="text-sm font-semibold mb-2">Expected Features:</h4>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {project.features.slice(0, 3).map((feature, index) => (
                          <li key={index} className="flex items-start opacity-60">
                            <span className="w-1 h-1 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <Button size="sm" variant="outline" className="w-full" disabled>
                      Coming Soon
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="text-center mt-16 py-8 border-t">
          <p className="text-gray-600">
            Data Analytics Portfolio • Built with React, TypeScript, and Tailwind CSS
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Showcasing advanced analytics across multiple domains with real business impact
          </p>
        </div>
      </div>
    </div>
  );
}