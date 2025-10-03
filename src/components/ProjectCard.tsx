import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Github, Star, TrendingUp, Users, Calendar } from "lucide-react";
import { Project } from "@/data/projects";

interface ProjectCardProps {
  project: Project;
  onViewProject: (projectId: string) => void;
}

export default function ProjectCard({ project, onViewProject }: ProjectCardProps) {
  const handleViewProject = () => {
    onViewProject(project.id);
  };

  const handleGithubClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (project.githubUrl) {
      window.open(project.githubUrl, '_blank');
    }
  };

  const handleDemoClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (project.demoUrl) {
      window.open(project.demoUrl, '_blank');
    }
  };

  // Safely get technologies array with fallback
  const technologies = project.technologies || [];
  const features = project.features || [];
  const metrics = project.metrics || [];

  return (
    <Card className="group cursor-pointer transition-all duration-300 hover:shadow-xl hover:-translate-y-1 border-0 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800">
      <div className="relative overflow-hidden rounded-t-lg">
        {project.image ? (
          <div className="h-48 bg-gray-100 dark:bg-gray-900">
            <img
              src={project.image}
              alt={`${project.title} thumbnail`}
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        ) : (
          <div className="h-48 bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <div className="text-white text-center">
              <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-80" />
              <p className="text-sm opacity-90">{project.domain}</p>
            </div>
          </div>
        )}
        
        {/* Status Badge */}
        <div className="absolute top-3 right-3">
          <Badge
            variant={project.status === 'completed' ? 'default' : project.status === 'in-progress' ? 'secondary' : 'outline'}
            className={`${
              project.status === 'completed' 
                ? 'bg-green-500 hover:bg-green-600' 
                : project.status === 'in-progress'
                ? 'bg-yellow-500 hover:bg-yellow-600'
                : 'bg-gray-500 hover:bg-gray-600'
            } text-white border-0`}
          >
            {project.status === 'completed' ? 'Completed' : 
             project.status === 'in-progress' ? 'In Progress' : 'Planned'}
          </Badge>
        </div>

        {/* Featured Badge */}
        {project.featured && (
          <div className="absolute top-3 left-3">
            <Badge className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white border-0">
              <Star className="w-3 h-3 mr-1" />
              Featured
            </Badge>
          </div>
        )}
      </div>

      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-xl font-bold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors line-clamp-2">
              {project.title}
            </CardTitle>
            <CardDescription className="text-gray-600 dark:text-gray-300 mt-2 line-clamp-3 leading-relaxed">
              {project.description}
            </CardDescription>
          </div>
        </div>

        {/* Domain Badge */}
        <div className="mt-3">
          <Badge variant="outline" className="text-blue-600 border-blue-200 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-800">
            {project.domain}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {/* Technologies */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Technologies</h4>
          <div className="flex flex-wrap gap-1">
            {technologies.slice(0, 4).map((tech, index) => (
              <Badge key={index} variant="secondary" className="text-xs bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300">
                {tech}
              </Badge>
            ))}
            {technologies.length > 4 && (
              <Badge variant="secondary" className="text-xs bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300">
                +{technologies.length - 4} more
              </Badge>
            )}
          </div>
        </div>

        {/* Key Features */}
        {features.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Key Features</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              {features.slice(0, 3).map((feature, index) => (
                <li key={index} className="flex items-start">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                  <span className="line-clamp-1">{feature}</span>
                </li>
              ))}
              {features.length > 3 && (
                <li className="text-blue-600 dark:text-blue-400 text-xs">
                  +{features.length - 3} more features
                </li>
              )}
            </ul>
          </div>
        )}

        {/* Metrics */}
        {metrics.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Key Metrics</h4>
            <div className="grid grid-cols-2 gap-2">
              {metrics.slice(0, 4).map((metric, index) => (
                <div key={index} className="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">{metric.value}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">{metric.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-4 border-t border-gray-100 dark:border-gray-800">
          <Button 
            onClick={handleViewProject}
            data-new-tab-url={`/project/${project.id}`}            
            className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white border-0"
          >
            View Details
          </Button>
          
          {project.githubUrl && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleGithubClick}
              data-new-tab-url={project.githubUrl}
              className="border-gray-200 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
            >
              <Github className="w-4 h-4" />
            </Button>
          )}
          
          {project.demoUrl && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleDemoClick}
              data-new-tab-url={project.demoUrl}
              className="border-gray-200 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
            >
              <ExternalLink className="w-4 h-4" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
