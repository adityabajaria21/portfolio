import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Github, TrendingUp, Database } from "lucide-react";
import { Project } from "@/data/projects";

interface ProjectCardProps {
  project: Project;
  onViewProject: (projectId: string) => void;
}

export default function ProjectCard({ project, onViewProject }: ProjectCardProps) {
  return (
    <Card className="group hover:shadow-xl transition-all duration-300 hover:-translate-y-2 border-0 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors">
              {project.title}
            </CardTitle>
            <CardDescription className="text-sm text-gray-600 dark:text-gray-300 mt-1">
              {project.domain} • {project.industry}
            </CardDescription>
          </div>
          {project.isDataScience && (
            <Badge variant="secondary" className="bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300">
              <TrendingUp className="w-3 h-3 mr-1" />
              ML/DS
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
          {project.description}
        </p>
        
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
        
        <div className="grid grid-cols-2 gap-3 text-xs">
          {project.keyMetrics.slice(0, 2).map((metric) => (
            <div key={metric.label} className="bg-gray-50 dark:bg-gray-800 p-2 rounded">
              <div className="font-semibold text-gray-900 dark:text-white">{metric.value}</div>
              <div className="text-gray-600 dark:text-gray-400">{metric.label}</div>
            </div>
          ))}
        </div>
        
        <div className="flex gap-2 pt-2">
          <Button 
            onClick={() => onViewProject(project.id)}
            className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
            size="sm"
          >
            <Database className="w-4 h-4 mr-2" />
            View Analysis
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => window.open(project.githubUrl, '_blank')}
          >
            <Github className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}