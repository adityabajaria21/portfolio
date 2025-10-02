import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Github, ExternalLink, TrendingUp, Database, BarChart3, Brain } from "lucide-react";
import { projects } from "@/data/projects";

export default function ProjectDashboard() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  
  const project = projects.find(p => p.id === projectId);
  
  if (!project) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Project Not Found</h1>
          <Button onClick={() => navigate('/')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-blue-950">
      {/* Header */}
      <div className="bg-white dark:bg-gray-900 shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button 
              variant="ghost" 
              onClick={() => navigate('/')}
              className="text-blue-600 hover:text-blue-700"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Portfolio
            </Button>
            <div className="flex gap-3">
              <Button 
                variant="outline"
                onClick={() => window.open(project.githubUrl, '_blank')}
              >
                <Github className="w-4 h-4 mr-2" />
                View Code
              </Button>
              <Button 
                onClick={() => window.open(project.dataSource, '_blank')}
              >
                <Database className="w-4 h-4 mr-2" />
                Dataset
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Project Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                {project.title}
              </h1>
              <div className="flex items-center gap-4 text-lg text-gray-600 dark:text-gray-300">
                <span>{project.domain}</span>
                <span>•</span>
                <span>{project.industry}</span>
              </div>
            </div>
            {project.isDataScience && (
              <Badge className="bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300 px-4 py-2">
                <Brain className="w-4 h-4 mr-2" />
                Machine Learning / Data Science
              </Badge>
            )}
          </div>
          
          <p className="text-xl text-gray-700 dark:text-gray-300 leading-relaxed">
            {project.description}
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Executive Summary */}
            <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950">
              <CardHeader>
                <CardTitle className="flex items-center text-2xl text-blue-900 dark:text-blue-100">
                  <TrendingUp className="w-6 h-6 mr-3" />
                  Executive Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
                  {project.executiveSummary}
                </p>
              </CardContent>
            </Card>

            {/* Key Insights */}
            <Card className="border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center text-2xl text-gray-900 dark:text-white">
                  <BarChart3 className="w-6 h-6 mr-3 text-green-600" />
                  Key Insights & Findings
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {project.insights.map((insight, index) => (
                    <div key={index} className="flex items-start space-x-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="w-6 h-6 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                        <span className="text-green-600 dark:text-green-400 text-sm font-bold">{index + 1}</span>
                      </div>
                      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Methodology & Technologies */}
            <Card className="border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center text-2xl text-gray-900 dark:text-white">
                  <Database className="w-6 h-6 mr-3 text-purple-600" />
                  Technologies & Methodology
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Tech Stack</h3>
                    <div className="flex flex-wrap gap-2">
                      {project.technologies.map((tech) => (
                        <Badge key={tech} variant="secondary" className="px-3 py-1">
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Data Source</h3>
                    <p className="text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                      {project.dataSource}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Key Metrics */}
            <Card className="border-0 shadow-lg bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950">
              <CardHeader>
                <CardTitle className="text-xl text-green-900 dark:text-green-100">
                  Key Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {project.keyMetrics.map((metric, index) => (
                    <div key={index} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {metric.value}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {metric.label}
                      </div>
                      {metric.change && (
                        <div className={`text-xs font-medium ${
                          metric.change.startsWith('+') || metric.change.startsWith('-') 
                            ? metric.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                            : 'text-blue-600'
                        }`}>
                          {metric.change}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Project Links */}
            <Card className="border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="text-xl text-gray-900 dark:text-white">
                  Project Resources
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button 
                  className="w-full justify-start" 
                  variant="outline"
                  onClick={() => window.open(project.githubUrl, '_blank')}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Source Code & Documentation
                </Button>
                <Button 
                  className="w-full justify-start" 
                  variant="outline"
                  onClick={() => window.open('#', '_blank')}
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  Live Dashboard
                </Button>
                <Button 
                  className="w-full justify-start" 
                  variant="outline"
                  onClick={() => window.open('#', '_blank')}
                >
                  <Database className="w-4 h-4 mr-2" />
                  Dataset & Schema
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}