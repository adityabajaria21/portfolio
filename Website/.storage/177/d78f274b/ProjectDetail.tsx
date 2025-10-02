import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, Calendar, Target, TrendingUp, ExternalLink, FileText, Code, Database, Award } from 'lucide-react';
import { projects } from '@/data/projects';

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  
  const project = projects.find(p => p.id === id);
  
  if (!project) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">Project Not Found</h1>
          <p className="text-gray-600 mb-8">The project you're looking for doesn't exist.</p>
          <Button onClick={() => navigate('/')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md border-b z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold text-blue-600">
              Aditya Bajaria
            </div>
            <Button 
              variant="outline"
              onClick={() => navigate('/')}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Portfolio
            </Button>
          </div>
        </div>
      </nav>

      {/* Project Detail Content */}
      <div className="pt-24 pb-20 px-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-blue-100 text-blue-800">
              {project.domain}
            </Badge>
            <h1 className="text-5xl font-bold mb-6 text-gray-800">
              {project.title}
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              {project.description}
            </p>
          </div>

          {/* Project Overview Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Calendar className="w-5 h-5 text-blue-600" />
                  Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Badge 
                  variant={project.status === 'completed' ? 'default' : 'secondary'}
                  className={project.status === 'completed' ? 'bg-green-100 text-green-800' : ''}
                >
                  {project.status.charAt(0).toUpperCase() + project.status.slice(1)}
                </Badge>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Target className="w-5 h-5 text-blue-600" />
                  Impact
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">{project.impact}</p>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  Technologies
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {project.technologies.map((tech, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {tech}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Problem Statement & Objective */}
          {project.problemStatement && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle className="text-2xl">Problem Statement</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 leading-relaxed mb-6">
                  {project.problemStatement}
                </p>
                {project.objective && (
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Objective</h3>
                    <p className="text-gray-600 leading-relaxed">
                      {project.objective}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Technologies Used */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl">
                <TrendingUp className="w-5 h-5 text-blue-600" />
                Technologies Used
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {project.technologies.map((tech, index) => (
                  <Badge key={index} variant="outline" className="px-4 py-2 text-sm font-medium">
                    {tech}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Key Features */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-2xl">Key Features</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {project.keyFeatures.map((feature, index) => (
                  <div key={index} className="text-gray-600 leading-relaxed">
                    {feature}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Outcomes and Impact */}
          {project.outcomesAndImpact && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Award className="w-6 h-6 text-blue-600" />
                  Outcomes and Impact
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 leading-relaxed">
                  {project.outcomesAndImpact}
                </p>
              </CardContent>
            </Card>
          )}

          {/* GitHub Links */}
          {project.githubLinks && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle className="text-2xl">Project Resources</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-4">
                  {project.githubLinks.report && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.report, '_blank')}
                    >
                      <FileText className="w-8 h-8 text-blue-600" />
                      <span className="font-semibold">Report</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.code && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.code, '_blank')}
                    >
                      <Code className="w-8 h-8 text-green-600" />
                      <span className="font-semibold">Code</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.dataset && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.dataset, '_blank')}
                    >
                      <Database className="w-8 h-8 text-purple-600" />
                      <span className="font-semibold">Dataset</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Additional Project Details */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Challenges</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {project.challenges.map((challenge, index) => (
                    <li key={index} className="flex items-start gap-2 text-gray-600">
                      <div className="w-2 h-2 bg-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                      <span>{challenge}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Results</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {project.results.map((result, index) => (
                    <li key={index} className="flex items-start gap-2 text-gray-600">
                      <div className="w-2 h-2 bg-green-600 rounded-full mt-2 flex-shrink-0"></div>
                      <span>{result}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Back Button */}
          <div className="text-center">
            <Button 
              onClick={() => navigate('/')}
              size="lg"
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Portfolio
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}