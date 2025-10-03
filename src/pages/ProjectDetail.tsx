import { useParams, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  ArrowLeft,
  Calendar,
  Target,
  TrendingUp,
  ExternalLink,
  FileText,
  Code,
  Database,
  BarChart3,
  Presentation,
  BookOpen,
  ClipboardList,
  FileCheck,
  NotebookPen,
  Monitor
} from 'lucide-react';
import { projects } from '@/data/projects';

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  
  const project = projects.find(p => p.id === id);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);
  
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
  const openLiveDashboard = () => {
    if (!project.githubLinks?.liveDashboard) {
      return;
    }
    window.open(project.githubLinks.liveDashboard, '_blank', 'noopener,noreferrer');
  };
  
  // Function to format text with Task 1 and Task 2 separation
  const formatTaskText = (text: string) => {
    if (text.includes('Task 1:') && text.includes('Task 2:')) {
      const parts = text.split('Task 2:');
      const task1 = parts[0].replace('Task 1:', '').trim();
      const task2 = parts[1].trim();
      
      return (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-3 text-blue-700">Task 1: Cardiovascular Disease in England</h3>
            <p className="text-gray-600 leading-relaxed">{task1}</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3 text-blue-700">Task 2: Customer Satisfaction in Furniture Retail</h3>
            <p className="text-gray-600 leading-relaxed">{task2}</p>
          </div>
        </div>
      );
    }
    return <p className="text-gray-600 leading-relaxed">{text}</p>;
  };

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
                {formatTaskText(project.problemStatement)}
                {project.objective && (
                  <div className="mt-8">
                    <h3 className="text-2xl font-semibold mb-6">Objective</h3>
                    {formatTaskText(project.objective)}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Technologies Used */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-2xl">Technologies Used</CardTitle>
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

          {/* Challenges and Results */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-2xl">Challenges</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {project.challenges.map((challenge, index) => (
                    <div key={index} className="flex items-start gap-3 text-gray-600 leading-relaxed">
                      <span className="text-orange-600 font-bold mt-1">•</span>
                      <span>{challenge}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-2xl">Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {project.results.map((result, index) => (
                    <div key={index} className="flex items-start gap-3 text-gray-600 leading-relaxed">
                      <span className="text-green-600 font-bold mt-1">•</span>
                      <span>{result}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Outcomes and Impact */}
          {project.outcomesAndImpact && Array.isArray(project.outcomesAndImpact) && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle className="text-2xl">Outcomes and Impact</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {project.outcomesAndImpact.map((outcome, index) => (
                    <div key={index} className="flex items-start gap-3 text-gray-600 leading-relaxed">
                      <span className="text-blue-600 font-bold mt-1">•</span>
                      <span>{outcome}</span>
                    </div>
                  ))}
                </div>
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
                  {project.githubLinks.dataset && (
                    <Button
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.dataset, '_blank')}
                    >
                      <Database className="w-8 h-8 text-purple-600" />
                      <span className="font-semibold">Data</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.instructions && (
                    <Button
                      variant="outline"
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.instructions, '_blank')}
                    >
                      <NotebookPen className="w-8 h-8 text-amber-600" />
                      <span className="font-semibold">Code Instructions</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}                  
                  {project.githubLinks.technicalDoc && (
                    <Button
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.technicalDoc, '_blank')}
                    >
                      <BookOpen className="w-8 h-8 text-blue-600" />
                      <span className="font-semibold">Technical Document</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.executiveSummary && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.executiveSummary, '_blank')}
                    >
                      <ClipboardList className="w-8 h-8 text-green-600" />
                      <span className="font-semibold">Executive Business Summary</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.code && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.code, '_blank')}
                    >
                      <Code className="w-8 h-8 text-gray-600" />
                      <span className="font-semibold">Code</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.codeAnalysis && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.codeAnalysis, '_blank')}
                    >
                      <FileCheck className="w-8 h-8 text-indigo-600" />
                      <span className="font-semibold">Code Analysis Document</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.dataDictionary && (
                    <Button
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.dataDictionary, '_blank')}
                    >
                      <FileText className="w-8 h-8 text-orange-600" />
                      <span className="font-semibold">Data Dictionary</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
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
                  {project.githubLinks.liveDashboard && (
                    <Button
                      variant="outline"
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={openLiveDashboard}
                    >
                      <Monitor className="w-8 h-8 text-sky-600" />
                      <span className="font-semibold">Live Dashboard</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.presentation && (
                    <Button
                      variant="outline"
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.presentation, '_blank')}
                    >
                      <Presentation className="w-8 h-8 text-teal-600" />
                      <span className="font-semibold">Executive Presentation</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  {project.githubLinks.dataset2 && (
                    <Button 
                      variant="outline" 
                      className="h-auto p-4 flex flex-col items-center gap-2"
                      onClick={() => window.open(project.githubLinks!.dataset2, '_blank')}
                    >
                      <Database className="w-8 h-8 text-orange-600" />
                      <span className="font-semibold">Dataset 02</span>
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

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