import React from 'react';
import { Project } from '../data/projects';
import { ExternalLink, Github } from 'lucide-react';

interface ProjectDetailProps {
  project: Project;
}

const ProjectDetail: React.FC<ProjectDetailProps> = ({ project }) => {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="border-b pb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
            {project.domain}
          </span>
          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium capitalize">
            {project.status}
          </span>
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-4">{project.title}</h1>
        <p className="text-lg text-gray-600 mb-4">{project.description}</p>
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span>{project.duration}</span>
        </div>
      </div>

      {/* Impact */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-green-800 mb-3">Impact</h2>
        <p className="text-green-700">{project.impact}</p>
      </div>

      {/* Problem Statement */}
      {project.problemStatement && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Problem Statement</h2>
          <p className="text-gray-700 leading-relaxed whitespace-pre-line">{project.problemStatement}</p>
        </div>
      )}

      {/* Objective */}
      {project.objective && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Objective</h2>
          <div className="text-gray-700 leading-relaxed whitespace-pre-line">{project.objective}</div>
        </div>
      )}

      {/* Technologies */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Technologies Used</h2>
        <div className="flex flex-wrap gap-2">
          {project.technologies.map((tech, index) => (
            <span
              key={index}
              className="px-3 py-1 bg-gray-100 text-gray-800 rounded-full text-sm"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>

      {/* Key Features */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Key Features</h2>
        <ul className="space-y-3">
          {project.keyFeatures.map((feature, index) => (
            <li key={index} className="text-gray-700 leading-relaxed">
              {feature}
            </li>
          ))}
        </ul>
      </div>

      {/* Challenges */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Challenges</h2>
        <ul className="space-y-2">
          {project.challenges.map((challenge, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-red-500 mt-1">•</span>
              <span className="text-gray-700">{challenge}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Results */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Results</h2>
        <ul className="space-y-2">
          {project.results.map((result, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-green-500 mt-1">•</span>
              <span className="text-gray-700">{result}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Outcomes & Impact */}
      {project.outcomesAndImpact && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Outcomes & Impact</h2>
          <ul className="space-y-3">
            {project.outcomesAndImpact.map((outcome, index) => (
              <li key={index} className="text-gray-700 leading-relaxed whitespace-pre-line">
                {outcome}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Live Dashboard */}
      {project.githubLinks?.liveDashboard && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Live Dashboard</h2>
          <div className="border rounded-lg p-4 bg-gray-50 overflow-hidden">
            <div 
              dangerouslySetInnerHTML={{ __html: project.githubLinks.liveDashboard }}
              className="w-full min-h-[600px]"
            />
          </div>
        </div>
      )}

      {/* GitHub Links */}
      {project.githubLinks && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {project.githubLinks.dataset && (
              <a
                href={project.githubLinks.dataset}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Dataset</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.code && (
              <a
                href={project.githubLinks.code}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Code</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.instructions && (
              <a
                href={project.githubLinks.instructions}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Code Instructions</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.executiveSummary && (
              <a
                href={project.githubLinks.executiveSummary}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Executive Business Summary</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.dataDictionary && (
              <a
                href={project.githubLinks.dataDictionary}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Data Dictionary</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.report && (
              <a
                href={project.githubLinks.report}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Report</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.dashboard && (
              <a
                href={project.githubLinks.dashboard}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Dashboard</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.presentation && (
              <a
                href={project.githubLinks.presentation}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Presentation</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.technicalDoc && (
              <a
                href={project.githubLinks.technicalDoc}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Technical Documentation</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.codeAnalysis && (
              <a
                href={project.githubLinks.codeAnalysis}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Code Analysis</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
            {project.githubLinks.dataset2 && (
              <a
                href={project.githubLinks.dataset2}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Github className="w-5 h-5 text-gray-600" />
                <span className="font-medium">Dataset 2</span>
                <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectDetail;