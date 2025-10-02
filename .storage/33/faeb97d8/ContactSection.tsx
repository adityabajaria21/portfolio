import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mail, Phone, Linkedin, Download, MapPin } from "lucide-react";

export default function ContactSection() {
  const handleResumeDownload = () => {
    // In a real implementation, this would link to the actual resume file
    window.open('#', '_blank');
  };

  return (
    <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 border-0 shadow-lg">
      <CardContent className="p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Let's Connect
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Ready to discuss data-driven solutions for your business
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Mail className="w-5 h-5 text-blue-600" />
              <a href="mailto:adityabajaria21@gmail.com" className="hover:text-blue-600 transition-colors">
                adityabajaria21@gmail.com
              </a>
            </div>
            
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Phone className="w-5 h-5 text-blue-600" />
              <a href="tel:+447587478594" className="hover:text-blue-600 transition-colors">
                +44 7587478594
              </a>
            </div>

                        <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Location className="w-5 h-5 text-blue-600" />
              <a href="London, United Kingdom" className="hover:text-blue-600 transition-colors">
                London, United Kingdo
              </a>
            </div>
            
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Linkedin className="w-5 h-5 text-blue-600" />
              <a 
                href="https://www.linkedin.com/in/adityabajaria/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:text-blue-600 transition-colors"
              >
                linkedin.com/in/adityabajaria
              </a>
            </div>
            
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <MapPin className="w-5 h-5 text-blue-600" />
              <span>London, United Kingdom</span>
            </div>
          </div>
          
          <div className="flex flex-col justify-center">
            <Button 
              onClick={handleResumeDownload}
              className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <Download className="w-5 h-5 mr-2" />
              Download Resume
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}