import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mail, Phone, Linkedin, MapPin, Download } from "lucide-react";

export default function ContactSection() {
  const handleResumeDownload = () => {
    // Link to the actual CV file
    window.open('/workspace/uploads/Aditya Bajaria CV (1).pdf', '_blank');
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-3xl font-bold text-center text-gray-900 dark:text-white">
          Get In Touch
        </CardTitle>
        <p className="text-center text-gray-600 dark:text-gray-300 mt-4">
          Ready to turn data into actionable insights? Let's connect and discuss how I can help drive your business forward.
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid md:grid-cols-2 gap-8">
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
              <MapPin className="w-5 h-5 text-blue-600" />
              <span className="text-gray-700 dark:text-gray-300">
                London, United Kingdom
              </span>
            </div>
            
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Linkedin className="w-5 h-5 text-blue-600" />
              <a 
                href="https://www.linkedin.com/in/adityabajaria/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:text-blue-600 transition-colors"
              >
                LinkedIn Profile
              </a>
            </div>
          </div>
          
          <div className="flex flex-col justify-center space-y-4">
            <Button 
              onClick={handleResumeDownload}
              size="lg"
              className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <Download className="w-5 h-5 mr-2" />
              Download Resume
            </Button>
            
            <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
              Available for full-time opportunities and consulting projects
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}