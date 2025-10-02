import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Mail, Phone, MapPin, Linkedin, Download } from 'lucide-react';

export default function ContactSection() {
  const handleResumeDownload = () => {
    window.open('/workspace/uploads/Aditya Bajaria CV (3).pdf', '_blank');
  };

  return (
    <div className="text-center">
      <h2 className="text-4xl font-bold mb-4">Get In Touch</h2>
      <p className="text-xl text-gray-600 mb-12 max-w-2xl mx-auto">
        Ready to collaborate on data-driven solutions? Let's discuss how I can help transform your business through analytics.
      </p>
      
      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mail className="w-5 h-5 text-blue-600" />
              Email
            </CardTitle>
          </CardHeader>
          <CardContent>
            <a 
              href="mailto:adityabajaria21@gmail.com" 
              className="text-blue-600 hover:underline text-lg"
            >
              adityabajaria21@gmail.com
            </a>
          </CardContent>
        </Card>
        
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Phone className="w-5 h-5 text-blue-600" />
              Phone
            </CardTitle>
          </CardHeader>
          <CardContent>
            <a 
              href="tel:+447587478594" 
              className="text-blue-600 hover:underline text-lg"
            >
              +44 7587478594
            </a>
          </CardContent>
        </Card>
        
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Linkedin className="w-5 h-5 text-blue-600" />
              LinkedIn
            </CardTitle>
          </CardHeader>
          <CardContent>
            <a 
              href="https://www.linkedin.com/in/adityabajaria/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline text-lg"
            >
              LinkedIn Profile
            </a>
          </CardContent>
        </Card>
        
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-blue-600" />
              Location
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 text-lg">London, United Kingdom</p>
          </CardContent>
        </Card>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Button 
          size="lg"
          className="bg-blue-600 hover:bg-blue-700 text-white"
          onClick={() => window.open('mailto:adityabajaria21@gmail.com', '_blank')}
        >
          <Mail className="w-5 h-5 mr-2" />
          Send Email
        </Button>
        <Button 
          size="lg"
          variant="outline"
          onClick={handleResumeDownload}
        >
          <Download className="w-5 h-5 mr-2" />
          Download Resume
        </Button>
      </div>
    </div>
  );
}