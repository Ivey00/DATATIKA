import Link from 'next/link';
import { BarChart, ActivitySquare, Clock, Image, ArrowRight, LayoutDashboard } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import PageLayout from '@/components/layout/PageLayout';

export default function Home() {
  const features = [
    {
      title: 'Target Prediction',
      description: 'Predict specific values using supervised learning models with customizable features and algorithms.',
      icon: <BarChart className="h-10 w-10 text-primary" />,
      path: '/target-prediction',
    },
    {
      title: 'Anomaly Detection',
      description: 'Detect anomalies in your data using unsupervised learning with visualizations of normal vs anomalous data.',
      icon: <ActivitySquare className="h-10 w-10 text-secondary" />,
      path: '/anomaly-detection',
    },
    {
      title: 'Numerical Classifier',
      description: 'Classify tabular data into predefined categories with customizable features and model selection.',
      icon: <BarChart className="h-10 w-10 rotate-90 text-primary" />,
      path: '/numerical-classifier',
    },
    {
      title: 'Time Series Forecasting',
      description: 'Predict future values based on time series data with customizable time intervals and model parameters.',
      icon: <Clock className="h-10 w-10 text-secondary" />,
      path: '/time-series',
    },
    {
      title: 'Image Classification',
      description: 'Classify images into multiple categories using directory-based training with advanced preprocessing options.',
      icon: <Image className="h-10 w-10 text-primary" />,
      path: '/image-classification',
    },
  ];

  return (
    <PageLayout>
      <section className="py-12 md:py-20 bg-background">
        <div className="container px-4 mx-auto">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 animate-fade-in">
              <span className="text-primary">DATATIKA </span> <span className="text-secondary">  Model Builder </span>
            </h1>
            <p className="text-lg text-foreground mb-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
              Upload your data, select features, choose algorithms, and get accurate predictions 
              and classifications for your industrial applications.
            </p>
            <div className="flex flex-wrap justify-center gap-4 animate-fade-in" style={{ animationDelay: '0.2s' }}>
              <Link href="/target-prediction">
                <Button size="lg" variant="secondary">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card 
                key={feature.title} 
                className="datatika-card animate-fade-in border-tertiary hover:border-secondary transition-colors duration-300" 
                style={{ animationDelay: `${0.3 + index * 0.1}s` }}
              >
                <CardHeader>
                  <div className="mb-4">
                    {feature.icon}
                  </div>
                  <CardTitle className="text-foreground">{feature.title}</CardTitle>
                  <CardDescription className="text-foreground/80">{feature.description}</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Link href={feature.path} className="w-full">
                    <Button variant="outline" className="w-full border-tertiary hover:bg-secondary hover:text-secondary-foreground">
                      Explore {feature.title}
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </PageLayout>
  );
}