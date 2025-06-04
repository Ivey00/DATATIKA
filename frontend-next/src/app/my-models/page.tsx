"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import PageLayout from '@/components/layout/PageLayout';
import { Database, ArrowRight } from 'lucide-react';

interface TrainedModel {
  id: number;
  name: string;
  model_type: string;
  dataset_name: string;
  metrics: Record<string, any>;
  hyperparameters: Record<string, any>;
  created_at: string;
}

export default function MyModels() {
  const router = useRouter();
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/auth/my-models', {
          credentials: 'include',
        });
        
        if (!response.ok) {
          if (response.status === 401) {
            router.push('/signin');
            return;
          }
          throw new Error('Failed to fetch models');
        }
        
        const data = await response.json();
        setModels(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [router]);

  const modelTypes = {
    'target_prediction': 'Target Prediction Models',
    'numerical_classification': 'Numerical Classification Models',
    'time_series': 'Time Series Models',
    'anomaly_detection': 'Anomaly Detection Models',
    'image_classification': 'Image Classification Models'
  };

  const groupedModels = models.reduce((acc, model) => {
    const type = model.model_type;
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(model);
    return acc;
  }, {} as Record<string, TrainedModel[]>);

  if (loading) {
    return (
      <PageLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Loading your models...</div>
        </div>
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-red-500">{error}</div>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-3 mb-8">
          <Database className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">My Models</h1>
        </div>

        {Object.entries(modelTypes).map(([type, title]) => (
          <div key={type} className="mb-12">
            <h2 className="text-2xl font-semibold mb-6">{title}</h2>
            {groupedModels[type]?.length ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {groupedModels[type].map((model) => (
                  <Card key={model.id} className="hover:border-primary transition-colors">
                    <CardHeader>
                      <CardTitle className="text-lg">{model.name}</CardTitle>
                      <CardDescription>
                        Dataset: {model.dataset_name}
                        <br />
                        Created: {new Date(model.created_at).toLocaleDateString()}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-between items-center">
                        <Link href={`/my-models/${model.id}`}>
                          <Button variant="outline">
                            View Details
                            <ArrowRight className="ml-2 h-4 w-4" />
                          </Button>
                        </Link>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="py-8">
                  <p className="text-center text-muted-foreground">
                    No {title.toLowerCase()} found.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        ))}
      </div>
    </PageLayout>
  );
} 