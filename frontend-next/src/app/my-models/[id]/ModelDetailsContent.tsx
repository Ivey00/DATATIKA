"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import PageLayout from '@/components/layout/PageLayout';
import { Database, ArrowLeft, Play } from 'lucide-react';

interface TrainedModel {
  id: number;
  name: string;
  model_type: string;
  dataset_name: string;
  metrics: Record<string, any>;
  hyperparameters: Record<string, any>;
  created_at: string;
}

interface ClassificationReportRow {
  class_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

function parseClassificationReport(report: string): {
  classes: ClassificationReportRow[];
  accuracy: number | null;
  macro_avg: ClassificationReportRow | null;
  weighted_avg: ClassificationReportRow | null;
} {
  try {
    const lines = report.split('\n').filter(line => line.trim());
    const classes: ClassificationReportRow[] = [];
    let accuracy: number | null = null;
    let macro_avg: ClassificationReportRow | null = null;
    let weighted_avg: ClassificationReportRow | null = null;

    lines.forEach(line => {
      const parts = line.split(/\s+/).filter(Boolean);
      
      if (parts.includes('accuracy')) {
        accuracy = parseFloat(parts[parts.length - 2]);
      } else if (line.includes('macro avg')) {
        macro_avg = {
          class_name: 'Macro Average',
          precision: parseFloat(parts[parts.length - 4]),
          recall: parseFloat(parts[parts.length - 3]),
          f1_score: parseFloat(parts[parts.length - 2]),
          support: parseInt(parts[parts.length - 1])
        };
      } else if (line.includes('weighted avg')) {
        weighted_avg = {
          class_name: 'Weighted Average',
          precision: parseFloat(parts[parts.length - 4]),
          recall: parseFloat(parts[parts.length - 3]),
          f1_score: parseFloat(parts[parts.length - 2]),
          support: parseInt(parts[parts.length - 1])
        };
      } else if (parts.length >= 5 && !line.includes('precision')) {
        // Handle class names that might contain spaces
        const supportIndex = parts.length - 1;
        const f1ScoreIndex = parts.length - 2;
        const recallIndex = parts.length - 3;
        const precisionIndex = parts.length - 4;
        const className = parts.slice(0, precisionIndex).join(' ');

        classes.push({
          class_name: className,
          precision: parseFloat(parts[precisionIndex]),
          recall: parseFloat(parts[recallIndex]),
          f1_score: parseFloat(parts[f1ScoreIndex]),
          support: parseInt(parts[supportIndex])
        });
      }
    });

    // Validate parsed data
    const isValid = 
      classes.length > 0 && 
      accuracy !== null && 
      macro_avg !== null && 
      weighted_avg !== null &&
      !isNaN(accuracy) &&
      typeof (macro_avg as ClassificationReportRow)?.precision === 'number' &&
      typeof (weighted_avg as ClassificationReportRow)?.precision === 'number';

    if (!isValid) {
      throw new Error('Invalid report format');
    }

    return {
      classes,
      accuracy,
      macro_avg,
      weighted_avg
    };
  } catch (error) {
    console.error('Error parsing classification report:', error);
    return {
      classes: [],
      accuracy: null,
      macro_avg: null,
      weighted_avg: null
    };
  }
}

function ClassificationReportTable({ report }: { report: string }) {
  const parsedReport = parseClassificationReport(report);

  if (parsedReport.classes.length === 0) {
    return (
      <div className="text-muted-foreground italic">
        Unable to parse classification report
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-muted">
            <th className="p-2 text-left border">Class</th>
            <th className="p-2 text-left border">Precision</th>
            <th className="p-2 text-left border">Recall</th>
            <th className="p-2 text-left border">F1 Score</th>
            <th className="p-2 text-left border">Support</th>
          </tr>
        </thead>
        <tbody>
          {parsedReport.classes.map((row, index) => (
            <tr key={index} className="hover:bg-muted/50">
              <td className="p-2 border font-medium">{row.class_name}</td>
              <td className="p-2 border">{row.precision.toFixed(3)}</td>
              <td className="p-2 border">{row.recall.toFixed(3)}</td>
              <td className="p-2 border">{row.f1_score.toFixed(3)}</td>
              <td className="p-2 border">{row.support}</td>
            </tr>
          ))}
          {parsedReport.accuracy !== null && (
            <tr className="bg-muted/30">
              <td colSpan={5} className="p-2 border font-medium">
                Accuracy: {parsedReport.accuracy.toFixed(3)}
              </td>
            </tr>
          )}
          {parsedReport.macro_avg && (
            <tr className="bg-muted/20">
              <td className="p-2 border font-medium">{parsedReport.macro_avg.class_name}</td>
              <td className="p-2 border">{parsedReport.macro_avg.precision.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.macro_avg.recall.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.macro_avg.f1_score.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.macro_avg.support}</td>
            </tr>
          )}
          {parsedReport.weighted_avg && (
            <tr className="bg-muted/20">
              <td className="p-2 border font-medium">{parsedReport.weighted_avg.class_name}</td>
              <td className="p-2 border">{parsedReport.weighted_avg.precision.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.weighted_avg.recall.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.weighted_avg.f1_score.toFixed(3)}</td>
              <td className="p-2 border">{parsedReport.weighted_avg.support}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelDetailsContent({ modelId }: { modelId: string }) {
  const router = useRouter();
  const [model, setModel] = useState<TrainedModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/auth/my-models/${modelId}`, {
          credentials: 'include',
        });
        
        if (!response.ok) {
          if (response.status === 401) {
            router.push('/signin');
            return;
          }
          throw new Error('Failed to fetch model details');
        }
        
        const data = await response.json();
        setModel(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchModel();
  }, [modelId, router]);

  const getPredictionPath = (modelType: string) => {
    const paths = {
      'target_prediction': '/my-models',
      'numerical_classification': '/my-models',
      'time_series': '/my-models',
      'anomaly_detection': '/my-models',
      'image_classification': '/my-models'
    };
    return `${paths[modelType as keyof typeof paths]}/${modelId}/predict`;
  };

  const formatMetricValue = (key: string, value: any) => {
    if (key === 'classification_report' && typeof value === 'string') {
      return <ClassificationReportTable report={value} />;
    }
    if (key === 'confusion_matrix' && Array.isArray(value)) {
      return (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <tbody>
              {value.map((row: number[], i: number) => (
                <tr key={i}>
                  {row.map((cell: number, j: number) => (
                    <td key={j} className="p-2 border text-center">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return JSON.stringify(value);
  };

  if (loading) {
    return (
      <PageLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Loading model details...</div>
        </div>
      </PageLayout>
    );
  }

  if (error || !model) {
    return (
      <PageLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-red-500">{error || 'Model not found'}</div>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-3 mb-8">
          <Button
            variant="ghost"
            onClick={() => router.back()}
            className="mr-2"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Database className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">{model.name}</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Information</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-4">
                <div>
                  <dt className="font-medium text-muted-foreground">Dataset</dt>
                  <dd>{model.dataset_name}</dd>
                </div>
                <div>
                  <dt className="font-medium text-muted-foreground">Type</dt>
                  <dd className="capitalize">{model.model_type.replace('_', ' ')}</dd>
                </div>
                <div>
                  <dt className="font-medium text-muted-foreground">Created</dt>
                  <dd>{new Date(model.created_at).toLocaleString()}</dd>
                </div>
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-4">
                {Object.entries(model.metrics).map(([key, value]) => (
                  <div key={key}>
                    <dt className="font-medium text-muted-foreground capitalize">
                      {key.replace('_', ' ')}
                    </dt>
                    <dd>{formatMetricValue(key, value)}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-4">
                {Object.entries(model.hyperparameters).map(([key, value]) => (
                  <div key={key}>
                    <dt className="font-medium text-muted-foreground capitalize">
                      {key.replace('_', ' ')}
                    </dt>
                    <dd>{JSON.stringify(value)}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
              <CardDescription>Use this model for predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                onClick={() => router.push(getPredictionPath(model.model_type))}
                className="w-full"
              >
                <Play className="h-4 w-4 mr-2" />
                Make Prediction
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </PageLayout>
  );
} 