"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/components/ui/use-toast';
import PageLayout from '@/components/layout/PageLayout';
import { ArrowLeft, Upload, Webhook } from 'lucide-react';

interface Prediction {
  [key: string]: any;
}

export default function PredictContent({ modelId }: { modelId: string }) {
  const router = useRouter();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [webhookUrl, setWebhookUrl] = useState('');

  // Function to read file as base64
  const readFileAsBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          const base64String = reader.result.split(',')[1];
          resolve(base64String);
        } else {
          reject(new Error("Failed to read file as base64"));
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Check file type
    const fileExt = file.name.split('.').pop()?.toLowerCase();
    if (fileExt !== 'csv' && fileExt !== 'xlsx' && fileExt !== 'xls') {
      toast({
        title: "Invalid file type",
        description: "Please upload a CSV or Excel file",
        variant: "destructive"
      });
      return;
    }
    
    setFile(file);
  };

  const handlePrediction = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select a file to make predictions",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      // First read the file content
      const fileReader = new FileReader();
      const fileLoadPromise = new Promise((resolve, reject) => {
        fileReader.onload = () => resolve(fileReader.result);
        fileReader.onerror = () => reject(fileReader.error);
      });

      // Read the file as text
      fileReader.readAsText(file);
      
      // Wait for the file to be read
      const content = await fileLoadPromise as string;
      
      console.log("CSV Content preview:", content.substring(0, 200)); // Debug log
      
      const requestBody = {
        file_content: content,
        file_name: file.name
      };
      
      console.log("Sending request with body:", JSON.stringify(requestBody).substring(0, 200)); // Debug log
      
      const response = await fetch(`/api/numerical-classifier/predict-saved-model/${modelId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error response from server:", errorData); // Debug log
        throw new Error(errorData.detail || errorData.message || 'Failed to make predictions');
      }

      const data = await response.json();
      console.log("Server response:", data); // Debug log

      if (data.success) {
        setPredictions(data.predictions);
        toast({
          title: "Predictions made successfully",
          description: `Made ${data.prediction_count} predictions`
        });
      } else {
        toast({
          title: "Error making predictions",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error making predictions:", error);
      toast({
        title: "Error making predictions",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleWebhookSetup = () => {
    const webhookEndpoint = `${window.location.origin}/api/numerical-classifier/webhook/${modelId}`;
    navigator.clipboard.writeText(webhookEndpoint);
    toast({
      title: "Webhook URL copied",
      description: "The webhook URL has been copied to your clipboard"
    });
  };

  return (
    <PageLayout>
      <div className="container mx-auto py-6">
        <Button
          variant="outline"
          className="mb-6"
          onClick={() => router.back()}
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <Card>
          <CardHeader>
            <CardTitle>Make Predictions</CardTitle>
            <CardDescription>
              Upload new data or set up a webhook to make predictions with your trained model
            </CardDescription>
          </CardHeader>

          <CardContent>
            <Tabs defaultValue="upload">
              <TabsList>
                <TabsTrigger value="upload">Upload Data</TabsTrigger>
                <TabsTrigger value="webhook">Webhook</TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="space-y-4">
                <div className="grid w-full max-w-sm items-center gap-1.5">
                  <Label htmlFor="file">Upload Data File</Label>
                  <Input
                    id="file"
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileUpload}
                  />
                </div>

                <Button
                  onClick={handlePrediction}
                  disabled={!file || loading}
                  className="mt-4"
                >
                  <Upload className="mr-2 h-4 w-4" />
                  Make Predictions
                </Button>

                {predictions.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-2">Predictions</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr className="bg-muted">
                            {Object.keys(predictions[0]).map((key) => (
                              <th key={key} className="p-2 text-left border">
                                {key}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {predictions.map((prediction, index) => (
                            <tr key={index} className="hover:bg-muted/50">
                              {Object.values(prediction).map((value: any, i) => (
                                <td key={i} className="p-2 border">
                                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="webhook" className="space-y-4">
                <div className="space-y-4">
                  <div>
                    <Label>Webhook URL</Label>
                    <div className="flex items-center gap-2 mt-1">
                      <Input
                        value={`${window.location.origin}/api/numerical-classifier/webhook/${modelId}`}
                        readOnly
                      />
                      <Button onClick={handleWebhookSetup}>
                        <Webhook className="mr-2 h-4 w-4" />
                        Copy
                      </Button>
                    </div>
                  </div>

                  <div className="prose prose-sm">
                    <h4>How to use the webhook:</h4>
                    <ol className="list-decimal list-inside space-y-2">
                      <li>Copy the webhook URL above</li>
                      <li>Send a POST request to this URL with your data in JSON format</li>
                      <li>Include the required features in your request body</li>
                      <li>The response will contain the predictions</li>
                    </ol>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  );
} 