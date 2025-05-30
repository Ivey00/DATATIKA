"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useToast } from "@/components/ui/use-toast";
import PageLayout from "@/components/layout/PageLayout";

// Define types for our data structures
interface ImageDimensions {
  width: number;
  height: number;
  channels: number;
}

interface DatasetInfo {
  extractedPath: string;
  classes: string[];
  imageCounts: Record<string, number>;
  totalImages: number;
}

interface SampledDatasetInfo {
  sampledCounts: Record<string, number>;
  totalSampled: number;
}

interface AlgorithmInfo {
  name: string;
  description: string;
  hyperparameters: Record<string, {
    description: string;
    suggested_values: any[];
  }>;
}

interface ModelEvaluation {
  metrics: Record<string, number>;
  classificationReport: Record<string, Record<string, number>>;
  confusionMatrix: number[][];
  visualizations: Record<string, string>;
}

interface Prediction {
  prediction: string;
  confidence?: number;
}

// Main component
export default function ImageClassification() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("dimensions");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  
  // State for each step
  const [dimensions, setDimensions] = useState<ImageDimensions>({
    width: 150,
    height: 150,
    channels: 3
  });
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [sampledDatasetInfo, setSampledDatasetInfo] = useState<SampledDatasetInfo | null>(null);
  const [samplesPerClass, setSamplesPerClass] = useState<number>(1000);
  const [visualizationData, setVisualizationData] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [algorithmInfo, setAlgorithmInfo] = useState<AlgorithmInfo | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [modelTrained, setModelTrained] = useState(false);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [testImage, setTestImage] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  
  // API base URL
  const API_BASE_URL = 'http://localhost:8000';
  
  // Function to fetch evaluation results
  const fetchEvaluationResults = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/evaluate-model`);
      const data = await response.json();
      
      if (data.success) {
        setEvaluation({
          metrics: data.metrics,
          classificationReport: data.classification_report,
          confusionMatrix: data.confusion_matrix,
          visualizations: data.visualizations
        });
      } else {
        toast({
          title: "Error fetching evaluation results",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error fetching evaluation results:", error);
      toast({
        title: "Error fetching evaluation results",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Function to handle setting image dimensions
  const handleSetDimensions = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/set-dimensions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          width: dimensions.width,
          height: dimensions.height,
          channels: dimensions.channels
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        toast({
          title: "Dimensions set successfully",
          description: `Set to ${dimensions.width}x${dimensions.height}x${dimensions.channels}`,
        });
        
        // Move to the next tab
        setActiveTab("upload");
      } else {
        toast({
          title: "Error setting dimensions",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error setting dimensions:", error);
      toast({
        title: "Error setting dimensions",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Check file type
    if (!file.name.endsWith('.zip')) {
      toast({
        title: "Invalid file type",
        description: "Please upload a ZIP file containing your dataset",
        variant: "destructive"
      });
      return;
    }
    
    setDatasetFile(file);
    setLoading(true);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Upload the file
      const response = await fetch(`${API_BASE_URL}/api/image-classification/upload-dataset`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setDatasetInfo({
          extractedPath: data.extracted_path,
          classes: data.classes,
          imageCounts: data.image_counts,
          totalImages: data.total_images
        });
        
        toast({
          title: "Dataset uploaded successfully",
          description: `Found ${data.classes.length} classes with ${data.total_images} total images`,
        });
        
        // Fetch available models
        await fetchAvailableModels();
        
        // Move to the next tab
        setActiveTab("sample");
      } else {
        toast({
          title: "Error uploading dataset",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error uploading dataset:", error);
      toast({
        title: "Error uploading dataset",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to fetch available models
  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/get-models`);
      const data = await response.json();
      
      if (data.success) {
        setAvailableModels(data.available_models);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  };
  
  // Function to handle dataset sampling
  const handleSampleDataset = async () => {
    if (!datasetInfo) return;
    
    setLoading(true);
    try {
      // Prepare source and target directories
      const sourceDirectories = datasetInfo.classes.map(className => 
        `${datasetInfo.extractedPath}/${className}`
      );
      
      const targetDirectories = datasetInfo.classes.map(className => 
        `${datasetInfo.extractedPath}/${className}_sampled`
      );
      
      const response = await fetch(`${API_BASE_URL}/api/image-classification/sample-dataset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          source_directories: sourceDirectories,
          target_directories: targetDirectories,
          samples_per_class: samplesPerClass
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setSampledDatasetInfo({
          sampledCounts: data.sampled_counts,
          totalSampled: data.total_sampled
        });
        
        toast({
          title: "Dataset sampled successfully",
          description: `Sampled ${data.total_sampled} images across ${datasetInfo.classes.length} classes`,
        });
      } else {
        toast({
          title: "Error sampling dataset",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error sampling dataset:", error);
      toast({
        title: "Error sampling dataset",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle data loading with polling
  const handleLoadDataset = async () => {
    if (!datasetInfo) return;
    
    setLoading(true);
    setProgress(0);
    setProgressMessage("Wait some minutes for load & preprocess your data...");
    
    try {
      // Start the data loading process
      const formData = new FormData();
      formData.append('data_dir', datasetInfo.extractedPath);
      datasetInfo.classes.forEach(category => {
        formData.append('categories', category);
      });
      
      const response = await fetch(`${API_BASE_URL}/api/image-classification/load-dataset`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.message);
      }
      
      // Start polling for progress
      let retryCount = 0;
      const maxRetries = 3;
      let pollTimeout: NodeJS.Timeout | null = null;
      
      const pollProgress = async () => {
        try {
          const progressResponse = await fetch(`${API_BASE_URL}/api/image-classification/check-progress`, {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          });
          
          if (!progressResponse.ok) {
            throw new Error(`HTTP error! status: ${progressResponse.status}`);
          }
          
          const progressData = await progressResponse.json();
          retryCount = 0; // Reset retry count on successful request
          
          setProgress(progressData.progress);
          setProgressMessage(progressData.message);
          
          if (progressData.error) {
            throw new Error(progressData.error);
          }
          
          if (!progressData.is_complete && progressData.task_running) {
            // Continue polling every second
            pollTimeout = setTimeout(pollProgress, 1000);
          } else {
            setLoading(false);
            
            if (progressData.progress === 100) {
              toast({
                title: "Dataset loaded successfully",
                description: "Dataset has been loaded and preprocessed",
              });
              
              // Move to the next tab
              setActiveTab("visualize");
            }
          }
        } catch (error) {
          console.error("Error checking progress:", error);
          
          // Retry logic
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`Retrying progress check (${retryCount}/${maxRetries})...`);
            pollTimeout = setTimeout(pollProgress, 2000); // Wait 2 seconds before retrying
          } else {
            setLoading(false);
        toast({
          title: "Error loading dataset",
              description: "Lost connection to server. The process may still be running in the background.",
          variant: "destructive"
        });
      }
        }
      };
      
      // Start polling
      pollProgress();
      
      // Cleanup function
      return () => {
        if (pollTimeout) {
          clearTimeout(pollTimeout);
        }
      };
      
    } catch (error) {
      console.error("Error loading dataset:", error);
      setLoading(false);
      
      toast({
        title: "Error loading dataset",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Function to handle data visualization
  const handleVisualizeData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/visualize-data`);
      const data = await response.json();
      
      if (data.success) {
        setVisualizationData(data.visualizations.pca_visualization || "");
        
        toast({
          title: "Data visualized successfully",
          description: "PCA visualization generated",
        });
        
        // Move to the next tab
        setActiveTab("model");
      } else {
        toast({
          title: "Error visualizing data",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error visualizing data:", error);
      toast({
        title: "Error visualizing data",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle model selection
  const handleModelSelection = async (model: string) => {
    setSelectedModel(model);
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/select-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm_name: model
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAlgorithmInfo({
          name: data.algorithm_name,
          description: data.algorithm_description,
          hyperparameters: data.hyperparameters
        });
        
        // Initialize hyperparameters with default values
        const initialHyperparams: Record<string, any> = {};
        Object.entries(data.hyperparameters).forEach(([key, value]: [string, any]) => {
          if (value.suggested_values && value.suggested_values.length > 0) {
            initialHyperparams[key] = value.suggested_values[0];
          }
        });
        
        setHyperparameters(initialHyperparams);
        
        toast({
          title: "Model selected",
          description: `Selected ${data.algorithm_name}`,
        });
      } else {
        toast({
          title: "Error selecting model",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error selecting model:", error);
      toast({
        title: "Error selecting model",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle hyperparameter configuration
  const handleConfigureModel = async () => {
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/image-classification/configure-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          hyperparameters: hyperparameters
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        toast({
          title: "Model configured",
          description: "Hyperparameters configured successfully",
        });
        
        // Move to the next tab
        setActiveTab("train");
      } else {
        toast({
          title: "Error configuring model",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error configuring model:", error);
      toast({
        title: "Error configuring model",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle model training
  const handleTrainModel = async () => {
    setLoading(true);
    setProgress(0);
    setProgressMessage("Wait some minutes to train your model...");
    
    try {
      // Prepare form data
      const formData = new FormData();
      formData.append('model_name', selectedModel);
      
      // Start the training process
      const response = await fetch(`${API_BASE_URL}/api/image-classification/train-model`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.message);
      }
      
      // Set up polling for progress updates
      let retryCount = 0;
      const maxRetries = 3;
      let pollTimeout: NodeJS.Timeout | null = null;
      
      const pollProgress = async () => {
        try {
          const progressResponse = await fetch(`${API_BASE_URL}/api/image-classification/check-progress`, {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          });
          
          if (!progressResponse.ok) {
            throw new Error(`HTTP error! status: ${progressResponse.status}`);
          }
          
          const progressData = await progressResponse.json();
          retryCount = 0; // Reset retry count on successful request
          
          setProgress(progressData.progress);
          setProgressMessage(progressData.message);
          
          if (progressData.error) {
            throw new Error(progressData.error);
          }
          
          if (!progressData.is_complete && progressData.task_running) {
            // Continue polling every second
            pollTimeout = setTimeout(pollProgress, 1000);
          } else {
            setLoading(false);
            
            if (progressData.progress === 100) {
              // Set model as trained
              setModelTrained(true);
              
              toast({
                title: "Model trained successfully",
                description: progressData.message || "Model has been trained and evaluated",
              });
              
              // Fetch evaluation results
              await fetchEvaluationResults();
              
              // Move to the next tab
              setActiveTab("evaluate");
            }
          }
        } catch (error) {
          console.error("Error checking progress:", error);
          
          // Retry logic
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`Retrying progress check (${retryCount}/${maxRetries})...`);
            pollTimeout = setTimeout(pollProgress, 2000); // Wait 2 seconds before retrying
          } else {
            setLoading(false);
            toast({
              title: "Error training model",
              description: "Lost connection to server. The process may still be running in the background.",
              variant: "destructive"
            });
          }
        }
      };
      
      // Start polling
      pollProgress();
      
      // Cleanup function
      return () => {
        if (pollTimeout) {
          clearTimeout(pollTimeout);
        }
      };
      
    } catch (error) {
      console.error("Error training model:", error);
      setLoading(false);
      
      toast({
        title: "Error training model",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Function to handle test image upload
  const handleTestImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Check file type
    const fileExt = file.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'].includes(fileExt || '')) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPG, PNG, BMP, etc.)",
        variant: "destructive"
      });
      return;
    }
    
    setTestImage(file);
  };
  
  // Function to handle image prediction
  const handlePredict = async () => {
    if (!testImage || !selectedModel) {
      toast({
        title: "Missing information",
        description: "Please select a test image and ensure a model is trained",
        variant: "destructive"
      });
      return;
    }
    
    setLoading(true);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', testImage);
      formData.append('model_name', selectedModel);
      
      // Make prediction
      const response = await fetch(`${API_BASE_URL}/api/image-classification/predict-image`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setPrediction({
          prediction: data.prediction,
          confidence: data.confidence
        });
        
        toast({
          title: "Prediction complete",
          description: `Predicted class: ${data.prediction}`,
        });
      } else {
        toast({
          title: "Error making prediction",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error making prediction:", error);
      toast({
        title: "Error making prediction",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle hyperparameter change
  const handleHyperparameterChange = (param: string, value: any) => {
    setHyperparameters(prev => ({
      ...prev,
      [param]: value
    }));
  };
  
  // Function to format hyperparameter value for display
  const formatHyperparamValue = (value: any): string => {
    if (value === null) return "None";
    if (typeof value === "boolean") return value ? "True" : "False";
    return value.toString();
  };
  
  // Function to render the dimensions setup step
  const renderDimensionsSetup = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Set Image Dimensions</CardTitle>
        <CardDescription className="text-foreground/80">
          Configure the dimensions for processing your images
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="width">Width (pixels)</Label>
            <Input 
              id="width" 
              type="number" 
              value={dimensions.width}
              onChange={(e) => setDimensions({...dimensions, width: parseInt(e.target.value)})}
              className="border-tertiary"
            />
          </div>
          
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="height">Height (pixels)</Label>
            <Input 
              id="height" 
              type="number" 
              value={dimensions.height}
              onChange={(e) => setDimensions({...dimensions, height: parseInt(e.target.value)})}
              className="border-tertiary"
            />
          </div>
          
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="channels">Channels</Label>
            <Select 
              value={dimensions.channels.toString()} 
              onValueChange={(value) => setDimensions({...dimensions, channels: parseInt(value)})}
            >
              <SelectTrigger id="channels" className="border-tertiary">
                <SelectValue placeholder="Select channels" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 (Grayscale)</SelectItem>
                <SelectItem value="3">3 (RGB)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-sm text-foreground/70">
              Number of color channels (1 for grayscale, 3 for RGB)
            </p>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" disabled className="border-tertiary">Back</Button>
        <Button 
          onClick={handleSetDimensions} 
          disabled={!dimensions.width || !dimensions.height || loading}
          variant="secondary"
        >
          Next: Upload Dataset
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the dataset upload step
  const renderDatasetUpload = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Upload Dataset</CardTitle>
        <CardDescription className="text-foreground/80">
          Upload a ZIP file containing your image dataset organized in class folders
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="datasetFile">Dataset ZIP File</Label>
            <Input 
              id="datasetFile" 
              type="file" 
              accept=".zip"
              onChange={handleFileUpload}
              disabled={loading}
              className="border-tertiary"
            />
            <p className="text-sm text-foreground/70">
              Upload a ZIP file containing folders named after each class (e.g., "Defected", "Non-Defected"), with images inside each folder
            </p>
          </div>
          
          {datasetInfo && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-secondary">Dataset Information</h3>
              <div className="mt-2 space-y-2">
                <p className="text-sm text-foreground/90">
                  <span className="font-medium">Classes:</span> {datasetInfo.classes.join(", ")}
                </p>
                <p className="text-sm text-foreground/90">
                  <span className="font-medium">Total Images:</span> {datasetInfo.totalImages}
                </p>
                <h4 className="text-md font-medium text-foreground mt-2">Images per Class:</h4>
                <div className="border border-tertiary rounded-md p-3">
                  {Object.entries(datasetInfo.imageCounts).map(([className, count]) => (
                    <div key={className} className="flex justify-between text-sm">
                      <span className="text-foreground/80">{className}:</span>
                      <span className="font-medium text-foreground">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("dimensions")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setActiveTab("sample")} 
          disabled={!datasetInfo || loading}
          variant="secondary"
        >
          Next: Sample Dataset
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the dataset sampling step
  const renderDatasetSampling = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Sample Dataset</CardTitle>
        <CardDescription className="text-foreground/80">
          Sample a subset of images from each class to balance the dataset
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="samplesPerClass">Samples per Class</Label>
            <Input 
              id="samplesPerClass" 
              type="number" 
              value={samplesPerClass}
              onChange={(e) => setSamplesPerClass(parseInt(e.target.value))}
              className="border-tertiary"
            />
            <p className="text-sm text-foreground/70">
              Maximum number of images to sample from each class. If a class has fewer images, all will be used.
            </p>
          </div>
          
          {sampledDatasetInfo && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-secondary">Sampling Results</h3>
              <div className="mt-2 space-y-2">
                <p className="text-sm text-foreground/90">
                  <span className="font-medium">Total Sampled Images:</span> {sampledDatasetInfo.totalSampled}
                </p>
                <h4 className="text-md font-medium text-foreground mt-2">Images Sampled per Class:</h4>
                <div className="border border-tertiary rounded-md p-3">
                  {Object.entries(sampledDatasetInfo.sampledCounts).map(([className, count]) => (
                    <div key={className} className="flex justify-between text-sm">
                      <span className="text-foreground/80">{className}:</span>
                      <span className="font-medium text-foreground">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          
          {loading && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2 bg-tertiary/30" />
              <p className="text-sm text-center text-foreground/80">
                {progressMessage || "Processing..."}
              </p>
        </div>
          )}
          
          <div className="flex space-x-4">
          <Button 
            onClick={handleSampleDataset} 
            disabled={!datasetInfo || loading}
            variant="default"
          >
            Sample Dataset
            </Button>
            
            <Button 
              onClick={handleLoadDataset} 
              disabled={!sampledDatasetInfo || loading}
              variant="secondary"
            >
              Load & Preprocess Data
            </Button>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("upload")} className="border-tertiary">
          Back
          </Button>
          <Button 
            onClick={() => setActiveTab("visualize")} 
          disabled={!sampledDatasetInfo || loading}
            variant="secondary"
          >
            Next: Visualize Data
          </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the data visualization step
  const renderDataVisualization = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Visualize Data</CardTitle>
        <CardDescription className="text-foreground/80">
          Visualize your dataset using Principal Component Analysis (PCA)
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <p className="text-sm text-foreground/90">
            PCA reduces the high-dimensional image data to 2D for visualization, helping you understand the separability of your classes.
          </p>
          
          <Button 
            onClick={handleVisualizeData} 
            disabled={loading || !sampledDatasetInfo}
            variant="default"
          >
            Generate PCA Visualization
          </Button>
          
          {visualizationData && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-secondary">PCA Visualization</h3>
              <div className="mt-2 border border-tertiary rounded-md p-4">
                <img 
                  src={`data:image/png;base64,${visualizationData}`} 
                  alt="PCA Visualization" 
                  className="w-full"
                />
              </div>
              <p className="text-sm text-foreground/70 mt-2">
                Each point represents an image. Different colors represent different classes.
              </p>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("sample")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setActiveTab("model")} 
          disabled={!visualizationData}
          variant="secondary"
        >
          Next: Select Model
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the model selection step
  const renderModelSelection = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Select Model</CardTitle>
        <CardDescription className="text-foreground/80">
          Choose a machine learning model for image classification
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div>
            <Label htmlFor="model">Classification Model</Label>
            <Select 
              value={selectedModel} 
              onValueChange={handleModelSelection}
            >
              <SelectTrigger id="model" className="border-tertiary">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map(model => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {algorithmInfo && (
            <>
              <div className="p-4 border border-tertiary rounded-md bg-tertiary/10">
                <h3 className="text-lg font-medium text-secondary mb-2">
                  {algorithmInfo.name}
                </h3>
                <p className="text-sm whitespace-pre-line text-foreground/90">
                  {algorithmInfo.description}
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium text-secondary mb-2">Hyperparameters</h3>
                <div className="space-y-4">
                  {Object.entries(algorithmInfo.hyperparameters).map(([param, info]) => (
                    <div key={param} className="space-y-2">
                      <Label htmlFor={`param-${param}`}>
                        {param}
                      </Label>
                      
                      <Select 
                        value={hyperparameters[param]?.toString() ?? ""}
                        onValueChange={(value: string) => {
                          // Convert value to appropriate type
                          let typedValue: any = value;
                          if (value === "null" || value === "None") typedValue = null;
                          else if (value === "true" || value === "True") typedValue = true;
                          else if (value === "false" || value === "False") typedValue = false;
                          else if (!isNaN(Number(value))) typedValue = Number(value);
                          
                          handleHyperparameterChange(param, typedValue);
                        }}
                      >
                        <SelectTrigger id={`param-${param}`} className="border-tertiary">
                          <SelectValue placeholder={`Select ${param}`} />
                        </SelectTrigger>
                        <SelectContent>
                          {info.suggested_values.map((value: any) => (
                            <SelectItem 
                              key={`${param}-${value}`} 
                              value={value === null ? "null" : value.toString()}
                            >
                              {formatHyperparamValue(value)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      
                      <p className="text-xs text-foreground/70">
                        {info.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("visualize")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={handleConfigureModel} 
          disabled={!selectedModel || !algorithmInfo || loading}
          variant="secondary"
        >
          Next: Train Model
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the model training step
  const renderModelTraining = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Train Model</CardTitle>
        <CardDescription className="text-foreground/80">
          Train your selected model on the dataset
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="p-4 border border-tertiary rounded-md bg-tertiary/10">
            <h3 className="text-lg font-medium text-secondary mb-2">Training Configuration</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-foreground/80">Model:</span>
                <span className="font-medium text-foreground">{selectedModel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Image Dimensions:</span>
                <span className="font-medium text-foreground">{dimensions.width}x{dimensions.height}x{dimensions.channels}</span>
              </div>
              
              <Separator className="bg-tertiary/50" />
              
              <h4 className="text-sm font-medium text-foreground">Hyperparameters:</h4>
              {Object.entries(hyperparameters).map(([param, value]) => (
                <div key={param} className="flex justify-between text-sm">
                  <span className="text-foreground/80">{param}:</span>
                  <span className="font-medium text-foreground">{formatHyperparamValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
          
          {loading && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2 bg-tertiary/30" />
              <p className="text-sm text-center text-foreground/80">
                {progressMessage || "Training..."}
              </p>
            </div>
          )}
          
          {modelTrained && (
            <Alert className="bg-primary/10 border-primary">
              <AlertTitle className="text-primary">Model Trained Successfully</AlertTitle>
              <AlertDescription className="text-foreground/90">
                Your model has been trained and is ready for evaluation and predictions.
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("model")} className="border-tertiary">
          Back
        </Button>
        <div className="space-x-2">
          <Button 
            onClick={handleTrainModel} 
            disabled={loading || !selectedModel}
            variant="default"
          >
            {modelTrained ? "Retrain Model" : "Train Model"}
          </Button>
          <Button 
            onClick={() => setActiveTab("evaluate")} 
            disabled={!modelTrained}
            variant="secondary"
          >
            Next: Evaluate Model
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
  
  // Function to render the model evaluation step
  const renderModelEvaluation = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Model Evaluation</CardTitle>
        <CardDescription className="text-foreground/80">
          Review the performance of your trained model
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {evaluation && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-primary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-primary">Accuracy</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {(evaluation.metrics.accuracy * 100).toFixed(2)}%
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-secondary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-secondary">F1 Score</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {(evaluation.metrics.f1 * 100).toFixed(2)}%
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-tertiary/20 border-b border-tertiary">
                  <CardTitle className="text-lg text-foreground">Precision</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {(evaluation.metrics.precision * 100).toFixed(2)}%
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-tertiary/20 border-b border-tertiary">
                  <CardTitle className="text-lg text-foreground">Recall</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {(evaluation.metrics.recall * 100).toFixed(2)}%
                  </p>
                </CardContent>
              </Card>
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Confusion Matrix</h3>
              {evaluation.visualizations.confusion_matrix && (
                <img 
                  src={`data:image/png;base64,${evaluation.visualizations.confusion_matrix}`} 
                  alt="Confusion Matrix" 
                  className="w-full max-w-md mx-auto"
                />
              )}
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Classification Report</h3>
              <div className="border border-tertiary rounded-md overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="bg-tertiary/20">
                      <th className="p-2 text-left text-foreground">Class</th>
                      <th className="p-2 text-left text-foreground">Precision</th>
                      <th className="p-2 text-left text-foreground">Recall</th>
                      <th className="p-2 text-left text-foreground">F1-Score</th>
                      <th className="p-2 text-left text-foreground">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(evaluation.classificationReport)
                      .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                      .map(([className, metrics]) => (
                        <tr key={className} className="border-t border-tertiary/50">
                          <td className="p-2 text-foreground">{className}</td>
                          <td className="p-2 text-foreground">{(metrics.precision * 100).toFixed(2)}%</td>
                          <td className="p-2 text-foreground">{(metrics.recall * 100).toFixed(2)}%</td>
                          <td className="p-2 text-foreground">{(metrics.f1_score * 100).toFixed(2)}%</td>
                          <td className="p-2 text-foreground">{metrics.support}</td>
                        </tr>
                      ))
                    }
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("train")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setActiveTab("predict")} 
          disabled={!evaluation}
          variant="secondary"
        >
          Next: Test Single Image
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the prediction step
  const renderPrediction = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Test Single Image</CardTitle>
        <CardDescription className="text-foreground/80">
          Test your trained model on a single image
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="testImage">Test Image</Label>
            <Input 
              id="testImage" 
              type="file" 
              accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
              onChange={handleTestImageUpload}
              disabled={loading}
              className="border-tertiary"
            />
            <p className="text-sm text-foreground/70">
              Upload a single image to test the model
            </p>
          </div>
          
          <Button 
            onClick={handlePredict} 
            disabled={!testImage || loading || !modelTrained}
            variant="default"
          >
            Make Prediction
          </Button>
          
          {prediction && (
            <div className="mt-4 p-4 border border-tertiary rounded-md bg-tertiary/10">
              <h3 className="text-lg font-medium text-secondary mb-2">Prediction Result</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-foreground/80">Predicted Class:</span>
                  <span className="font-medium text-foreground">{prediction.prediction}</span>
                </div>
                
                {prediction.confidence !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-foreground/80">Confidence:</span>
                    <span className="font-medium text-foreground">{(prediction.confidence * 100).toFixed(2)}%</span>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {testImage && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-secondary mb-2">Test Image</h3>
              <div className="border border-tertiary rounded-md p-4">
                <img 
                  src={URL.createObjectURL(testImage)} 
                  alt="Test Image" 
                  className="max-h-64 mx-auto"
                />
              </div>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("evaluate")} className="border-tertiary">
          Back
        </Button>
        <Button variant="secondary" disabled>
          Save Model
        </Button>
      </CardFooter>
    </Card>
  );
  
  return (
    <PageLayout>
      <div className="container mx-auto py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold"><span className="text-primary">Image</span> <span className="text-secondary">Classification</span></h1>
          <p className="text-foreground">
            Train machine learning models for image classification on industrial data
          </p>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-7 border border-tertiary">
            <TabsTrigger value="dimensions" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">1. Set Dimensions</TabsTrigger>
            <TabsTrigger value="upload" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">2. Upload Dataset</TabsTrigger>
            <TabsTrigger value="sample" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">3. Sample Data</TabsTrigger>
            <TabsTrigger value="visualize" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">4. Visualize</TabsTrigger>
            <TabsTrigger value="model" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">5. Select Model</TabsTrigger>
            <TabsTrigger value="train" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">6. Train Model</TabsTrigger>
            <TabsTrigger value="evaluate" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">7. Evaluate & Test</TabsTrigger>
          </TabsList>
          
          <div className="mt-6">
            <TabsContent value="dimensions">
              {renderDimensionsSetup()}
            </TabsContent>
            
            <TabsContent value="upload">
              {renderDatasetUpload()}
            </TabsContent>
            
            <TabsContent value="sample">
              {renderDatasetSampling()}
            </TabsContent>
            
            <TabsContent value="visualize">
              {renderDataVisualization()}
            </TabsContent>
            
            <TabsContent value="model">
              {renderModelSelection()}
            </TabsContent>
            
            <TabsContent value="train">
              {renderModelTraining()}
            </TabsContent>
            
            <TabsContent value="evaluate">
              {renderModelEvaluation()}
            </TabsContent>
            
            <TabsContent value="predict">
              {renderPrediction()}
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </PageLayout>
  );
}