"use client";

import { useState } from "react";
import { apiRequest } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useToast } from "@/components/ui/use-toast";
import PageLayout from "@/components/layout/PageLayout";

// Define types for our data structures
interface ColumnInfo {
  dtype: string;
  unique_values: number;
  missing_values: number;
  suggested_type: string;
}

interface DataInfo {
  shape: [number, number];
  columns: string[];
  sample: any[];
  columnInfo: Record<string, ColumnInfo>;
}

interface FeatureDefinition {
  datetime_column: string;
  target: string;
  additional_features: string[];
  categorical_features: string[];
  numerical_features: string[];
  item_id_column: string | null;
}

interface TimeUnitDefinition {
  timeUnit: string;
  forecastHorizon: number;
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
  visualizations: Record<string, string>;
}

interface Prediction {
  [key: string]: any;
}

interface ModelSaveRequest {
  model_name: string;
  dataset_name: string;
  save_directory: string;
  hyperparameters: { [key: string]: any };
  algorithm_name: string;
}

// Add retry mechanism for API requests
const apiRequestWithRetry = async (url: string, options?: RequestInit, maxRetries = 3) => {
  let lastError;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options?.headers,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      lastError = error;
      console.error(`Attempt ${attempt + 1} failed:`, error);
      
      // Only retry on network errors or 5xx server errors
      if (error instanceof Error && error.message.includes('fetch failed') ||
          error instanceof Error && error.message.includes('status: 5')) {
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        continue;
      }
      
      throw error;
    }
  }
  
  throw lastError;
};

// Main component
export default function TimeSeries() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("import");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [modelName, setModelName] = useState("");
  const [saveDirectory, setSaveDirectory] = useState("");
  
  // State for each step
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [featureDefinition, setFeatureDefinition] = useState<FeatureDefinition>({
    datetime_column: "",
    target: "",
    additional_features: [],
    categorical_features: [],
    numerical_features: [],
    item_id_column: null
  });
  const [timeUnitDefinition, setTimeUnitDefinition] = useState<TimeUnitDefinition>({
    timeUnit: "day",
    forecastHorizon: 7
  });
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("");
  const [algorithmInfo, setAlgorithmInfo] = useState<AlgorithmInfo | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [modelTrained, setModelTrained] = useState(false);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [futurePeriods, setFuturePeriods] = useState(7);
  const [visualizations, setVisualizations] = useState<Record<string, string>>({});
  const [plotType, setPlotType] = useState<string>("all");
  
  // Available algorithms
  const algorithms = [
    { value: "xgboost", label: "XGBoost" },
    { value: "lightgbm", label: "LightGBM" },
    { value: "random_forest", label: "Random Forest" },
    { value: "gradient_boosting", label: "Gradient Boosting" },
    { value: "prophet", label: "Prophet" }
  ];

  // Available time units
  const timeUnits = [
    { value: "hour", label: "Hour" },
    { value: "day", label: "Day" },
    { value: "month", label: "Month" }
  ];

  // Function to handle file upload
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
    
    setDataFile(file);
    
    // Read and upload the file
    setLoading(true);
    try {
      // Convert file to base64
      const fileContent = await readFileAsBase64(file);
      
      // Send to API using keepAlive agent
      const data = await apiRequest('/api/time-series/import-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_content: fileContent,
          file_name: file.name
        })
      });
      
      if (data.success) {
        // Process the data
        setDataInfo({
          shape: data.data_shape,
          columns: Object.keys(data.data_types),
          sample: data.data_sample,
          columnInfo: {} // Will be populated in the next step
        });
        
        // Get column info
        await fetchColumnInfo();
        
        toast({
          title: "Data imported successfully",
          description: `Imported ${data.data_shape[0]} rows and ${data.data_shape[1]} columns`,
        });
        
        // Move to the next tab
        setActiveTab("features");
      } else {
        toast({
          title: "Error importing data",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error importing data:", error);
      toast({
        title: "Error importing data",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to read file as base64
  const readFileAsBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Remove the data URL prefix (e.g., "data:application/octet-stream;base64,")
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
  
  // Function to fetch column info
  const fetchColumnInfo = async () => {
    try {
      const data = await apiRequestWithRetry('/api/time-series/column-info');
      
      if (data.column_info) {
        setDataInfo(prev => prev ? {
          ...prev,
          columnInfo: data.column_info
        } : null);
      }
    } catch (error) {
      console.error("Error fetching column info:", error);
      toast({
        title: "Error fetching column info",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
        variant: "destructive"
      });
    }
  };
  
  // Function to handle feature definition
  const handleFeatureDefinition = async () => {
    if (!featureDefinition.datetime_column || !featureDefinition.target) {
      toast({
        title: "Error",
        description: "Please select datetime and target columns.",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await apiRequest('/api/time-series/define-features', {
        method: 'POST',
        body: JSON.stringify({
          datetime_column: featureDefinition.datetime_column,
          target: featureDefinition.target,
          additional_features: featureDefinition.additional_features,
          categorical_features: featureDefinition.categorical_features,
          numerical_features: featureDefinition.numerical_features,
          item_id_column: featureDefinition.item_id_column
        })
      });

      if (response.success) {
        toast({
          title: "Success",
          description: "Features defined successfully."
        });
        setActiveTab("visualize");
      } else {
        toast({
          title: "Error",
          description: response.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to define features.",
        variant: "destructive"
      });
    }
  };
  
  // Function to handle time unit definition
  const handleTimeUnitDefinition = async () => {
    setLoading(true);
    try {
      const data = await apiRequest(
        `/api/time-series/define-time-unit?time_unit=${timeUnitDefinition.timeUnit}&forecast_horizon=${timeUnitDefinition.forecastHorizon}`,
        { method: 'POST' }
      );
      
      if (data.success) {
        toast({
          title: "Time unit defined successfully",
          description: `Time unit: ${data.time_unit}, Forecast horizon: ${data.forecast_horizon}`,
        });
        return true;
      } else {
        toast({
          title: "Error defining time unit",
          description: data.message,
          variant: "destructive"
        });
        return false;
      }
    } catch (error) {
      console.error("Error defining time unit:", error);
      toast({
        title: "Error defining time unit",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
      return false;
    } finally {
      setLoading(false);
    }
  };
  
  // Function to fetch visualizations
  const fetchVisualizations = async (type: string) => {
    setLoading(true);
    try {
      // Make sure features are defined
      await apiRequest('/api/time-series/define-features', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          features: featureDefinition.additional_features.concat(featureDefinition.categorical_features).concat(featureDefinition.numerical_features),
          target: featureDefinition.target,
          categorical_features: featureDefinition.categorical_features,
          numerical_features: featureDefinition.numerical_features,
          datetime_column: featureDefinition.datetime_column,
          item_id_column: featureDefinition.item_id_column
        })
      });
      
      // Now fetch visualizations
      const data = await apiRequest(`/api/time-series/visualize-data?visualization_type=${type}`);
      
      if (data.success) {
        setVisualizations(data.visualizations);
        setPlotType(type);
        toast({
          title: "Visualizations generated",
          description: `Generated ${Object.keys(data.visualizations).length} visualizations`,
        });
      } else {
        toast({
          title: "Error generating visualizations",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error fetching visualizations:", error);
      toast({
        title: "Error generating visualizations",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle algorithm selection
  const handleAlgorithmSelection = async (algorithm: string) => {
    setSelectedAlgorithm(algorithm);
    setLoading(true);
    
    try {
      const data = await apiRequest('/api/time-series/select-algorithm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm_name: algorithm
        })
      });
      
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
          title: "Algorithm selected",
          description: `Selected ${data.algorithm_name}`,
        });
      } else {
        toast({
          title: "Error selecting algorithm",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error selecting algorithm:", error);
      toast({
        title: "Error selecting algorithm",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle hyperparameter configuration
  const handleHyperparameterConfig = async () => {
    setLoading(true);
    
    try {
      const data = await apiRequest('/api/time-series/configure-hyperparameters', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          hyperparameters: hyperparameters
        })
      });
      
      if (data.success) {
        toast({
          title: "Hyperparameters configured",
          description: "Hyperparameters configured successfully",
        });
        
        // Move to the next tab
        setActiveTab("training");
      } else {
        toast({
          title: "Error configuring hyperparameters",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error configuring hyperparameters:", error);
      toast({
        title: "Error configuring hyperparameters",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle model training
  const handleModelTraining = async () => {
    setLoading(true);
    setProgress(0);
    
    try {
      // First preprocess the data
      const preprocessData = await apiRequestWithRetry('/api/time-series/preprocess-data');
      
      if (!preprocessData.success) {
        throw new Error(preprocessData.message);
      }
      
      setProgress(20);
      
      // Then train the model with retry
      const trainData = await apiRequestWithRetry('/api/time-series/train-model');
      
      if (!trainData.success) {
        throw new Error(trainData.message);
      }

      // Start polling for progress with retry and timeout
      const pollProgress = async () => {
        const maxPollingTime = 30 * 60 * 1000; // 30 minutes
        const startTime = Date.now();
        
        const poll = async () => {
          try {
            if (Date.now() - startTime > maxPollingTime) {
              throw new Error("Training timeout: Operation took too long");
            }

            const response = await apiRequestWithRetry('/api/time-series/check-progress');
            
            if (response.error) {
              throw new Error(response.error);
            }
            
            setProgress(response.progress);
            
            if (response.is_running) {
              // Continue polling if task is still running
              await new Promise(resolve => setTimeout(resolve, 1000));
              return poll();
            } else {
              // Task completed, evaluate the model with retry
              const evaluateData = await apiRequestWithRetry('/api/time-series/evaluate-model');
              
              if (!evaluateData.success) {
                throw new Error(evaluateData.message);
              }
              
              setProgress(100);
              setModelTrained(true);
              setEvaluation({
                metrics: evaluateData.metrics,
                visualizations: evaluateData.visualizations
              });
              
              toast({
                title: "Model trained successfully",
                description: `R² Score: ${(evaluateData.metrics.R2 * 100).toFixed(2)}%`,
              });
              
              // Move to the next tab
              setActiveTab("evaluation");
            }
          } catch (error) {
            console.error("Error in polling:", error);
            
            // Check if we should retry based on error type
            if (error instanceof Error && 
                (error.message.includes('fetch failed') || error.message.includes('status: 5'))) {
              await new Promise(resolve => setTimeout(resolve, 2000));
              return poll();
            }
            
            toast({
              title: "Error training model",
              description: error instanceof Error ? error.message : "An unexpected error occurred",
              variant: "destructive"
            });
            setLoading(false);
            throw error;
          }
        };
        
        return poll();
      };
      
      // Start polling
      await pollProgress();
      
    } catch (error) {
      console.error("Error training model:", error);
      toast({
        title: "Error training model",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle prediction file upload
  const handlePredictionFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
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
    
    setPredictionFile(file);
  };
  
  // Function to handle predictions from file
  const handlePredictionsFromFile = async () => {
    if (!predictionFile) {
      toast({
        title: "No file selected",
        description: "Please select a file for prediction",
        variant: "destructive"
      });
      return;
    }
    
    setLoading(true);
    
    try {
      // Convert file to base64
      const fileContent = await readFileAsBase64(predictionFile);
      
      // Send to API
      const data = await apiRequest('/api/time-series/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_content: fileContent,
          file_name: predictionFile.name
        })
      });
      
      if (data.success) {
        setPredictions(data.predictions);
        
        toast({
          title: "Predictions made successfully",
          description: `Made ${data.prediction_count} predictions`,
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
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle future predictions
  const handleFuturePredictions = async () => {
    setLoading(true);
    
    try {
      // Send to API
      const data = await apiRequest(`/api/time-series/predict-future?future_periods=${futurePeriods}`, {
        method: 'POST'
      });
      
      if (data.success) {
        setPredictions(data.predictions);
        
        toast({
          title: "Future predictions made successfully",
          description: `Made ${data.prediction_count} future predictions`,
        });
      } else {
        toast({
          title: "Error making future predictions",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error making future predictions:", error);
      toast({
        title: "Error making future predictions",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle column selection for features
  const handleColumnSelection = (column: string, isSelected: boolean) => {
    if (isSelected) {
      setFeatureDefinition(prev => ({
        ...prev,
        additional_features: [...prev.additional_features, column]
      }));
    } else {
      setFeatureDefinition(prev => ({
        ...prev,
        additional_features: prev.additional_features.filter(f => f !== column)
      }));
    }
  };
  
  // Function to handle column type selection
  const handleColumnTypeSelection = (column: string, type: 'categorical' | 'numerical') => {
    if (type === 'categorical') {
      setFeatureDefinition(prev => ({
        ...prev,
        categorical_features: [...prev.categorical_features.filter(f => f !== column), column],
        numerical_features: prev.numerical_features.filter(f => f !== column)
      }));
    } else {
      setFeatureDefinition(prev => ({
        ...prev,
        numerical_features: [...prev.numerical_features.filter(f => f !== column), column],
        categorical_features: prev.categorical_features.filter(f => f !== column)
      }));
    }
  };
  
  // Function to handle target column selection
  const handleTargetSelection = (column: string) => {
    setFeatureDefinition(prev => ({
      ...prev,
      target: column,
      // Remove target from features if it's there
      additional_features: prev.additional_features.filter(f => f !== column),
      // Remove target from datetime if it's there
      datetime_column: prev.datetime_column === column ? "" : prev.datetime_column
    }));
  };
  
  // Function to handle datetime column selection
  const handleDatetimeSelection = (column: string) => {
    setFeatureDefinition(prev => ({
      ...prev,
      datetime_column: column,
      // Remove datetime from features if it's there
      additional_features: prev.additional_features.filter(f => f !== column),
      // Remove datetime from target if it's there
      target: prev.target === column ? "" : prev.target
    }));
  };
  
  // Function to handle item ID column selection
  const handleItemIdSelection = (column: string | null) => {
    setFeatureDefinition(prev => ({
      ...prev,
      item_id_column: column,
      // Remove item ID from features if it's there
      additional_features: column ? prev.additional_features.filter(f => f !== column) : prev.additional_features
    }));
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
  
  // Function to handle model saving
  const handleSaveModel = async () => {
    if (!modelName || !saveDirectory) {
      toast({
        title: "Missing information",
        description: "Please provide both model name and save directory",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/time-series/save-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify({
          model_name: modelName,
          dataset_name: dataFile?.name || "unknown_dataset",
          save_directory: saveDirectory,
          hyperparameters: hyperparameters,
          algorithm_name: selectedAlgorithm
        })
      });

      const data = await response.json();

      if (data.success) {
        toast({
          title: "Success",
          description: "Model saved successfully",
        });
      } else {
        toast({
          title: "Error",
          description: data.message || "Failed to save model",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error saving model:", error);
      toast({
        title: "Error",
        description: "Failed to save model",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to render the data import step
  const renderDataImport = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Import Data</CardTitle>
        <CardDescription className="text-foreground/80">
          Upload a CSV or Excel file containing your time series data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="dataFile">Data File</Label>
          <Input 
            id="dataFile" 
            type="file" 
            accept=".csv,.xlsx,.xls"
            onChange={handleFileUpload}
            disabled={loading}
            className="border-tertiary"
          />
          <p className="text-sm text-foreground/70">
            Supported formats: CSV, Excel (.xlsx, .xls)
          </p>
        </div>
        
        {dataInfo && (
          <div className="mt-4">
            <h3 className="text-lg font-medium text-secondary">Data Preview</h3>
            <p className="text-sm text-foreground/70">
              Rows: {dataInfo.shape[0]}, Columns: {dataInfo.shape[1]}
            </p>
            
            <div className="mt-2 border border-tertiary rounded-md">
              <ScrollArea className="h-[200px]">
                <table className="w-full">
                  <thead>
                    <tr className="bg-tertiary/20">
                      {dataInfo.columns.map(col => (
                        <th key={col} className="p-2 text-left text-sm font-medium text-foreground">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataInfo.sample.map((row, i) => (
                      <tr key={i} className="border-t border-tertiary/50">
                        {dataInfo.columns.map(col => (
                          <td key={`${i}-${col}`} className="p-2 text-sm text-foreground/90">
                            {row[col]?.toString() || ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </ScrollArea>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" disabled className="border-tertiary">Back</Button>
        <Button 
          onClick={() => setActiveTab("features")} 
          disabled={!dataInfo || loading}
          variant="secondary"
        >
          Next: Define Features
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the feature definition step
  const renderFeatureDefinition = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Define Features</CardTitle>
        <CardDescription className="text-foreground/80">
          Define datetime, target and features for time series forecasting.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {dataInfo && (
          <div className="space-y-4">
            <div>
              <Label htmlFor="datetimeColumn">Datetime Column</Label>
              <Select 
                value={featureDefinition.datetime_column} 
                onValueChange={handleDatetimeSelection}
              >
                <SelectTrigger id="datetimeColumn" className="border-tertiary">
                  <SelectValue placeholder="Select datetime column" />
                </SelectTrigger>
                <SelectContent>
                  {dataInfo.columns.map(col => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-foreground/70 mt-1">
                The column containing date and time information
              </p>
            </div>
            
            <div>
              <Label htmlFor="targetColumn">Target Column</Label>
              <Select 
                value={featureDefinition.target} 
                onValueChange={handleTargetSelection}
              >
                <SelectTrigger id="targetColumn" className="border-tertiary">
                  <SelectValue placeholder="Select target column" />
                </SelectTrigger>
                <SelectContent>
                  {dataInfo.columns.map(col => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-foreground/70 mt-1">
                The column you want to forecast
              </p>
            </div>
            
            <div>
              <Label htmlFor="itemIdColumn">Item/Machine ID Column (Optional)</Label>
              <Select 
                value={featureDefinition.item_id_column || "none"} 
                onValueChange={(val: string) => handleItemIdSelection(val === "none" ? null : val)}
              >
                <SelectTrigger id="itemIdColumn" className="border-tertiary">
                  <SelectValue placeholder="Select ID column (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  {dataInfo.columns.map(col => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-foreground/70 mt-1">
                Column that identifies individual machines or items
              </p>
            </div>
            
            <div>
              <Label htmlFor="timeUnit">Time Unit</Label>
              <Select 
                value={timeUnitDefinition.timeUnit} 
                onValueChange={(value) => setTimeUnitDefinition(prev => ({ ...prev, timeUnit: value }))}
              >
                <SelectTrigger id="timeUnit" className="border-tertiary">
                  <SelectValue placeholder="Select time unit" />
                </SelectTrigger>
                <SelectContent>
                  {timeUnits.map(unit => (
                    <SelectItem key={unit.value} value={unit.value}>
                      {unit.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-foreground/70 mt-1">
                The time unit for forecasting
              </p>
            </div>
            
            <div>
              <Label htmlFor="forecastHorizon">Forecast Horizon</Label>
              <Input 
                id="forecastHorizon" 
                type="number"
                min={1}
                value={timeUnitDefinition.forecastHorizon}
                onChange={(e) => setTimeUnitDefinition(prev => ({
                  ...prev,
                  forecastHorizon: parseInt(e.target.value) || 1
                }))}
                className="border-tertiary"
              />
              <p className="text-sm text-foreground/70 mt-1">
                Number of time units to forecast into the future
              </p>
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Select Features</h3>
              <div className="space-y-2">
                {dataInfo.columns.map(col => {
                  const isFeature = featureDefinition.additional_features.includes(col);
                  const isTarget = featureDefinition.target === col;
                  const isDatetime = featureDefinition.datetime_column === col;
                  const isItemId = featureDefinition.item_id_column === col;
                  const isCategorical = featureDefinition.categorical_features.includes(col);
                  const isNumerical = featureDefinition.numerical_features.includes(col);
                  const columnInfo = dataInfo.columnInfo[col];
                  
                  return (
                    <div key={col} className="flex items-center justify-between p-2 border border-tertiary rounded-md">
                      <div className="flex items-center space-x-2">
                        <Checkbox 
                          id={`feature-${col}`}
                          checked={isFeature}
                          disabled={isTarget || isDatetime || isItemId}
                          onCheckedChange={(checked: boolean) => handleColumnSelection(col, checked)}
                          className="border-tertiary data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label htmlFor={`feature-${col}`} className="cursor-pointer">
                          {col}
                        </Label>
                        {columnInfo && (
                          <span className="text-xs text-foreground/70">
                            ({columnInfo.suggested_type})
                          </span>
                        )}
                      </div>
                      
                      {isFeature && (
                        <div className="flex items-center space-x-2">
                          <Label htmlFor={`type-${col}-cat`} className="cursor-pointer text-sm">
                            Categorical
                          </Label>
                          <Checkbox 
                            id={`type-${col}-cat`}
                            checked={isCategorical}
                            onCheckedChange={(checked: boolean) => {
                              if (checked) handleColumnTypeSelection(col, 'categorical');
                            }}
                            className="border-tertiary data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                          />
                          
                          <Label htmlFor={`type-${col}-num`} className="cursor-pointer text-sm">
                            Numerical
                          </Label>
                          <Checkbox 
                            id={`type-${col}-num`}
                            checked={isNumerical}
                            onCheckedChange={(checked: boolean) => {
                              if (checked) handleColumnTypeSelection(col, 'numerical');
                            }}
                            className="border-tertiary data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("import")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={handleFeatureDefinition} 
          disabled={!featureDefinition.datetime_column || !featureDefinition.target || loading}
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
        <CardTitle className="text-primary">Data Visualization</CardTitle>
        <CardDescription className="text-foreground/80">
          Visualize your time series data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-secondary mb-2">Generate Visualizations</h3>
            <p className="text-sm text-foreground/70 mb-4">
              Generate visualizations to better understand your time series data. You can analyze numerical features, categorical features, and time patterns.
            </p>
            
            <div className="flex flex-wrap gap-2 mb-4">
              <Button 
                variant={plotType === "all" ? "default" : "outline"} 
                onClick={() => {
                  setPlotType("all");
                  setVisualizations({});
                }}
                className="border-tertiary"
              >
                All Visualizations
              </Button>
              <Button 
                variant={plotType === "numerical" ? "default" : "outline"} 
                onClick={() => {
                  setPlotType("numerical");
                  setVisualizations({});
                }}
                className="border-tertiary"
              >
                Numerical
              </Button>
              <Button 
                variant={plotType === "categorical" ? "default" : "outline"} 
                onClick={() => {
                  setPlotType("categorical");
                  setVisualizations({});
                }}
                className="border-tertiary"
              >
                Categorical
              </Button>
              <Button 
                variant={plotType === "time" ? "default" : "outline"} 
                onClick={() => {
                  setPlotType("time");
                  setVisualizations({});
                }}
                className="border-tertiary"
              >
                Time-based
              </Button>
              
              <Button 
                onClick={() => fetchVisualizations(plotType)}
                disabled={loading}
                variant="secondary"
                className="ml-2"
              >
                {loading ? "Generating..." : "Generate Visualizations"}
              </Button>
            </div>
            
            {loading && (
              <div className="space-y-2 mb-4">
                <Progress value={45} className="h-2 bg-tertiary/30" />
                <p className="text-sm text-center text-foreground/80">
                  Generating visualizations...
                </p>
              </div>
            )}
            
            {Object.keys(visualizations).length > 0 ? (
              <Accordion type="single" collapsible className="w-full">
                {Object.entries(visualizations).map(([key, value]) => (
                  <AccordionItem key={key} value={key} className="border-tertiary">
                    <AccordionTrigger className="text-foreground hover:text-primary">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </AccordionTrigger>
                    <AccordionContent>
                      <img 
                        src={`data:image/png;base64,${value}`} 
                        alt={key} 
                        className="w-full"
                      />
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            ) : (
              <p className="text-sm text-foreground/70">
                Click "Generate Visualizations" to see visualizations of your data
              </p>
            )}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("features")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setActiveTab("algorithm")} 
          variant="secondary"
        >
          Next: Select Algorithm
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the algorithm selection step
  const renderAlgorithmSelection = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Select Algorithm</CardTitle>
        <CardDescription className="text-foreground/80">
          Choose a time series forecasting algorithm and configure its hyperparameters.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div>
            <Label htmlFor="algorithm">Forecasting Algorithm</Label>
            <Select 
              value={selectedAlgorithm} 
              onValueChange={handleAlgorithmSelection}
            >
              <SelectTrigger id="algorithm" className="border-tertiary">
                <SelectValue placeholder="Select algorithm" />
              </SelectTrigger>
              <SelectContent>
                {algorithms.map(algo => (
                  <SelectItem key={algo.value} value={algo.value}>
                    {algo.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {algorithmInfo && (
            <>
              <div className="p-4 border border-tertiary rounded-md bg-tertiary/10">
                <h3 className="text-lg font-medium text-secondary mb-2">
                  {algorithmInfo.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
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
                        {param.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </Label>
                      
                      <Select 
                        value={hyperparameters[param]?.toString() || ""}
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
          onClick={handleHyperparameterConfig} 
          disabled={!selectedAlgorithm || !algorithmInfo || loading}
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
          Preprocess data and train the time series forecasting model.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="p-4 border border-tertiary rounded-md bg-tertiary/10">
            <h3 className="text-lg font-medium text-secondary mb-2">Training Configuration</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-foreground/80">Algorithm:</span>
                <span className="font-medium text-foreground">
                  {selectedAlgorithm.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Features:</span>
                <span className="font-medium text-foreground">{featureDefinition.additional_features.length + featureDefinition.categorical_features.length + featureDefinition.numerical_features.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Target:</span>
                <span className="font-medium text-foreground">{featureDefinition.target}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Datetime Column:</span>
                <span className="font-medium text-foreground">{featureDefinition.datetime_column}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Time Unit:</span>
                <span className="font-medium text-foreground">{timeUnitDefinition.timeUnit}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Forecast Horizon:</span>
                <span className="font-medium text-foreground">{timeUnitDefinition.forecastHorizon} {timeUnitDefinition.timeUnit}s</span>
              </div>
              
              <Separator className="bg-tertiary/50" />
              
              <h4 className="text-sm font-medium text-foreground">Hyperparameters:</h4>
              {Object.entries(hyperparameters).map(([param, value]) => (
                <div key={param} className="flex justify-between text-sm">
                  <span className="text-foreground/80">{param.replace(/_/g, ' ')}:</span>
                  <span className="font-medium text-foreground">{formatHyperparamValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
          
          {loading && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2 bg-tertiary/30" />
              <p className="text-sm text-center text-foreground/80">
                {progress < 33 ? "Preprocessing data..." : 
                 progress < 66 ? "Training model..." : 
                 "Evaluating model..."}
              </p>
            </div>
          )}
          
          {modelTrained && (
            <Alert className="bg-primary/10 border-primary">
              <AlertTitle className="text-primary">Model Trained Successfully</AlertTitle>
              <AlertDescription className="text-foreground/90">
                Your time series forecasting model has been trained and is ready for evaluation and predictions.
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("algorithm")} className="border-tertiary">
          Back
        </Button>
        <div className="space-x-2">
          <Button 
            onClick={handleModelTraining} 
            disabled={loading || !selectedAlgorithm}
            variant="default"
          >
            {modelTrained ? "Retrain Model" : "Train Model"}
          </Button>
          <Button 
            onClick={() => setActiveTab("evaluation")} 
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
          Review the performance of your trained time series forecasting model.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Model Evaluation</h2>
          
          {evaluation && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Metrics</h3>
                  <div className="space-y-2">
                    {Object.entries(evaluation.metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span>{key}:</span>
                        <span>{typeof value === 'number' ? value.toFixed(4) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Save Model Section */}
                <div className="p-4 border rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Save Model</h3>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="modelName">Model Name</Label>
                      <Input
                        id="modelName"
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="Enter model name"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="saveDirectory">Save Directory</Label>
                      <Input
                        id="saveDirectory"
                        value={saveDirectory}
                        onChange={(e) => setSaveDirectory(e.target.value)}
                        placeholder="Enter save directory path"
                      />
                    </div>
                    <Button 
                      onClick={handleSaveModel}
                      disabled={loading}
                      variant="secondary"
                    >
                      {loading ? "Saving..." : "Save Model"}
                    </Button>
                  </div>
                </div>
              </div>

              {/* Visualizations */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Visualizations</h3>
                {Object.entries(evaluation.visualizations).map(([key, base64Image]) => (
                  <div key={key} className="border rounded-lg p-4">
                    <h4 className="text-md font-medium mb-2">{key}</h4>
                    <img 
                      src={`data:image/png;base64,${base64Image}`} 
                      alt={key} 
                      className="w-full"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("training")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setActiveTab("prediction")} 
          disabled={!evaluation}
          variant="secondary"
        >
          Next: Make Predictions
        </Button>
      </CardFooter>
    </Card>
  );
  
  // Function to render the prediction step
  const renderPrediction = () => (
    <Card className="border-tertiary">
      <CardHeader className="border-b border-tertiary">
        <CardTitle className="text-primary">Make Predictions</CardTitle>
        <CardDescription className="text-foreground/80">
          Use your trained model to forecast future values or predict based on new data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-secondary mb-2">Future Forecasting</h3>
            <p className="text-sm text-foreground/70 mb-4">
              Forecast future values based on the trained time series model.
            </p>
            
            <div className="grid w-full max-w-sm items-center gap-1.5 mb-4">
              <Label htmlFor="futurePeriods">Number of Future Periods</Label>
              <Input 
                id="futurePeriods" 
                type="number"
                min={1}
                value={futurePeriods}
                onChange={(e) => setFuturePeriods(parseInt(e.target.value) || 1)}
                className="border-tertiary"
              />
              <p className="text-sm text-foreground/70">
                Number of {timeUnitDefinition.timeUnit}s to forecast into the future
              </p>
            </div>
            
            <Button 
              onClick={handleFuturePredictions} 
              disabled={!modelTrained || loading}
              variant="default"
              className="mb-4"
            >
              Forecast Future Values
            </Button>
          </div>
          
          <Separator className="bg-tertiary/50" />
          
          <div>
            <h3 className="text-lg font-medium text-secondary mb-2">Predict from Data</h3>
            <p className="text-sm text-foreground/70 mb-4">
              Upload a file with new data to make predictions.
            </p>
            
            <div className="grid w-full max-w-sm items-center gap-1.5 mb-4">
              <Label htmlFor="predictionFile">Prediction Data</Label>
              <Input 
                id="predictionFile" 
                type="file" 
                accept=".csv,.xlsx,.xls"
                onChange={handlePredictionFileUpload}
                disabled={loading}
                className="border-tertiary"
              />
              <p className="text-sm text-foreground/70">
                Upload a file with the same features as your training data (with or without the target column)
              </p>
            </div>
            
            <Button 
              onClick={handlePredictionsFromFile} 
              disabled={!predictionFile || loading || !modelTrained}
              variant="default"
              className="mb-4"
            >
              Make Predictions from File
            </Button>
          </div>
          
          {predictions.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-secondary mb-2">Prediction Results</h3>
              <div className="border border-tertiary rounded-md">
                <ScrollArea className="h-[300px]">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-tertiary/20">
                        {Object.keys(predictions[0]).map(key => (
                          <th key={key} className="p-2 text-left text-sm font-medium text-foreground">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((row, i) => (
                        <tr key={i} className="border-t border-tertiary/50">
                          {Object.entries(row).map(([key, value]) => (
                            <td key={`${i}-${key}`} className="p-2 text-sm text-foreground">
                              {value?.toString() || ""}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            </div>
          )}
          
          <div className="mt-4">
            <h3 className="text-lg font-medium text-secondary mb-2">Compare Predictions with Actual Values</h3>
            <p className="text-sm text-foreground/70">
              If your prediction data includes the target column, you can upload it to compare predictions with actual values.
            </p>
            <Button 
              onClick={() => {
                if (predictionFile) {
                  // Implement comparison functionality
                  toast({
                    title: "Feature coming soon",
                    description: "Comparison functionality will be available in a future update",
                  });
                } else {
                  toast({
                    title: "No file selected",
                    description: "Please select a file with actual values for comparison",
                    variant: "destructive"
                  });
                }
              }} 
              disabled={!predictionFile || loading || !modelTrained}
              variant="outline"
              className="mt-2 border-tertiary"
            >
              Compare with Actual Values
            </Button>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("evaluation")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={handleSaveModel}
          variant="secondary"
        >
          Save Model
        </Button>
      </CardFooter>
    </Card>
  );
  
  return (
    <PageLayout>
      <div className="container mx-auto py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold"><span className="text-primary">Time Series</span> <span className="text-secondary">Forecasting</span></h1>
          <p className="text-foreground">
            Train time series forecasting models on industrial data to predict future values.
          </p>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6 border border-tertiary">
            <TabsTrigger value="import" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">1. Import</TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">2. Features</TabsTrigger>
            <TabsTrigger value="visualize" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">3. Visualize</TabsTrigger>
            <TabsTrigger value="algorithm" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">4. Algorithm</TabsTrigger>
            <TabsTrigger value="training" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">5. Train</TabsTrigger>
            <TabsTrigger value="evaluation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">6. Evaluate & Predict</TabsTrigger>
          </TabsList>
          
          <div className="mt-6">
            <TabsContent value="import">
              {renderDataImport()}
            </TabsContent>
            
            <TabsContent value="features">
              {renderFeatureDefinition()}
            </TabsContent>
            
            <TabsContent value="visualize">
              {renderDataVisualization()}
            </TabsContent>
            
            <TabsContent value="algorithm">
              {renderAlgorithmSelection()}
            </TabsContent>
            
            <TabsContent value="training">
              {renderModelTraining()}
            </TabsContent>
            
            <TabsContent value="evaluation">
              {renderModelEvaluation()}
            </TabsContent>
            
            <TabsContent value="prediction">
              {renderPrediction()}
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </PageLayout>
  );
}