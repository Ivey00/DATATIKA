"use client";

import { useState, useEffect } from "react";
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
  features: string[];
  target: string;
  categoricalFeatures: string[];
  numericalFeatures: string[];
  itemIdColumn: string | null;
  datetimeColumn: string | null;
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
  hyperparameters: {[key: string]: any};
  algorithm_name: string;
}

// Main component
export default function TargetPrediction() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("import");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [modelName, setModelName] = useState("");
  const [saveDirectory, setSaveDirectory] = useState("");
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  
  // State for each step
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [featureDefinition, setFeatureDefinition] = useState<FeatureDefinition>({
    features: [],
    target: "",
    categoricalFeatures: [],
    numericalFeatures: [],
    itemIdColumn: null,
    datetimeColumn: null
  });
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("");
  const [algorithmInfo, setAlgorithmInfo] = useState<AlgorithmInfo | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [modelTrained, setModelTrained] = useState(false);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [visualizations, setVisualizations] = useState<Record<string, string>>({});
  const [plotType, setPlotType] = useState<string>("numerical");
  const [progressMessage, setProgressMessage] = useState<string>("");
  
  // Available algorithms
  const algorithms = [
    { value: "LinearRegression", label: "Linear Regression" },
    { value: "Ridge", label: "Ridge Regression" },
    { value: "Lasso", label: "Lasso Regression" },
    { value: "ElasticNet", label: "Elastic Net" },
    { value: "RandomForestRegressor", label: "Random Forest" },
    { value: "GradientBoostingRegressor", label: "Gradient Boosting" },
    { value: "LGBMRegressor", label: "LightGBM" },
    { value: "KNeighborsRegressor", label: "K-Nearest Neighbors" },
    { value: "DecisionTreeRegressor", label: "Decision Tree" },
    { value: "ExtraTreesRegressor", label: "Extra Trees" },
    { value: "AdaBoostRegressor", label: "AdaBoost" },
    { value: "BayesianRidge", label: "Bayesian Ridge" },
    { value: "HuberRegressor", label: "Huber Regressor" },
    { value: "PassiveAggressiveRegressor", label: "Passive Aggressive" },
    { value: "Lars", label: "Least Angle Regression (LARS)" },
    { value: "LassoLars", label: "Lasso LARS" },
    { value: "OrthogonalMatchingPursuit", label: "Orthogonal Matching Pursuit" },
    { value: "DummyRegressor", label: "Dummy Regressor" }
  ];

  // Add fetchEvaluationResults function
  const fetchEvaluationResults = async () => {
    try {
      const response = await apiRequestWithRetry('/api/target-prediction/evaluate-model');
      
      if (response.success) {
        setEvaluation({
          metrics: response.metrics,
          visualizations: response.visualizations
        });
      } else {
        toast({
          title: "Error fetching evaluation results",
          description: response.message,
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

  // Add retry logic for API requests
  const apiRequestWithRetry = async (url: string, options?: RequestInit, maxRetries = 3) => {
    let lastError;
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(url, options);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
      } catch (error) {
        lastError = error;
        // Only retry on network errors or 5xx server errors
        if (error instanceof Error && 
            (error.message.includes('socket hang up') || 
             error.message.includes('network') ||
             (error as any).status >= 500)) {
          // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  };

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
      const formData = new FormData();
      formData.append('file', file);
      
      // Convert file to base64
      const fileContent = await readFileAsBase64(file);
      
      // Send to API using keepAlive agent
      const data = await apiRequest('/api/target-prediction/import-data', {
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
      const data = await apiRequest('/api/target-prediction/column-info');
      
      if (data.column_info) {
        setDataInfo(prev => prev ? {
          ...prev,
          columnInfo: data.column_info
        } : null);
      }
    } catch (error) {
      console.error("Error fetching column info:", error);
    }
  };
  
  // Function to handle feature definition
  const handleFeatureDefinition = async () => {
    if (!featureDefinition.features.length || !featureDefinition.target) {
      toast({
        title: "Missing information",
        description: "Please select features and target column",
        variant: "destructive"
      });
      return;
    }
    
    setLoading(true);
    try {
      const data = await apiRequest('/api/target-prediction/define-features', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          features: featureDefinition.features,
          target: featureDefinition.target,
          categorical_features: featureDefinition.categoricalFeatures,
          numerical_features: featureDefinition.numericalFeatures,
          item_id_column: featureDefinition.itemIdColumn,
          datetime_column: featureDefinition.datetimeColumn || null
        })
      });
      
      if (data.success) {
        toast({
          title: "Features defined successfully",
          description: `Defined ${data.features.length} features and target: ${data.target}`,
        });
        
        // Move to the next tab
        setActiveTab("algorithm");
      } else {
        toast({
          title: "Error defining features",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error defining features:", error);
      toast({
        title: "Error defining features",
        description: "An unexpected error occurred",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to fetch visualizations
  const fetchVisualizations = async (type: string) => {
    setLoading(true);
    try {
      // Only proceed with time-based visualizations if datetime column is selected
      if (type === "time" && !featureDefinition.datetimeColumn) {
        toast({
          title: "Datetime column required",
          description: "Please select a datetime column for time-based visualizations",
          variant: "destructive"
        });
        return;
      }

      // First, ensure features are defined on the backend
      const defineData = await apiRequestWithRetry('/api/target-prediction/define-features', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          features: featureDefinition.features,
          target: featureDefinition.target,
          categorical_features: featureDefinition.categoricalFeatures,
          numerical_features: featureDefinition.numericalFeatures,
          item_id_column: featureDefinition.itemIdColumn,
          datetime_column: featureDefinition.datetimeColumn || null
        })
      });
      
      if (!defineData.success) {
        toast({
          title: "Error defining features",
          description: defineData.message || "Failed to define features for visualization",
          variant: "destructive"
        });
        return;
      }
      
      // Now fetch the visualizations with retry
      const data = await apiRequestWithRetry(`/api/target-prediction/visualize-data?plot_type=${type}`);
      
      if (data.success) {
        setVisualizations(data.visualizations);
        setPlotType(type);
        toast({
          title: "Visualizations generated",
          description: data.message,
        });
      } else {
        toast({
          title: "Error generating visualizations",
          description: data.message || "Failed to generate visualizations",
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
      const data = await apiRequest('/api/target-prediction/select-algorithm', {
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
      const data = await apiRequest('/api/target-prediction/configure-hyperparameters', {
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
    setProgressMessage("Starting model training...");
    
    try {
      // Start the training process
      const response = await apiRequestWithRetry('/api/target-prediction/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.success) {
        throw new Error(response.message);
      }

      // Set up polling for progress updates
      let retryCount = 0;
      const maxRetries = 3;
      let pollTimeout: NodeJS.Timeout | null = null;

      const pollProgress = async () => {
        try {
          const progressData = await apiRequestWithRetry('/api/target-prediction/check-progress', {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          });

          retryCount = 0; // Reset retry count on successful request

          // Update progress state
          setProgress(progressData.progress);
          setProgressMessage(progressData.message || "Processing...");

          if (progressData.error) {
            setLoading(false);
            throw new Error(progressData.error);
          }

          // Continue polling if task is still running
          if (!progressData.is_complete && progressData.task_running) {
            pollTimeout = setTimeout(pollProgress, 1000);
          } else {
            // Task is complete
            if (progressData.progress === 100) {
              setModelTrained(true);
              setLoading(false);

              toast({
                title: "Model trained successfully",
                description: progressData.message || "Model has been trained and evaluated",
              });

              // Fetch evaluation results
              await fetchEvaluationResults();

              // Move to the next tab
              setActiveTab("evaluate");
            } else if (!progressData.task_running) {
              // Task stopped without completing
              setLoading(false);
              toast({
                title: "Training incomplete",
                description: "The training process stopped unexpectedly",
                variant: "destructive"
              });
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
              description: error instanceof Error ? error.message : "Lost connection to server",
              variant: "destructive"
            });
          }
        }
      };

      // Start polling immediately
      await pollProgress();

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
  
  // Function to handle predictions
  const handlePredictions = async () => {
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
      const data = await apiRequest('/api/target-prediction/predict', {
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
  
  // Function to handle column selection for features
  const handleColumnSelection = (column: string, isSelected: boolean) => {
    if (isSelected) {
      setFeatureDefinition(prev => ({
        ...prev,
        features: [...prev.features, column]
      }));
    } else {
      setFeatureDefinition(prev => ({
        ...prev,
        features: prev.features.filter(f => f !== column)
      }));
    }
  };
  
  // Function to handle column type selection
  const handleColumnTypeSelection = (column: string, type: 'categorical' | 'numerical') => {
    if (type === 'categorical') {
      setFeatureDefinition(prev => ({
        ...prev,
        categoricalFeatures: [...prev.categoricalFeatures.filter(f => f !== column), column],
        numericalFeatures: prev.numericalFeatures.filter(f => f !== column)
      }));
    } else {
      setFeatureDefinition(prev => ({
        ...prev,
        numericalFeatures: [...prev.numericalFeatures.filter(f => f !== column), column],
        categoricalFeatures: prev.categoricalFeatures.filter(f => f !== column)
      }));
    }
  };
  
  // Function to handle target column selection
  const handleTargetSelection = (column: string) => {
    setFeatureDefinition(prev => ({
      ...prev,
      target: column,
      // Remove target from features if it's there
      features: prev.features.filter(f => f !== column)
    }));
  };
  
  // Function to handle item ID column selection
  const handleItemIdSelection = (column: string | null) => {
    setFeatureDefinition(prev => ({
      ...prev,
      itemIdColumn: column
    }));
  };
  
  // Function to handle datetime column selection
  const handleDatetimeSelection = (column: string | null) => {
    setFeatureDefinition(prev => ({
      ...prev,
      datetimeColumn: column
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
        description: "Please provide both model name and save location",
        variant: "destructive"
      });
      return;
    }

    try {
      const datasetName = dataFile?.name || "unknown_dataset";

      // Prepare hyperparameters data
      const currentHyperparameters = algorithmInfo ? Object.fromEntries(
        Object.entries(hyperparameters).map(([key, value]) => {
          // Convert string values to appropriate types
          if (value === "null" || value === "None") return [key, null];
          if (value === "true" || value === "True") return [key, true];
          if (value === "false" || value === "False") return [key, false];
          if (!isNaN(Number(value))) return [key, Number(value)];
          return [key, value];
        })
      ) : {};

      setLoading(true);
      const data = await apiRequest('/api/target-prediction/save-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_name: modelName,
          dataset_name: datasetName,
          save_directory: saveDirectory,
          hyperparameters: currentHyperparameters,
          algorithm_name: selectedAlgorithm
        } as ModelSaveRequest)
      });

      if (data.success) {
        toast({
          title: "Model saved successfully",
          description: `Model saved to ${data.model_path}`,
        });
        setShowSaveDialog(false);
      } else {
        toast({
          title: "Error saving model",
          description: data.message,
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error saving model:", error);
      toast({
        title: "Error saving model",
        description: error instanceof Error ? error.message : "An unexpected error occurred",
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
          Upload a CSV or Excel file containing your industrial data.
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
          Select the features, target column, and specify column types.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {dataInfo && (
          <div className="space-y-4">
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
                The column you want to predict (continuous value for regression)
              </p>
            </div>
            
            <div>
              <Label htmlFor="itemIdColumn">Machine/Item ID Column (Optional)</Label>
              <Select 
                value={featureDefinition.itemIdColumn || "none"} 
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
              <Label htmlFor="datetimeColumn">Datetime Column</Label>
              <Select 
                value={featureDefinition.datetimeColumn || "none"} 
                onValueChange={(val: string) => handleDatetimeSelection(val === "none" ? null : val)}
              >
                <SelectTrigger id="datetimeColumn" className="border-tertiary">
                  <SelectValue placeholder="Select datetime column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  {dataInfo.columns.map(col => (
                    <SelectItem key={col} value={col}>
                      {col} {dataInfo.columnInfo[col]?.suggested_type === "datetime" ? "(datetime)" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-foreground/70 mt-1">
                Column containing timestamps for time-based analysis
              </p>
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Select Features</h3>
              <div className="space-y-2">
                {dataInfo.columns.map(col => {
                  const isFeature = featureDefinition.features.includes(col);
                  const isTarget = featureDefinition.target === col;
                  const isCategorical = featureDefinition.categoricalFeatures.includes(col);
                  const isNumerical = featureDefinition.numericalFeatures.includes(col);
                  const columnInfo = dataInfo.columnInfo[col];
                  
                  return (
                    <div key={col} className="flex items-center justify-between p-2 border border-tertiary rounded-md">
                      <div className="flex items-center space-x-2">
                        <Checkbox 
                          id={`feature-${col}`}
                          checked={isFeature}
                          disabled={isTarget}
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
            
            <div className="mt-6">
              <h3 className="text-lg font-medium text-secondary">Data Visualizations</h3>
              <p className="text-sm text-foreground/70 mb-2">
                Select a visualization type and click to generate plots. 
                {plotType === "time" && (
                  <>
                    <br />
                    Time-based plots will use {featureDefinition.datetimeColumn || "[No datetime column selected]"} as the time axis.
                  </>
                )}
              </p>
              
              <div className="flex flex-wrap gap-2 mb-4">
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
                  disabled={!featureDefinition.datetimeColumn}
                  className="border-tertiary"
                >
                  Time-based
                </Button>
                {plotType === "time" && !featureDefinition.datetimeColumn && (
                  <p className="text-sm text-foreground/70 mt-2">
                    Please select a datetime column to enable time-based visualizations
                  </p>
                )}
                
                <Button 
                  onClick={() => {
                    setLoading(true);
                    fetchVisualizations(plotType)
                      .finally(() => setLoading(false));
                  }}
                  disabled={loading || !featureDefinition.features.length || !featureDefinition.target}
                  variant="secondary"
                  className="ml-2"
                >
                  {loading ? "Generating..." : `Generate ${plotType.charAt(0).toUpperCase() + plotType.slice(1)} Plots`}
                </Button>
              </div>
              
              {loading && plotType && (
                <div className="space-y-2 mb-4">
                  <Progress value={45} className="h-2 bg-tertiary/30" />
                  <p className="text-sm text-center text-foreground/80">
                    Generating {plotType} visualizations...
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
                  Define features and target to see visualizations
                </p>
              )}
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
          disabled={!featureDefinition.features.length || !featureDefinition.target || loading}
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
          Choose a regression algorithm and configure its hyperparameters.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div>
            <Label htmlFor="algorithm">Regression Algorithm</Label>
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
        <Button variant="outline" onClick={() => setActiveTab("features")} className="border-tertiary">
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
          Preprocess data and train the regression model.
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
                <span className="font-medium text-foreground">{featureDefinition.features.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Target:</span>
                <span className="font-medium text-foreground">{featureDefinition.target}</span>
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
                Your model has been trained and is ready for evaluation and predictions.
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
          Review the performance of your trained regression model.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {evaluation && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-primary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-primary">R² Score</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {(evaluation.metrics.r2 * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-foreground/70 mt-1">
                    Coefficient of determination (1.0 is perfect prediction)
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-secondary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-secondary">RMSE</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.rmse.toFixed(4)}
                  </p>
                  <p className="text-xs text-foreground/70 mt-1">
                    Root Mean Squared Error (lower is better)
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-tertiary/20 border-b border-tertiary">
                  <CardTitle className="text-lg text-foreground">MAE</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.mae.toFixed(4)}
                  </p>
                  <p className="text-xs text-foreground/70 mt-1">
                    Mean Absolute Error (lower is better)
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-tertiary/20 border-b border-tertiary">
                  <CardTitle className="text-lg text-foreground">MSE</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.mse.toFixed(4)}
                  </p>
                  <p className="text-xs text-foreground/70 mt-1">
                    Mean Squared Error (lower is better)
                  </p>
                </CardContent>
              </Card>
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Actual vs Predicted Values</h3>
              {evaluation.visualizations.actual_vs_predicted && (
                <img 
                  src={`data:image/png;base64,${evaluation.visualizations.actual_vs_predicted}`} 
                  alt="Actual vs Predicted Values" 
                  className="w-full max-w-md mx-auto"
                />
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Residuals Plot</h3>
              {evaluation.visualizations.residuals && (
                <img 
                  src={`data:image/png;base64,${evaluation.visualizations.residuals}`} 
                  alt="Residuals Plot" 
                  className="w-full max-w-md mx-auto"
                />
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Residuals Distribution</h3>
              {evaluation.visualizations.residuals_distribution && (
                <img 
                  src={`data:image/png;base64,${evaluation.visualizations.residuals_distribution}`} 
                  alt="Residuals Distribution" 
                  className="w-full max-w-md mx-auto"
                />
              )}
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Interpretation</h3>
              <div className="space-y-2 text-sm text-foreground/90">
                <p>
                  <strong>R² Score:</strong> Indicates how well the model explains the variance in the target variable. 
                  Values closer to 1.0 indicate better fit.
                </p>
                <p>
                  <strong>RMSE (Root Mean Squared Error):</strong> Measures the average magnitude of errors in predictions. 
                  It gives higher weight to larger errors.
                </p>
                <p>
                  <strong>MAE (Mean Absolute Error):</strong> Measures the average magnitude of errors without considering their direction.
                  It's less sensitive to outliers than RMSE.
                </p>
                <p>
                  <strong>Residuals Plot:</strong> Shows the difference between actual and predicted values. 
                  Ideally, residuals should be randomly distributed around zero.
                </p>
              </div>
            </div>
          </div>
        )}
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
          Use your trained model to predict target values for new data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
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
            onClick={handlePredictions} 
            disabled={!predictionFile || loading || !modelTrained}
            variant="default"
          >
            Make Predictions
          </Button>
          
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

      {showSaveDialog && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" />
          <div className="fixed inset-0 flex items-center justify-center">
            <Card className="w-[400px] border-tertiary bg-background">
              <CardHeader>
                <CardTitle>Save Model</CardTitle>
                <CardDescription>
                  Enter a name for your model and specify the save location.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="modelName">Model Name</Label>
                    <Input
                      id="modelName"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="my_regression_model"
                      className="border-tertiary"
                    />
                    <p className="text-sm text-foreground/70 mt-1">
                      The name of your model (e.g., temperature_predictor)
                    </p>
                  </div>
                  <div>
                    <Label htmlFor="saveDirectory">Save Location</Label>
                    <Input
                      id="saveDirectory"
                      value={saveDirectory}
                      onChange={(e) => setSaveDirectory(e.target.value)}
                      placeholder="C:/Users/username/models"
                      className="border-tertiary"
                    />
                    <p className="text-sm text-foreground/70 mt-1">
                      Full path where you want to save the model
                    </p>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-end space-x-2">
                <Button
                  variant="outline"
                  onClick={() => setShowSaveDialog(false)}
                  className="border-tertiary"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveModel}
                  disabled={loading || !modelName || !saveDirectory}
                >
                  Save Model
                </Button>
              </CardFooter>
            </Card>
          </div>
        </div>
      )}

      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("evaluation")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={() => setShowSaveDialog(true)}
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
          <h1 className="text-3xl font-bold"><span className="text-primary">Target</span> <span className="text-secondary">Prediction</span></h1>
          <p className="text-foreground">
            Train regression models on industrial data to predict continuous target values.
          </p>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5 border border-tertiary">
            <TabsTrigger value="import" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">1. Import Data</TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">2. Define Features</TabsTrigger>
            <TabsTrigger value="algorithm" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">3. Select Algorithm</TabsTrigger>
            <TabsTrigger value="training" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">4. Train Model</TabsTrigger>
            <TabsTrigger value="evaluation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">5. Evaluate & Predict</TabsTrigger>
          </TabsList>
          
          <div className="mt-6">
            <TabsContent value="import">
              {renderDataImport()}
            </TabsContent>
            
            <TabsContent value="features">
              {renderFeatureDefinition()}
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
