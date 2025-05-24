"use client";

import { useState, useEffect } from "react";
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
  column_type: string;
  missing_values: number;
  unique_values_list?: any[];
  min?: number;
  max?: number;
  mean?: number;
}

interface DataInfo {
  shape: [number, number];
  columns: string[];
  sample: any[];
  columnInfo: Record<string, ColumnInfo>;
}

interface FeatureDefinition {
  features: string[];
  categoricalFeatures: string[];
  numericalFeatures: string[];
  datetimeColumn: string;
  itemIdColumn: string | null;
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

// Main component
export default function AnomalyDetection() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("import");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  
  // State for each step
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [featureDefinition, setFeatureDefinition] = useState<FeatureDefinition>({
    features: [],
    categoricalFeatures: [],
    numericalFeatures: [],
    datetimeColumn: "",
    itemIdColumn: null
  });
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("");
  const [algorithmInfo, setAlgorithmInfo] = useState<AlgorithmInfo | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [modelTrained, setModelTrained] = useState(false);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [visualizations, setVisualizations] = useState<Record<string, string>>({});
  
  // Available algorithms
  const algorithms = [
    { value: "K-Means", label: "K-Means Clustering" },
    { value: "DBSCAN", label: "DBSCAN" },
    { value: "Gaussian Mixture", label: "Gaussian Mixture Model" },
    { value: "Isolation Forest", label: "Isolation Forest" },
    { value: "Local Outlier Factor", label: "Local Outlier Factor" },
    { value: "One-Class SVM", label: "One-Class SVM" },
    { value: "Agglomerative Clustering", label: "Agglomerative Clustering" }
  ];

  // Setup SSE for progress updates
  useEffect(() => {
    let eventSource: EventSource | null = null;
    
    if (activeTab === "training" && loading) {
      eventSource = new EventSource('/api/anomaly-detection/progress');
      
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setProgress(data.progress);
        setProgressMessage(data.message);
        
        // When progress reaches 100%, set modelTrained to true and close the connection
        if (data.progress === 100) {
          console.log("Training completed via SSE, updating modelTrained state");
          setModelTrained(true); // Enable the "Next: Evaluate Model" button
          setLoading(false); // Stop the loading state
          
          // Show success toast
          toast({
            title: "Model trained successfully",
            description: "Your model has been trained and is ready for evaluation.",
          });
          
          // Automatically fetch evaluation data
          fetch('/api/anomaly-detection/evaluate-model')
            .then(res => res.json())
            .then(data => {
              if (data.success) {
                setEvaluation({
                  metrics: data.metrics,
                  visualizations: data.visualizations
                });
                console.log("Evaluation data loaded automatically");
              }
            })
            .catch(error => {
              console.error("Error fetching evaluation data:", error);
            });
          
          eventSource?.close();
        }
      };
      
      eventSource.onerror = (error) => {
        console.error("SSE connection error:", error);
        eventSource?.close();
      };
    }
    
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [activeTab, loading, toast]);

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
      
      // Send to API
      const response = await fetch('/api/anomaly-detection/import-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_content: fileContent,
          file_name: file.name
        })
      });
      
      const data = await response.json();
      
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
      const response = await fetch('/api/anomaly-detection/column-info');
      const data = await response.json();
      
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
    if (!featureDefinition.features.length || !featureDefinition.datetimeColumn) {
      toast({
        title: "Missing information",
        description: "Please select features and datetime column",
        variant: "destructive"
      });
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch('/api/anomaly-detection/define-features', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          features: featureDefinition.features,
          target: "anomaly", // For anomaly detection, target is always "anomaly"
          categorical_features: featureDefinition.categoricalFeatures,
          numerical_features: featureDefinition.numericalFeatures,
          datetime_column: featureDefinition.datetimeColumn,
          item_id_column: featureDefinition.itemIdColumn
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        toast({
          title: "Features defined successfully",
          description: `Defined ${data.features.length} features with datetime column: ${data.datetime_column}`,
        });
        
        // Fetch visualizations
        await fetchVisualizations();
        
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
  const fetchVisualizations = async () => {
    try {
      const response = await fetch('/api/anomaly-detection/visualize-data');
      const data = await response.json();
      
      if (data.success) {
        setVisualizations(data.visualizations);
      }
    } catch (error) {
      console.error("Error fetching visualizations:", error);
    }
  };
  
  // Function to handle algorithm selection
  const handleAlgorithmSelection = async (algorithm: string) => {
    setSelectedAlgorithm(algorithm);
    setLoading(true);
    
    try {
      const response = await fetch('/api/anomaly-detection/select-algorithm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm_name: algorithm
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
      const response = await fetch('/api/anomaly-detection/configure-hyperparameters', {
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
      // First preprocess the data
      const preprocessResponse = await fetch('/api/anomaly-detection/preprocess-data');
      const preprocessData = await preprocessResponse.json();
      
      if (!preprocessData.success) {
        throw new Error(preprocessData.message);
      }
      
      // Then train the model
      const trainResponse = await fetch('/api/anomaly-detection/train-model');
      const trainData = await trainResponse.json();
      
      if (!trainData.success) {
        throw new Error(trainData.message);
      }
      
      // Wait for the model to finish training (progress will be updated via SSE)
      // The evaluate model endpoint will be called after training is complete
      await new Promise(resolve => {
        const checkStatus = () => {
          if (progress === 100) {
            resolve(true);
          } else {
            setTimeout(checkStatus, 1000);
          }
        };
        checkStatus();
      });
      
      // Finally evaluate the model
      const evaluateResponse = await fetch('/api/anomaly-detection/evaluate-model');
      const evaluateData = await evaluateResponse.json();
      
      if (!evaluateData.success) {
        throw new Error(evaluateData.message);
      }
      
      setModelTrained(true);
      setEvaluation({
        metrics: evaluateData.metrics,
        visualizations: evaluateData.visualizations
      });
      
      toast({
        title: "Model trained successfully",
        description: `Detected ${evaluateData.metrics.anomaly_count} anomalies (${evaluateData.metrics.anomaly_percentage.toFixed(2)}%)`,
      });
      
      // Move to the next tab
      setActiveTab("evaluation");
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
      const response = await fetch('/api/anomaly-detection/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_content: fileContent,
          file_name: predictionFile.name
        })
      });
      
      const data = await response.json();
      
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
  
  // Function to handle datetime column selection
  const handleDatetimeSelection = (column: string) => {
    setFeatureDefinition(prev => ({
      ...prev,
      datetimeColumn: column,
      // Remove datetime from features if it's there
      features: prev.features.filter(f => f !== column)
    }));
  };
  
  // Function to handle item ID column selection
  const handleItemIdSelection = (column: string | null) => {
    setFeatureDefinition(prev => ({
      ...prev,
      itemIdColumn: column,
      // Remove item ID from features if it's there and not null
      features: column ? prev.features.filter(f => f !== column) : prev.features
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
  
  // Function to save model and results
  const handleSaveResults = async () => {
    setLoading(true);
    
    try {
      // Save results
      const resultsResponse = await fetch('/api/anomaly-detection/save-results');
      const resultsData = await resultsResponse.json();
      
      if (!resultsData.success) {
        throw new Error(resultsData.message);
      }
      
      // Save model
      const modelResponse = await fetch('/api/anomaly-detection/save-model');
      const modelData = await modelResponse.json();
      
      if (!modelData.success) {
        throw new Error(modelData.message);
      }
      
      toast({
        title: "Model and results saved",
        description: `Results saved to ${resultsData.model_path}`,
      });
    } catch (error) {
      console.error("Error saving model and results:", error);
      toast({
        title: "Error saving",
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
          Upload a CSV or Excel file containing your industrial machine data.
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
                            {row[col]?.toString() ?? ""}
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
          Select the features, datetime column, item ID column, and specify column types.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {dataInfo && (
          <div className="space-y-4">
            <div>
              <Label htmlFor="datetimeColumn">Datetime Column</Label>
              <Select 
                value={featureDefinition.datetimeColumn} 
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
                The column containing timestamps for your time series data
              </p>
            </div>
            
            <div>
              <Label htmlFor="itemIdColumn">Item/Machine ID Column (Optional)</Label>
              <Select 
                value={featureDefinition.itemIdColumn ?? "none"} 
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
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Select Features</h3>
              <p className="text-sm text-foreground/70 mb-2">
                Select the features to use for anomaly detection. These are typically numerical readings like temperature, pressure, vibration, etc.
              </p>
              <div className="space-y-2">
                {dataInfo.columns.map(col => {
                  const isFeature = featureDefinition.features.includes(col);
                  const isDateTime = featureDefinition.datetimeColumn === col;
                  const isItemId = featureDefinition.itemIdColumn === col;
                  const isCategorical = featureDefinition.categoricalFeatures.includes(col);
                  const isNumerical = featureDefinition.numericalFeatures.includes(col);
                  const columnInfo = dataInfo.columnInfo[col];
                  
                  return (
                    <div key={col} className="flex items-center justify-between p-2 border border-tertiary rounded-md">
                      <div className="flex items-center space-x-2">
                        <Checkbox 
                          id={`feature-${col}`}
                          checked={isFeature}
                          disabled={isDateTime || isItemId}
                          onCheckedChange={(checked: boolean) => handleColumnSelection(col, checked)}
                          className="border-tertiary data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label htmlFor={`feature-${col}`} className="cursor-pointer">
                          {col}
                        </Label>
                        {columnInfo && (
                          <span className="text-xs text-foreground/70">
                            ({columnInfo.column_type})
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
            
            {Object.keys(visualizations).length > 0 && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-secondary">Data Visualizations</h3>
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
              </div>
            )}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("import")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={handleFeatureDefinition} 
          disabled={!featureDefinition.features.length || !featureDefinition.datetimeColumn || loading}
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
          Choose an anomaly detection algorithm and configure its hyperparameters.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div>
            <Label htmlFor="algorithm">Anomaly Detection Algorithm</Label>
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
                        {param.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
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
          Preprocess data and train the anomaly detection model.
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
                  {selectedAlgorithm}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Features:</span>
                <span className="font-medium text-foreground">{featureDefinition.features.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/80">Datetime Column:</span>
                <span className="font-medium text-foreground">{featureDefinition.datetimeColumn}</span>
              </div>
              {featureDefinition.itemIdColumn && (
                <div className="flex justify-between">
                  <span className="text-foreground/80">Item ID Column:</span>
                  <span className="font-medium text-foreground">{featureDefinition.itemIdColumn}</span>
                </div>
              )}
              
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
                {progressMessage || "Processing..."}
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
        <CardTitle className="text-primary">Model Evaluation & Predictions</CardTitle>
        <CardDescription className="text-foreground/80">
          Review the anomaly detection results and make predictions on new data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        {evaluation && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-primary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-primary">Total Points</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.anomaly_count + evaluation.metrics.normal_count}
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-secondary/10 border-b border-tertiary">
                  <CardTitle className="text-lg text-secondary">Anomalies</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.anomaly_count}
                  </p>
                </CardContent>
              </Card>
              
              <Card className="border-tertiary">
                <CardHeader className="py-2 bg-tertiary/20 border-b border-tertiary">
                  <CardTitle className="text-lg text-foreground">Anomaly %</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <p className="text-3xl font-bold text-foreground">
                    {evaluation.metrics.anomaly_percentage.toFixed(2)}%
                  </p>
                </CardContent>
              </Card>
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div>
              <h3 className="text-lg font-medium text-secondary mb-2">Anomaly Visualizations</h3>
              <Accordion type="single" collapsible className="w-full">
                {Object.entries(evaluation.visualizations).map(([key, value]) => (
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
            </div>
            
            <Separator className="bg-tertiary/50" />
            
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-secondary">Make Predictions on New Data</h3>
              
              <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="predictionFile">New Data for Prediction</Label>
                <Input 
                  id="predictionFile" 
                  type="file" 
                  accept=".csv,.xlsx,.xls"
                  onChange={handlePredictionFileUpload}
                  disabled={loading}
                  className="border-tertiary"
                />
                <p className="text-sm text-foreground/70">
                  Upload a file with the same features as your training data
                </p>
              </div>
              
              <Button 
                onClick={handlePredictions} 
                disabled={!predictionFile || loading || !modelTrained}
                variant="default"
              >
                Detect Anomalies
              </Button>
              
              {predictions.length > 0 && (
                <div>
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
                            <tr key={i} className={`border-t border-tertiary/50 ${row.anomaly === 'Yes' ? 'bg-red-100 dark:bg-red-900/20' : ''}`}>
                              {Object.entries(row).map(([key, value]) => (
                                <td key={`${i}-${key}`} className="p-2 text-sm text-foreground">
                                  {value?.toString() ?? ""}
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
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t border-tertiary">
        <Button variant="outline" onClick={() => setActiveTab("training")} className="border-tertiary">
          Back
        </Button>
        <Button 
          onClick={handleSaveResults} 
          disabled={!evaluation}
          variant="secondary"
        >
          Save Results
        </Button>
      </CardFooter>
    </Card>
  );
  
  return (
    <PageLayout>
      <div className="container mx-auto py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold"><span className="text-primary">Anomaly</span> <span className="text-secondary">Detection</span></h1>
          <p className="text-foreground">
            Detect anomalies in industrial machine data using unsupervised learning.
          </p>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 border border-tertiary">
            <TabsTrigger value="import" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">1. Import Data</TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">2. Define Features</TabsTrigger>
            <TabsTrigger value="algorithm" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">3. Select Algorithm</TabsTrigger>
            <TabsTrigger value="training" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">4. Train & Evaluate</TabsTrigger>
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
          </div>
        </Tabs>
      </div>
    </PageLayout>
  );
}