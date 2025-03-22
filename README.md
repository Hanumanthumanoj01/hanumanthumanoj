# ML 24/25-02 # ML Investigate Input reconstruction by using Classifiers
###### _Through out this project we contribute to  implement the Spatial Pooler SDR Reconstruction in NeoCortexAPI_This project implements input reconstruction using  classifiers (KNN and HtmClassifier) to regenerate scalar inputs from Spatial Pooler SDRs

[![N|Logo](https://ddobric.github.io/neocortexapi/images/logo-NeoCortexAPI.svg )](https://ddobric.github.io/neocortexapi/)


In this Documentation we will describe our contribution in this project..

#### Instruction for Running the Project
- Clone the Repository and Run
- You will get the project here
[NeoCortexApi-Team](https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/NeoCortexApi.Experiments)

#### Two Existing Classifiers
- **`HtmClassifier.cs`**: 
[HtmClassifier.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/HtmClassifier.cs)
- **`KnnClassifier.cs`**: 
[KnnClassifier.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs)

#### Experiment Code
- **`SpatialPoolerInputReconstructionExperiment.cs`**: The implementation of the Spatial Pooler Input Reconstruction Experiment can be found here: 
[SpatialPoolerInputReconstructionExperiment.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/NeoCortexApi.Experiments/SpatialPoolerInputReconstructionExperiment.cs)
- **`SpatialPoolerInputReconstructionExperimentTests.cs`**: The Unit Test for the experiment can be found here:
[SpatialPoolerInputReconstructionExperimentTests.cs](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs)

###### Image Input sets are already uploaded here
- [Documentation](https://github.com/prnshubn/neocortexapi-team-untitled/tree/master/source/Documentation_Team_Untitled)

###### All the output will be saved here
- [neocortexapi_team.yet to make it **********]

## Introduction
This project explores the concept of input reconstruction using classifiers within HTM (Hierarchical Temporal Memory). The goal is to analyze how well HTM and KNN classifiers can reconstruct the original input based on Sparse Distributed Representations (SDRs). This investigation is inspired by the SpatialLearning experiment and extends it by incorporating input reconstruction.

# Methodology
The Spatial Pooler Input Reconstruction Experiment investigates how well classifiers can reconstruct original input values from Sparse Distributed Representations (SDRs). The experiment follows a structured pipeline starting with data encoding, SDR generation, classifier training, and input reconstruction. First, numerical input values are encoded using a Scalar Encoder, which transforms continuous values into binary representations. These encoded values are then passed through the Spatial Pooler (SP), which learns stable patterns and generates SDRs. The Spatial Pooler applies synaptic learning rules to form a structured representation of input data, which serves as the basis for reconstruction.

Once the SDRs are generated, two classifiers—HTM Classifier and KNN Classifier—are trained to associate SDRs with their corresponding input values. The HTM Classifier learns temporal sequences, meaning it adapts over time to improve predictions, whereas the KNN Classifier memorizes SDRs and reconstructs inputs based on similarity to previously seen patterns. The classifiers are trained using an 80-20 split on the dataset, with 80% of the input values used for learning and the remaining 20% used for testing. During inference, classifiers attempt to predict the original input from unseen SDRs. The reconstructed values are then compared with actual inputs using similarity metrics.

To evaluate classifier performance, the results are visualized through similarity graphs and heatmaps. These visualizations show the accuracy of HTM and KNN predictions in reconstructing inputs from SDRs. By comparing their performance, we gain insights into which classifier is more effective for input reconstruction within HTM-based systems. The findings contribute to a better understanding of classification-based reconstruction techniques and their potential for enhancing Sparse Distributed Representations in machine learning applications.

**Fig: Methodology Flowchart**
![Methodology Flowchart](******** yet to add it ********)

## Training the HTM and KNN Classifiers

After SDRs are generated, the next step is training two different classifiers to learn and predict input values.

- **HTM Classifier:** This classifier learns temporal patterns over time. It associates SDRs with input values and refines its predictions as more data is observed.

 - **KNN Classifier (K-Nearest Neighbors):** This classifier memorizes SDRs and predicts new inputs by comparing them with previously stored representations, selecting the closest match.

## Training Process:
* The dataset is split into training (80%) and testing (20%) subsets.
* For each training value, an SDR is generated using the trained Spatial Pooler.
* During training, classifiers store SDR-input mappings to be used later during reconstruction.
* The trained classifiers are tested on unseen data, where they take an SDR as input and attempt to reconstruct the original input value.


## Reconstruct() Method:

The ReconstructionExperiment method runs the input reconstruction experiment by setting up the required components, training the Spatial Pooler (SP), and performing input reconstruction using HTM and KNN classifiers. It starts by defining HTM configurations and Scalar Encoder settings to convert input values into Sparse Distributed Representations (SDRs). The method then trains the Spatial Pooler to generate stable SDRs and passes them to the classifiers for learning. Once trained, the classifiers attempt to reconstruct the original input values from SDRs. Finally, the method evaluates reconstruction accuracy and visualizes results for comparison.
``` csharp
     public void ReconstructionExperiment(double max, int seedValue=0)
        {
            if (max < 10) throw new ArgumentException("max must be 10 or greater", nameof(max));
            
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPoolerInputReconstructionExperiment)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            HtmConfig cfg = new(new[] { inputBits }, new[] { numColumns })
            {
                CellsPerColumn = 10,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = true,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10
            };

            // Scalar Encoder settings
            Dictionary<string, object> settings = new()
            {
                { "W", 21 },
                { "N", inputBits },
                { "Radius", -1.0 },
                { "MinVal", 1.0 },
                { "MaxVal", max },
                { "Periodic", false },
                { "Name", "scalar" },
                { "ClipInput", false }
            };

            EncoderBase encoder = new ScalarEncoder(settings);
            List<double> inputValues = Enumerable.Range(1, (int)max).Select(i => (double)i).ToList();

            // Train the Spatial Pooler
            SpatialPooler sp = TrainSpatialPooler(cfg, encoder, inputValues);

            // Perform Reconstruction Experiment
            ClassifierPart(sp, encoder, inputValues, seedValue);
        }

```
[Reconstruction in SP](Need to be added ) - Lines (1442 to 1482)

#### Reconstruct() Workflow:
- **SDR Generation:** The input value is first encoded using the Scalar Encoder to create a Sparse Distributed Representation (SDR). This SDR is passed through the trained Spatial Pooler, which computes the active mini-columns.
   
- **Classifier Input Preparation:** The active mini-columns are converted into an array of Cells, as both the HTM Classifier and KNN Classifier require inputs in this format.
   
- **Prediction using Classifiers:** 
* The KNN Classifier retrieves the most similar SDR from its stored dataset and predicts the corresponding input value.
* The HTM Classifier uses temporal learning to predict the input value based on learned sequences.
   
- **Similarity Calculation:** The reconstructed values from HTM and KNN Classifiers are compared to the original input value using percentage similarity metrics 
   
- **Result Storage:** The reconstructed values, internal similarity scores, and percentage similarity are stored for later visualization and evaluation.
   
- **Visualization & Analysis:** The experiment results are plotted in similarity graphs, highlighting the accuracy and performance of both classifiers.

# TrainSpatialPooler
Trains the Spatial Pooler to generate stable Sparse Distributed Representations (SDRs) for inputs.
```csharp
     private SpatialPooler TrainSpatialPooler(HtmConfig cfg, EncoderBase encoder, List<double> inputs)
{
    Connections mem = new(cfg);
    SpatialPooler sp = new(new HomeostaticPlasticityController(mem, inputs.Count * 40));
    sp.Init(mem, new DistributedMemory());

    foreach (double input in inputs)
    {
        int[] sdr = encoder.Encode(input);
        int[] actCols = sp.Compute(sdr, true);
    }

    Console.WriteLine("STABLE STATE REACHED");
    return sp;
}

```
[Running Reconstruct Method ](git link with line number to be based ) - Lines (243 to 329)


# TrainSpatialPooler
The ReconstructionPart method generates SDRs for input data, predicts reconstructed values using HTM and KNN classifiers, compares them with original inputs using similarity metrics, and visualizes the results.
```csharp
   private void ReconstructionPart(List<double> dataset, EncoderBase encoder, SpatialPooler sp,
    KNeighborsClassifier<string, string> knnClassifier, HtmClassifier<string, string> htmClassifier, double max, string datasetType)
{
    Results.Clear();
    foreach (double data in dataset)
    {
        int[] sdr = encoder.Encode(data);
        int[] actCols = sp.Compute(sdr, false);
        Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

        ClassifierResult<string> knnPrediction = knnClassifier.GetPredictedInputValues(cells)[0];
        ClassifierResult<string> htmPrediction = htmClassifier.GetPredictedInputValues(cells)[0];

        double knnPercentageSimilarity = CalculatePercentageSimilarity(data, double.Parse(knnPrediction.PredictedInput));
        double htmPercentageSimilarity = CalculatePercentageSimilarity(data, double.Parse(htmPrediction.PredictedInput));

        Console.WriteLine($"KNN Prediction: {knnPrediction.PredictedInput}, Similarity: {knnPercentageSimilarity:P}");
        Console.WriteLine($"HTM Prediction: {htmPrediction.PredictedInput}, Similarity: {htmPercentageSimilarity:P}");

        Results[data] = (double.Parse(knnPrediction.PredictedInput), double.Parse(htmPrediction.PredictedInput), 
                         knnPrediction.Similarity, htmPrediction.Similarity / 100, 
                         knnPercentageSimilarity, htmPercentageSimilarity);
    }
}

# Hierarchical Temporal Memory (HTM) Spatial Pooler
Encoded inputs undergo transformation using the HTM Spatial Pooler to generate SDRs, which serve as the basis for classification and reconstruction.

- **Learning with Classifiers: ** 
```csharp
cls.Learn(key, actCells.ToArray());
```
- ** Predicting SDR from Classifiers: **
```csharp    var predictedInputValues = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3); ```


# Reconstruct Inputs
Once trained, classifiers attempt to predict the original input based on SDRs.

```csharp
var knnPrediction = knnClassifier.GetPredictedInputValues(cells)[0];
var htmPrediction = htmClassifier.GetPredictedInputValues(cells)[0];
```


# ClassifierPart()
Trains HTM and KNN classifiers using SDRs, then reconstructs input values for evaluation.
```csharp
private void ClassifierPart(SpatialPooler sp, EncoderBase encoder, List<double> inputValues, int seedValue)
{
    KNeighborsClassifier<string, string> knnClassifier = new();
    HtmClassifier<string, string> htmClassifier = new();

    foreach (double trainData in inputValues.Take((int)(inputValues.Count * 0.8)))
    {
        int[] sdr = encoder.Encode(trainData);
        int[] actCols = sp.Compute(sdr, false);
        Cell[] cells = actCols.Select(idx => new Cell { Index = idx }).ToArray();

        knnClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
        htmClassifier.Learn(trainData.ToString("F2", CultureInfo.InvariantCulture), cells);
    }

    // Run reconstruction on test data (20% unseen values)
    ReconstructionPart(inputValues.Skip((int)(inputValues.Count * 0.8)).ToList(), encoder, sp, knnClassifier, htmClassifier, inputValues.Max(), "Test");
}
```

Graphs to be added ****
Results are analyzed based on accuracy, stability, and computational efficiency.
Performance of HTM vs.s KNN







Similarity functions for validation










## Spatial Pooler Reconstruction Tests
## UnitTest of SdrReconstructionTests
We tested SpatialPoolerInputReconstructionExperimentTests.cs with 5 test cases, and all passed successfully. These tests validate the correct execution of the Spatial Pooler training, input reconstruction, and classifier accuracy.
[SdrReconstructionTests](https://github.com/prnshubn/neocortexapi-team-untitled/blob/master/source/UnitTestsProject/SpatialPoolerInputReconstructionExperimentTests.cs)

### Test_Experiment_Completes_Without_Exception
- **Test Category:** SpatialPoolerReconstruction
- **Description:** Verifies whether the ReconstructionExperiment() method runs without throwing errors. Ensures that all components (Scalar Encoder, Spatial Pooler, HTM & KNN Classifiers) initialize and execute correctly.

### Test_Experiment_With_Improper_Max_Value
- **Test Category:** ReconstructionExceptionHandling
- **Description:** Ensures that the ReconstructionExperiment() method throws an ArgumentException when an invalid max value (less than 10) is provided.

### Test_SpatialPoolerTraining_ReachesStableState
- **Test Category:** SpatialPoolerStability
- **Description:** Checks whether the Spatial Pooler reaches a stable state during training. Captures the console output and verifies that "STABLE STATE REACHED" appears, confirming that the Spatial Pooler has learned stable SDRs.

### Test_Reconstruction_ProducesPredictions
- **Test Category:** ClassifierPrediction
- **Description:** Verifies that the HTM and KNN classifiers successfully predict reconstructed inputs. Ensures that the console output contains predictions from both classifiers and similarity percentages.

### Test_ReconstructionPart_Results_Have_Valid_Similarity
- **Test Category:** ReconstructionAccuracy
- **Description:** Validates that similarity scores between reconstructed and actual inputs fall within the valid range (0% - 100%). Ensures that both HTM and KNN classifiers return meaningful predictions.
