/*****************************************************
 * GenerateAndPredict.cpp
 *
 * Refactored main program that calls utility modules and
 * the run_patient_group_predictions.py script for final model predictions.
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 *
 * Usage Examples:
 *   GenerateAndPredict.exe --population=100
 *   GenerateAndPredict.exe --population=100 --enable-xai
 *   GenerateAndPredict.exe --population=100 --threads=8
 *   GenerateAndPredict.exe --population=100 --performance-mode
 *   GenerateAndPredict.exe --population=100 --extreme-performance
 *   GenerateAndPredict.exe --population=100 --memory-util=90 --cpu-util=95
 *
 * DEBUGGING GUIDE:
 * ================
 * This application comes with several debugging tools to diagnose issues:
 *
 * 1. debug_patient_data.exe - Tests patient data processing and index calculations
 *    Usage: debug_patient_data.exe 
 *    Output: patient_debug.csv - Contains patient IDs and their calculated indices
 *    Use when: You suspect patient health indices aren't calculating correctly
 *
 * 2. run_debug.exe - Validates medical dictionaries and tests condition code matching
 *    Usage: run_debug.exe
 *    Output: Console logs showing dictionary contents and condition matching rates
 *    Use when: You need to verify that condition codes map correctly to weights
 *
 * 3. thread_counter_debug.exe - Tests the ThreadSafeCounter class for correct operation
 *    Usage: thread_counter_debug.exe
 *    Output: Console logs showing counter operations and verification
 *    Use when: You suspect threading issues with patient index calculations
 *
 * COMMON DEBUGGING SCENARIOS:
 * ==========================
 * 1. Zero/missing patient health indicators:
 *    - Run debug_patient_data.exe to see if indices are being calculated
 *    - Check patient_debug.csv for summary of affected patients
 *    - Run run_debug.exe to verify condition code mappings and match rates
 *    - Check medical dictionaries in MedicalDictionaries.cpp for correct entries
 *
 * 2. Python model prediction failures:
 *    - Verify PatientFeatures.csv format using the feature_validator.py script
 *    - Ensure Python environment has pytorch_tabnet, torch and sklearn installed
 *    - Check that models exist in expected locations using verifyModelsExist()
 *    - Look at Python output logs for specific error messages
 *
 * 3. Performance issues:
 *    - Try reducing thread count with: --threads=4
 *    - For large files, use --performance-mode
 *    - On memory-constrained systems, try lower memory-util: --memory-util=50
 *
 * COMPILING DEBUG TOOLS:
 * ====================
 * Use build.bat to compile all components including debug tools:
 *   build.bat
 *
 * To compile individually:
 *   g++ -std=c++17 debug_patient_data.cpp FileProcessing.cpp MedicalDictionaries.cpp DataStructures.cpp -o debug_patient_data
 *   g++ -std=c++17 run_debug.cpp FileProcessing.cpp MedicalDictionaries.cpp DataStructures.cpp HealthIndex.cpp -o run_debug
 *   g++ -std=c++17 ThreadSafeCounterDebug.cpp DataStructures.cpp -o thread_counter_debug
 *****************************************************/

 #define NOMINMAX  // Prevent Windows macro conflicts

 #include <Python.h>
 #include <iostream>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <cstdlib>
 #include <chrono>
 #include <thread>
 #include <filesystem>
 #include <set>  // Added for std::set
 #include <functional> // Added for std::function
 #include <fstream> // Added for std::ifstream
 #include <algorithm> // Added for std::transform

 // Define SCRIPT_DIR - path to the script directory (current directory by default)
 #ifdef _WIN32
 const std::string SCRIPT_DIR = ".";
 #else
 const std::string SCRIPT_DIR = ".";
 #endif
 
 // Headers for refactored modules/utility code - reordered to prevent redefinition issues
 #include "DataStructures.h"       // Include first as it contains BatchProcessor
 // Comment out BatchProcessor.h to avoid the redefinition
 // #include "BatchProcessor.h"     // This causes redefinition with DataStructures.h
 #include "Utilities.h"            
 #include "FileProcessing.h"       
 #include "PatientSubsets.h"       
 #include "FeatureUtils.h"         
 #include "HealthIndex.h"          
 #include "MedicalDictionaries.h"  
 #include "ThreadPool.h"           
 #include "ResourceMonitor.h"
 #include "SystemResources.h"
 #include "BatchProcessorTuner.h"
 
 #ifdef _WIN32
 #include <windows.h>
 #else
 #include <unistd.h>
 #endif
 
 namespace fs = std::filesystem;

 // Function to calculate age based on birthdate - declare it before main
 int calculateAge(const std::string& birthdate);
 
 // Add this function to select only the features expected by the model
 static std::vector<std::string> getRequiredFeatures() {
     // Use the utility function to ensure consistency with model expectations
     return getModelExpectedFeatures();
 }

// Add this declaration before main()
std::string getCurrentTimestampString();
 
 //-----------------------------------------------------------------------------------
 // Command-line argument parsing-----------------------------------------------------
 //-----------------------------------------------------------------------------------
 static void parseCommandLineArgs(
     int argc, char* argv[],
     int &populationSize,
     bool &enableXAI,
     bool &perfMode,
     bool &extremePerf,
     float &memUtil,
     float &cpuUtil,
     unsigned int &threadCount,
     bool &forceCPU)  // Added parameter
 {   
     // Default values (some are also defined as extern in Utilities.h, but
     // we can override them here)
     populationSize   = 100;
     enableXAI        = false;
     perfMode         = false;
     extremePerf      = false;
     memUtil          = 70.0f;   // as percent
     cpuUtil          = 80.0f;   // as percent
     threadCount      = DEFAULT_THREAD_COUNT;
     forceCPU         = false;   // Default to using GPU if available
     
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];
         if (arg.rfind("--population=", 0) == 0) {
             populationSize = std::stoi(arg.substr(13));
         } else if (arg == "--enable-xai") {
             enableXAI = true;
         } else if (arg == "--performance-mode") {
             perfMode = true;
         } else if (arg == "--extreme-performance") {
             extremePerf = true;
         } else if (arg.rfind("--memory-util=", 0) == 0) {
             memUtil = std::stof(arg.substr(14));
         } else if (arg.rfind("--cpu-util=", 0) == 0) {
             cpuUtil = std::stof(arg.substr(11));
         } else if (arg.rfind("--threads=", 0) == 0) {
             threadCount = static_cast<unsigned int>(std::stoi(arg.substr(10)));
         } else if (arg == "--force-cpu") {
             forceCPU = true;
         }   
         // Add more CLI parsing here if needed
     }   
 }
 
 //-----------------------------------------------------------------------------------
 // Example function to call the tabnet_inference.py script using the system command.
 // Alternatively, you can embed Python via Py_Initialize, or any other approach.
 //-----------------------------------------------------------------------------------
 static bool predictUsingPythonScript(const std::string &modelID,
                                      const std::string &inputCSV,
                                      bool forceCPU = false)
 {                                    
     // Construct the system call for your Python script. 
     // Example usage: python tabnet_inference.py <model_id> <csv_for_inference> [--force-cpu]
 #ifdef _WIN32
     const std::string pyCmdBase = "python";  // or "python.exe", depending on environment
 #else
     const std::string pyCmdBase = "python3"; // or "python", depending on your environment
 #endif
     std::ostringstream cmd;
     cmd << pyCmdBase
         << " tabnet_inference.py"
         << " " << modelID
         << " " << inputCSV;
     if (forceCPU) {
         cmd << " --force-cpu";
     }   
     std::cout << "[INFO] Invoking: " << cmd.str() << "\n";
     int ret = std::system(cmd.str().c_str());
     if (ret != 0) {
         std::cerr << "[ERROR] tabnet_inference.py returned error code: " << ret << "\n";
         return false;
     }   
     return true;
 }
 
 //-----------------------------------------------------------------------------------
 // Function to find the most recent CSV files with a specific prefix in the Data directory
 //-----------------------------------------------------------------------------------
 static std::vector<std::string> findMostRecentCSVs(const std::string &filePrefix) {
     std::vector<std::string> matchingFiles;
     std::string latestTimestamp;
     
     // Search in the Data directory
     fs::path dataDir = "Data";
     if (!fs::exists(dataDir)) {
         std::cerr << "[WARNING] Directory does not exist: " << dataDir << "\n";
         return matchingFiles;
     }   
     
     // First pass: find the most recent timestamp
     for (const auto &entry : fs::directory_iterator(dataDir)) {
         if (entry.is_regular_file() && entry.path().extension() == ".csv") {
             std::string filename = entry.path().filename().string();
             if (filename.find(filePrefix) == 0) { // Starts with prefix
                 // Extract timestamp from filename (assuming format prefix_diff_timestamp.csv)
                 size_t pos = filename.find("_diff_");
                 if (pos != std::string::npos) {
                     std::string timestamp = filename.substr(pos + 6, 15); // Extract timestamp
                     if (timestamp > latestTimestamp) {
                         latestTimestamp = timestamp;
                     }   
                 }   
             }   
         }   
     }   
     
     // Second pass: collect files with the latest timestamp
     if (!latestTimestamp.empty()) {
         for (const auto &entry : fs::directory_iterator(dataDir)) {
             if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                 std::string filename = entry.path().filename().string();
                 if (filename.find(filePrefix) == 0 && filename.find(latestTimestamp) != std::string::npos) {
                     matchingFiles.push_back(entry.path().string());
                 }   
             }   
         }   
     }   
     
     return matchingFiles;
 }
 
 // Global variable to track inference status
 static bool GPU_INFERENCE_FAILED = false;
 
 //-----------------------------------------------------------------------------------
 // Check if Python is installed and required packages are available
 //-----------------------------------------------------------------------------------
 static bool verifyPythonEnvironment() {
     std::cout << "[INFO] Verifying Python environment...\n";
     
 #ifdef _WIN32
     const std::string checkCmd = "python -c \"import sys; import pytorch_tabnet; import torch; import sklearn; "
                                  "print('Python:', sys.version); "
                                  "print('Torch:', torch.__version__); "
                                  "print('All required packages available.')\"";
 #else
     const std::string checkCmd = "python3 -c \"import sys; import pytorch_tabnet; import torch; import sklearn; "
                                  "print('Python:', sys.version); "
                                  "print('Torch:', torch.__version__); "
                                  "print('All required packages available.')\"";
 #endif
 
     int result = std::system(checkCmd.c_str());
     if (result != 0) {
         std::cerr << "[ERROR] Failed to verify Python environment. Please ensure Python and required packages are installed.\n";
         return false;
     }
     
     std::cout << "[INFO] Python environment verified successfully.\n";
     return true;
 }
 
 //-----------------------------------------------------------------------------------
 // Check if model files exist before attempting inference
 //-----------------------------------------------------------------------------------
 static bool verifyModelsExist() {
     std::vector<std::string> requiredModels = {
         "combined_diabetes_tabnet_model",
         "combined_all_ckd_tabnet_model",
         "combined_none_tabnet_model"
     };
     
     bool allFound = true;
     for (const auto& model : requiredModels) {
         bool found = false;
         
         // Extract the base part before "_model" if it exists
         std::string baseModelName = model;
         size_t modelPos = model.find("_model");
         if (modelPos != std::string::npos) {
             baseModelName = model.substr(0, modelPos);
         }
         
         // Create paths to search with additional paths from Python implementation
         std::vector<std::string> searchPaths;
         
         // Add main script directory paths
         searchPaths.push_back((fs::path(SCRIPT_DIR) / (model + ".zip")).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / model).string());
         
         // Add Data/finals paths (matches Python's FINALS_DIR)
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "finals" / (model + ".zip")).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "finals" / model).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "finals" / baseModelName / (model + ".zip")).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "finals" / baseModelName / model).string());
         
         // Add Data/models paths
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "models" / (model + ".zip")).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "models" / model).string());
         
         // Add paths that match the exact TabNet model format used in Python
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "finals" / baseModelName / (model + "_model.zip")).string());
         searchPaths.push_back((fs::path(SCRIPT_DIR) / "Data" / "models" / (model + "_model.zip")).string());
         
         for (const auto& path : searchPaths) {
             if (fs::exists(path)) {
                 std::cout << "[INFO] Found model: " << path << "\n";
                 found = true;
                 break;
             }
         }
         
         if (!found) {
             std::cerr << "[ERROR] Could not find model: " << model << "\n";
             allFound = false;
         }
     }
     
     return allFound;
 }
 
 //-----------------------------------------------------------------------------------
 // Function to run model predictions for different patient subsets using Python
 //-----------------------------------------------------------------------------------
 static bool runMultiModelPredictions(const std::string &featuresCSV, bool forceCPU = false)
 {
     // First verify Python environment and model files
     if (!verifyPythonEnvironment()) {
         std::cerr << "[ERROR] Python environment check failed. Cannot run inference.\n";
         return false;
     }
     
     if (!verifyModelsExist()) {
         std::cerr << "[WARNING] Some model files not found. Inference may fail.\n";
         // Continue anyway as models might be in an unexpected location
     }
     
     // Ensure input file exists and is readable
     fs::path inputPath(featuresCSV);
     if (!fs::exists(inputPath)) {
         std::cerr << "[ERROR] Input CSV not found: " << featuresCSV << "\n";
         return false;
     }
     
     // Create output directory if it doesn't exist
     fs::path outDir = "Data/new_predictions";
     if (!fs::exists(outDir)) {
         fs::create_directories(outDir);
     }
     
     // Define model configurations similar to the Python MODEL_CONFIG_MAP
     // We'll pass this information to the Python script
 #ifdef _WIN32
     const std::string pyCmdBase = "python";
     const std::string tempOutputFile = "temp_py_output.txt";
 #else
     const std::string pyCmdBase = "python3";
     const std::string tempOutputFile = "/tmp/temp_py_output.txt";
 #endif

     // Locate the Python script
     fs::path pythonScriptPath = "run_patient_group_predictions.py";
     if (!fs::exists(pythonScriptPath)) {
         // Try SCRIPT_DIR location
         pythonScriptPath = fs::path(SCRIPT_DIR) / "run_patient_group_predictions.py";
         if (!fs::exists(pythonScriptPath)) {
             // Try vitai_scripts subdirectory
             pythonScriptPath = "vitai_scripts/run_patient_group_predictions.py";
             if (!fs::exists(pythonScriptPath)) {
                 std::cerr << "[ERROR] Required Python script not found: run_patient_group_predictions.py\n";
                 return false;
             }
         }
     }
     std::cout << "[INFO] Found Python script at: " << pythonScriptPath << std::endl;

     std::ostringstream cmd;
     // Use exact path to Python script with proper quotes
     cmd << pyCmdBase << " \"" << pythonScriptPath.string() << "\""
         << " --input-csv=\"" << fs::absolute(inputPath).string() << "\"";
     if (forceCPU) {
         cmd << " --force-cpu";
     }   
     // Redirect output to a temporary file for analysis
     cmd << " > " << tempOutputFile << " 2>&1";
     std::cout << "[INFO] Running multi-model predictions for patient groups...\n";
     std::cout << "[INFO] Invoking: " << cmd.str() << "\n";
     
     int ret = std::system(cmd.str().c_str());
     if (ret != 0) {
         std::cerr << "[ERROR] run_patient_group_predictions.py returned error code: " << ret << "\n";
         // Read and display error output to help diagnose the issue
         std::ifstream outputFile(tempOutputFile);
         if (outputFile.is_open()) {
             std::cout << "\n[ERROR OUTPUT] --------------------\n";
             std::string line;
             while (std::getline(outputFile, line)) {
                 std::cout << line << std::endl;
             }   
             std::cout << "[END ERROR OUTPUT] ----------------\n\n";
             outputFile.close();
         }   
         return false;
     }   
     
     // Read the output file to check for errors
     std::ifstream outputFile(tempOutputFile);
     if (!outputFile.is_open()) {
         std::cerr << "[ERROR] Could not open Python script output file.\n";
         return false;
     }   
     std::string line;
     bool errorFound = false;
     while (std::getline(outputFile, line)) {
         std::cout << line << std::endl;  // Echo the Python output to console
         
         // Check for error indicators in the output
         if (line.find("ERROR:") != std::string::npos || 
             line.find("Inference error:") != std::string::npos ||
             line.find("Exception:") != std::string::npos) {
             errorFound = true;
         }   
     }   
     outputFile.close();
     
     // Clean up the temporary file
     std::remove(tempOutputFile.c_str());
     
     if (errorFound) {
         std::cerr << "[ERROR] Errors detected in model inference process.\n";
         return false;
     }   
     std::cout << "[INFO] Multi-model inference completed successfully.\n";
     
     // Verify that output files were created
     std::string timestamp = std::to_string(std::time(nullptr)); // Simple timestamp
     // Use the actual timestamp format from Python (YYYYMMDD_HHMMSS) for better matching
     std::string currentDate = std::string("_") + getCurrentTimestampString();
     
     std::vector<std::string> expectedOutputs = {
         "combined_diabetes_tabnet_predictions_",
         "combined_all_ckd_tabnet_predictions_",
         "combined_none_tabnet_predictions_"
     };
     
     bool outputsFound = false;
     for (const auto& prefix : expectedOutputs) {
         // Check for files with this prefix created in the last 5 minutes
         for (const auto& entry : fs::directory_iterator("Data/new_predictions")) {
             std::string filename = entry.path().filename().string();
             if (filename.find(prefix) == 0) {
                 fs::file_time_type fileTime = fs::last_write_time(entry.path());
                 auto now = fs::file_time_type::clock::now();
                 auto diff = std::chrono::duration_cast<std::chrono::minutes>(now - fileTime).count();
                 if (diff < 5) { // File created in last 5 minutes
                     outputsFound = true;
                     std::cout << "[INFO] Found output file: " << filename << "\n";
                     break;
                 }
             }
         }
     }
     
     if (!outputsFound) {
         std::cerr << "[WARNING] No recent output files found. Inference may have failed.\n";
     }
     
     return true;
 }
 
 //-----------------------------------------------------------------------------------
 // main()
 //-----------------------------------------------------------------------------------
 int main(int argc, char* argv[])
 {
     // 1. Parse command-line arguments
     int populationSize;
     bool enableXAI;
     bool perfMode;
     bool extremePerf;
     float memTargetPercent;
     float cpuTargetPercent;
     unsigned int userThreads;
     bool forceCPU;  // Added variable
     
     parseCommandLineArgs(argc, argv,
                          populationSize,
                          enableXAI,
                          perfMode,
                          extremePerf,
                          memTargetPercent,
                          cpuTargetPercent,
                          userThreads,
                          forceCPU);  // Added parameter
     
     // Initialize medical dictionaries before using them
     initializeDirectLookups();
     initializeElixhauserLookups(); // Add this call to initialize Elixhauser dictionary
     initializeObsAbnormalDirect();
     
     // 3. Always run Synthea to generate required data files
     // Ensure we have a valid population size (minimum 1)
     if (populationSize <= 0) {
         std::cout << "[WARNING] Invalid population size specified, defaulting to 100.\n";
         populationSize = 100;
     }   
     std::cout << "[INFO] Running Synthea to generate " << populationSize << " patient records...\n";
     runSynthea(populationSize);  // from Utilities.cpp
     copySyntheaOutput();         // from Utilities.cpp
     
     // 4. Identify CSV files (patients, conditions, etc.) - MODIFIED SECTION
     std::vector<std::string> patientFiles = findMostRecentCSVs("patients");
     std::vector<std::string> condFiles = findMostRecentCSVs("conditions");
     std::vector<std::string> encFiles = findMostRecentCSVs("encounters");
     std::vector<std::string> medFiles = findMostRecentCSVs("medications");
     std::vector<std::string> obsFiles = findMostRecentCSVs("observations");
     std::vector<std::string> procFiles = findMostRecentCSVs("procedures");
     
     // Log the files found
     std::cout << "[INFO] Found " << patientFiles.size() << " patient files\n";
     std::cout << "[INFO] Found " << condFiles.size() << " condition files\n";
     std::cout << "[INFO] Found " << encFiles.size() << " encounter files\n";
     std::cout << "[INFO] Found " << medFiles.size() << " medication files\n";
     std::cout << "[INFO] Found " << obsFiles.size() << " observation files\n";
     std::cout << "[INFO] Found " << procFiles.size() << " procedure files\n";
     
     // 5. Read patient data
     std::cout << "[INFO] Reading patient records...\n";
     std::vector<PatientRecord> allPatients;
     try {
         for (auto &pf : patientFiles) {
             processPatientsInBatches(pf, [&](const PatientRecord &p){
                 allPatients.push_back(p);
             }); 
         }   
     } catch (const std::exception &e) {
         std::cerr << "[ERROR] Failed to process patient files: " << e.what() << "\n";
         return 1;
     }   
     
     // Update ages after loading patient data but before calculating health metrics
     std::cout << "[INFO] Calculating patient ages based on birthdate..." << std::endl;
     for (auto &p : allPatients) {
         p.Age = calculateAge(p.Birthdate);
     }
     
     // 6. Use concurrency for calculating indexes/counters (Charlson, Elixhauser, meds, etc.)
     // Shared concurrency-safe counters
     ThreadSafeCounter charlsonCounter;
     ThreadSafeCounter elixhauserCounter;
     ThreadSafeCounter comorbCounter;
     ThreadSafeCounter hospitalCounter;
     ThreadSafeCounter medsCounter;
     ThreadSafeCounter abnormalObsCounter;
     
     // Conditions -> Charlson / Elixhauser / comorbidity
     std::cout << "[INFO] Processing conditions files (" << condFiles.size() << " files)...\n";
     int fileCounter = 0;
     bool verboseDebug = true;  // Enable verbose debug output to see what's happening
     int debugCount = 0;         // Counter to limit the number of debug messages
     
     // Add a debug counter for code matches
     int totalConditions = 0;
     int matchedCharlson = 0;
     int matchedElixhauser = 0;
     int matchedDescription = 0;
     
     // Process conditions with enhanced debug output
     for (auto &condFile : condFiles) {
         fileCounter++;
         std::cout << "[INFO] Processing condition file " << fileCounter << "/" << condFiles.size() << ": " 
                   << condFile << std::endl;
         
         processConditionsInBatches(condFile, [&](const ConditionRow &cRow) {
             totalConditions++;
             
             // Debug output for first 20 conditions
             if (debugCount < 20) {
                 std::cout << "[DEBUG] Processing condition: " << cRow.CODE << " - " << cRow.DESCRIPTION << std::endl;
                 debugCount++;
             }
             
             // Try exact code match first
             auto charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(cRow.CODE);
             if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
                 charlsonCounter.addFloat(cRow.PATIENT, charlsonIt->second);
                 matchedCharlson++;
                 if (debugCount < 40) {
                     std::cout << "[DEBUG] Added Charlson weight " << charlsonIt->second 
                             << " for patient " << cRow.PATIENT 
                             << " (code match: " << cRow.CODE << ")"<< std::endl;
                     debugCount++;
                 }
             }
             
             auto elixhauserIt = ELIXHAUSER_CODE_TO_WEIGHT.find(cRow.CODE);
             if (elixhauserIt != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                 elixhauserCounter.addFloat(cRow.PATIENT, elixhauserIt->second);
                 matchedElixhauser++;
             }
             
             // Try prefix match if exact match fails
             if (charlsonIt == CHARLSON_CODE_TO_WEIGHT.end() && cRow.CODE.length() >= 3) {
                 std::string prefix = cRow.CODE.substr(0, 3);
                 charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(prefix);
                 if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
                     charlsonCounter.addFloat(cRow.PATIENT, charlsonIt->second);
                     matchedCharlson++;
                     if (debugCount < 60) {
                         std::cout << "[DEBUG] Added Charlson weight " << charlsonIt->second 
                                 << " for patient " << cRow.PATIENT 
                                 << " (prefix match: " << prefix << " from " << cRow.CODE << ")" << std::endl;
                         debugCount++;
                     }
                 }
             }
             
             // Normalize description and try text matching (like Python's approach)
             std::string lowerDesc = cRow.DESCRIPTION;
             std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(), 
                          [](unsigned char c){ return std::tolower(c); });
                          
             // Check for keywords in description like Python
             if (lowerDesc.find("diabetes") != std::string::npos) {
                 charlsonCounter.addFloat(cRow.PATIENT, 2.0f); // Match Python weight of 2
                 elixhauserCounter.addFloat(cRow.PATIENT, 0.5f);
                 matchedDescription++;
                 if (debugCount < 80) {
                     std::cout << "[DEBUG] Added diabetes weight via description match for patient "
                               << cRow.PATIENT << std::endl;
                     debugCount++;
                 }
             }
             else if (lowerDesc.find("heart failure") != std::string::npos) {
                 charlsonCounter.addFloat(cRow.PATIENT, 3.0f); // Match Python cardiovascular weight
                 elixhauserCounter.addFloat(cRow.PATIENT, 1.5f);
                 matchedDescription++;
             }
             else if (lowerDesc.find("copd") != std::string::npos || 
                      lowerDesc.find("chronic obstructive pulmonary") != std::string::npos) {
                 charlsonCounter.addFloat(cRow.PATIENT, 2.0f); // Match Python respiratory weight
                 elixhauserCounter.addFloat(cRow.PATIENT, 0.9f);
                 matchedDescription++;
             }
         });
     }
     
     // Print statistics about condition processing
     std::cout << "[INFO] Condition processing stats:" << std::endl;
     std::cout << "  Total conditions: " << totalConditions << std::endl;
     std::cout << "  Matched Charlson code: " << matchedCharlson << " (" 
               << (matchedCharlson * 100.0f / totalConditions) << "%)" << std::endl;
     std::cout << "  Matched Elixhauser code: " << matchedElixhauser << " (" 
               << (matchedElixhauser * 100.0f / totalConditions) << "%)" << std::endl;
     std::cout << "  Matched by description: " << matchedDescription << " (" 
               << (matchedDescription * 100.0f / totalConditions) << "%)" << std::endl;

     std::cout << std::endl;
     
     // Encounters -> hospitalization count
     for (auto &encFile : encFiles) {
         processEncountersInBatches(encFile, [&](const EncounterRow &enc){
             if (enc.ENCOUNTERCLASS == "inpatient") {
                 hospitalCounter.increment(enc.PATIENT);
             }   
         }); 
     }   
     
     // Medications -> medication count
     for (auto &mFile : medFiles) {
         processMedicationsInBatches(mFile, [&](const MedicationRow &mr){
             // Each line can be considered a medication row
             medsCounter.increment(mr.PATIENT);
         }); 
     }   
     
     // Observations -> count abnormal observations
     std::cout << "[INFO] Processing observation files...\n";
     size_t nonNumericCount = 0;
     std::set<std::string> uniqueNonNumericDescriptions;
     std::map<std::string, int> nonNumericByDescription;
     bool verboseWarnings = false;  // Set to true to see individual warnings
     
     for (auto &oFile : obsFiles) {
         processObservationsInBatches(oFile, [&](const ObservationRow &obs){
             try {
                 double value = std::stod(obs.VALUE);
                 if (isAbnormalObsFast(obs.DESCRIPTION, value)) {
                     abnormalObsCounter.increment(obs.PATIENT);
                 }   
             } catch (const std::exception& e) {
                 // Track non-numeric values instead of printing each warning
                 nonNumericCount++;
                 uniqueNonNumericDescriptions.insert(obs.DESCRIPTION);
                 nonNumericByDescription[obs.DESCRIPTION]++;
                 // Only print individual warnings if verbose mode is enabled
                 if (verboseWarnings) {
                     std::cerr << "[WARNING] Non-numeric observation value: " << obs.VALUE 
                               << " for description: " << obs.DESCRIPTION << "\n";
                 }   
             }   
         }); 
     }   
     
     // Print a summary of non-numeric observations
     if (nonNumericCount > 0) {
         std::cout << "[WARNING] Found " << nonNumericCount << " non-numeric observation values across " 
                   << uniqueNonNumericDescriptions.size() << " different description types.\n";
         // Optionally show top descriptions with non-numeric values
         std::cout << "[INFO] Top description with non-numeric values: \n";
         std::vector<std::pair<std::string, int>> sortedDescriptions(
             nonNumericByDescription.begin(), nonNumericByDescription.end());
         std::sort(sortedDescriptions.begin(), sortedDescriptions.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
         const int maxToShow = 3;  // Limit the number of descriptions shown
         for (int i = 0; i < std::min(maxToShow, static_cast<int>(sortedDescriptions.size())); ++i) {
             std::cout << "  - Description ID: " << sortedDescriptions[i].first 
                       << " (" << sortedDescriptions[i].second << " occurrences)\n";
         }   
     }   
     
     // Debug counters before updating patients
     std::cout << "[DEBUG] First 5 patients from charlsonCounter:\n";
     debugCount = 0;  // Reset the counter instead of redefining it
     for (auto& p : allPatients) {
         if (debugCount < 5) {
             std::cout << "   Patient " << p.Id << ": CharlsonIndex=" 
                       << charlsonCounter.getFloat(p.Id) 
                       << ", Hospitalizations=" << hospitalCounter.getInt(p.Id)
                       << ", Medications=" << medsCounter.getInt(p.Id) << "\n";
             debugCount++;
         }
     }
     
     // Debug counter values after processing all files
     std::cout << "[DEBUG] Random sample of counter values:\n";
     for (int i = 0; i < 10 && i < allPatients.size(); ++i) {
         std::string patientId = allPatients[i].Id;
         std::cout << "Patient " << patientId << ":\n"
                   << "  Charlson: " << charlsonCounter.getFloat(patientId) << "\n"
                   << "  Elixhauser: " << elixhauserCounter.getFloat(patientId) << "\n" 
                   << "  Comorbidity: " << comorbCounter.getFloat(patientId) << "\n"
                   << "  Hospitalizations: " << hospitalCounter.getInt(patientId) << "\n"
                   << "  Medications: " << medsCounter.getInt(patientId) << "\n"
                   << "  Abnormal Obs: " << abnormalObsCounter.getInt(patientId) << "\n";
     }
     
     // 7. Update allPatients with these concurrency-safe counters
     // Using original capitalized field names from PatientRecord struct
     for (auto &p : allPatients) {
         p.CharlsonIndex              = charlsonCounter.getFloat(p.Id);  // Use getFloat instead of casting from getInt
         p.ElixhauserIndex            = elixhauserCounter.getFloat(p.Id);  // Use getFloat directly
         p.Comorbidity_Score          = comorbCounter.getFloat(p.Id);  // Use getFloat directly
         p.Hospitalizations_Count     = hospitalCounter.getInt(p.Id);
         p.Medications_Count          = medsCounter.getInt(p.Id);
         p.Abnormal_Observations_Count= abnormalObsCounter.getInt(p.Id);
         
         // Recompute health index
         p.Health_Index = computeHealthIndex(p);
     }   
     
     // Debug the updated patient records
     std::cout << "[DEBUG] First 5 patient records after updating:\n";
     debugCount = 0;  // Reset the counter again
     for (auto& p : allPatients) {
         if (debugCount < 5) {
             std::cout << "   Patient " << p.Id << ": CharlsonIndex=" << p.CharlsonIndex 
                       << ", Hospitalizations=" << p.Hospitalizations_Count 
                       << ", Medications=" << p.Medications_Count << "\n";
             debugCount++;
         }
     }
     
     // 8. (Optional) Identify subsets (diabetes, CKD, etc.) using PatientSubsets
     std::cout << "[INFO] Loading ConditionRows for subset identification...\n";
     std::vector<ConditionRow> allConds;
     for (auto &condFile : condFiles) {
         processConditionsInBatches(condFile, [&](const ConditionRow &c){
             allConds.push_back(c);
         }); 
     }   
     
     // Find diabetic patients with more comprehensive checks
     std::set<std::string> diabeticSet;
     std::set<std::string> ckdSet;
     
     // Check both CODE and DESCRIPTION for better identification
     std::cout << "[INFO] Identifying patient subgroups with enhanced criteria...\n";
     for (const auto& cond : allConds) {
         std::string code = cond.CODE;
         std::string description = cond.DESCRIPTION;
         
         // Convert to lowercase for case-insensitive matching
         std::transform(description.begin(), description.end(), description.begin(), 
                      [](unsigned char c){ return std::tolower(c); });
         
         // Check for diabetes - match Python implementation (subset_utils.py)
         // Python uses: mask = conditions["DESCRIPTION"].str.lower().str.contains("diabetes", na=False)
         if (description.find("diabetes") != std::string::npos) {
             diabeticSet.insert(cond.PATIENT);
         }   
         
         // Check for CKD - match Python implementation (subset_ckd)
         // Python defines the same exact codes and case-insensitive text search
         std::set<std::string> ckdCodes = {"431855005", "431856006", "433144002", "431857002", "46177005"};
         
         if (ckdCodes.find(code) != std::string::npos || 
             description.find("chronic kidney disease") != std::string::npos) {
             ckdSet.insert(cond.PATIENT);
         }
     }   
     
     // Remove arbitrary percentage allocation and replace with proper fallback check
     if (diabeticSet.empty()) {
         std::cout << "[WARNING] No diabetic patients found by condition codes/descriptions.\n";
         std::cout << "[INFO] This might indicate issues with the condition data or coding system.\n";
     }   
     std::cout << "[INFO] Found " << diabeticSet.size() << " diabetic patients using clinical criteria.\n";
     std::cout << "[INFO] Found " << ckdSet.size() << " CKD patients using clinical criteria.\n";
     
     // For verification, we can also log patient IDs to a file for debugging
     std::ofstream diabeticFile("diabetic_patients.txt");
     if (diabeticFile.is_open()) {
         diabeticFile << "# Diabetic patients identified in C++\n";
         for (const auto& id : diabeticSet) {
             diabeticFile << id << "\n";
         }   
         diabeticFile.close();
         std::cout << "[INFO] Wrote diabetic patient IDs to diabetic_patients.txt\n";
     }   
     
     // 9. Create feature CSV for inference or training
     std::string featureConfig = "combined_all"; 
     // Get expected features with exact capitalization 
     auto featureCols = getRequiredFeatures(); 
     // Add this before writeFeaturesCSV to normalize features
     std::cout << "[INFO] Normalizing features to match Python scaling..." << std::endl;
     normalizePatientFeatures(allPatients);  // Add this function to FeatureUtils.cpp
     writeFeaturesCSV(allPatients, "PatientFeatures.csv", featureCols);
     // Validate the feature CSV before running predictions
     validateFeatureCSV("PatientFeatures.csv");
     
     // Still write a comprehensive CSV of everything including Health_Index for other purposes    
     saveFinalDataCSV(allPatients, "AllPatientsData.csv");
     
     // Before running predictions, ensure the Python script exists
     // Check multiple locations for required Python scripts
     fs::path pythonScriptPath = "run_patient_group_predictions.py";
     if (!fs::exists(pythonScriptPath)) {
         // Try SCRIPT_DIR location
         pythonScriptPath = fs::path(SCRIPT_DIR) / "run_patient_group_predictions.py";
         if (!fs::exists(pythonScriptPath)) {
             // Try vitai_scripts subdirectory
             pythonScriptPath = "vitai_scripts/run_patient_group_predictions.py";
             if (!fs::exists(pythonScriptPath)) {
                 std::cerr << "[ERROR] Required Python script not found: run_patient_group_predictions.py\n";
                 return 1;
             }
         }
     }
     std::cout << "[INFO] Found Python script at: " << pythonScriptPath << std::endl;
     
     // Run model inspector to check our model requirements
     std::cout << "[INFO] Running model inspector to verify embedding dimensions...\n";
     std::system("python model_inspector.py");
     
     // 10. Call Python script for patient grouping and multi-model predictions
     bool ok = false;

     // First try the adapted tabnet solution
     std::cout << "[INFO] Running inference with TabNet adapter...\n";
     // Change the model parameter to match what tabnet_adapter.py expects
     std::string adapterCmd = "python tabnet_adapter.py --input=\"" + fs::absolute("PatientFeatures.csv").string() + 
                             "\" --model=\"combined_diabetes_tabnet\"";
     // Fix: Use output-dir instead of output_dir, and remove the = sign
     adapterCmd += " --output-dir \"Data/new_predictions\"";
     if (forceCPU) {
         adapterCmd += " --force-cpu";
     }
     int adapterResult = std::system(adapterCmd.c_str());
     if (adapterResult == 0) {
         ok = true;
         std::cout << "[INFO] TabNet adapter ran successfully.\n";
     } else {
         std::cout << "[INFO] TabNet adapter failed, falling back to standard method.\n";
         // Fall back to standard method
         ok = runMultiModelPredictions("PatientFeatures.csv", forceCPU);
     }
     
     // (Optional) If XAI is enabled, do something here
     if (enableXAI) {
         std::cout << "[INFO] Running XAI for each patient group...\n";
         // Construct the command for XAI
         std::ostringstream xaiCmd;
         xaiCmd << "python run_xai_analysis.py";
         if (forceCPU || GPU_INFERENCE_FAILED) {  // Force CPU for XAI if GPU inference failed
             xaiCmd << " --force-cpu";
         }   
         int xaiRet = std::system(xaiCmd.str().c_str());
         if (xaiRet != 0) {
             std::cerr << "[ERROR] XAI analysis failed with code: " << xaiRet << "\n";
         }   
     }   
     
     // Update return code based on inference success
     if (GPU_INFERENCE_FAILED) {
         std::cout << "[INFO] GenerateAndPredict completed with inference issues. Check logs for details.\n";
         return 1;  // Return non-zero to indicate issues occurred
     } else {
         std::cout << "[INFO] GenerateAndPredict completed successfully.\n";
         return 0;
     }   
 }

// Function to calculate age based on birthdate - moved before main
int calculateAge(const std::string& birthdate) {
    if (birthdate.empty() || birthdate.length() < 10) {
        return 30; // Default age if birthdate is invalid
    }
    
    try {
        // Extract year from birthdate (format: YYYY-MM-DD)
        int birthYear = std::stoi(birthdate.substr(0, 4));
        
        // Get current year
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        struct tm* currentTime = std::localtime(&now_c);
        int currentYear = currentTime->tm_year + 1900;
        
        // Calculate age
        int age = currentYear - birthYear;
        
        // Basic validation
        if (age < 0) age = 0;
        if (age > 120) age = 120;
        
        return age;
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Error calculating age from birthdate: " << birthdate << ": " << e.what() << std::endl;
        return 30; // Default on error
    }
}

// Add this helper function to get timestamp string in the same format as Python
std::string getCurrentTimestampString() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &now_c);
#else
    localtime_r(&local_tm, &now_c);
#endif
    
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_tm);
    return std::string(buffer);
}