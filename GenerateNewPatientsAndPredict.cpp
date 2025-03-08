/*****************************************************
 * GenerateAndPredict.cpp
 *
 * A single monolithic C++ program that replicates:
 *   1) data_preprocessing.py (merge old & new CSVs, set NewData)
 *   2) health_index.py (compute comorbidity-based features and a Health_Index)
 *   3) data_prep.py (merge Charlson/Elixhauser, produce final integrated dataset)
 *   4) subset_utils.py (filter by diabetes/ckd/none)
 *   5) feature_utils.py (select columns: composite, cci, eci, combined, etc.)
 *   6) final pipeline (run Synthea, copy files, do transformations).
 *   7) Embeds Python to load TabNet models and produce predictions
 *      in Data/new_predictions/<model_id>/...
 *   8) Optionally runs the XAI script if --enable-xai is set.
 *
 * It does *not* write or read .pkl files. Instead, it stores an
 * equivalent final dataset in CSV form named 'patient_data_with_all_indices.csv'.
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 *****************************************************/

 #include <Python.h> // For embedded Python
 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <map>
 #include <unordered_map>
 #include <set>
 #include <algorithm>
 #include <functional>
 #include <sys/stat.h>
 #include <cstdlib>
 #include <cstdio>
 #include <chrono>
 #include <ctime>
 
 // For CSV parsing (external single-header library)
 #include "csv.hpp"
 
 #ifdef _WIN32
   #include <direct.h>
   #define MKDIR(x) _mkdir(x)
 #else
   #include <unistd.h>
   #define MKDIR(x) mkdir(x, 0755)
 #endif
 
 /****************************************************
  * Global Config & Structures
  ****************************************************/
 static const std::string DATA_DIR  = "Data";
 static const std::string SYN_DIR   = "synthea-master";
 static const std::string SYN_OUT   = "output/csv";
 
 // Similar to your Python MODEL_CONFIG_MAP: model_id -> (subset, feature_config)
 struct ModelConfig {
     std::string subset;
     std::string feature_config;
 };
 static std::map<std::string, ModelConfig> MODEL_CONFIG_MAP = {
     {"combined_diabetes_tabnet", {"diabetes", "combined"}},
     {"combined_all_ckd_tabnet",  {"ckd",      "combined_all"}},
     {"combined_none_tabnet",     {"none",     "combined"}}
 };
 
 // --------------------------------------------------
 // Data structures to hold loaded records
 // --------------------------------------------------
 struct PatientRecord {
     std::string Id;
     std::string BIRTHDATE;
     std::string DEATHDATE;
     std::string GENDER;
     std::string RACE;
     std::string ETHNICITY;
     double HEALTHCARE_EXPENSES=0.0;
     double HEALTHCARE_COVERAGE=0.0;
     double INCOME=0.0;
     std::string MARITAL;
     bool   NewData=false;   // Mark if loaded from a "_diff_" file
 
     // We also keep a simplistic approach for age & deceased
     double AGE=0.0;
     bool   DECEASED=false;
 
     // For Charlson/Elixhauser
     double CharlsonIndex=0.0;
     double ElixhauserIndex=0.0;
 
     // For additional health metrics
     double Comorbidity_Score=0.0;
     int    Hospitalizations_Count=0;
     int    Medications_Count=0;
     int    Abnormal_Observations_Count=0;
 
     // The final "Health_Index"
     double Health_Index=0.0;
 };
 
 struct ConditionRow {
     std::string PATIENT;
     std::string CODE;
     std::string DESCRIPTION;
 };
 struct EncounterRow {
     std::string Id;
     std::string PATIENT;
     std::string ENCOUNTERCLASS;
 };
 struct MedicationRow {
     std::string PATIENT;
     std::string ENCOUNTER;
     std::string CODE;
     std::string DESCRIPTION;
 };
 struct ObservationRow {
     std::string PATIENT;
     std::string ENCOUNTER;
     std::string CODE;
     std::string DESCRIPTION;
     double VALUE=0.0;
     std::string UNITS;
 };
 struct ProcedureRow {
     std::string PATIENT;
     std::string ENCOUNTER;
     std::string CODE;
     std::string DESCRIPTION;
 };
 
 /****************************************************
  * Utility: mkdir, timestamp, runSynthea, copySynthea
  ****************************************************/
 static void makeDirIfNeeded(const std::string &dir) {
     struct stat st;
     if (stat(dir.c_str(), &st) != 0) {
 #ifdef _WIN32
         _mkdir(dir.c_str());
 #else
         mkdir(dir.c_str(), 0755);
 #endif
     }
 }
 
 static std::string getTimestamp() {
     auto now = std::chrono::system_clock::now();
     auto t_c = std::chrono::system_clock::to_time_t(now);
     struct tm *parts = std::localtime(&t_c);
     char buf[64];
     std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", parts);
     return std::string(buf);
 }
 
 static void runSynthea(int popSize) {
 #ifdef _WIN32
     // On Windows, you might have run_synthea.bat
     // e.g. "cd synthea-master && run_synthea.bat -p 100"
     std::string cmd = "cd " + SYN_DIR + " && run_synthea.bat -p " + std::to_string(popSize);
 #else
     // On Linux / macOS
     std::string cmd = "cd " + SYN_DIR + " && ./run_synthea -p " + std::to_string(popSize);
 #endif
     std::cout << "[INFO] Running Synthea: " << cmd << "\n";
     int ret = std::system(cmd.c_str());
     if (ret != 0) {
         std::cerr << "[ERROR] Synthea generation failed.\n";
         std::exit(1);
     }
     std::cout << "[INFO] Synthea generation complete.\n";
 }
 
 static void copySyntheaOutput() {
     std::string synOutput = SYN_DIR + "/" + SYN_OUT;
     struct stat st;
     if (stat(synOutput.c_str(), &st) != 0) {
         std::cerr << "[ERROR] Synthea output dir " << synOutput << " not found.\n";
         std::exit(1);
     }
     makeDirIfNeeded(DATA_DIR);
 
     // The main CSV files we want
     std::vector<std::string> needed = {
         "patients.csv", "encounters.csv", "conditions.csv",
         "medications.csv", "observations.csv", "procedures.csv"
     };
     std::string stamp = getTimestamp();
 
     for (auto &fname : needed) {
         std::string src = synOutput + "/" + fname;
         if (stat(src.c_str(), &st) != 0) {
             std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
             continue;
         }
         // rename => e.g., "patients_diff_20250308_130045.csv"
         auto dotPos = fname.rfind('.');
         std::string base = (dotPos == std::string::npos ? fname : fname.substr(0, dotPos));
         std::string ext  = (dotPos == std::string::npos ? ""    : fname.substr(dotPos));
         std::string newName = base + "_diff_" + stamp + ext;
         std::string dst = DATA_DIR + "/" + newName;
 
 #ifdef _WIN32
         std::string cmd = "copy " + src + " " + dst;
 #else
         std::string cmd = "cp " + src + " " + dst;
 #endif
         std::cout << "[INFO] Copying " << src << " => " << dst << "\n";
         int ret = std::system(cmd.c_str());
         if (ret != 0) {
             std::cerr << "[ERROR] copy cmd failed: " << cmd << "\n";
         }
     }
 }
 
 /****************************************************
  * Loading CSVs: we want to load both the original
  * (non-diff) and all diff files, then combine them.
  ****************************************************/
 
 // Helper to see if a filename is "diff"
 static bool isDiffFile(const std::string &fname) {
     return (fname.find("_diff_") != std::string::npos);
 }
 
 // List all CSVs that start with <prefix> in Data/
 static std::vector<std::string> listCSVFiles(const std::string &prefix) {
     // In a real scenario you might do platform-specific directory listing;
     // for simplicity, we’ll guess up to 50 versions.
     // You can refine if you want to read the entire directory.
     // We'll check <prefix>*.csv or <prefix>_diff_*.csv
     std::vector<std::string> found;
     for (int i = 0; i < 200; i++) {
         // e.g. "patients.csv", "patients_diff_20250308_123455.csv", ...
         // We can try a pattern approach.
         // We'll attempt two forms: "prefix.csv" or "prefix_diff_i.csv"
         // but to replicate Python you'd check all that match prefix*
         // For demonstration: we'll do prefix alone and prefix_diff_...
         std::ostringstream oss;
         if (i == 0) {
             // "patients.csv" or "encounters.csv", etc.
             oss << DATA_DIR << "/" << prefix << ".csv";
         } else {
             // "patients_diff_1.csv", "patients_diff_2.csv", ...
             oss << DATA_DIR << "/" << prefix << "_diff_" << i << ".csv";
         }
         std::string path = oss.str();
         struct stat st;
         if (stat(path.c_str(), &st) == 0) {
             found.push_back(path);
         }
     }
     // We do the same pattern with a timestamp approach or just rely on the above "i" approach.
     // In practice you can scan the directory for all files that start with <prefix>.
     return found;
 }
 
 /****************************************************
  * 1) LOADING PATIENTS
  *    We replicate "data_preprocessing.py" logic:
  *    - If file name has "_diff_", set NewData = true
  *    - Otherwise NewData = false
  ****************************************************/
 static void loadPatients(std::vector<PatientRecord> &allPatients) {
     auto patientFiles = listCSVFiles("patients"); // e.g. "patients.csv", "patients_diff_..."
 
     for (auto &path : patientFiles) {
         bool isDiff = isDiffFile(path);
         try {
             csv::CSVReader reader(path);
             for (auto &row : reader) {
                 PatientRecord p;
                 p.Id        = row["Id"].get<>();
                 p.BIRTHDATE = row["BIRTHDATE"].get<>();
                 p.DEATHDATE = row["DEATHDATE"].get<>();
                 p.GENDER    = row["GENDER"].get<>();
                 p.RACE      = row["RACE"].get<>();
                 p.ETHNICITY = row["ETHNICITY"].get<>();
                 p.MARITAL   = row["MARITAL"].get<>();
 
                 try { p.HEALTHCARE_EXPENSES = row["HEALTHCARE_EXPENSES"].get<double>(); } catch(...) {}
                 try { p.HEALTHCARE_COVERAGE = row["HEALTHCARE_COVERAGE"].get<double>(); } catch(...) {}
                 try { p.INCOME = row["INCOME"].get<double>(); } catch(...) {}
 
                 p.NewData = isDiff;  // mark all from this file the same
                 // approximate: if DEATHDATE is non-empty => DECEASED
                 if (!p.DEATHDATE.empty() && p.DEATHDATE != "NaN") {
                     p.DECEASED = true;
                 }
                 // naive approximation for AGE (just ignoring date)
                 // you could parse BIRTHDATE properly if desired
                 p.AGE = 50.0; // placeholder
 
                 allPatients.push_back(p);
             }
         } catch(...) {
             std::cerr << "[ERROR] parsing " << path << "\n";
         }
     }
 }
 
 /****************************************************
  * 2) Loading Conditions (and so forth)
  ****************************************************/
 static void loadConditions(std::vector<ConditionRow> &conds) {
     auto condFiles = listCSVFiles("conditions");
     for (auto &path : condFiles) {
         try {
             csv::CSVReader rd(path);
             for (auto &row : rd) {
                 ConditionRow c;
                 c.PATIENT     = row["PATIENT"].get<>();
                 c.CODE        = row["CODE"].get<>();
                 c.DESCRIPTION = row["DESCRIPTION"].get<>();
                 conds.push_back(c);
             }
         } catch(...) {}
     }
 }
 static void loadEncounters(std::vector<EncounterRow> &encs) {
     auto encFiles = listCSVFiles("encounters");
     for (auto &path : encFiles) {
         try {
             csv::CSVReader rd(path);
             for (auto &row : rd) {
                 EncounterRow e;
                 e.Id            = row["Id"].get<>();
                 e.PATIENT       = row["PATIENT"].get<>();
                 e.ENCOUNTERCLASS= row["ENCOUNTERCLASS"].get<>();
                 encs.push_back(e);
             }
         } catch(...) {}
     }
 }
 static void loadMedications(std::vector<MedicationRow> &meds) {
     auto files = listCSVFiles("medications");
     for (auto &path : files) {
         try {
             csv::CSVReader rd(path);
             for (auto &row : rd) {
                 MedicationRow m;
                 m.PATIENT    = row["PATIENT"].get<>();
                 m.ENCOUNTER  = row["ENCOUNTER"].get<>();
                 m.CODE       = row["CODE"].get<>();
                 m.DESCRIPTION= row["DESCRIPTION"].get<>();
                 meds.push_back(m);
             }
         } catch(...) {}
     }
 }
 static void loadObservations(std::vector<ObservationRow> &obs) {
     auto files = listCSVFiles("observations");
     for (auto &path : files) {
         try {
             csv::CSVReader rd(path);
             for (auto &row : rd) {
                 ObservationRow o;
                 o.PATIENT     = row["PATIENT"].get<>();
                 o.ENCOUNTER   = row["ENCOUNTER"].get<>();
                 o.CODE        = row["CODE"].get<>();
                 o.DESCRIPTION = row["DESCRIPTION"].get<>();
                 try {
                     o.VALUE = row["VALUE"].get<double>();
                 } catch(...) {
                     o.VALUE = 0.0;
                 }
                 o.UNITS = row["UNITS"].get<>();
                 obs.push_back(o);
             }
         } catch(...) {}
     }
 }
 static void loadProcedures(std::vector<ProcedureRow> &procs) {
     auto files = listCSVFiles("procedures");
     for (auto &path : files) {
         try {
             csv::CSVReader rd(path);
             for (auto &row : rd) {
                 ProcedureRow p;
                 p.PATIENT     = row["PATIENT"].get<>();
                 p.ENCOUNTER   = row["ENCOUNTER"].get<>();
                 p.CODE        = row["CODE"].get<>();
                 p.DESCRIPTION = row["DESCRIPTION"].get<>();
                 procs.push_back(p);
             }
         } catch(...) {}
     }
 }
 
 /****************************************************
  * 3) Charlson / Elixhauser logic
  *    (We’ll embed partial mappings)
  ****************************************************/
 static std::map<std::string,std::string> CHARLSON_MAP = {
     {"22298006","Myocardial infarction"},
     {"88805009","Congestive heart failure"},
     {"62479008","AIDS/HIV"}
 };
 static std::map<std::string,int> CHARLSON_WEIGHTS = {
     {"Myocardial infarction",1},
     {"Congestive heart failure",1},
     {"AIDS/HIV",1}
 };
 
 static std::map<std::string,std::string> ELIX_MAP = {
     {"88805009","Congestive heart failure"}
 };
 static std::map<std::string,int> ELIX_WEIGHTS = {
     {"Congestive heart failure",7}
 };
 
 static int getCharlsonWeight(const std::string &code) {
     auto it = CHARLSON_MAP.find(code);
     if (it != CHARLSON_MAP.end()) {
         // e.g. Myocardial infarction
         auto wIt = CHARLSON_WEIGHTS.find(it->second);
         if (wIt != CHARLSON_WEIGHTS.end()) {
             return wIt->second;
         }
     }
     return 0;
 }
 static int getElixWeight(const std::string &code) {
     auto it = ELIX_MAP.find(code);
     if (it != ELIX_MAP.end()) {
         auto wIt = ELIX_WEIGHTS.find(it->second);
         if (wIt != ELIX_WEIGHTS.end()) {
             return wIt->second;
         }
     }
     return 0;
 }
 
 /****************************************************
  * 4) Health Index logic (basic approximation)
  ****************************************************/
 // SNOMED groups => codes => weights
 static std::map<std::string,std::vector<std::string>> SNOMED_GROUPS = {
     {"Cardiovascular Diseases",{"53741008","445118002","22298006"}},
     {"Respiratory Diseases",   {"19829001","233604007"}},
     {"Diabetes",               {"44054006","73211009"}},
     {"Cancer",                 {"363346000","254637007"}}
 };
 static std::map<std::string,double> GROUP_WEIGHTS = {
     {"Cardiovascular Diseases",3.0},
     {"Respiratory Diseases",  2.0},
     {"Diabetes",              2.0},
     {"Cancer",                3.0},
     {"Other",                 1.0}
 };
 
 // For observations => abnormal
 struct ObsThreshold { double minVal; double maxVal; };
 static std::map<std::string,ObsThreshold> OBS_THRESHOLDS = {
     {"Systolic Blood Pressure",{90,120}},
     {"Body Mass Index",{18.5,24.9}}
     // You can expand these as needed
 };
 // Maps from internal strings to a "standard" that has thresholds
 static std::map<std::string,std::string> OBS_DESC_MAP = {
     {"Systolic Blood Pressure","Systolic Blood Pressure"},
     {"Body mass index (BMI) [Ratio]","Body Mass Index"}
 };
 
 static double findGroupWeight(const std::string &code) {
     // Check each named group
     for (auto &kv : SNOMED_GROUPS) {
         for (auto &c : kv.second) {
             if (c == code) {
                 return GROUP_WEIGHTS[kv.first];
             }
         }
     }
     return GROUP_WEIGHTS["Other"];
 }
 
 static bool isAbnormalObs(const std::string &desc, double val) {
     // If desc matches one of our known obs, check threshold
     auto it = OBS_DESC_MAP.find(desc);
     if (it != OBS_DESC_MAP.end()) {
         std::string standard = it->second;
         auto thr = OBS_THRESHOLDS.find(standard);
         if (thr != OBS_THRESHOLDS.end()) {
             if (val < thr->second.minVal || val > thr->second.maxVal) {
                 return true;
             }
         }
     }
     return false;
 }
 
 // A simple formula to mimic the python approach
 static double computeHealthIndex(const PatientRecord &p) {
     double base = 10.0;
     double penalty1 = 0.4 * p.Comorbidity_Score;
     double penalty2 = 1.0 * p.Hospitalizations_Count;
     double penalty3 = 0.2 * p.Medications_Count;
     double penalty4 = 0.3 * p.Abnormal_Observations_Count;
     double penalty5 = 0.1 * p.CharlsonIndex + 0.05 * p.ElixhauserIndex;
 
     double raw = base - (penalty1 + penalty2 + penalty3 + penalty4 + penalty5);
     if (raw < 1) raw = 1;
     if (raw > 10) raw = 10;
     return raw;
 }
 
 /****************************************************
  * 5) Subset Utils
  ****************************************************/
 static std::set<std::string> findDiabeticPatients(const std::vector<ConditionRow> &conds) {
     std::set<std::string> out;
     for (auto &c : conds) {
         std::string lower = c.DESCRIPTION;
         for (auto &ch : lower) ch = tolower(ch);
         if (lower.find("diabetes") != std::string::npos) {
             out.insert(c.PATIENT);
         }
     }
     return out;
 }
 static std::set<std::string> findCKDPatients(const std::vector<ConditionRow> &conds) {
     std::set<std::string> out;
     for (auto &c : conds) {
         std::string lower = c.DESCRIPTION;
         for (auto &ch : lower) ch = tolower(ch);
         if (lower.find("chronic kidney disease") != std::string::npos ||
             lower.find("ckd") != std::string::npos) {
             out.insert(c.PATIENT);
         }
     }
     return out;
 }
 
 static std::vector<PatientRecord> filterSubpopulation(
     const std::vector<PatientRecord> &allP,
     const std::string &subsetType,
     const std::vector<ConditionRow> &conds
 ) {
     if (subsetType == "none") {
         return allP;
     } else if (subsetType == "diabetes") {
         auto diabs = findDiabeticPatients(conds);
         std::vector<PatientRecord> sub;
         for (auto &p : allP) {
             if (diabs.count(p.Id)) {
                 sub.push_back(p);
             }
         }
         return sub;
     } else if (subsetType == "ckd") {
         auto ckdSet = findCKDPatients(conds);
         std::vector<PatientRecord> sub;
         for (auto &p : allP) {
             if (ckdSet.count(p.Id)) {
                 sub.push_back(p);
             }
         }
         return sub;
     }
     // default => entire set
     return allP;
 }
 
 /****************************************************
  * 6) Feature Utils
  ****************************************************/
 static std::vector<std::string> getFeatureCols(const std::string &feature_config) {
     // Base columns from your Python feature_utils:
     std::vector<std::string> base = {
         "Id","GENDER","RACE","ETHNICITY","MARITAL",
         "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","INCOME",
         "AGE","DECEASED",
         "Hospitalizations_Count","Medications_Count","Abnormal_Observations_Count"
     };
     // Now add depending on feature_config
     if (feature_config == "composite") {
         base.push_back("Health_Index");
     } else if (feature_config == "cci") {
         base.push_back("CharlsonIndex");
     } else if (feature_config == "eci") {
         base.push_back("ElixhauserIndex");
     } else if (feature_config == "combined") {
         base.push_back("Health_Index");
         base.push_back("CharlsonIndex");
     } else if (feature_config == "combined_eci") {
         base.push_back("Health_Index");
         base.push_back("ElixhauserIndex");
     } else if (feature_config == "combined_all") {
         base.push_back("Health_Index");
         base.push_back("CharlsonIndex");
         base.push_back("ElixhauserIndex");
     } else {
         std::cerr << "[ERROR] invalid feature_config: " << feature_config << "\n";
     }
     return base;
 }
 
 static void writeFeaturesCSV(const std::vector<PatientRecord> &pats,
                              const std::string &outFile,
                              const std::vector<std::string> &cols)
 {
     std::ofstream ofs(outFile);
     if (!ofs.is_open()) {
         std::cerr << "[ERROR] can't open " << outFile << "\n";
         return;
     }
     // Header
     for (size_t i = 0; i < cols.size(); i++) {
         ofs << cols[i];
         if (i + 1 < cols.size()) ofs << ",";
     }
     ofs << "\n";
 
     for (auto &p : pats) {
         for (size_t c = 0; c < cols.size(); c++) {
             if (c > 0) ofs << ",";
             const auto &col = cols[c];
             if (col == "Id") {
                 ofs << p.Id;
             } else if (col == "GENDER") {
                 ofs << p.GENDER;
             } else if (col == "RACE") {
                 ofs << p.RACE;
             } else if (col == "ETHNICITY") {
                 ofs << p.ETHNICITY;
             } else if (col == "MARITAL") {
                 ofs << p.MARITAL;
             } else if (col == "HEALTHCARE_EXPENSES") {
                 ofs << p.HEALTHCARE_EXPENSES;
             } else if (col == "HEALTHCARE_COVERAGE") {
                 ofs << p.HEALTHCARE_COVERAGE;
             } else if (col == "INCOME") {
                 ofs << p.INCOME;
             } else if (col == "AGE") {
                 ofs << p.AGE;
             } else if (col == "DECEASED") {
                 ofs << (p.DECEASED ? "1" : "0");
             } else if (col == "Hospitalizations_Count") {
                 ofs << p.Hospitalizations_Count;
             } else if (col == "Medications_Count") {
                 ofs << p.Medications_Count;
             } else if (col == "Abnormal_Observations_Count") {
                 ofs << p.Abnormal_Observations_Count;
             } else if (col == "Health_Index") {
                 ofs << p.Health_Index;
             } else if (col == "CharlsonIndex") {
                 ofs << p.CharlsonIndex;
             } else if (col == "ElixhauserIndex") {
                 ofs << p.ElixhauserIndex;
             } else {
                 ofs << 0; // fallback
             }
         }
         ofs << "\n";
     }
     ofs.close();
     std::cout << "[INFO] Wrote features => " << outFile << "\n";
 }
 
 /****************************************************
  * 7) Save final data akin to "patient_data_with_all_indices.pkl"
  *    but in CSV form
  ****************************************************/
 static void saveFinalDataCSV(const std::vector<PatientRecord> &pats,
                              const std::string &outfile)
 {
     std::ofstream ofs(outfile);
     if (!ofs.is_open()) {
         std::cerr << "[ERROR] can't open " << outfile << "\n";
         return;
     }
     ofs << "Id,BIRTHDATE,DEATHDATE,GENDER,RACE,ETHNICITY,"
         << "HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME,MARITAL,NewData,"
         << "CharlsonIndex,ElixhauserIndex,Comorbidity_Score,"
         << "Hospitalizations_Count,Medications_Count,Abnormal_Observations_Count,"
         << "Health_Index,AGE,DECEASED\n";
 
     for (auto &p : pats) {
         ofs << p.Id << "," << p.BIRTHDATE << "," << p.DEATHDATE << ","
             << p.GENDER << "," << p.RACE << "," << p.ETHNICITY << ","
             << p.HEALTHCARE_EXPENSES << "," << p.HEALTHCARE_COVERAGE << ","
             << p.INCOME << "," << p.MARITAL << ","
             << (p.NewData ? "True" : "False") << ","
             << p.CharlsonIndex << "," << p.ElixhauserIndex << "," << p.Comorbidity_Score << ","
             << p.Hospitalizations_Count << "," << p.Medications_Count << "," << p.Abnormal_Observations_Count << ","
             << p.Health_Index << ","
             << p.AGE << ","
             << (p.DECEASED ? "1" : "0")
             << "\n";
     }
     ofs.close();
     std::cout << "[INFO] Wrote final data => " << outfile << "\n";
 }
 
 /****************************************************
  * 8) Embedding Python for TabNet Inference
  ****************************************************/
 static bool runPythonInference(const std::string &model_id,
                                const std::string &csvPath)
 {
     // We assume a Python script: "tabnet_inference.py"
     // that takes sys.argv = [script, model_id, csvPath]
     // loads the TabNet model from e.g. Data/finals/<model_id>/<model_id>_model.zip
     // reads <csvPath> for features
     // writes predictions to Data/new_predictions/<model_id>/
     std::string script = "tabnet_inference.py"; // must exist
 
     // Build Python code string that fakes sys.argv and runs the script
     std::ostringstream code;
     code << "import sys, runpy\n"
          << "sys.argv = ['" << script << "', '" << model_id << "', '" << csvPath << "']\n"
          << "runpy.run_path('" << script << "', run_name='__main__')\n";
 
     int rc = PyRun_SimpleString(code.str().c_str());
     if (rc != 0) {
         std::cerr << "[ERROR] Python inference for " << model_id
                   << " returned code " << rc << "\n";
         return false;
     }
     return true;
 }
 
 /****************************************************
  * 9) XAI script
  ****************************************************/
 static void runExplainability() {
     // If you have final_explain_xai_clustered_lime.py
     std::string script = "Explain_Xai/final_explain_xai_clustered_lime.py";
     FILE* fp = fopen(script.c_str(), "r");
     if (!fp) {
         std::cerr << "[WARN] can't open XAI script: " << script << "\n";
         return;
     }
     std::cout << "[INFO] Running XAI script: " << script << "\n";
     int ret = PyRun_SimpleFile(fp, script.c_str());
     fclose(fp);
     if (ret != 0) {
         std::cerr << "[ERROR] XAI script returned " << ret << "\n";
     }
 }
 
 /****************************************************
  * main()
  ****************************************************/
 int main(int argc, char* argv[]) {
     int popSize = 100;
     bool enableXAI = false;
 
     // parse command line
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];
         if (arg.rfind("--population=", 0) == 0) {
             popSize = std::stoi(arg.substr(13));
         } else if (arg == "--enable-xai") {
             enableXAI = true;
         }
     }
     std::cout << "[INFO] popSize=" << popSize
               << ", XAI=" << (enableXAI ? "true" : "false") << "\n";
 
     // 1) Generate new data via Synthea
     runSynthea(popSize);
     copySyntheaOutput();
 
     // 2) Load entire dataset (both old & diff) for each CSV type
     std::vector<PatientRecord> allPatients;
     loadPatients(allPatients);
 
     std::vector<ConditionRow> conds;
     loadConditions(conds);
 
     std::vector<EncounterRow> encs;
     loadEncounters(encs);
 
     std::vector<MedicationRow> meds;
     loadMedications(meds);
 
     std::vector<ObservationRow> obs;
     loadObservations(obs);
 
     std::vector<ProcedureRow> procs;
     loadProcedures(procs);
 
     // 3) Compute Charlson & Elixhauser
     // We'll accumulate them by patient
     std::unordered_map<std::string, double> charlMap, elixMap;
     for (auto &c : conds) {
         charlMap[c.PATIENT] += getCharlsonWeight(c.CODE);
         elixMap[c.PATIENT]  += getElixWeight(c.CODE);
     }
 
     // 4) Additional aggregator logic
     //    4a) Comorbidity_Score from SNOMED groups
     std::unordered_map<std::string, double> groupScore;
     for (auto &c : conds) {
         groupScore[c.PATIENT] += findGroupWeight(c.CODE);
     }
     //    4b) Hospitalizations => inpatient
     std::unordered_map<std::string, int> hospCount;
     for (auto &e : encs) {
         if (e.ENCOUNTERCLASS == "inpatient") {
             hospCount[e.PATIENT]++;
         }
     }
     //    4c) Distinct medication codes
     std::unordered_map<std::string, std::set<std::string>> medSet;
     for (auto &m : meds) {
         medSet[m.PATIENT].insert(m.CODE);
     }
     //    4d) Abnormal observations
     std::unordered_map<std::string, int> abnMap;
     for (auto &oRec : obs) {
         if (isAbnormalObs(oRec.DESCRIPTION, oRec.VALUE)) {
             abnMap[oRec.PATIENT]++;
         }
     }
 
     // 5) Merge into the PatientRecord
     //    We'll index them by Id for quick updates
     std::unordered_map<std::string, PatientRecord*> pMap;
     for (auto &p : allPatients) {
         pMap[p.Id] = &p;
     }
     for (auto &kv : charlMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->CharlsonIndex = kv.second;
         }
     }
     for (auto &kv : elixMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->ElixhauserIndex = kv.second;
         }
     }
     for (auto &kv : groupScore) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Comorbidity_Score = kv.second;
         }
     }
     for (auto &kv : hospCount) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Hospitalizations_Count = kv.second;
         }
     }
     for (auto &kv : medSet) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Medications_Count = (int)kv.second.size();
         }
     }
     for (auto &kv : abnMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Abnormal_Observations_Count = kv.second;
         }
     }
 
     // 6) Compute Health_Index
     for (auto &p : allPatients) {
         p.Health_Index = computeHealthIndex(p);
     }
 
     // 7) Save final big CSV
     std::string finalCSV = DATA_DIR + "/patient_data_with_all_indices.csv";
     saveFinalDataCSV(allPatients, finalCSV);
 
     // 8) We only run predictions on newly generated patients
     std::vector<PatientRecord> newPatients;
     for (auto &p : allPatients) {
         if (p.NewData) {
             newPatients.push_back(p);
         }
     }
     std::cout << "[INFO] Found " << newPatients.size() << " newly generated patients.\n";
 
     // Initialize Python
     Py_Initialize();
 
     // For each final model, filter subset, select features, write CSV, run inference
     for (auto &it : MODEL_CONFIG_MAP) {
         std::string model_id   = it.first;
         std::string subsetType = it.second.subset;
         std::string featconf   = it.second.feature_config;
 
         std::cout << "\n[INFO] Predicting with model=" << model_id
                   << ", subset=" << subsetType
                   << ", feature_config=" << featconf << "\n";
 
         // Filter subpopulation
         auto sub = filterSubpopulation(newPatients, subsetType, conds);
         if (sub.empty()) {
             std::cout << "[INFO] No new patients belong to subpop=" << subsetType
                       << " => skip.\n";
             continue;
         }
         // Get columns
         auto cols = getFeatureCols(featconf);
         // Write the CSV for inference
         std::string outDir = DATA_DIR + "/new_predictions/" + model_id;
         makeDirIfNeeded(DATA_DIR + "/new_predictions");
         makeDirIfNeeded(outDir);
 
         std::string infCSV = outDir + "/input_for_inference.csv";
         writeFeaturesCSV(sub, infCSV, cols);
 
         // Invoke Python to do TabNet inference
         bool ok = runPythonInference(model_id, infCSV);
         if (!ok) {
             std::cerr << "[WARN] Inference for " << model_id << " failed.\n";
         }
     }
 
     // Optionally, run advanced XAI
     if (enableXAI) {
         runExplainability();
     }
 
     // Finalize Python
     Py_Finalize();
 
     std::cout << "[INFO] Done. Generated new data, merged Charlson/Elixhauser, "
               << "computed health index, saved final CSV, ran TabNet, XAI="
               << (enableXAI ? "true" : "false") << ".\n";
     return 0;
 }
 