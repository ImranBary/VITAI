/*****************************************************
 * GenerateAndPredict.cpp
 *
 * A single monolithic C++ file that replicates the logic of:
 *  1) data_preprocessing.py
 *  2) health_index.py
 *  3) data_prep.py (which merges Charlson/Elixhauser & final index)
 *  4) subset_utils.py (filter subpop: none, diabetes, ckd)
 *  5) feature_utils.py (select features: composite, cci, eci, combined, combined_eci, combined_all)
 *  6) final pipeline that:
 *     - Runs Synthea to generate N patients
 *     - Copies to Data/ as _diff_ files
 *     - "Preprocesses" them: building sequences, computing Charlson/Elix, health index, etc.
 *     - Produces a final CSV akin to "patient_data_with_all_indices.csv" (no pickle).
 *     - For each final TabNet model, filters subpop, selects features, calls embedded Python
 *       to load the TabNet model and produce predictions in Data/new_predictions/<model_id>/...
 *  7) Optionally runs a final XAI script if desired.
 *
 * This file does NOT write or read .pkl files. All data is stored in memory or CSV.
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 *****************************************************/

 #include <Python.h> // Embedded Python
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
 
 // For CSV parsing
 #include "csv.hpp"
 
 #ifdef _WIN32
   #include <direct.h>
   #define MKDIR(x) _mkdir(x)
 #else
   #include <unistd.h>
   #define MKDIR(x) mkdir(x, 0755)
 #endif
 
 /****************************************************
  * 1) Global Config & Structures
  ****************************************************/
 static const std::string DATA_DIR  = "Data";
 static const std::string SYN_DIR   = "synthea-master";
 static const std::string SYN_OUT   = "output/csv";
 
 // Like your Python MODEL_CONFIG_MAP
 // model_id => (subset_type, feature_config)
 struct ModelConfig {
     std::string subset;
     std::string feature_config;
 };
 static std::map<std::string, ModelConfig> MODEL_CONFIG_MAP = {
     {"combined_diabetes_tabnet", {"diabetes", "combined"}},
     {"combined_all_ckd_tabnet",  {"ckd",      "combined_all"}},
     {"combined_none_tabnet",     {"none",     "combined"}}
 };
 
 struct PatientRecord {
     // Basic columns
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
     bool   NewData=false;
 
     // Preprocessing indices
     double CharlsonIndex=0.0;
     double ElixhauserIndex=0.0;
 
     // Additional fields
     double Comorbidity_Score=0.0;
     int    Hospitalizations_Count=0;
     int    Medications_Count=0;
     int    Abnormal_Observations_Count=0;
 
     // We'll store an approximate "AGE" and "DECEASED" just to align with feature_utils
     double AGE=50.0; // naive
     bool   DECEASED=false;
 
     // final health index
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
  * 2) Utility: runSynthea, copySyntheaOutput
  ****************************************************/
 static void makeDirIfNeeded(const std::string &d) {
     struct stat st;
     if (stat(d.c_str(), &st)!=0) {
     #ifdef _WIN32
         _mkdir(d.c_str());
     #else
         mkdir(d.c_str(), 0755);
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
     // On Windows, maybe runSynthea.bat, etc.
     std::string cmd = "cd " + SYN_DIR + " && ./run_synthea -p " + std::to_string(popSize);
     std::cout << "[INFO] Running Synthea: " << cmd << "\n";
     int ret = std::system(cmd.c_str());
     if (ret!=0) {
         std::cerr << "[ERROR] Synthea generation failed.\n";
         std::exit(1);
     }
     std::cout << "[INFO] Synthea generation complete.\n";
 }
 
 static void copySyntheaOutput() {
     std::string synOutput = SYN_DIR + "/" + SYN_OUT;
     struct stat st;
     if (stat(synOutput.c_str(), &st)!=0) {
         std::cerr << "[ERROR] Synthea output dir not found.\n";
         std::exit(1);
     }
     makeDirIfNeeded(DATA_DIR);
 
     std::vector<std::string> needed = {
         "patients.csv","encounters.csv","conditions.csv","medications.csv","observations.csv","procedures.csv"
     };
     std::string stamp = getTimestamp();
     for (auto &fname : needed) {
         std::string src = synOutput + "/" + fname;
         if (stat(src.c_str(), &st)!=0) {
             std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
             continue;
         }
         // rename => e.g. "patients_diff_YYYYMMDD_HHMMSS.csv"
         auto dotPos = fname.rfind('.');
         std::string base = (dotPos==std::string::npos? fname : fname.substr(0,dotPos));
         std::string ext  = (dotPos==std::string::npos? "" : fname.substr(dotPos));
         std::string newName = base + "_diff_" + stamp + ext;
         std::string dst = DATA_DIR + "/" + newName;
     #ifdef _WIN32
         std::string cmd = "copy " + src + " " + dst;
     #else
         std::string cmd = "cp " + src + " " + dst;
     #endif
         std::cout << "[INFO] Copying " << src << " => " << dst << "\n";
         int ret = std::system(cmd.c_str());
         if (ret!=0) {
             std::cerr << "[ERROR] copy cmd failed: " << cmd << "\n";
         }
     }
 }
 
 /****************************************************
  * 3) Load the "diff" CSVs (like data_preprocessing)
  ****************************************************/
 static std::vector<PatientRecord> loadNewPatients() {
     // We'll look for up to 50 "patients_diff_i.csv"
     std::vector<PatientRecord> out;
     for (int i=0; i<50; i++) {
         std::ostringstream oss;
         oss << DATA_DIR << "/patients_diff_" << i << ".csv";
         std::string path = oss.str();
         struct stat st;
         if (stat(path.c_str(), &st)==0) {
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
                     p.HEALTHCARE_EXPENSES = row["HEALTHCARE_EXPENSES"].get<double>();
                     p.HEALTHCARE_COVERAGE = row["HEALTHCARE_COVERAGE"].get<double>();
                     p.INCOME = row["INCOME"].get<double>();
                     p.NewData= true;
                     // We skip advanced date parsing, so let's do
                     if (!p.DEATHDATE.empty() && p.DEATHDATE!="NaN") {
                         p.DECEASED= true;
                     }
                     out.push_back(p);
                 }
             } catch(...) {
                 std::cerr << "[ERROR] parsing " << path << "\n";
             }
         }
     }
     return out;
 }
 
 template<typename T>
 std::vector<T> loadDiffRows(const std::string &prefix, std::function<void(const csv::CSVRow&,T&)> parseFn) {
     std::vector<T> out;
     for (int i=0; i<50; i++) {
         std::ostringstream oss;
         oss << DATA_DIR << "/" << prefix << "_diff_" << i << ".csv";
         std::string path = oss.str();
         struct stat st;
         if (stat(path.c_str(), &st)==0) {
             try {
                 csv::CSVReader rr(path);
                 for (auto &row : rr) {
                     T rec;
                     parseFn(row, rec);
                     out.push_back(rec);
                 }
             } catch(...) {}
         }
     }
     return out;
 }
 
 /****************************************************
  * 4) Charlson/Elixhauser logic (like charlson_comorbidity.py, elixhauser_comorbidity.py)
  ****************************************************/
 // For simplicity, we embed small partial maps:
 static std::map<std::string,std::string> CHARLSON_MAP = {
     {"22298006","Myocardial infarction"},
     {"88805009","Congestive heart failure"},
     {"62479008","AIDS/HIV"}
     // expand as needed
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
     if (it!=CHARLSON_MAP.end()) {
         std::string cat= it->second;
         auto wIt= CHARLSON_WEIGHTS.find(cat);
         if (wIt!=CHARLSON_WEIGHTS.end()) {
             return wIt->second;
         }
     }
     return 0;
 }
 static int getElixWeight(const std::string &code) {
     auto it = ELIX_MAP.find(code);
     if (it!=ELIX_MAP.end()) {
         std::string cat= it->second;
         auto wIt= ELIX_WEIGHTS.find(cat);
         if (wIt!=ELIX_WEIGHTS.end()) {
             return wIt->second;
         }
     }
     return 0;
 }
 
 /****************************************************
  * 5) Health Index logic (like health_index.py)
  ****************************************************/
 // We'll define some "groups" => codes => weights
 static std::map<std::string,std::vector<std::string>> SNOMED_GROUPS = {
     {"Cardiovascular Diseases",{"53741008","445118002","22298006"}},
     {"Respiratory Diseases",   {"19829001","233604007"}},
     {"Diabetes",               {"44054006","73211009"}},
     {"Cancer",                 {"363346000","254637007"}}
     // ...
 };
 static std::map<std::string,double> GROUP_WEIGHTS = {
     {"Cardiovascular Diseases",3.0},
     {"Respiratory Diseases",  2.0},
     {"Diabetes",              2.0},
     {"Cancer",                3.0},
     {"Other",                 1.0}
 };
 
 static double findGroupWeight(const std::string &code) {
     for (auto &kv : SNOMED_GROUPS) {
         for (auto &c : kv.second) {
             if (c==code) {
                 return kv.second.empty()?1.0: GROUP_WEIGHTS[kv.first];
             }
         }
     }
     return GROUP_WEIGHTS["Other"];
 }
 
 // Observations => abnormal
 struct ObsThreshold { double minVal; double maxVal; };
 static std::map<std::string,ObsThreshold> OBS_THRESHOLDS = {
     {"Systolic Blood Pressure",{90,120}},
     {"Body Mass Index",{18.5,24.9}}
     // ...
 };
 static std::map<std::string,std::string> OBS_DESC_MAP = {
     {"Systolic Blood Pressure","Systolic Blood Pressure"},
     {"Body mass index (BMI) [Ratio]","Body Mass Index"}
     // ...
 };
 
 static bool isAbnormalObs(const std::string &desc, double val) {
     auto it= OBS_DESC_MAP.find(desc);
     if (it!=OBS_DESC_MAP.end()) {
         std::string standard= it->second;
         auto thr= OBS_THRESHOLDS.find(standard);
         if (thr!=OBS_THRESHOLDS.end()) {
             if (val<thr->second.minVal || val>thr->second.maxVal) {
                 return true;
             }
         }
     }
     return false;
 }
 
 // Weighted approach for Health_Index
 static double computeHealthIndex(const PatientRecord &p) {
     double base=10.0;
     double penalty1= 0.4 * p.Comorbidity_Score;
     double penalty2= 1.0 * p.Hospitalizations_Count;
     double penalty3= 0.2 * p.Medications_Count;
     double penalty4= 0.3 * p.Abnormal_Observations_Count;
     double penalty5= 0.1*p.CharlsonIndex + 0.05*p.ElixhauserIndex;
 
     double raw= base - (penalty1+penalty2+penalty3+penalty4+penalty5);
     if (raw<1) raw=1;
     if (raw>10)raw=10;
     return raw;
 }
 
 /****************************************************
  * 6) Subset Utils: "diabetes", "ckd", or "none"
  ****************************************************/
 static std::set<std::string> findDiabeticPatients(const std::vector<ConditionRow> &conds) {
     std::set<std::string> out;
     for (auto &c : conds) {
         // if "diabetes" in description
         std::string lower=c.DESCRIPTION;
         for (auto &ch : lower) ch= tolower(ch);
         if (lower.find("diabetes")!=std::string::npos) {
             out.insert(c.PATIENT);
         }
     }
     return out;
 }
 
 static std::set<std::string> findCKDPatients(const std::vector<ConditionRow> &conds) {
     std::set<std::string> out;
     // check for codes or "chronic kidney disease" in text
     // just an example
     for (auto &c : conds) {
         std::string lower=c.DESCRIPTION;
         for (auto &ch : lower) ch= tolower(ch);
         if (lower.find("chronic kidney disease")!=std::string::npos ||
             lower.find("ckd")!=std::string::npos) {
             out.insert(c.PATIENT);
         }
     }
     return out;
 }
 
 static std::vector<PatientRecord> filterSubpopulation(
     const std::vector<PatientRecord> &allP,
     const std::string &subsetType,
     const std::vector<ConditionRow> &conds
 ){
     if (subsetType=="none") {
         return allP;
     }
     else if (subsetType=="diabetes") {
         auto diabs= findDiabeticPatients(conds);
         std::vector<PatientRecord> sub;
         for (auto &p : allP) {
             if (diabs.count(p.Id)) {
                 sub.push_back(p);
             }
         }
         return sub;
     }
     else if (subsetType=="ckd") {
         auto ckdSet= findCKDPatients(conds);
         std::vector<PatientRecord> sub;
         for (auto &p : allP) {
             if (ckdSet.count(p.Id)) {
                 sub.push_back(p);
             }
         }
         return sub;
     }
     // default => no filter
     return allP;
 }
 
 /****************************************************
  * 7) Feature Utils (like feature_utils.py)
  ****************************************************/
 static std::vector<std::string> getFeatureCols(const std::string &feature_config) {
     // base columns from python:
     // ["Id","GENDER","RACE","ETHNICITY","MARITAL","HEALTHCARE_EXPENSES",
     //  "HEALTHCARE_COVERAGE","INCOME","AGE","DECEASED","Hospitalizations_Count",
     //  "Medications_Count","Abnormal_Observations_Count"]
     // then add "Health_Index"/"CharlsonIndex"/"ElixhauserIndex" depending on config
     std::vector<std::string> base = {
         "Id","GENDER","RACE","ETHNICITY","MARITAL",
         "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","INCOME",
         "AGE","DECEASED",
         "Hospitalizations_Count","Medications_Count","Abnormal_Observations_Count"
     };
     if (feature_config=="composite") {
         // need Health_Index
         base.push_back("Health_Index");
     }
     else if (feature_config=="cci") {
         // need CharlsonIndex
         base.push_back("CharlsonIndex");
     }
     else if (feature_config=="eci") {
         // need ElixhauserIndex
         base.push_back("ElixhauserIndex");
     }
     else if (feature_config=="combined") {
         // need Health_Index + CharlsonIndex
         base.push_back("Health_Index");
         base.push_back("CharlsonIndex");
     }
     else if (feature_config=="combined_eci") {
         // need Health_Index + ElixhauserIndex
         base.push_back("Health_Index");
         base.push_back("ElixhauserIndex");
     }
     else if (feature_config=="combined_all") {
         // Health + Charlson + Elix
         base.push_back("Health_Index");
         base.push_back("CharlsonIndex");
         base.push_back("ElixhauserIndex");
     }
     else {
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
     // write header
     for (size_t i=0; i<cols.size(); i++) {
         ofs << cols[i];
         if (i+1<cols.size()) ofs << ",";
     }
     ofs << "\n";
     for (auto &p : pats) {
         // We must write columns in the same order as 'cols'
         for (size_t c=0; c<cols.size(); c++) {
             if (c>0) ofs << ",";
             std::string col= cols[c];
             if (col=="Id") {
                 ofs << p.Id;
             } else if (col=="GENDER") {
                 ofs << p.GENDER;
             } else if (col=="RACE") {
                 ofs << p.RACE;
             } else if (col=="ETHNICITY") {
                 ofs << p.ETHNICITY;
             } else if (col=="MARITAL") {
                 ofs << p.MARITAL;
             } else if (col=="HEALTHCARE_EXPENSES") {
                 ofs << p.HEALTHCARE_EXPENSES;
             } else if (col=="HEALTHCARE_COVERAGE") {
                 ofs << p.HEALTHCARE_COVERAGE;
             } else if (col=="INCOME") {
                 ofs << p.INCOME;
             } else if (col=="AGE") {
                 ofs << p.AGE; // naive
             } else if (col=="DECEASED") {
                 ofs << (p.DECEASED?"1":"0");
             } else if (col=="Hospitalizations_Count") {
                 ofs << p.Hospitalizations_Count;
             } else if (col=="Medications_Count") {
                 ofs << p.Medications_Count;
             } else if (col=="Abnormal_Observations_Count") {
                 ofs << p.Abnormal_Observations_Count;
             } else if (col=="Health_Index") {
                 ofs << p.Health_Index;
             } else if (col=="CharlsonIndex") {
                 ofs << p.CharlsonIndex;
             } else if (col=="ElixhauserIndex") {
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
  * 8) Save "patient_data_with_all_indices.csv"
  *    akin to data_prep's final pickle
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
             << (p.NewData?"True":"False") << ","
             << p.CharlsonIndex << "," << p.ElixhauserIndex << "," << p.Comorbidity_Score << ","
             << p.Hospitalizations_Count << "," << p.Medications_Count << "," << p.Abnormal_Observations_Count << ","
             << p.Health_Index << ","
             << p.AGE << ","
             << (p.DECEASED?"1":"0")
             << "\n";
     }
     ofs.close();
     std::cout << "[INFO] Wrote final data => " << outfile << "\n";
 }
 
 /****************************************************
  * 9) Embedding Python for predictions
  ****************************************************/
 static bool runPythonInference(const std::string &model_id, const std::string &csvPath) {
     // We'll do a naive approach: we have "tabnet_inference.py" that:
     //  - sys.argv = [script, model_id, csvPath]
     //  - loads model from Data/finals/<model_id>/<model_id>_model.zip
     //  - loads csvPath for features
     //  - writes Data/new_predictions/<model_id>/<...>.csv
     std::string script= "tabnet_inference.py"; // you create it
 
     std::ostringstream code;
     code << "import sys, runpy\n"
          << "sys.argv = ['" << script << "', '" << model_id << "', '" << csvPath << "']\n"
          << "runpy.run_path('" << script << "', run_name='__main__')\n";
 
     int rc = PyRun_SimpleString(code.str().c_str());
     if (rc!=0) {
         std::cerr << "[ERROR] Python inference for " << model_id
                   << " returned code " << rc << "\n";
         return false;
     }
     return true;
 }
 
 // XAI
 static void runExplainability() {
     std::string script= "Explain_Xai/final_explain_xai_clustered_lime.py";
     FILE* fp= fopen(script.c_str(),"r");
     if (!fp) {
         std::cerr << "[WARN] can't open XAI script: " << script << "\n";
         return;
     }
     std::cout << "[INFO] Running XAI script: " << script << "\n";
     int ret = PyRun_SimpleFile(fp, script.c_str());
     fclose(fp);
     if (ret!=0) {
         std::cerr << "[ERROR] XAI script returned " << ret << "\n";
     }
 }
 
 /****************************************************
  * MAIN
  ****************************************************/
 int main(int argc, char* argv[]) {
     int popSize=100;
     bool enableXAI=false;
     // parse cmd
     for (int i=1; i<argc; i++) {
         std::string arg= argv[i];
         if (arg.rfind("--population=",0)==0) {
             popSize= std::stoi(arg.substr(13));
         } else if (arg=="--enable-xai") {
             enableXAI= true;
         }
     }
     std::cout << "[INFO] popSize=" << popSize
               << ", XAI=" << (enableXAI?"true":"false") << "\n";
 
     // 1) Run Synthea, copy outputs
     runSynthea(popSize);
     copySyntheaOutput();
 
     // 2) Load new patients, conditions, etc.
     auto patients= loadNewPatients();
     auto conds= loadDiffRows<ConditionRow>("conditions",[](auto &row, auto &c){
        c.PATIENT= row["PATIENT"].get<>();
        c.CODE   = row["CODE"].get<>();
        c.DESCRIPTION= row["DESCRIPTION"].get<>();
     });
     auto encs= loadDiffRows<EncounterRow>("encounters",[](auto &row, auto &e){
        e.Id= row["Id"].get<>();
        e.PATIENT= row["PATIENT"].get<>();
        e.ENCOUNTERCLASS= row["ENCOUNTERCLASS"].get<>();
     });
     auto meds= loadDiffRows<MedicationRow>("medications",[](auto &row, auto &m){
        m.PATIENT= row["PATIENT"].get<>();
        m.ENCOUNTER= row["ENCOUNTER"].get<>();
        m.CODE= row["CODE"].get<>();
        m.DESCRIPTION= row["DESCRIPTION"].get<>();
     });
     auto obs= loadDiffRows<ObservationRow>("observations",[](auto &row, auto &o){
        o.PATIENT= row["PATIENT"].get<>();
        o.ENCOUNTER= row["ENCOUNTER"].get<>();
        o.CODE= row["CODE"].get<>();
        o.DESCRIPTION= row["DESCRIPTION"].get<>();
        try {
          o.VALUE= row["VALUE"].get<double>();
        } catch(...) {
          o.VALUE=0.0;
        }
        o.UNITS= row["UNITS"].get<>();
     });
     auto procs= loadDiffRows<ProcedureRow>("procedures",[](auto &row, auto &p){
        p.PATIENT= row["PATIENT"].get<>();
        p.ENCOUNTER= row["ENCOUNTER"].get<>();
        p.CODE= row["CODE"].get<>();
        p.DESCRIPTION= row["DESCRIPTION"].get<>();
     });
 
     // 3) Compute Charlson/Elixhauser from conditions
     std::unordered_map<std::string,double> charlMap, elixMap;
     for (auto &c : conds) {
         charlMap[c.PATIENT]+= getCharlsonWeight(c.CODE);
         elixMap[c.PATIENT] += getElixWeight(c.CODE);
     }
 
     // 4) Additional aggregator
     // 4a) Comorbidity_Score from SNOMED groups
     std::unordered_map<std::string,double> groupScore;
     for (auto &c : conds) {
         groupScore[c.PATIENT]+= findGroupWeight(c.CODE);
     }
     // 4b) Hospitalizations => inpatient
     std::unordered_map<std::string,int> hospCount;
     for (auto &e : encs) {
         if (e.ENCOUNTERCLASS=="inpatient") {
             hospCount[e.PATIENT]++;
         }
     }
     // 4c) Distinct med codes
     std::unordered_map<std::string,std::set<std::string>> medSet;
     for (auto &m : meds) {
         medSet[m.PATIENT].insert(m.CODE);
     }
     // 4d) Abnormal obs
     std::unordered_map<std::string,int> abnMap;
     for (auto &oRec : obs) {
         if (isAbnormalObs(oRec.DESCRIPTION, oRec.VALUE)) {
             abnMap[oRec.PATIENT]++;
         }
     }
 
     // Merge all into patient records
     std::unordered_map<std::string,PatientRecord*> pMap;
     for (auto &p : patients) {
         pMap[p.Id]= &p;
     }
     for (auto &kv : charlMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->CharlsonIndex= kv.second;
         }
     }
     for (auto &kv : elixMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->ElixhauserIndex= kv.second;
         }
     }
     for (auto &kv : groupScore) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Comorbidity_Score= kv.second;
         }
     }
     for (auto &kv : hospCount) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Hospitalizations_Count= kv.second;
         }
     }
     for (auto &kv : medSet) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Medications_Count= (int)kv.second.size();
         }
     }
     for (auto &kv : abnMap) {
         if (pMap.count(kv.first)) {
             pMap[kv.first]->Abnormal_Observations_Count= kv.second;
         }
     }
 
     // 5) Compute health index
     for (auto &p : patients) {
         p.Health_Index= computeHealthIndex(p);
     }
 
     // 6) Save final, akin to "patient_data_with_all_indices.pkl" => CSV
     std::string finalCSV= DATA_DIR + "/patient_data_with_all_indices.csv";
     saveFinalDataCSV(patients, finalCSV);
 
     // 7) We "predict" with each final model
     // We'll embed Python for TabNet, but first we isolate newly generated
     std::vector<PatientRecord> newPatients;
     for (auto &p : patients) {
         if (p.NewData) {
             newPatients.push_back(p);
         }
     }
     std::cout << "[INFO] Found " << newPatients.size() << " newly generated patients.\n";
 
     // init python
     Py_Initialize();
 
     // For each final model
     for (auto &it : MODEL_CONFIG_MAP) {
         std::string model_id= it.first;
         std::string subset= it.second.subset;
         std::string featconf= it.second.feature_config;
 
         std::cout << "\n[INFO] Predicting with " << model_id
                   << ", subset=" << subset
                   << ", featconf=" << featconf << "\n";
         // filter subpop
         auto sub= filterSubpopulation(newPatients, subset, conds);
         if (sub.empty()) {
             std::cout << "[INFO] No new patients for subpop=" << subset
                       << ", skip.\n";
             continue;
         }
         // pick feature cols
         auto cols= getFeatureCols(featconf);
         // write a CSV for these patients
         std::string outDir= DATA_DIR + "/new_predictions/" + model_id;
         makeDirIfNeeded(DATA_DIR + "/new_predictions");
         makeDirIfNeeded(outDir);
         std::string infCSV= outDir + "/input_for_inference.csv";
         writeFeaturesCSV(sub, infCSV, cols);
 
         // call python
         bool ok= runPythonInference(model_id, infCSV);
         if (!ok) {
             std::cerr << "[WARN] Inference for " << model_id << " failed.\n";
         }
     }
 
     // optional XAI
     if (enableXAI) {
         runExplainability();
     }
 
     Py_Finalize();
 
     std::cout << "[INFO] Done. Generated new data, merged charlson/elix, "
               << "computed health index, saved final CSV, ran TabNet.\n";
     return 0;
 }
 