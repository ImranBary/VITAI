/*****************************************************
 * GenerateAndPredict.cpp
 *
 * C++ program replicating the logic of:
 *   1) data_preprocessing.py (merge old & new CSVs, marking new data as NewData)
 *   2) health_index.py (computing comorbidity-based features + Health_Index)
 *   3) data_prep.py (merging Charlson/Elixhauser, final integrated dataset)
 *   4) subset_utils.py (filter: diabetes/ckd/none)
 *   5) feature_utils.py (select columns: composite, cci, eci, combined, etc.)
 *   6) final "generate_new_patients_and_predict.py" pipeline
 *   7) Embedded Python calls for TabNet model inference + optional XAI
 *
 * Charlson Comorbidity Index now uses a full dictionary-based approach
 * (no CSV file). We replicate the "SNOMED_TO_CHARLSON" and category weights
 * from the Python script you provided.
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 * To build the file use command:
 * cl /EHsc /std:c++17 GenerateAndPredict.cpp ^ /I"C:\Users\imran\miniconda3\envs\tf_gpu_env\include" ^ /link /LIBPATH:"C:\Users\imran\miniconda3\envs\tf_gpu_env\libs" python39.lib /MACHINE:X64
 * Usage: 
 * GenerateAndPredict.exe --population=100
 * or
 * GenerateAndPredict.exe --population=100 --enable-xai
 * 
 *****************************************************/








/******************************************
 * TODO: Confirm if the logic used for the health index is similar to
 * the one used in helath_index.py
 ******************************************/
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
#include <filesystem> // C++17
namespace fs = std::filesystem;

// For CSV parsing (using an external single-header CSV library)
// #define CSV_IO_NO_THREAD 
#define CSV_IMPLEMENTATION
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

// Data structures to hold loaded records
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

    // Simplistic approach for age & deceased
    double AGE=0.0;
    bool   DECEASED=false;

    // Charlson & Elixhauser indices
    double CharlsonIndex=0.0;
    double ElixhauserIndex=0.0;

    // Additional metrics
    double Comorbidity_Score=0.0;
    int    Hospitalizations_Count=0;
    int    Medications_Count=0;
    int    Abnormal_Observations_Count=0;

    // Final "Health_Index"
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
    // Windows example
    std::string cmd = "cd " + SYN_DIR + " && run_synthea.bat -p " + std::to_string(popSize);
#else
    // Linux/macOS example
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

// Copy the main Synthea output CSVs to our Data directory with a timestamp 
static void copySyntheaOutput() {
   // e.g. "synthea-master/output/csv"
   fs::path synOutput = fs::path(SYN_DIR) / SYN_OUT;

   // Make sure the Synthea output dir actually exists:
   if (!fs::exists(synOutput)) {
       std::cerr << "[ERROR] Synthea output dir " << synOutput << " not found.\n";
       std::exit(1);
   }
   makeDirIfNeeded(DATA_DIR); // keep your existing mkdir logic or use filesystem as well

   // The main CSV files we want to copy
   std::vector<std::string> needed = {
       "patients.csv", "encounters.csv", "conditions.csv",
       "medications.csv", "observations.csv", "procedures.csv"
   };

   std::string stamp = getTimestamp(); // your existing function
   for (auto &fname : needed) {
       fs::path src = synOutput / fname;
       if (!fs::exists(src)) {
           std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
           continue;
       }
       // e.g. "patients_diff_20250309_165742.csv"
       auto dotPos = fname.rfind('.');
       std::string base = (dotPos == std::string::npos
                           ? fname
                           : fname.substr(0, dotPos));
       std::string ext  = (dotPos == std::string::npos
                           ? ""
                           : fname.substr(dotPos));
       std::string newName = base + "_diff_" + stamp + ext;

       fs::path dst = fs::path(DATA_DIR) / newName;

       std::cout << "[INFO] Copying " << src << " => " << dst << "\n";
       try {
           // Overwrites if already exists:
           fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
       } catch (const fs::filesystem_error &ex) {
           std::cerr << "[ERROR] copy failed: " << ex.what() << "\n";
       }
   }
}

/****************************************************
 * Helpers to load CSVs
 ****************************************************/

// check if file name includes "_diff_"
static bool isDiffFile(const std::string &fname) {
    return (fname.find("_diff_") != std::string::npos);
}

// Returns all CSV files in the Data/ directory that start with `prefix` and end with ".csv".
// e.g. If prefix="patients", it finds "patients.csv", "patients_diff_20250309_170905.csv", etc.
static std::vector<std::string> listCSVFiles(const std::string &prefix)
{
   std::vector<std::string> found;

   // Point to the "Data" directory (or use a global constant if you prefer)
   fs::path dataDir("Data");

   // Check if Data/ exists and is a directory
   if (!fs::exists(dataDir) || !fs::is_directory(dataDir))
   {
       std::cerr << "[ERROR] Data directory not found or not a directory: " 
                 << dataDir << std::endl;
       return found;
   }

   // Iterate over files in Data/
   for (auto &entry : fs::directory_iterator(dataDir))
   {
       if (!entry.is_regular_file()) 
           continue;  // skip directories, etc.

       // Extract just the filename (e.g. "patients_diff_20250309_170905.csv")
       std::string filename = entry.path().filename().string();

       // Check if filename starts with `prefix` and ends with ".csv"
       //  - rfind(prefix, 0) == 0 means the prefix is found at position 0 (i.e., "starts with")
       //  - compare(...) == 0 on the last 4 chars ensures ".csv" at the end
       if (filename.rfind(prefix, 0) == 0 && 
           filename.size() > 4 &&
           filename.compare(filename.size() - 4, 4, ".csv") == 0)
       {
           // Full path to the file
           found.push_back(entry.path().string());
       }
   }
   return found;
}

// 1) Load patients
static void loadPatients(std::vector<PatientRecord> &allPatients) {
    auto patientFiles = listCSVFiles("patients"); 
    for (auto &path : patientFiles) {
        bool diff = isDiffFile(path);
        try {
            csv::CSVReader rd(path);
            for (auto &row : rd) {
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

                p.NewData = diff;
                if (!p.DEATHDATE.empty() && p.DEATHDATE != "NaN") {
                    p.DECEASED = true;
                }
                // naive approach for AGE
                p.AGE = 50.0;
                allPatients.push_back(p);
            }
        } catch(...) {
            std::cerr << "[ERROR] parsing " << path << "\n";
        }
    }
}

// 2) Load conditions, encounters, meds, obs, procs
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
                e.Id             = row["Id"].get<>();
                e.PATIENT        = row["PATIENT"].get<>();
                e.ENCOUNTERCLASS = row["ENCOUNTERCLASS"].get<>();
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
                m.PATIENT     = row["PATIENT"].get<>();
                m.ENCOUNTER   = row["ENCOUNTER"].get<>();
                m.CODE        = row["CODE"].get<>();
                m.DESCRIPTION = row["DESCRIPTION"].get<>();
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
 * Charlson Comorbidity: dictionary-based approach
 * from your Python "charlson_comorbidity.py"
 ****************************************************/

// 1) Full SNOMED->CharlsonCategory mapping
static std::map<long long, std::string> SNOMED_TO_CHARLSON = {
    // MYOCARDIAL INFARCTION (weight 1)
    {22298006, "Myocardial infarction"},
    {401303003, "Myocardial infarction"},
    {401314000, "Myocardial infarction"},
    {129574000, "Myocardial infarction"},

    // CONGESTIVE HEART FAILURE (weight 1)
    {88805009,  "Congestive heart failure"},
    {84114007,  "Congestive heart failure"},

    // PERIPHERAL VASCULAR DISEASE (weight 1)
    // (No explicit codes in that snippet, so omitted)

    // CEREBROVASCULAR DISEASE (weight 1)
    {230690007, "Cerebrovascular disease"},

    // DEMENTIA (weight 1)
    {26929004,  "Dementia"},
    {230265002, "Dementia"},

    // CHRONIC PULMONARY DISEASE (weight 1)
    {185086009, "Chronic pulmonary disease"},
    {87433001,  "Chronic pulmonary disease"},
    {195967001, "Chronic pulmonary disease"},
    {233678006, "Chronic pulmonary disease"},

    // CONNECTIVE TISSUE DISEASE (weight 1)
    {69896004,  "Connective tissue disease"},
    {200936003, "Connective tissue disease"},

    // ULCER DISEASE (weight 1)
    // (No explicit codes in that snippet)

    // MILD LIVER DISEASE (weight 1)
    {128302006, "Mild liver disease"},
    {61977001,  "Mild liver disease"},

    // DIABETES WITHOUT END-ORGAN DAMAGE (weight 1)
    {44054006,  "Diabetes without end-organ damage"},

    // DIABETES WITH END-ORGAN DAMAGE (weight 2)
    {368581000119106LL, "Diabetes with end-organ damage"},
    {422034002,         "Diabetes with end-organ damage"},
    {127013003,         "Diabetes with end-organ damage"},
    {90781000119102LL,  "Diabetes with end-organ damage"},
    {157141000119108LL, "Diabetes with end-organ damage"},
    {60951000119105LL,  "Diabetes with end-organ damage"},
    {97331000119101LL,  "Diabetes with end-organ damage"},
    {1501000119109LL,   "Diabetes with end-organ damage"},
    {1551000119108LL,   "Diabetes with end-organ damage"},

    // HEMIPLEGIA/PARAPLEGIA (weight 2)
    // (No explicit codes in that snippet)

    // MODERATE OR SEVERE KIDNEY DISEASE (weight 2)
    {431855005, "Moderate or severe kidney disease"},
    {431856006, "Moderate or severe kidney disease"},
    {433144002, "Moderate or severe kidney disease"},
    {431857002, "Moderate or severe kidney disease"},
    {46177005,  "Moderate or severe kidney disease"},
    {129721000119106LL, "Moderate or severe kidney disease"},

    // ANY TUMOUR (weight 2)
    {254637007, "Any tumour, leukaemia, lymphoma"},
    {254632001, "Any tumour, leukaemia, lymphoma"},
    {93761005,  "Any tumour, leukaemia, lymphoma"},
    {363406005, "Any tumour, leukaemia, lymphoma"},
    {109838007, "Any tumour, leukaemia, lymphoma"},
    {126906006, "Any tumour, leukaemia, lymphoma"},
    {92691004,  "Any tumour, leukaemia, lymphoma"},
    {254837009, "Any tumour, leukaemia, lymphoma"},
    {109989006, "Any tumour, leukaemia, lymphoma"},
    {93143009,  "Any tumour, leukaemia, lymphoma"},
    {91861009,  "Any tumour, leukaemia, lymphoma"},

    // MODERATE OR SEVERE LIVER DISEASE (weight 3)
    // (No explicit codes in the snippet, but you can add if needed)

    // METASTATIC SOLID TUMOUR (weight 6)
    {94503003,  "Metastatic solid tumour"},
    {94260004,  "Metastatic solid tumour"},

    // AIDS/HIV (weight 6)
    {62479008,  "AIDS/HIV"},
    {86406008,  "AIDS/HIV"}
};

// 2) Category->CharlsonWeight
static std::map<std::string,int> CHARLSON_CATEGORY_WEIGHTS = {
    {"Myocardial infarction", 1},
    {"Congestive heart failure", 1},
    {"Peripheral vascular disease", 1},
    {"Cerebrovascular disease", 1},
    {"Dementia", 1},
    {"Chronic pulmonary disease", 1},
    {"Connective tissue disease", 1},
    {"Ulcer disease", 1},
    {"Mild liver disease", 1},
    {"Diabetes without end-organ damage", 1},

    {"Hemiplegia", 2},
    {"Moderate or severe kidney disease", 2},
    {"Diabetes with end-organ damage", 2},
    {"Any tumour, leukaemia, lymphoma", 2},

    {"Moderate or severe liver disease", 3},
    {"Metastatic solid tumour", 6},
    {"AIDS/HIV", 6}
};

/****************************************************
 * Elixhauser (unchanged from earlier approach)
 ****************************************************/
static std::map<long long, std::string> SNOMED_TO_ELIXHAUSER = {
   // Congestive Heart Failure
   {88805009,  "Congestive heart failure"},
   {84114007,  "Congestive heart failure"},

   // Cardiac Arrhythmias
   {49436004,  "Cardiac arrhythmias"},

   // Valvular Disease
   {48724000,  "Valvular disease"},
   {91434003,  "Pulmonic valve regurgitation"},
   {79619009,  "Mitral valve stenosis"},
   {111287006, "Tricuspid valve regurgitation"},
   {49915006,  "Tricuspid valve stenosis"},
   {60573004,  "Aortic valve stenosis"},
   {60234000,  "Aortic valve regurgitation"},

   // Pulmonary Circulation Disorders
   {65710008,  "Pulmonary circulation disorders"},
   {706870000, "Acute pulmonary embolism"},
   {67782005,  "Acute respiratory distress syndrome"},

   // Peripheral Vascular Disorders
   // NOTE: Overwritten by the next entry in the Python dictionary (key repeated)
   // {698754002, "Peripheral vascular disorders"},

   // Hypertension
   {59621000,  "Hypertension, uncomplicated"},

   // Paralysis
   // Overwrites the earlier "Peripheral vascular disorders" for 698754002
   {698754002, "Paralysis"},
   {128188000, "Paralysis"},

   // Other Neurological Disorders
   // NOTE: Overwritten by the next entry in the Python dictionary (key repeated)
   // {69896004,  "Other neurological disorders"},
   {128613002, "Seizure disorder"},

   // Chronic Pulmonary Disease
   {195967001, "Chronic pulmonary disease"},
   {233678006, "Chronic pulmonary disease"},

   // Diabetes, Complicated
   {368581000119106, "Diabetes, complicated"},
   {422034002,        "Diabetes, complicated"},
   {90781000119102,   "Diabetes, complicated"},

   // Diabetes, Uncomplicated
   {44054006,  "Diabetes, uncomplicated"},

   // Renal Failure
   {129721000119106, "Renal failure"},
   {433144002,       "Renal failure"},

   // Liver Disease
   {128302006, "Liver disease"},
   {61977001,  "Liver disease"},

   // Peptic Ulcer Disease
   // (Not identified in the dataset)

   // AIDS/HIV
   {62479008,  "AIDS/HIV"},
   {86406008,  "AIDS/HIV"},

   // Lymphoma
   {93143009,  "Lymphoma"},

   // Metastatic Cancer
   {94503003,  "Metastatic cancer"},
   {94260004,  "Metastatic cancer"},

   // Solid Tumour Without Metastasis
   {126906006, "Solid tumour without metastasis"},
   {254637007, "Solid tumour without metastasis"},

   // Rheumatoid Arthritis / Collagen Vascular Diseases
   // Overwrites the earlier "Other neurological disorders" for 69896004
   {69896004,  "Rheumatoid arthritis/collagen vascular diseases"},
   {200936003, "Rheumatoid arthritis/collagen vascular diseases"},

   // Coagulopathy
   {234466008, "Coagulopathy"},

   // Obesity
   {408512008, "Obesity"},
   {162864005, "Obesity"},

   // Weight Loss
   {278860009, "Weight loss"},

   // Fluid and Electrolyte Disorders
   {389087006, "Fluid and electrolyte disorders"},

   // Blood Loss Anaemia
   // (Not identified in the dataset)

   // Deficiency Anaemias
   {271737000, "Deficiency anaemias"},

   // Alcohol Abuse
   {7200002,   "Alcohol abuse"},

   // Drug Abuse
   {6525002,   "Drug abuse"},

   // Psychoses
   {47505003,  "Psychoses"},

   // Depression
   {370143000, "Depression"},
   {36923009,  "Depression"}
};


// We apply the typical van Walraven weighting:
static std::map<std::string,int> ELIXHAUSER_CATEGORY_WEIGHTS = {
   {"Congestive heart failure", 7},
   {"Cardiac arrhythmias", 5},
   {"Valvular disease", 4},
   {"Pulmonary circulation disorders", 6},
   {"Peripheral vascular disorders", 2},
   {"Hypertension, uncomplicated", -1},
   {"Hypertension, complicated", 0},
   {"Paralysis", 7},
   {"Other neurological disorders", 6},
   {"Chronic pulmonary disease", 3},
   {"Diabetes, uncomplicated", 0},
   {"Diabetes, complicated", 7},
   {"Hypothyroidism", 0},
   {"Renal failure", 5},
   {"Liver disease", 11},
   {"Peptic ulcer disease", 0},
   {"AIDS/HIV", 0},
   {"Lymphoma", 9},
   {"Metastatic cancer", 14},
   {"Solid tumour without metastasis", 8},
   {"Rheumatoid arthritis/collagen vascular diseases", 4},
   {"Coagulopathy", 11},
   {"Obesity", 0},
   {"Weight loss", 6},
   {"Fluid and electrolyte disorders", 5},
   {"Blood loss anaemia", 3},
   {"Deficiency anaemias", 0},
   {"Alcohol abuse", 0},
   {"Drug abuse", 0},
   {"Psychoses", 0},
   {"Depression", -3}
};

/****************************************************
 * 1) Compute Charlson (dictionary-based)
 ****************************************************/
static void computeCharlsonIndex(const std::vector<ConditionRow> &conds,
                                 std::unordered_map<std::string,double> &charlsonScore)
{
    // We'll store the best weight for each (patient, category).
    // i.e., if a patient has multiple codes in the same category,
    // we only count that category once and take the max weight if needed.
    // But typically Charlson is "one category = fixed weight".
    // This approach matches the Python logic of grouping by (PATIENT, category).
    std::map<std::pair<std::string,std::string>, int> patCatWeight; 

    for (auto &c : conds) {
        // Convert code to a long long
        long long codeLL=0;
        try {
            codeLL = std::stoll(c.CODE);
        } catch(...) {
            // if code is not numeric
            continue;
        }
        // Look up category
        auto it = SNOMED_TO_CHARLSON.find(codeLL);
        if (it == SNOMED_TO_CHARLSON.end()) {
            continue; // code not in dictionary
        }
        const std::string &category = it->second;

        // Category -> weight
        auto wIt = CHARLSON_CATEGORY_WEIGHTS.find(category);
        if (wIt == CHARLSON_CATEGORY_WEIGHTS.end()) {
            continue;
        }
        int weight = wIt->second;

        auto key = std::make_pair(c.PATIENT, category);
        auto existing = patCatWeight.find(key);
        if (existing == patCatWeight.end()) {
            patCatWeight[key] = weight;
        } else {
            // If there's some reason we might want max; typically it's the same weight
            // but let's do max() to emulate Python's .groupby max
            patCatWeight[key] = std::max(existing->second, weight);
        }
    }

    // Now sum across categories for each patient
    // so patient_cci_sum = sum of each category’s weight
    for (auto &kv : patCatWeight) {
        const std::string &patient = kv.first.first;
        int w = kv.second;
        charlsonScore[patient] += w;
    }
}

/****************************************************
 * 2) Compute Elixhauser
 ****************************************************/
static void computeElixhauserIndex(const std::vector<ConditionRow> &conds,
                                   std::unordered_map<std::string,double> &elixScore)
{
    // Similar approach: group by (patient, category), take max weight
    std::map<std::pair<std::string,std::string>, int> patCatWeight;

    for (auto &c : conds) {
        long long codeLL=0;
        try {
            codeLL = std::stoll(c.CODE);
        } catch(...) { continue; }
        auto it = SNOMED_TO_ELIXHAUSER.find(codeLL);
        if (it == SNOMED_TO_ELIXHAUSER.end()) {
            continue;
        }
        const std::string &category = it->second;
        auto wIt = ELIXHAUSER_CATEGORY_WEIGHTS.find(category);
        if (wIt == ELIXHAUSER_CATEGORY_WEIGHTS.end()) {
            continue;
        }
        int weight = wIt->second;

        auto key = std::make_pair(c.PATIENT, category);
        auto existing = patCatWeight.find(key);
        if (existing == patCatWeight.end()) {
            patCatWeight[key] = weight;
        } else {
            patCatWeight[key] = std::max(existing->second, weight);
        }
    }
    // sum for each patient
    for (auto &kv : patCatWeight) {
        const std::string &patient = kv.first.first;
        elixScore[patient] += kv.second;
    }
}

/****************************************************
 * Health Index logic
 ****************************************************/
struct ObsThreshold { double minVal; double maxVal; };
static std::map<std::string,ObsThreshold> OBS_THRESHOLDS = {
    {"Systolic Blood Pressure",{90,120}},
    {"Body Mass Index",{18.5,24.9}}
};
static std::map<std::string,std::string> OBS_DESC_MAP = {
    {"Systolic Blood Pressure","Systolic Blood Pressure"},
    {"Body mass index (BMI) [Ratio]","Body Mass Index"}
};

// We'll define some "groups" => codes => weights for Comorbidity_Score
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

static double findGroupWeight(const std::string &codeStr) {
    // We'll do a simplistic check if code is in these groups
    for (auto &kv : SNOMED_GROUPS) {
        for (auto &c : kv.second) {
            if (c == codeStr) {
                return GROUP_WEIGHTS[kv.first];
            }
        }
    }
    return GROUP_WEIGHTS["Other"];
}
static bool isAbnormalObs(const std::string &desc, double val) {
    auto it = OBS_DESC_MAP.find(desc);
    if (it != OBS_DESC_MAP.end()) {
        // e.g. "Body Mass Index"
        auto thr = OBS_THRESHOLDS.find(it->second);
        if (thr != OBS_THRESHOLDS.end()) {
            if (val < thr->second.minVal || val > thr->second.maxVal) {
                return true;
            }
        }
    }
    return false;
}

static double computeHealthIndex(const PatientRecord &p) {
    // Same approximate formula as we used before
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
 * Subset Utils
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
    return allP;
}

/****************************************************
 * Feature Utils
 ****************************************************/
static std::vector<std::string> getFeatureCols(const std::string &feature_config) {
    // Base columns
    std::vector<std::string> base = {
        "Id","GENDER","RACE","ETHNICITY","MARITAL",
        "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","INCOME",
        "AGE","DECEASED",
        "Hospitalizations_Count","Medications_Count","Abnormal_Observations_Count"
    };
    // Additional columns based on config
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
        std::cerr << "[ERROR] cannot open " << outFile << "\n";
        return;
    }
    // Header
    for (size_t i=0; i<cols.size(); i++) {
        ofs << cols[i];
        if (i+1<cols.size()) ofs << ",";
    }
    ofs << "\n";

    for (auto &p : pats) {
        for (size_t c=0; c<cols.size(); c++) {
            if (c>0) ofs << ",";
            const auto &col= cols[c];
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
                ofs << p.AGE;
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
                ofs << 0;
            }
        }
        ofs << "\n";
    }
    ofs.close();
    std::cout << "[INFO] Wrote features => " << outFile << "\n";
}

/****************************************************
 * Save final data akin to "patient_data_with_all_indices.pkl"
 * but in CSV form
 ****************************************************/
static void saveFinalDataCSV(const std::vector<PatientRecord> &pats,
                             const std::string &outfile)
{
    std::ofstream ofs(outfile);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] cannot open " << outfile << "\n";
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
 * Embedding Python for TabNet Inference
 ****************************************************/
static bool runPythonInference(const std::string &model_id,
                               const std::string &csvPath)
{
    // We'll assume we have "tabnet_inference.py" that:
    //   sys.argv = [script, model_id, csvPath]
    //   loads the TabNet model from Data/finals/<model_id>/<model_id>_model.zip
    //   reads <csvPath> for features
    //   writes predictions in Data/new_predictions/<model_id>/...
    // 
    std::string script = "tabnet_inference.py";

    // Build Python code
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
 * Optional XAI
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

    // Parse command line
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

    // 1) Run Synthea & copy CSV outputs
    runSynthea(popSize);
    copySyntheaOutput();

    // 2) Load entire dataset (old + new)
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
    std::unordered_map<std::string, double> charlMap, elixMap;
    computeCharlsonIndex(conds, charlMap);
    computeElixhauserIndex(conds, elixMap);

    // 4) Additional aggregator logic
    //  (Comorbidity_Score from simpler "groups", Hospitalizations, etc.)
    // 4a) Comorbidity_Score
    std::unordered_map<std::string,double> groupScore;
    for (auto &c : conds) {
        // We'll treat c.CODE as a string
        double w = findGroupWeight(c.CODE);
        groupScore[c.PATIENT] += w;
    }
    // 4b) Hospitalizations => inpatient
    std::unordered_map<std::string,int> hospCount;
    for (auto &e : encs) {
        if (e.ENCOUNTERCLASS == "inpatient") {
            hospCount[e.PATIENT]++;
        }
    }
    // 4c) Distinct medication codes
    std::unordered_map<std::string, std::set<std::string>> medSet;
    for (auto &m : meds) {
        medSet[m.PATIENT].insert(m.CODE);
    }
    // 4d) Abnormal observations
    std::unordered_map<std::string,int> abnMap;
    for (auto &oRec : obs) {
        if (isAbnormalObs(oRec.DESCRIPTION, oRec.VALUE)) {
            abnMap[oRec.PATIENT]++;
        }
    }

    // 5) Merge into PatientRecord
    // Index them by Id
    std::unordered_map<std::string,PatientRecord*> pMap;
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

    // 7) Save final big CSV (akin to "patient_data_with_all_indices.pkl", but in CSV)
    std::string finalCSV = DATA_DIR + "/patient_data_with_all_indices.csv";
    saveFinalDataCSV(allPatients, finalCSV);

    // 8) Subset to new patients (NewData==true), run predictions for each final model
    std::vector<PatientRecord> newPatients;
    for (auto &p : allPatients) {
        if (p.NewData) {
            newPatients.push_back(p);
        }
    }
    std::cout << "[INFO] Found " << newPatients.size() << " newly generated patients.\n";
    
    // Need to set python home - causing errors I don't know why

    Py_SetPythonHome(L"C:\\Users\\imran\\miniconda3\\envs\\tf_gpu_env");

    // Initialize Python
    Py_Initialize();

    // For each final TabNet model
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
            std::cout << "[INFO] No new patients in subpop=" << subsetType
                      << " => skip.\n";
            continue;
        }
        // pick columns
        auto cols = getFeatureCols(featconf);
        // write CSV for inference
        std::string outDir = DATA_DIR + "/new_predictions/" + model_id;
        makeDirIfNeeded(DATA_DIR + "/new_predictions");
        makeDirIfNeeded(outDir);

        std::string infCSV = outDir + "/input_for_inference.csv";
        writeFeaturesCSV(sub, infCSV, cols);

        // call Python
        bool ok = runPythonInference(model_id, infCSV);
        if (!ok) {
            std::cerr << "[WARN] Inference for " << model_id << " failed.\n";
        }
    }

    // optionally run XAI
    if (enableXAI) {
        runExplainability();
    }

    // finalize Python
    Py_Finalize();

    std::cout << "[INFO] Done. Generated new data, computed Charlson/Elixhauser, "
              << "computed health index, saved final CSV, ran TabNet, XAI="
              << (enableXAI ? "true" : "false") << ".\n";
    return 0;
}