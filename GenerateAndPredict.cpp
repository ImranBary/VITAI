/*****************************************************
 * GenerateAndPredict.cpp
 *
 * Refactored for batch/stream processing using BatchProcessor
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 *
 * Usage Examples:
 *   GenerateAndPredict.exe --population=100
 *   GenerateAndPredict.exe --population=100 --enable-xai
 *   GenerateAndPredict.exe --population=100 --threads=8  (specify thread count)
 *****************************************************/

#include <Python.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <bitset>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <condition_variable>

#include "BatchProcessor.h"    // Your batch-based CSV helper
#include "ThreadSafeCounter.h" // If you want concurrency safety
#include "MedicalDictionaries.h" // New include for medical dictionaries

#ifdef _WIN32
#include <direct.h>  // for _mkdir
#else
#include <unistd.h>
#endif
#include <queue>

namespace fs = std::filesystem;

// Auto-detect optimal thread count based on hardware
const unsigned int DEFAULT_THREAD_COUNT = std::max(2u, std::thread::hardware_concurrency());

// Thread count will be set from command line or system detection
unsigned int THREAD_COUNT = DEFAULT_THREAD_COUNT;

// Run time logging
auto PROGRAM_START_TIME = std::chrono::high_resolution_clock::now();
void logElapsedTime(const std::string& operation) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - PROGRAM_START_TIME).count();
    std::cout << "[TIME] " << operation << ": " << duration / 1000.0 << " seconds\n";
}

// Thread pool for parallel processing
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if(stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }
    
    // Wait for all tasks to complete
    void wait_all() {
        // Not the most efficient but works for our case
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        while(true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if(tasks.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

/****************************************************
 * GLOBAL CONSTANTS / DIRECTORIES
 ****************************************************/
static const std::string DATA_DIR  = "Data";
static const std::string SYN_DIR   = "synthea-master";
static const std::string SYN_OUT   = "output/csv";

/****************************************************
 * MODEL CONFIG
 ****************************************************/
struct ModelConfig {
    std::string subset;
    std::string feature_config;
};
static std::map<std::string, ModelConfig> MODEL_CONFIG_MAP = {
    {"combined_diabetes_tabnet", {"diabetes", "combined"}},
    {"combined_all_ckd_tabnet",  {"ckd",      "combined_all"}},
    {"combined_none_tabnet",     {"none",     "combined"}}
};

/****************************************************
 * DATA STRUCTURES
 ****************************************************/
struct PatientRecord {
    std::string Id;
    std::string BIRTHDATE;
    std::string DEATHDATE;
    std::string GENDER;
    std::string RACE;
    std::string ETHNICITY;
    float HEALTHCARE_EXPENSES = 0.0f;
    float HEALTHCARE_COVERAGE = 0.0f;
    float INCOME              = 0.0f;
    std::string MARITAL;
    bool   NewData = false;  // Mark if loaded from a "_diff_" file

    float AGE = 0.0f;
    bool  DECEASED = false;

    float CharlsonIndex   = 0.0f;
    float ElixhauserIndex = 0.0f;

    float   Comorbidity_Score = 0.0f;
    uint16_t Hospitalizations_Count     = 0;
    uint16_t Medications_Count          = 0;
    uint16_t Abnormal_Observations_Count= 0;

    float Health_Index = 0.0f;
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
    double VALUE = 0.0;
    std::string UNITS;
};

struct ProcedureRow {
    std::string PATIENT;
    std::string ENCOUNTER;
    std::string CODE;
    std::string DESCRIPTION;
};

/****************************************************
 * UTILITY: MKDIR, TIMESTAMP, RUN SYNTHEA, COPY
 ****************************************************/
static void makeDirIfNeeded(const std::string &dir)
{
    // Consider using std::filesystem::create_directories instead for better error handling
    // and support for creating nested directories in one call
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
        return;
    }
}

static std::string getTimestamp()
{
    // Thread-safe version using localtime_s for Windows, localtime_r for others
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
    
#ifdef _WIN32
    localtime_s(&timeinfo, &t_c);
#else
    localtime_r(&t_c, &timeinfo);
#endif
    
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &timeinfo);
    return std::string(buf);
}

static void runSynthea(int popSize)
{
#ifdef _WIN32
    std::string cmd = "cd " + SYN_DIR + " && run_synthea.bat -p " + std::to_string(popSize);
#else
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

// Check if a file name includes "_diff_"
static bool isDiffFile(const std::string &fname) {
    return (fname.find("_diff_") != std::string::npos);
}

static void copySyntheaOutput()
{
    fs::path synOutput = fs::path(SYN_DIR) / SYN_OUT;
    if (!fs::exists(synOutput)) {
        std::cerr << "[ERROR] Synthea output dir " << synOutput << " not found.\n";
        std::exit(1);
    }
    makeDirIfNeeded(DATA_DIR);

    std::vector<std::string> needed = {
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    };

    std::string stamp = getTimestamp();
    for (auto &fname : needed) {
        fs::path src = synOutput / fname;
        if (!fs::exists(src)) {
            std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
            continue;
        }
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
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        } catch(const fs::filesystem_error &ex) {
            std::cerr << "[ERROR] copy failed: " << ex.what() << "\n";
        }
    }
}

/****************************************************
 * LISTING CSV FILES
 ****************************************************/
static std::vector<std::string> listCSVFiles(const std::string &prefix)
{
    std::vector<std::string> found;
    fs::path dataDir("Data");

    if (!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
        std::cerr << "[ERROR] Data directory not found or not a directory: " 
                  << dataDir << std::endl;
        return found;
    }

    for (auto &entry : fs::directory_iterator(dataDir)) {
        if (!entry.is_regular_file()) 
            continue;

        std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0 &&
            filename.size() > 4 &&
            filename.compare(filename.size() - 4, 4, ".csv") == 0)
        {
            found.push_back(entry.path().string());
        }
    }
    return found;
}

/****************************************************
 * PARALLEL FILE PROCESSING
 ****************************************************/
// Process multiple files in parallel using a thread pool
template<typename FileProcessor>
static void processFilesInParallel(const std::vector<std::string>& files, FileProcessor processor) {
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto& file : files) {
        results.emplace_back(pool.enqueue(processor, file));
    }
    
    // Wait for all to complete
    for (auto& result : results) {
        result.get();
    }
}

// Specialized parallel CSV processing with thread-safe accumulators
static void processConditionsFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter& charlsonCounter,
    ThreadSafeCounter& elixhauserCounter,
    ThreadSafeCounter& comorbidityCounter) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto& file : files) {
        results.emplace_back(pool.enqueue([&](const std::string& path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            
            std::string line;
            std::getline(csv, line); // Skip header
            
            // Extract header positions
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, codeIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "CODE") codeIdx = static_cast<int>(i);
            }
            
            if (patientIdx == -1 || codeIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(patientIdx, codeIdx)) continue;
                
                std::string patientId = values[patientIdx];
                std::string code = values[codeIdx];
                
                // Fast direct lookups using the MedicalDictionaries header
                auto charlson_it = CHARLSON_CODE_TO_WEIGHT.find(code);
                if (charlson_it != CHARLSON_CODE_TO_WEIGHT.end()) {
                    charlsonCounter.addFloat(patientId, charlson_it->second);
                }
                
                auto elixhauser_it = ELIXHAUSER_CODE_TO_WEIGHT.find(code);
                if (elixhauser_it != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                    elixhauserCounter.addFloat(patientId, elixhauser_it->second);
                }
                
                double comorbidity = findGroupWeightFast(code);
                if (comorbidity > 0) {
                    comorbidityCounter.addFloat(patientId, static_cast<float>(comorbidity));
                }
            }
        }, file));
    }
    
    // Wait for all to complete
    for (auto& result : results) {
        result.get();
    }
}

static void processEncountersFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter& hospitalizationCounter) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto& file : files) {
        results.emplace_back(pool.enqueue([&](const std::string& path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            
            std::string line;
            std::getline(csv, line); // Skip header
            
            // Extract header positions
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, classIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "ENCOUNTERCLASS") classIdx = static_cast<int>(i);
            }
            
            if (patientIdx == -1 || classIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(patientIdx, classIdx)) continue;
                
                if (values[classIdx] == "inpatient") {
                    hospitalizationCounter.addInt(values[patientIdx], 1);
                }
            }
        }, file));
    }
    
    // Wait for all to complete
    for (auto& result : results) {
        result.get();
    }
}

// Template for processing different file types
class MedicationTrackerThreadSafe {
private:
    mutable std::mutex mutex;  // Make mutex mutable so it can be locked in const methods
    std::unordered_map<std::string, std::unordered_set<size_t>> medHashes;
    
public:
    void addMedication(const std::string& patientId, const std::string& code) {
        size_t h = std::hash<std::string>{}(code);
        std::lock_guard<std::mutex> lock(mutex);
        medHashes[patientId].insert(h);
    }
    
    uint16_t getCount(const std::string& patientId) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = medHashes.find(patientId);
        return (it != medHashes.end()) ? static_cast<uint16_t>(it->second.size()) : 0;
    }
};

static void processMedicationsFilesParallel(
    const std::vector<std::string>& files,
    MedicationTrackerThreadSafe& medicationTracker) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto& file : files) {
        results.emplace_back(pool.enqueue([&](const std::string& path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            
            std::string line;
            std::getline(csv, line); // Skip header
            
            // Extract header positions
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, codeIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "CODE") codeIdx = static_cast<int>(i);
            }
            
            if (patientIdx == -1 || codeIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(patientIdx, codeIdx)) continue;
                
                medicationTracker.addMedication(values[patientIdx], values[codeIdx]);
            }
        }, file));
    }
    
    // Wait for all to complete
    for (auto& result : results) {
        result.get();
    }
}

static void processObservationsFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter& abnormalCounter) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto& file : files) {
        results.emplace_back(pool.enqueue([&](const std::string& path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            
            std::string line;
            std::getline(csv, line); // Skip header
            
            // Extract header positions
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, descIdx = -1, valueIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "DESCRIPTION") descIdx = static_cast<int>(i);
                else if (header[i] == "VALUE") valueIdx = static_cast<int>(i);
            }
            
            if (patientIdx == -1 || descIdx == -1 || valueIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(std::max(patientIdx, descIdx), valueIdx)) continue;
                
                double value = 0.0;
                try {
                    value = std::stod(values[valueIdx]);
                } catch (...) {
                    continue;
                }
                
                // Using the function from MedicalDictionaries
                if (isAbnormalObsFast(values[descIdx], value)) {
                    abnormalCounter.addInt(values[patientIdx], 1);
                }
            }
        }, file));
    }
    
    // Wait for all to complete
    for (auto& result : results) {
        result.get();
    }
}

/****************************************************
 * HEALTH INDEX
 ****************************************************/
static double computeHealthIndex(const PatientRecord &p)
{
    double base     = 10.0;
    double penalty1 = 0.4 * p.Comorbidity_Score;
    double penalty2 = 1.0 * p.Hospitalizations_Count;
    double penalty3 = 0.2 * p.Medications_Count;
    double penalty4 = 0.3 * p.Abnormal_Observations_Count;
    double penalty5 = 0.1 * p.CharlsonIndex + 0.05 * p.ElixhauserIndex;

    double raw = base - (penalty1 + penalty2 + penalty3 + penalty4 + penalty5);
    if (raw < 1.0)  raw = 1.0;
    if (raw > 10.0) raw = 10.0;
    return raw;
}

/****************************************************
 * SUBSET UTILS - OPTIMIZED
 ****************************************************/
// Faster subset identification using unordered_set
static std::unordered_set<std::string> findDiabeticPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    for (const auto& c : conds) {
        std::string lower = c.DESCRIPTION;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("diabetes") != std::string::npos) {
            out.insert(c.PATIENT);
        }
    }
    return out;
}

static std::unordered_set<std::string> findCKDPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    for (const auto& c : conds) {
        std::string lower = c.DESCRIPTION;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("chronic kidney disease") != std::string::npos ||
            lower.find("ckd") != std::string::npos) {
            out.insert(c.PATIENT);
        }
    }
    return out;
}

static std::vector<PatientRecord> filterSubpopulationOptimized(
    const std::vector<PatientRecord> &allP,
    const std::string &subsetType,
    const std::vector<ConditionRow> &conds)
{
    if (subsetType == "none") {
        return allP;
    }
    
    std::unordered_set<std::string> patientSet;
    if (subsetType == "diabetes") {
        patientSet = findDiabeticPatientsOptimized(conds);
    } else if (subsetType == "ckd") {
        patientSet = findCKDPatientsOptimized(conds);
    }
    
    if (patientSet.empty()) {
        return allP; // Fallback if no patients found
    }
    
    std::vector<PatientRecord> sub;
    sub.reserve(allP.size());
    for (const auto &p : allP) {
        if (patientSet.count(p.Id)) {
            sub.push_back(p);
        }
    }
    return sub;
}

/****************************************************
 * FEATURE UTILS
 ****************************************************/
static std::vector<std::string> getFeatureCols(const std::string &feature_config)
{
    std::vector<std::string> base = {
        "Id","GENDER","RACE","ETHNICITY","MARITAL",
        "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","INCOME",
        "AGE","DECEASED",
        "Hospitalizations_Count","Medications_Count","Abnormal_Observations_Count"
    };
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
    // header
    for (size_t i=0; i<cols.size(); i++) {
        ofs << cols[i];
        if (i+1 < cols.size()) ofs << ",";
    }
    ofs << "\n";

    for (auto &p : pats) {
        for (size_t c=0; c<cols.size(); c++) {
            if (c > 0) ofs << ",";
            const auto &col= cols[c];
            if      (col=="Id")                { ofs << p.Id; }
            else if (col=="GENDER")            { ofs << p.GENDER; }
            else if (col=="RACE")              { ofs << p.RACE; }
            else if (col=="ETHNICITY")         { ofs << p.ETHNICITY; }
            else if (col=="MARITAL")           { ofs << p.MARITAL; }
            else if (col=="HEALTHCARE_EXPENSES"){ ofs << p.HEALTHCARE_EXPENSES; }
            else if (col=="HEALTHCARE_COVERAGE"){ ofs << p.HEALTHCARE_COVERAGE; }
            else if (col=="INCOME")            { ofs << p.INCOME; }
            else if (col=="AGE")               { ofs << p.AGE; }
            else if (col=="DECEASED")          { ofs << (p.DECEASED ? "1":"0"); }
            else if (col=="Hospitalizations_Count") { ofs << p.Hospitalizations_Count; }
            else if (col=="Medications_Count")       { ofs << p.Medications_Count; }
            else if (col=="Abnormal_Observations_Count") { ofs << p.Abnormal_Observations_Count; }
            else if (col=="Health_Index")      { ofs << p.Health_Index; }
            else if (col=="CharlsonIndex")     { ofs << p.CharlsonIndex; }
            else if (col=="ElixhauserIndex")   { ofs << p.ElixhauserIndex; }
            else {
                ofs << 0;
            }
        }
        ofs << "\n";
    }
    ofs.close();
    std::cout << "[INFO] Wrote features => " << outFile << "\n";
}

/****************************************************
 * SAVE FINAL DATA
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
            << p.Health_Index << "," << p.AGE << ","
            << (p.DECEASED ? "1" : "0")
            << "\n";
    }
    ofs.close();
    std::cout << "[INFO] Wrote final data => " << outfile << "\n";
}

/****************************************************
 * EMBEDDED PYTHON / TABNET
 ****************************************************/
static bool runPythonInference(const std::string &model_id,
                               const std::string &csvPath)
{
    std::string script = "tabnet_inference.py";
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
 * OPTIONAL XAI
 ****************************************************/
static void runExplainability()
{
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
 * BATCH-BASED PROCESSING
 ****************************************************/

static void processConditionsInBatches(const std::string &path,
                                       std::function<void(const ConditionRow&)> callback)
{
    BatchProcessor::processFile<ConditionRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ConditionRow {
            ConditionRow c;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            c.PATIENT     = getValue("PATIENT");
            c.CODE        = getValue("CODE");
            c.DESCRIPTION = getValue("DESCRIPTION");
            return c;
        },
        callback
    );
}

static void processEncountersInBatches(const std::string &path,
                                       std::function<void(const EncounterRow&)> callback)
{
    BatchProcessor::processFile<EncounterRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> EncounterRow {
            EncounterRow e;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            e.Id             = getValue("Id");
            e.PATIENT        = getValue("PATIENT");
            e.ENCOUNTERCLASS = getValue("ENCOUNTERCLASS");
            return e;
        },
        callback
    );
}

static void processMedicationsInBatches(const std::string &path,
                                        std::function<void(const MedicationRow&)> callback)
{
    BatchProcessor::processFile<MedicationRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> MedicationRow {
            MedicationRow m;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            m.PATIENT     = getValue("PATIENT");
            m.ENCOUNTER   = getValue("ENCOUNTER");
            m.CODE        = getValue("CODE");
            m.DESCRIPTION = getValue("DESCRIPTION");
            return m;
        },
        callback
    );
}

static void processObservationsInBatches(const std::string &path,
                                         std::function<void(const ObservationRow&)> callback)
{
    BatchProcessor::processFile<ObservationRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ObservationRow {
            ObservationRow o;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            o.PATIENT     = getValue("PATIENT");
            o.ENCOUNTER   = getValue("ENCOUNTER");
            o.CODE        = getValue("CODE");
            o.DESCRIPTION = getValue("DESCRIPTION");
            try {
                o.VALUE = std::stod(getValue("VALUE"));
            } catch(...) {}
            o.UNITS = getValue("UNITS");
            return o;
        },
        callback
    );
}

static void processProceduresInBatches(const std::string &path,
                                       std::function<void(const ProcedureRow&)> callback)
{
    BatchProcessor::processFile<ProcedureRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ProcedureRow {
            ProcedureRow p;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            p.PATIENT     = getValue("PATIENT");
            p.ENCOUNTER   = getValue("ENCOUNTER");
            p.CODE        = getValue("CODE");
            p.DESCRIPTION = getValue("DESCRIPTION");
            return p;
        },
        callback
    );
}

// For patients
static void processPatientsInBatches(const std::string &path,
                                     std::function<void(const PatientRecord&)> callback)
{
    BatchProcessor::processFile<PatientRecord>(
        path,
        [&path](const std::vector<std::string> &header,
                const std::vector<std::string> &values) -> PatientRecord {
            PatientRecord p;
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = (int)i;
            }
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };

            p.Id         = getValue("Id");
            p.BIRTHDATE  = getValue("BIRTHDATE");
            p.DEATHDATE  = getValue("DEATHDATE");
            p.GENDER     = getValue("GENDER");
            p.RACE       = getValue("RACE");
            p.ETHNICITY  = getValue("ETHNICITY");
            p.MARITAL    = getValue("MARITAL");

            try { p.HEALTHCARE_EXPENSES = std::stof(getValue("HEALTHCARE_EXPENSES")); } catch(...) {}
            try { p.HEALTHCARE_COVERAGE = std::stof(getValue("HEALTHCARE_COVERAGE")); } catch(...) {}
            try { p.INCOME             = std::stof(getValue("INCOME")); } catch(...) {}

            p.NewData = isDiffFile(path);
            if (!p.DEATHDATE.empty() && p.DEATHDATE != "NaN") {
                p.DECEASED = true;
            }
            // naive approach to Age
            p.AGE = 50.0f;

            return p;
        },
        callback
    );
}

/****************************************************
 * BATCH COMPUTATIONS: CHARLSON, ELIXHAUSER, ETC.
 ****************************************************/

// Charlson
static void computeCharlsonIndexBatched(std::function<void(const std::string&, float)> scoreCallback)
{
    // Use the dictionaries from the header
    std::unordered_map<long long,int> codeToWeight;
    for (auto &kv : SNOMED_TO_CHARLSON) {
        long long codeLL = kv.first;
        const std::string &cat = kv.second;
        auto wIt = CHARLSON_CATEGORY_WEIGHTS.find(cat);
        if (wIt != CHARLSON_CATEGORY_WEIGHTS.end()) {
            codeToWeight[codeLL] = wIt->second;
        }
    }

    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            long long codeLL = 0;
            try {
                codeLL = std::stoll(c.CODE);
            } catch(...) { return; }
            auto it = codeToWeight.find(codeLL);
            if (it != codeToWeight.end()) {
                // Add its weight
                scoreCallback(c.PATIENT, (float)it->second);
            }
        });
    }
}

// Elixhauser
static void computeElixhauserIndexBatched(std::function<void(const std::string&, float)> scoreCallback)
{
    std::unordered_map<long long,int> codeToWeight;
    for (auto &kv : SNOMED_TO_ELIXHAUSER) {
        long long codeLL = kv.first;
        const std::string &cat = kv.second;
        auto wIt = ELIXHAUSER_CATEGORY_WEIGHTS.find(cat);
        if (wIt != ELIXHAUSER_CATEGORY_WEIGHTS.end()) {
            codeToWeight[codeLL] = wIt->second;
        }
    }

    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            long long codeLL = 0;
            try {
                codeLL = std::stoll(c.CODE);
            } catch(...) { return; }
            auto it = codeToWeight.find(codeLL);
            if (it != codeToWeight.end()) {
                scoreCallback(c.PATIENT, (float)it->second);
            }
        });
    }
}

// Simple group-based comorbidity
static void computeComorbidityScoreBatched(std::function<void(const std::string&, float)> scoreCallback)
{
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            float w = (float)findGroupWeight(c.CODE);
            if (w > 0.f) {
                scoreCallback(c.PATIENT, w);
            }
        });
    }
}

// If you want the "optimized" approach instead, just swap the function body
static void computeComorbidityScoreBatchedOptimized(std::function<void(const std::string&, float)> scoreCallback)
{
    initializeCodeLookups(); // Make sure lookups are built from MedicalDictionaries
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            float w = (float)findGroupWeightOptimized(c.CODE);
            if (w > 0.f) {
                scoreCallback(c.PATIENT, w);
            }
        });
    }
}

// Hospitalizations
static void countHospitalizationsInBatches(std::function<void(const std::string&, uint16_t)> countCallback)
{
    auto encFiles = listCSVFiles("encounters");
    for (auto &path : encFiles) {
        processEncountersInBatches(path, [&](const EncounterRow &e){
            if (e.ENCOUNTERCLASS == "inpatient") {
                countCallback(e.PATIENT, 1);
            }
        });
    }
}

// Medications
static void countMedicationsInBatches(std::function<void(const std::string&, const std::string&)> medCallback)
{
    auto medFiles = listCSVFiles("medications");
    for (auto &path : medFiles) {
        processMedicationsInBatches(path, [&](const MedicationRow &m){
            medCallback(m.PATIENT, m.CODE);
        });
    }
}

// Abnormal Observations
static void countAbnormalObservationsInBatches(std::function<void(const std::string&)> abnormalCallback)
{
    auto obsFiles = listCSVFiles("observations");
    for (auto &path : obsFiles) {
        processObservationsInBatches(path, [&](const ObservationRow &o){
            // Using function from MedicalDictionaries.h
            if (isAbnormalObs(o.DESCRIPTION, o.VALUE)) {
                abnormalCallback(o.PATIENT);
            }
        });
    }
}

// Or the "optimized" version
static void countAbnormalObservationsInBatchesOptimized(std::function<void(const std::string&)> abnormalCallback)
{
    initializeObsLookups(); // Use function from MedicalDictionaries.h
    auto obsFiles = listCSVFiles("observations");
    for (auto &path : obsFiles) {
        processObservationsInBatches(path, [&](const ObservationRow &o){
            if (isAbnormalObsOptimized(o.DESCRIPTION, o.VALUE)) {
                abnormalCallback(o.PATIENT);
            }
        });
    }
}

/****************************************************
 * MEDICATION TRACKER (OPTIONAL)
 ****************************************************/
class MedicationTracker {
private:
    std::unordered_map<std::string, std::unordered_set<size_t>> medHashes;
public:
    void addMedication(const std::string &patientId, const std::string &code) {
        size_t h = std::hash<std::string>{}(code);
        medHashes[patientId].insert(h);
    }
    uint16_t getCount(const std::string &patientId) {
        auto it = medHashes.find(patientId);
        if (it != medHashes.end()) {
            return (uint16_t) it->second.size();
        }
        return 0;
    }
};

/****************************************************
 * main()
 ****************************************************/
int main(int argc, char* argv[])
{
    PROGRAM_START_TIME = std::chrono::high_resolution_clock::now();
    
    int popSize = 100;
    bool enableXAI = false;

    // parse
    for (int i=1; i<argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--population=", 0) == 0) {
            popSize = std::stoi(arg.substr(13));
        } else if (arg == "--enable-xai") {
            enableXAI = true;
        } else if (arg.rfind("--threads=", 0) == 0) {
            THREAD_COUNT = std::stoi(arg.substr(10));
        }
    }
    
    std::cout << "[INFO] popSize=" << popSize
              << ", XAI=" << (enableXAI ? "true" : "false")
              << ", threads=" << THREAD_COUNT << "\n";

    // 1) Synthea & copy
    logElapsedTime("Starting Synthea");
    runSynthea(popSize);
    copySyntheaOutput();
    logElapsedTime("Synthea data generated and copied");

    // 2) Initialize optimized lookups for fast processing
    std::cout << "[INFO] Building optimized lookup tables..." << std::endl;
    initializeDirectLookups(); // Now imported from MedicalDictionaries.h
    initializeObsAbnormalDirect(); // Now imported from MedicalDictionaries.h
    logElapsedTime("Lookup tables initialized");

    // 3) Create thread-safe data structures for parallel processing
    ThreadSafeCounter charlsonCounter;
    ThreadSafeCounter elixhauserCounter;
    ThreadSafeCounter comorbidityCounter;
    ThreadSafeCounter hospitalizationCounter;
    ThreadSafeCounter abnormalObsCounter;
    MedicationTrackerThreadSafe medicationTracker;

    // 4) Process files in parallel with optimized data structures
    std::cout << "[INFO] Processing condition files in parallel..." << std::endl;
    auto condFiles = listCSVFiles("conditions");
    processConditionsFilesParallel(condFiles, charlsonCounter, elixhauserCounter, comorbidityCounter);
    logElapsedTime("Conditions processing complete");
    
    std::cout << "[INFO] Processing encounter files in parallel..." << std::endl;
    auto encFiles = listCSVFiles("encounters");
    processEncountersFilesParallel(encFiles, hospitalizationCounter);
    logElapsedTime("Encounters processing complete");
    
    std::cout << "[INFO] Processing medication files in parallel..." << std::endl;
    auto medFiles = listCSVFiles("medications");
    processMedicationsFilesParallel(medFiles, medicationTracker);
    logElapsedTime("Medications processing complete");
    
    std::cout << "[INFO] Processing observation files in parallel..." << std::endl;
    auto obsFiles = listCSVFiles("observations");
    processObservationsFilesParallel(obsFiles, abnormalObsCounter);
    logElapsedTime("Observations processing complete");

    // 5) Process patient files and combine all data
    std::cout << "[INFO] Building final patient records...\n";
    std::vector<PatientRecord> allPatients;
    std::vector<PatientRecord> newPatients;

    auto patientFiles = listCSVFiles("patients");
    for (auto &path : patientFiles) {
        bool diff = isDiffFile(path);
        processPatientsInBatches(path, [&](const PatientRecord &baseP) {
            PatientRecord p = baseP;
            p.CharlsonIndex   = charlsonCounter.getFloat(p.Id);
            p.ElixhauserIndex = elixhauserCounter.getFloat(p.Id);
            p.Comorbidity_Score = comorbidityCounter.getFloat(p.Id);
            p.Hospitalizations_Count = hospitalizationCounter.getInt(p.Id);
            p.Medications_Count = medicationTracker.getCount(p.Id);
            p.Abnormal_Observations_Count = abnormalObsCounter.getInt(p.Id);
            p.Health_Index = (float) computeHealthIndex(p);

            allPatients.push_back(p);
            if (diff) {
                newPatients.push_back(p);
            }
        });
    }
    
    std::cout << "[INFO] In total, loaded " << allPatients.size()
              << " patients, of which " << newPatients.size()
              << " are newly generated.\n";
    logElapsedTime("Patient records assembled");

    // 6) Save final CSV
    std::string finalCSV = DATA_DIR + "/patient_data_with_all_indices.csv";
    saveFinalDataCSV(allPatients, finalCSV);
    logElapsedTime("Final CSV saved");

    // 7) Predictions on new patients
    std::cout << "[INFO] Running predictions on new patients...\n";

    Py_SetPythonHome(L"C:\\Users\\imran\\miniconda3\\envs\\tf_gpu_env");
    Py_Initialize();

    // If we need condition data for subpop
    bool needsSubpop = false;
    for (auto &mc : MODEL_CONFIG_MAP) {
        if (mc.second.subset != "none") { 
            needsSubpop = true; 
            break; 
        }
    }
    
    std::vector<ConditionRow> allConditions;
    if (needsSubpop) {
        for (auto &path : condFiles) {
            processConditionsInBatches(path, [&](const ConditionRow &c){
                allConditions.push_back(c);
            });
        }
    }

    // For each model
    for (auto &it : MODEL_CONFIG_MAP) {
        std::string model_id = it.first;
        std::string subsetType = it.second.subset;
        std::string featconf = it.second.feature_config;

        std::cout << "\n[INFO] Predicting with model=" << model_id
                 << ", subset=" << subsetType
                 << ", feature_config=" << featconf << "\n";

        std::vector<PatientRecord> sub;
        if (subsetType=="none") {
            sub = newPatients;
        } else {
            // Filter using optimized version
            sub = filterSubpopulationOptimized(newPatients, subsetType, allConditions);
        }
        
        if (sub.empty()) {
            std::cout << "[INFO] No new patients in subpop=" << subsetType
                     << " => skip.\n";
            continue;
        }

        auto cols = getFeatureCols(featconf);
        std::string outDir = DATA_DIR + "/new_predictions/" + model_id;
        makeDirIfNeeded(DATA_DIR + "/new_predictions");
        makeDirIfNeeded(outDir);

        std::string infCSV = outDir + "/input_for_inference.csv";
        writeFeaturesCSV(sub, infCSV, cols);

        bool ok = runPythonInference(model_id, infCSV);
        if (!ok) {
            std::cerr << "[WARN] Inference for " << model_id << " failed.\n";
        }
    }

    // 8) Optional XAI
    if (enableXAI) {
        runExplainability();
    }

    Py_Finalize();

    std::cout << "\n[INFO] Done. Generated new data, computed indexes, "
             << "computed health index, saved final CSV, ran TabNet. XAI="
             << (enableXAI ? "true" : "false") << ".\n";
    
    logElapsedTime("Total execution time");
    return 0;
}
