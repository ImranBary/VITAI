#pragma once

#ifndef MEDICAL_DICTIONARIES_H
#define MEDICAL_DICTIONARIES_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>

/****************************************************
 * CHARLSON INDEX DICTIONARIES
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
 * ELIXHAUSER INDEX DICTIONARIES
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
   {698754002, "Peripheral vascular disorders"},

   // Hypertension
   {59621000,  "Hypertension, uncomplicated"},

   // Paralysis
   // Overwrites the earlier "Peripheral vascular disorders" for 698754002
   {698754002, "Paralysis"},
   {128188000, "Paralysis"},

   // Other Neurological Disorders
   {69896004,  "Other neurological disorders"},
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
 * GROUP-BASED COMORBIDITY
 ****************************************************/
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

/****************************************************
 * OBSERVATIONS
 ****************************************************/
struct ObsThreshold { double minVal; double maxVal; };

static std::map<std::string, ObsThreshold> OBS_THRESHOLDS = {
    {"Systolic Blood Pressure",{90,120}},
    {"Body Mass Index",{18.5,24.9}}
};

static std::map<std::string,std::string> OBS_DESC_MAP = {
    {"Systolic Blood Pressure","Systolic Blood Pressure"},
    {"Body mass index (BMI) [Ratio]","Body Mass Index"}
};

/****************************************************
 * OPTIMIZED LOOKUPS
 ****************************************************/
// Pre-compute direct lookup tables for faster code access
extern const std::unordered_map<std::string, float> CHARLSON_CODE_TO_WEIGHT;
extern const std::unordered_map<std::string, float> ELIXHAUSER_CODE_TO_WEIGHT;
static std::unordered_map<std::string, double> CODE_TO_GROUP_WEIGHT;
static std::unordered_map<std::string, std::string> CODE_TO_GROUP_NAME;
static std::unordered_map<std::string, double> GROUP_NAME_TO_WEIGHT;
static std::unordered_map<std::string, std::string> OBS_DESC_MAP_OPTIMIZED;
static std::unordered_map<std::string, ObsThreshold> OBS_THRESHOLDS_OPTIMIZED;
static std::unordered_map<std::string, std::pair<double, double>> OBS_ABNORMAL_DIRECT;

static bool CODE_LOOKUPS_INITIALIZED = false;
static bool OBS_LOOKUPS_INITIALIZED = false;

/****************************************************
 * LOOKUP INITIALIZATION FUNCTIONS
 ****************************************************/
// Basic function for group weight lookup
static double findGroupWeight(const std::string &codeStr)
{
    for (auto &kv : SNOMED_GROUPS) {
        for (auto &c : kv.second) {
            if (c == codeStr) {
                return GROUP_WEIGHTS[kv.first];
            }
        }
    }
    return GROUP_WEIGHTS["Other"];
}

// Initialize code lookups at runtime once
static void initializeCodeLookups()
{
    if (CODE_LOOKUPS_INITIALIZED) return;
    CODE_LOOKUPS_INITIALIZED = true;

    // Fill CODE_TO_GROUP_NAME
    for (auto &kv : SNOMED_GROUPS) {
        const std::string &groupName = kv.first;
        for (auto &code : kv.second) {
            CODE_TO_GROUP_NAME[code] = groupName;
        }
    }
    // Fill GROUP_NAME_TO_WEIGHT
    for (auto &kv : GROUP_WEIGHTS) {
        GROUP_NAME_TO_WEIGHT[kv.first] = kv.second;
    }
}

// Optimized group weight finder that uses direct lookup
static double findGroupWeightOptimized(const std::string &codeStr)
{
    auto it = CODE_TO_GROUP_NAME.find(codeStr);
    if (it != CODE_TO_GROUP_NAME.end()) {
        return GROUP_NAME_TO_WEIGHT[it->second];
    }
    return GROUP_WEIGHTS["Other"];
}

// Optimized group weight finder using direct weight lookup
static double findGroupWeightFast(const std::string &codeStr) {
    auto it = CODE_TO_GROUP_WEIGHT.find(codeStr);
    return (it != CODE_TO_GROUP_WEIGHT.end()) ? it->second : GROUP_WEIGHTS["Other"];
}

// Initialize observation lookups
static void initializeObsLookups()
{
    if (OBS_LOOKUPS_INITIALIZED) return;
    OBS_LOOKUPS_INITIALIZED = true;

    for (auto &kv : OBS_DESC_MAP) {
        OBS_DESC_MAP_OPTIMIZED[kv.first] = kv.second;
    }
    for (auto &kv : OBS_THRESHOLDS) {
        OBS_THRESHOLDS_OPTIMIZED[kv.first] = kv.second;
    }
}

// Observation abnormality check
static bool isAbnormalObs(const std::string &desc, double val)
{
    auto it = OBS_DESC_MAP.find(desc);
    if (it != OBS_DESC_MAP.end()) {
        auto thr = OBS_THRESHOLDS.find(it->second);
        if (thr != OBS_THRESHOLDS.end()) {
            if (val < thr->second.minVal || val > thr->second.maxVal) {
                return true;
            }
        }
    }
    return false;
}

// Optimize observation abnormality check
static void initializeObsAbnormalDirect() {
    // Pre-compute the direct map from description to min/max values
    for (const auto& desc_it : OBS_DESC_MAP) {
        auto threshold_it = OBS_THRESHOLDS.find(desc_it.second);
        if (threshold_it != OBS_THRESHOLDS.end()) {
            OBS_ABNORMAL_DIRECT[desc_it.first] = std::make_pair(
                threshold_it->second.minVal, 
                threshold_it->second.maxVal
            );
        }
    }
}

static bool isAbnormalObsFast(const std::string &desc, double val) {
    auto it = OBS_ABNORMAL_DIRECT.find(desc);
    if (it != OBS_ABNORMAL_DIRECT.end()) {
        return (val < it->second.first || val > it->second.second);
    }
    return false;
}

static bool isAbnormalObsOptimized(const std::string &desc, double val)
{
    auto it = OBS_DESC_MAP_OPTIMIZED.find(desc);
    if (it != OBS_DESC_MAP_OPTIMIZED.end()) {
        auto thr = OBS_THRESHOLDS_OPTIMIZED.find(it->second);
        if (thr != OBS_THRESHOLDS_OPTIMIZED.end()) {
            if (val < thr->second.minVal || val > thr->second.maxVal) {
                return true;
            }
        }
    }
    return false;
}

// Initialize all direct lookups for faster processing
static void initializeDirectLookups() {
    // Don't modify the const maps - they're initialized in the .cpp file
    // Instead, just initialize the non-const maps
    
    // Initialize group weight direct map
    for (const auto& group : SNOMED_GROUPS) {
        double weight = GROUP_WEIGHTS[group.first];
        for (const auto& code : group.second) {
            CODE_TO_GROUP_WEIGHT[code] = weight;
        }
    }
    
    // Initialize the other lookups as well
    initializeCodeLookups();
    initializeObsLookups();
    initializeObsAbnormalDirect();
    
    std::cout << "[INFO] Direct lookup maps initialized\n";
}

// Function to quickly find the comorbidity group weight for a given code
double findGroupWeightFast(const std::string& code);

// Function to quickly determine if an observation is abnormal
bool isAbnormalObsFast(const std::string& description, double value);

#endif // MEDICAL_DICTIONARIES_H
