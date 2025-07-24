#include "System.hpp"
#include <iostream>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include "ConfigParser.hpp"

// Struttura per definire i parametri di benchmark
struct BenchmarkParams {
    int numParticles;
    double years;
    double dtYears;
    ExecutionMode executionMode;
    ForceMethod forceMethod;
    int numThreads;
    IntegrationMethod integrationMethod;
    std::string initType;
};

// Struttura per i risultati del benchmark con statistiche
struct BenchmarkResult {
    BenchmarkParams params;
    int numRuns;
    double meanExecutionTime;
    double stdExecutionTime;
    double meanStepsPerSecond;
    double stdStepsPerSecond;
    double meanParticleStepsPerSecond;
    double stdParticleStepsPerSecond;
    long totalSteps;
    std::string errorMessage;
    bool success;
    double cvStepsPerSecond;  // Coefficient of variation
    double minStepsPerSecond;
    double maxStepsPerSecond;
};

// Funzione per calcolare statistiche
struct Statistics {
    double mean;
    double std;
    double min;
    double max;
    double cv;  // coefficient of variation
};

Statistics calculateStats(const std::vector<double>& values) {
    Statistics stats;
    if (values.empty()) return stats;
    
    double sum = 0.0;
    stats.min = values[0];
    stats.max = values[0];
    
    for (double val : values) {
        sum += val;
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
    }
    
    stats.mean = sum / values.size();
    
    if (values.size() > 1) {
        double variance = 0.0;
        for (double val : values) {
            variance += (val - stats.mean) * (val - stats.mean);
        }
        stats.std = sqrt(variance / (values.size() - 1));
        stats.cv = stats.std / stats.mean;  // coefficient of variation
    } else {
        stats.std = 0.0;
        stats.cv = 0.0;
    }
    
    return stats;
}

// Funzione per eseguire un singolo benchmark
BenchmarkResult runSingleBenchmark(const BenchmarkParams& params) {
    BenchmarkResult result;
    result.params = params;
    result.numRuns = 1;  // Single run
    result.success = false;
    result.errorMessage = "";
    
    try {
        // Imposta il numero di thread per OpenMP
        omp_set_num_threads(params.numThreads);
        
        // Calcola il numero di step
        long totalSteps = static_cast<long>(params.years / params.dtYears);
        result.totalSteps = totalSteps;
        
        // Inizializza direttamente senza usare Config per evitare problemi
        InitResult initResult;
        
        if (params.initType == "random") {
            // Usa parametri realistici per l'inizializzazione random
            int nParticles = params.numParticles;
            double minMass = 1e-10;     // Dal config.json
            double maxMass = 1e-5;      // Dal config.json
            double L = 1e2;             // Dal config.json
            double maxVelocity = 1.0;   // Dal config.json
            double minDistance = 1e-3;  // Dal config.json
            
            initResult = Initializer::initRandomParticles(nParticles, minMass, maxMass, L, maxVelocity, minDistance);
        } else {
            throw std::runtime_error("Solo inizializzazione 'random' supportata nel benchmark");
        }
        
        // Crea il sistema
        System sys(std::move(initResult.particles), params.integrationMethod, 
                  params.forceMethod, params.executionMode, OutputMode::BENCHMARK);
        
        // Esegui la simulazione e misura il tempo
        auto start = std::chrono::high_resolution_clock::now();
        
        // Per il benchmark, non salvare output (file vuoto)
        sys.runSimulation(params.dtYears, totalSteps, totalSteps + 1, "", "");
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calcola i risultati usando i nuovi campi con statistiche
        std::chrono::duration<double> elapsed = end - start;
        double execTime = elapsed.count();
        double stepsPerSec = totalSteps / execTime;
        double particleStepsPerSec = (static_cast<double>(params.numParticles) * totalSteps) / execTime;
        
        // Assegna ai campi mean dato che è un singolo run
        result.meanExecutionTime = execTime;
        result.stdExecutionTime = 0.0;  // No std per singolo run
        result.meanStepsPerSecond = stepsPerSec;
        result.stdStepsPerSecond = 0.0;
        result.meanParticleStepsPerSecond = particleStepsPerSec;
        result.stdParticleStepsPerSecond = 0.0;
        result.cvStepsPerSecond = 0.0;  // No CV per singolo run
        result.minStepsPerSecond = stepsPerSec;
        result.maxStepsPerSecond = stepsPerSec;
        
        result.success = true;
        
    } catch (const std::exception& ex) {
        result.errorMessage = ex.what();
        result.success = false;
    }
    
    return result;
}

// Funzione per eseguire multipli benchmark con statistiche
BenchmarkResult runMultipleBenchmarks(const BenchmarkParams& params, int numRuns = 15) {
    BenchmarkResult result;
    result.params = params;
    result.numRuns = numRuns;
    result.success = false;
    
    std::vector<double> executionTimes;
    std::vector<double> stepsPerSecond;
    std::vector<double> particleStepsPerSecond;
    
    try {
        // Imposta il numero di thread per OpenMP
        omp_set_num_threads(params.numThreads);
        
        // Calcola il numero di step
        long totalSteps = static_cast<long>(params.years / params.dtYears);
        result.totalSteps = totalSteps;
        
        // Esegui multipli run
        for (int run = 0; run < numRuns; ++run) {
            // Inizializza sistema per ogni run (per evitare effetti di cache)
            InitResult initResult;
            
            if (params.initType == "random") {
                int nParticles = params.numParticles;
                double minMass = 1e-10;
                double maxMass = 1e-5;
                double L = 1e2;
                double maxVelocity = 1.0;
                double minDistance = 1e-3;
                
                initResult = Initializer::initRandomParticles(nParticles, minMass, maxMass, L, maxVelocity, minDistance);
            } else {
                throw std::runtime_error("Solo inizializzazione 'random' supportata nel benchmark");
            }
            
            // Crea il sistema
            System sys(std::move(initResult.particles), params.integrationMethod, 
                      params.forceMethod, params.executionMode, OutputMode::BENCHMARK);
            
            // Misura il tempo per questo run
            auto start = std::chrono::high_resolution_clock::now();
            sys.runSimulation(params.dtYears, totalSteps, totalSteps + 1, "", "");
            auto end = std::chrono::high_resolution_clock::now();
            
            // Raccogli le metriche
            std::chrono::duration<double> elapsed = end - start;
            double execTime = elapsed.count();
            double stepsPerSec = totalSteps / execTime;
            double particleStepsPerSec = (static_cast<double>(params.numParticles) * totalSteps) / execTime;
            
            executionTimes.push_back(execTime);
            stepsPerSecond.push_back(stepsPerSec);
            particleStepsPerSecond.push_back(particleStepsPerSec);
        }
        
        // Calcola statistiche
        auto execTimeStats = calculateStats(executionTimes);
        auto stepsPerSecStats = calculateStats(stepsPerSecond);
        auto particleStepsPerSecStats = calculateStats(particleStepsPerSecond);
        
        result.meanExecutionTime = execTimeStats.mean;
        result.stdExecutionTime = execTimeStats.std;
        result.meanStepsPerSecond = stepsPerSecStats.mean;
        result.stdStepsPerSecond = stepsPerSecStats.std;
        result.meanParticleStepsPerSecond = particleStepsPerSecStats.mean;
        result.stdParticleStepsPerSecond = particleStepsPerSecStats.std;
        result.cvStepsPerSecond = stepsPerSecStats.cv;
        result.minStepsPerSecond = stepsPerSecStats.min;
        result.maxStepsPerSecond = stepsPerSecStats.max;
        
        result.success = true;
        
    } catch (const std::exception& ex) {
        result.errorMessage = ex.what();
        result.success = false;
    }
    
    return result;
}

// Funzione per salvare i risultati in CSV
void saveBenchmarkResults(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Header CSV con statistiche
    file << "NumParticles,Years,DtYears,ExecutionMode,ForceMethod,IntegrationMethod,NumThreads,InitType,";
    file << "NumRuns,TotalSteps,";
    file << "MeanExecutionTime,StdExecutionTime,";
    file << "MeanStepsPerSecond,StdStepsPerSecond,CVStepsPerSecond,MinStepsPerSecond,MaxStepsPerSecond,";
    file << "MeanParticleStepsPerSecond,StdParticleStepsPerSecond,";
    file << "Success,ErrorMessage\n";
    
    // Dati
    for (const auto& result : results) {
        const auto& p = result.params;
        
        std::string execMode = (p.executionMode == ExecutionMode::CPU) ? "CPU" : "GPU";
        std::string forceMethod;
        switch (p.forceMethod) {
            case ForceMethod::PAIRWISE: forceMethod = "PAIRWISE"; break;
            case ForceMethod::PAIRWISE_AVX2_FP32: forceMethod = "PAIRWISE_AVX2/FP32"; break;
            case ForceMethod::ADAPTIVE_MUTUAL: forceMethod = "ADAPTIVE_MUTUAL"; break;
            case ForceMethod::BARNES_HUT: forceMethod = "BARNES_HUT"; break;
        }
        std::string integMethod = (p.integrationMethod == IntegrationMethod::EULER) ? "EULER" : "VELOCITY_VERLET";
        
        file << p.numParticles << ","
             << p.years << ","
             << p.dtYears << ","
             << execMode << ","
             << forceMethod << ","
             << integMethod << ","
             << p.numThreads << ","
             << p.initType << ","
             << result.numRuns << ","
             << result.totalSteps << ","
             << std::fixed << std::setprecision(6) << result.meanExecutionTime << ","
             << std::fixed << std::setprecision(6) << result.stdExecutionTime << ","
             << std::fixed << std::setprecision(2) << result.meanStepsPerSecond << ","
             << std::fixed << std::setprecision(2) << result.stdStepsPerSecond << ","
             << std::fixed << std::setprecision(4) << result.cvStepsPerSecond << ","
             << std::fixed << std::setprecision(2) << result.minStepsPerSecond << ","
             << std::fixed << std::setprecision(2) << result.maxStepsPerSecond << ","
             << std::scientific << std::setprecision(3) << result.meanParticleStepsPerSecond << ","
             << std::scientific << std::setprecision(3) << result.stdParticleStepsPerSecond << ","
             << (result.success ? "true" : "false") << ","
             << "\"" << result.errorMessage << "\"\n";
    }
    
    file.close();
}

// Funzione per eseguire il benchmark automatico
void runAutomaticBenchmark() {
    std::cout << "=== BENCHMARK AUTOMATICO N-BODY CON STATISTICHE ===" << std::endl;
    
    // Parametri ridotti per velocità in presentazione
    std::vector<int> particleCounts = {2, 5, 8, 10, 25, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 100000};  // Ridotto per velocità
    std::vector<int> threadCounts = {2};  // Ridotto
    std::vector<ExecutionMode> executionModes = {ExecutionMode::CPU};
    std::vector<ForceMethod> forceMethods = {ForceMethod::BARNES_HUT};
    std::vector<IntegrationMethod> integrationMethods = {IntegrationMethod::VELOCITY_VERLET};
    
    double years = 0.02;     // 2 steps
    double dtYears = 0.01;
    std::string initType = "random";
    int numRuns = 10;  // 5 run per configurazione per statistiche decenti
    std::cout << "Numero di run per configurazione: " << numRuns << std::endl;
    std::cout << "Parametri di benchmark: " << std::endl;
    std::cout << " - Anni: " << years << std::endl;
    std::cout << " - Passo (anni): " << dtYears << std::endl;
    // Parametri fissi della simulazione - uso valori più realistici per test rapidi
    // double years = 0.1;      // Solo 0.1 anni per test rapidi
    // double dtYears = 0.01;   // Step time più grande per meno iterazioni
    // std::string initType = "random";
    
    // Genera tutte le combinazioni
    std::vector<BenchmarkParams> paramSets;
    for (int particles : particleCounts) {
        for (ExecutionMode execMode : executionModes) {
            for (ForceMethod forceMethod : forceMethods) {
                for (IntegrationMethod integMethod : integrationMethods) {
                    if (execMode == ExecutionMode::CPU) {
                        // Per CPU, testa diversi numeri di thread
                        for (int threads : threadCounts) {
                            BenchmarkParams params;
                            params.numParticles = particles;
                            params.years = years;
                            params.dtYears = dtYears;
                            params.executionMode = execMode;
                            params.forceMethod = forceMethod;
                            params.numThreads = threads;
                            params.integrationMethod = integMethod;
                            params.initType = initType;
                            paramSets.push_back(params);
                        }
                    } else {
                        // Per GPU, usa solo 1 thread (GPU non usa OpenMP)
                        BenchmarkParams params;
                        params.numParticles = particles;
                        params.years = years;
                        params.dtYears = dtYears;
                        params.executionMode = execMode;
                        params.forceMethod = forceMethod;
                        params.numThreads = 1;
                        params.integrationMethod = integMethod;
                        params.initType = initType;
                        paramSets.push_back(params);
                    }
                }
            }
        }
    }
    
    std::cout << "Numero totale di test: " << paramSets.size() << std::endl;
    
    // Esegui i benchmark
    std::vector<BenchmarkResult> results;
    auto overallStart = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < paramSets.size(); ++i) {
        const auto& params = paramSets[i];
        
        double progress = static_cast<double>(i) / paramSets.size() * 100.0;
        std::cout << "\r[" << std::fixed << std::setprecision(1) << progress << "%] "
                  << "Test " << (i + 1) << "/" << paramSets.size() 
                  << " - N=" << params.numParticles 
                  << " " << (params.executionMode == ExecutionMode::CPU ? "CPU" : "GPU")
                  << " T=" << params.numThreads << " (" << numRuns << " runs)    " << std::flush;
        
        // Esegui multiple runs con statistiche
        auto testStart = std::chrono::high_resolution_clock::now();
        BenchmarkResult result = runMultipleBenchmarks(params, numRuns);
        auto testEnd = std::chrono::high_resolution_clock::now();
        
        results.push_back(result);
        
        // Mostra risultato con CV (coefficient of variation)
        if (result.success) {
            std::chrono::duration<double> testElapsed = testEnd - testStart;
            std::cout << " ✓ " << std::fixed << std::setprecision(2) << testElapsed.count() << "s"
                      << " (CV=" << std::setprecision(1) << result.cvStepsPerSecond * 100 << "%)";
        } else {
            std::cout << " ✗ FAILED";
        }
    }
    
    auto overallEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalElapsed = overallEnd - overallStart;
    
    std::cout << "\n\n=== BENCHMARK COMPLETATO ===" << std::endl;
    std::cout << "Tempo totale: " << std::fixed << std::setprecision(2) << totalElapsed.count() << " secondi" << std::endl;
    
    // Conta successi e fallimenti
    int successes = 0, failures = 0;
    for (const auto& result : results) {
        if (result.success) successes++;
        else failures++;
    }
    std::cout << "Test riusciti: " << successes << "/" << results.size() << std::endl;
    if (failures > 0) {
        std::cout << "Test falliti: " << failures << std::endl;
    }
    
    // Salva i risultati
    std::string outputFile = "presentation/benchmarks/benchmark_results_" + 
                            std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count()) + ".csv";
    
    saveBenchmarkResults(results, outputFile);
    std::cout << "Risultati salvati in: " << outputFile << std::endl;
    
    // Mostra alcuni risultati di esempio
    std::cout << "\n=== ESEMPI DI RISULTATI CON STATISTICHE ===" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), results.size()); ++i) {
        const auto& r = results[i];
        if (r.success) {
            std::cout << "N=" << r.params.numParticles 
                      << " " << (r.params.executionMode == ExecutionMode::CPU ? "CPU" : "GPU")
                      << " T=" << r.params.numThreads
                      << ": " << std::scientific << std::setprecision(2) << r.meanParticleStepsPerSecond 
                      << " ± " << r.stdParticleStepsPerSecond
                      << " particle-steps/s (CV=" << std::fixed << std::setprecision(1) << r.cvStepsPerSecond * 100 << "%)" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    // Controlla se è stata richiesta la modalità benchmark automatica
    if (argc > 1 && std::string(argv[1]) == "--auto-benchmark") {
        runAutomaticBenchmark();
        return 0;
    }
    
    try
    {
        // Load configuration from JSON file
        auto cfg = parseConfig("config.json");
        
        // Set the number of threads for OpenMP
        int numThreads = cfg.threads;
        omp_set_num_threads(numThreads);
        std::cout << "OpenMP is configured to use " << numThreads << " threads.\n";
        
        // Optimize OpenMP thread binding for better performance
        omp_set_nested(0);  // Disable nested parallelism
        
        // For CPU-bound computations, bind threads to cores for better cache utilization
        #ifdef _OPENMP
            #if _OPENMP >= 200805
                omp_set_schedule(omp_sched_guided, 0);
            #endif
        #endif

        // Set the simulation parameters from config
        IntegrationMethod integrationMethod = cfg.integrationMethod;
        ForceMethod forceMethod = cfg.forceMethod;
        ExecutionMode executionMode = cfg.executionMode;
        OutputMode outputMode = cfg.outputMode;
        
        // Add automatic fallback to CPU if GPU fails
        bool gpuFailed = false;

        std::string outputDir = cfg.outputDir;
        std::string outputFile = cfg.outputFile;
        
        double years = cfg.years;
        double dtYears = cfg.dtYears;
        long int n = static_cast<long int>(years / dtYears); // Number of steps

        // Automatic output frequency to get a target number of prints
        int desiredPrintCount = 5e2; //1e4 = 0,6MB per body in file

        std::string timeMetadata = std::string("# years: ") + std::to_string(years) + "\n";
        
        auto init_start = std::chrono::high_resolution_clock::now();
        
        // Use the helper function to create initializer based on config
        InitResult result = createInitializerFromConfig(cfg);

        auto init_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> init_elapsed = init_end - init_start;
        std::cout << "Initialization time: " << init_elapsed.count() << " seconds\n";

        // The metadata string is now available as result.metadata
        std::string metadata = timeMetadata + result.metadata;

        // Add a block to convert the integration method enum to a string
        std::string methodName;
        switch (integrationMethod) {
            case IntegrationMethod::EULER:
                methodName = "Euler";
                break;
            case IntegrationMethod::VELOCITY_VERLET:
                methodName = "Velocity Verlet"; 
                break;
        }
        long int stepFreq = std::max(1L, n / desiredPrintCount);

        metadata += "# automatic steps: " + std::to_string(n) + "\n";
        metadata += "# print frequency (steps): " + std::to_string(stepFreq) + "\n";
        // Append integration method info to the metadata
        metadata += "# integration method: " + methodName + "\n";

        // Add force method metadata
        std::string forceMethodName;
        switch (forceMethod) {
            case ForceMethod::PAIRWISE:
                forceMethodName = "Pairwise";
                break;
            case ForceMethod::PAIRWISE_AVX2_FP32:
                forceMethodName = "Pairwise AVX2/FP32";
                break;
            case ForceMethod::ADAPTIVE_MUTUAL:
                forceMethodName = "Adaptive Mutual";
                break;
            case ForceMethod::BARNES_HUT:
                forceMethodName = "Barnes-Hut";
                break;
        }
        metadata += "# force method: " + forceMethodName + "\n";

        // Add execution mode metadata
        std::string executionModeName;
        switch (executionMode) {
            case ExecutionMode::CPU:
                executionModeName = "CPU";
                break;
            case ExecutionMode::GPU:
                executionModeName = "GPU";
                break;
        }
        metadata += "# execution mode: " + executionModeName + "\n";

        // Add output mode metadata
        std::string outputModeName;
        switch (outputMode) {
            case OutputMode::BENCHMARK:
                outputModeName = "None (Benchmark)";
                break;
            case OutputMode::FILE_CSV:
                outputModeName = "CSV File";
                break;
            case OutputMode::VISUALIZATION:
                outputModeName = "Real-time Visualization";
                break;
        }
        metadata += "# output mode: " + outputModeName + "\n";

        std::string unifiedOutputFile = outputDir + outputFile;
        
        // Initialize system with execution mode and output mode
        System sys(std::move(result.particles), integrationMethod, forceMethod, executionMode, outputMode);
        
        // Compute total simulation time in years:
        long double totalYears = years;

        metadata += "# time basis: years\n";
        metadata += "# dt (years): " + std::to_string(dtYears) + "\n";
        metadata += "# total years: " + std::to_string(totalYears) + "\n";

            // Output some relevant information
            std::cout << "Initialization mode: " << cfg.initSelected << "\n";
            std::cout << "Force method: " << forceMethodName << "\n";
            std::cout << "Execution mode: " << executionModeName << "\n";
            std::cout << "Output mode: " << outputModeName << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        try {
                sys.runSimulation(dtYears, n, stepFreq, unifiedOutputFile, metadata);
        }
        catch (const std::exception& ex) {
            // If we're using GPU and got an error, try falling back to CPU
            if (executionMode == ExecutionMode::GPU) {
                std::cerr << "GPU execution failed: " << ex.what() << "\n";
                std::cerr << "Falling back to CPU execution...\n";
                
                // Switch to CPU mode
                sys.setExecutionMode(ExecutionMode::CPU);
                metadata += "# NOTE: Fell back to CPU execution due to GPU error\n";
                sys.runSimulation(dtYears, n, stepFreq, unifiedOutputFile, metadata);
                gpuFailed = true;
            } else {
                // If already using CPU or some other issue, rethrow
                throw;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\nSimulation time: " << elapsed.count() << " seconds";
        
        if (gpuFailed) {
            std::cout << " (using CPU fallback)";
        } else {
            std::cout << " (using " << (executionMode == ExecutionMode::GPU ? "GPU" : "CPU") << ")";
        }
        
        std::cout << std::endl;
        
        return 0;
    }
    catch(const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

