// train.js - Enhanced JavaScript compatible with existing structure

document.addEventListener('DOMContentLoaded', function() {
    console.log("Enhanced train.js: Model training scripts loaded with advanced features.");

    // MANTÉM: Seletores originais + novos elementos avançados
    const trainForm = document.getElementById('trainForm');
    const trainingSpinner = document.getElementById('training-spinner');
    const customLoadingOverlay = document.getElementById('customLoadingOverlay');
    const loadingText = customLoadingOverlay ? customLoadingOverlay.querySelector('.loading-text') : null;
    const loadingSubtitle = customLoadingOverlay ? customLoadingOverlay.querySelector('.loading-subtitle') : null;

    // ORIGINAL: Elementos básicos de feedback
    const trainingFeedbackArea = document.getElementById('trainingFeedbackArea');
    const trainingProgressBar = document.getElementById('trainingProgressBar');
    const trainingStatusDisplay = document.getElementById('trainingStatus');

    // ENHANCED: Novos elementos avançados (graceful degradation se não existirem)
    const phaseTitle = document.getElementById('phaseTitle');
    const phaseDescription = document.getElementById('phaseDescription');
    const epochCounter = document.getElementById('epochCounter');
    const lossDisplay = document.getElementById('lossDisplay');
    const accuracyDisplay = document.getElementById('accuracyDisplay');
    const learningRateDisplay = document.getElementById('learningRateDisplay');
    const timeElapsedDisplay = document.getElementById('timeElapsedDisplay');
    const etaDisplay = document.getElementById('etaDisplay');
    const metricsChart = document.getElementById('metricsChart');
    const validationMetrics = document.getElementById('validationMetrics');
    const crossValidationResults = document.getElementById('crossValidationResults');
    const checkpointIndicator = document.getElementById('checkpointIndicator');
    const downloadContainer = document.getElementById('downloadContainer');

    // Training state management
    let trainingState = {
        isTraining: false,
        startTime: null,
        currentEpoch: 0,
        totalEpochs: 0,
        trainingId: null,
        pollInterval: null,
        timeUpdateInterval: null,
        config: {},
        metrics: {
            loss: [],
            accuracy: [],
            val_loss: [],
            val_accuracy: [],
            learning_rate: []
        }
    };

    // Training phases with enhanced feedback
    const TRAINING_PHASES = {
        INITIALIZING: {
            message: 'Inicializando Treinamento...',
            subtitle: 'Preparando ambiente, carregando dados e configurando modelo.',
            progress: 5
        },
        DATA_PREPARATION: {
            message: 'Preparando Dados...',
            subtitle: 'Aplicando normalização, augmentação e divisão dos dados.',
            progress: 15
        },
        MODEL_CREATION: {
            message: 'Criando Modelo...',
            subtitle: 'Construindo arquitetura neural e compilando modelo.',
            progress: 25
        },
        CROSS_VALIDATION: {
            message: 'Executando Validação Cruzada...',
            subtitle: 'Avaliando performance com múltiplos folds.',
            progress: 35
        },
        TRAINING: {
            message: 'Treinando Modelo...',
            subtitle: 'Otimizando parâmetros através das épocas.',
            progress: 50
        },
        VALIDATION: {
            message: 'Validando Modelo...',
            subtitle: 'Avaliando performance em dados de validação.',
            progress: 85
        },
        SAVING: {
            message: 'Salvando Modelo...',
            subtitle: 'Persistindo modelo treinado e metadados.',
            progress: 95
        },
        COMPLETED: {
            message: 'Treinamento Concluído!',
            subtitle: 'Modelo salvo com sucesso e pronto para uso.',
            progress: 100
        }
    };

    // ENHANCED: Form submission logic (mantém compatibilidade)
    if (trainForm) {
        trainForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const submitButton = this.querySelector('button[type="submit"]');

            // Initialize enhanced training state
            initializeTrainingState();

            // ENHANCED: Extract configuration from form
            const trainConfig = extractTrainingConfiguration();
            trainingState.config = trainConfig;
            trainingState.totalEpochs = trainConfig.epochs;

            // MANTÉM: UI state changes originais
            if (submitButton) submitButton.classList.add('d-none');
            if (trainingSpinner) trainingSpinner.classList.remove('d-none');

            // ENHANCED: Show loading overlay with phase
            showTrainingPhase('INITIALIZING');

            // MANTÉM: URL original do formulário
            const trainUrl = this.action;

            // ENHANCED: Send configuration data
            fetch(trainUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(trainConfig) // Agora envia configuração real
            })
            .then(response => handleTrainingResponse(response))
            .then(data => handleTrainingSuccess(data))
            .catch(error => handleTrainingError(error));
        });
    }

    // ENHANCED: Initialize training state
    function initializeTrainingState() {
        trainingState.isTraining = true;
        trainingState.startTime = new Date();
        trainingState.currentEpoch = 0;
        trainingState.trainingId = null;

        // Reset metrics
        trainingState.metrics = {
            loss: [],
            accuracy: [],
            val_loss: [],
            val_accuracy: [],
            learning_rate: []
        };

        // Clear all displays
        clearAllDisplays();

        // Start time update interval if element exists
        if (timeElapsedDisplay) {
            if (trainingState.timeUpdateInterval) {
                clearInterval(trainingState.timeUpdateInterval);
            }
            trainingState.timeUpdateInterval = setInterval(updateTimeElapsed, 1000);
        }
    }

    // ENHANCED: Extract training configuration from form
    function extractTrainingConfiguration() {
        const formData = new FormData(trainForm);

        const config = {
            // Basic configuration
            architecture: formData.get('architecture') || 'advanced_cnn_lstm',
            epochs: parseInt(formData.get('epochs')) || 100,
            batch_size: parseInt(formData.get('batch_size')) || 32,
            learning_rate: parseFloat(formData.get('learning_rate')) || 0.001,
            dropout_rate: parseFloat(formData.get('dropout_rate')) || 0.3,
            cv_folds: parseInt(formData.get('cv_folds')) || 5,

            // Advanced options (checkboxes)
            use_cross_validation: formData.get('use_cross_validation') === 'on',
            use_mixed_precision: formData.get('use_mixed_precision') === 'on',
            enable_augmentation: formData.get('enable_augmentation') === 'on',
            use_cosine_annealing: formData.get('use_cosine_annealing') === 'on'
        };

        console.log('Configuração extraída do formulário:', config);
        return config;
    }

    // MANTÉM: Response handling original com melhorias
    function handleTrainingResponse(response) {
        const clonedResponse = response.clone();
        if (!response.ok) {
            return clonedResponse.json().then(errorData => {
                throw new Error(errorData.message || `Erro do servidor: ${response.status}`);
            }).catch(() => {
                return clonedResponse.text().then(text => {
                    throw new Error(text || 'Erro de rede ou resposta inesperada do servidor.');
                });
            });
        }
        return response.json();
    }

    // ENHANCED: Success handling
    function handleTrainingSuccess(data) {
        console.log('Training request sent:', data);

        if (data.status === 'success') {
            trainingState.trainingId = data.training_id;

            // MANTÉM: Hide overlay original
            if (customLoadingOverlay) customLoadingOverlay.style.display = 'none';

            // ENHANCED: Show feedback area with smooth scroll
            if (trainingFeedbackArea) {
                trainingFeedbackArea.classList.remove('d-none');
                trainingFeedbackArea.scrollIntoView({ behavior: 'smooth' });
            }

            // ENHANCED: Update status with message
            updateTrainingStatus(data.message || 'Treinamento iniciado com sucesso!');

            // ENHANCED: Handle cross-validation
            if (data.cross_validation_started || trainingState.config.use_cross_validation) {
                showTrainingPhase('CROSS_VALIDATION');
                if (data.cv_info) {
                    updateCrossValidationStatus(data.cv_info);
                }
            } else {
                showTrainingPhase('TRAINING');
            }

            // MANTÉM/ENHANCED: Polling logic
            if (trainingState.trainingId) {
                startEnhancedPolling();
            } else {
                // Fallback sem training_id
                if (window.showToast) {
                    window.showToast(data.message || 'Treinamento iniciado com sucesso!', 'success');
                }
                resetTrainingUI();
            }

        } else {
            // MANTÉM: Error handling original
            if (window.showToast) {
                window.showToast(data.message || 'Falha ao iniciar o treinamento. Tente novamente mais tarde.', 'error');
            }
            resetTrainingUI();
        }
    }

    // MANTÉM: Error handling original
    function handleTrainingError(error) {
        console.error('Erro ao enviar formulário de treinamento:', error);
        if (window.showToast) {
            window.showToast(`Erro: ${error.message}`, 'error');
        }
        resetTrainingUI();
    }

    // ENHANCED: Show training phase
    function showTrainingPhase(phaseName) {
        const phase = TRAINING_PHASES[phaseName];
        if (!phase) return;

        // Update phase displays if they exist
        if (phaseTitle) phaseTitle.textContent = phase.message;
        if (phaseDescription) phaseDescription.textContent = phase.subtitle;

        // Update loading overlay if visible
        if (customLoadingOverlay && customLoadingOverlay.style.display === 'flex') {
            if (loadingText) loadingText.textContent = phase.message;
            if (loadingSubtitle) loadingSubtitle.textContent = phase.subtitle;
        }

        updateProgressBar(phase.progress);
    }

    // ENHANCED: Polling with comprehensive status tracking
    function startEnhancedPolling() {
        if (trainingState.pollInterval) {
            clearInterval(trainingState.pollInterval);
        }

        trainingState.pollInterval = setInterval(() => {
            if (!trainingState.isTraining) {
                clearInterval(trainingState.pollInterval);
                return;
            }

            // MANTÉM: Endpoint original
            fetch(`/api/training_status/${trainingState.trainingId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => processEnhancedStatus(data))
                .catch(error => handlePollingError(error));
        }, 2000); // Mais frequente que o original (3s -> 2s)
    }

    // ENHANCED: Process comprehensive status updates
    function processEnhancedStatus(data) {
        // MANTÉM: Basic progress update
        const progress = data.progress || 0;
        updateProgressBar(progress);

        // ENHANCED: Epoch information
        if (data.current_epoch !== undefined) {
            trainingState.currentEpoch = data.current_epoch;
            updateEpochCounter(data.current_epoch, trainingState.totalEpochs || data.total_epochs);
        }

        // ENHANCED: Training phase
        if (data.phase && TRAINING_PHASES[data.phase]) {
            showTrainingPhase(data.phase);
        }

        // ENHANCED: Metrics update
        if (data.metrics) {
            updateTrainingMetrics(data.metrics);
        }

        // ENHANCED: Learning rate
        if (data.learning_rate !== undefined) {
            updateLearningRate(data.learning_rate);
        }

        // ENHANCED: ETA
        if (data.eta_seconds) {
            updateETA(data.eta_seconds);
        }

        // ENHANCED: Cross-validation
        if (data.cv_info) {
            updateCrossValidationStatus(data.cv_info);
        }

        if (data.cv_results) {
            updateCrossValidationResults(data.cv_results);
        }

        // ENHANCED: Model checkpoints
        if (data.is_best_epoch) {
            showCheckpointSaved(data.current_epoch);
        }

        // MANTÉM/ENHANCED: Status message
        const statusMessage = data.status_message ||
            `Época ${trainingState.currentEpoch}/${trainingState.totalEpochs} - ${data.status || 'Processando...'}`;
        updateTrainingStatus(statusMessage);

        // MANTÉM: Completion check
        if (data.status === 'completed' || data.status === 'failed') {
            handleTrainingCompletion(data);
        }
    }

    // ENHANCED: Training completion handling
    function handleTrainingCompletion(data) {
        // Stop intervals
        clearInterval(trainingState.pollInterval);
        if (trainingState.timeUpdateInterval) {
            clearInterval(trainingState.timeUpdateInterval);
        }
        trainingState.isTraining = false;

        console.log('Training finished:', data.status);

        const isSuccess = data.status === 'completed';
        const message = isSuccess ? 'Treinamento concluído com sucesso!' : 'Treinamento falhou!';
        const toastType = isSuccess ? 'success' : 'error';

        if (isSuccess) {
            showTrainingPhase('COMPLETED');

            if (data.final_metrics) {
                updateFinalResults(data.final_metrics);
            }

            showDownloadOptions(trainingState.trainingId);

            // Hide loading overlay after completion phase
            setTimeout(() => {
                if (customLoadingOverlay) customLoadingOverlay.style.display = 'none';
            }, 2000);
        }

        if (window.showToast) {
            window.showToast(message, toastType);
        }

        // Reset UI for failures only
        if (!isSuccess) {
            setTimeout(() => resetTrainingUI(), 1000);
        }
    }

    // MANTÉM: Polling error handling
    function handlePollingError(error) {
        console.error('Error polling status:', error);
        clearInterval(trainingState.pollInterval);
        trainingState.isTraining = false;

        if (window.showToast) {
            window.showToast(`Erro ao verificar status: ${error.message}`, 'error');
        }
        resetTrainingUI();
    }

    // ENHANCED: Update functions
    function updateProgressBar(progress) {
        if (trainingProgressBar) {
            const clampedProgress = Math.max(0, Math.min(100, progress));
            trainingProgressBar.style.width = `${clampedProgress}%`;
            trainingProgressBar.setAttribute('aria-valuenow', clampedProgress);
            trainingProgressBar.textContent = `${clampedProgress.toFixed(1)}%`;

            //