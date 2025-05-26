// fake_detector.js - CORRIGIDO - JavaScript específico para a página de Detecção de Deepfake

document.addEventListener('DOMContentLoaded', function() {
    console.log("fake_detector.js: Scripts específicos da página carregados.");

    // Elementos DOM
    const audioFileInput = document.getElementById('audio_file');
    const selectedFileNameDisplay = document.getElementById('selectedFileName');
    const audioDropZone = document.getElementById('audioDropZone');
    const predictionForm = document.getElementById('predictionForm');
    const submitPredictionBtn = document.getElementById('submitPredictionBtn');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const progressBarInner = uploadProgressBar ? uploadProgressBar.querySelector('.progress-bar') : null;
    const fileErrorMessages = document.getElementById('fileErrorMessages');
    const errorMessageText = document.getElementById('errorMessageText');
    const analysisResultSection = document.getElementById('analysis-result-section');

    // Constantes para validação
    const MAX_FILE_SIZE_MB = 50;
    const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
    const ALLOWED_MIME_TYPES = [
        'audio/mpeg', // MP3
        'audio/wav',  // WAV
        'audio/flac', // FLAC
        'audio/ogg'   // OGG
    ];

    let fileIsValid = false;
    let isSubmitting = false; // Flag para evitar múltiplas submissões

    // === FUNÇÕES AUXILIARES ===

    function showFileError(message) {
        if (errorMessageText && fileErrorMessages) {
            errorMessageText.textContent = message;
            fileErrorMessages.style.display = 'block';
            fileErrorMessages.classList.add('animate__fadeIn');
            fileIsValid = false;
            updateSubmitButtonState();
        }
    }

    function hideFileError() {
        if (fileErrorMessages) {
            fileErrorMessages.style.display = 'none';
            fileErrorMessages.classList.remove('animate__fadeIn');
        }
    }

    function updateSubmitButtonState() {
        if (submitPredictionBtn) {
            const isModelAvailable = submitPredictionBtn.dataset.modelExists === 'true';
            submitPredictionBtn.disabled = !(fileIsValid && isModelAvailable && !isSubmitting);
        }
    }

    function resetSubmitButton() {
        if (submitPredictionBtn) {
            isSubmitting = false;
            submitPredictionBtn.disabled = false;
            const btnContent = submitPredictionBtn.querySelector('.btn-content');
            if (btnContent) {
                const isModelAvailable = submitPredictionBtn.dataset.modelExists === 'true';
                btnContent.innerHTML = isModelAvailable
                    ? '<i class="bi bi-journal-check me-2"></i>Iniciar Análise de Áudio'
                    : '<i class="bi bi-lock-fill me-2"></i>Modelo de Detecção Indisponível';
            }
            updateSubmitButtonState();
        }
    }

    function showProgressBar() {
        if (uploadProgressBar && progressBarInner) {
            uploadProgressBar.style.display = 'block';
            progressBarInner.style.width = '0%';
            progressBarInner.textContent = '0%';
        }
    }

    function hideProgressBar() {
        if (uploadProgressBar) {
            uploadProgressBar.style.display = 'none';
        }
    }

    function updateProgress(percent) {
        if (progressBarInner) {
            const roundedPercent = Math.round(percent);
            progressBarInner.style.width = roundedPercent + '%';
            progressBarInner.textContent = roundedPercent + '%';
        }
    }

    // === VALIDAÇÃO DE ARQUIVO ===

    function validateFile(file) {
        hideFileError();

        if (!file) {
            showFileError('Nenhum arquivo selecionado.');
            return false;
        }

        // Validação de tipo MIME
        if (!ALLOWED_MIME_TYPES.includes(file.type)) {
            const supportedFormats = ALLOWED_MIME_TYPES
                .map(t => t.split('/')[1].toUpperCase())
                .join(', ');
            showFileError(`Formato inválido. Suportados: ${supportedFormats}.`);
            return false;
        }

        // Validação de tamanho
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showFileError(`Arquivo muito grande. Máximo: ${MAX_FILE_SIZE_MB}MB.`);
            return false;
        }

        // Validação adicional de nome de arquivo
        if (file.name.length > 255) {
            showFileError('Nome do arquivo muito longo.');
            return false;
        }

        return true;
    }

    // === MANIPULAÇÃO DE ARQUIVOS ===

    if (audioFileInput && selectedFileNameDisplay && audioDropZone) {
        audioFileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                const file = this.files[0];
                fileIsValid = validateFile(file);

                if (fileIsValid) {
                    selectedFileNameDisplay.textContent = `Arquivo selecionado: ${file.name}`;
                    selectedFileNameDisplay.style.display = 'block';
                    audioDropZone.setAttribute('aria-label', `Arquivo selecionado: ${file.name}. Clique para alterar.`);
                } else {
                    selectedFileNameDisplay.textContent = '';
                    selectedFileNameDisplay.style.display = 'none';
                    audioDropZone.setAttribute('aria-label', 'Selecione ou arraste o arquivo de áudio');
                    this.value = ''; // Limpar input se arquivo inválido
                }
            } else {
                selectedFileNameDisplay.textContent = '';
                selectedFileNameDisplay.style.display = 'none';
                audioDropZone.setAttribute('aria-label', 'Selecione ou arraste o arquivo de áudio');
                hideFileError();
                fileIsValid = false;
            }
            updateSubmitButtonState();
        });

        // === DRAG AND DROP ===

        audioDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            audioDropZone.classList.add('drag-over');
            e.dataTransfer.dropEffect = 'copy';
        });

        audioDropZone.addEventListener('dragleave', (e) => {
            // Só remove a classe se realmente saiu da zona de drop
            if (!audioDropZone.contains(e.relatedTarget)) {
                audioDropZone.classList.remove('drag-over');
            }
        });

        audioDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            audioDropZone.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                // Pegar apenas o primeiro arquivo
                const dt = new DataTransfer();
                dt.items.add(files[0]);
                audioFileInput.files = dt.files;
                audioFileInput.dispatchEvent(new Event('change'));
            }
        });

        // Clique na zona de drop
        audioDropZone.addEventListener('click', (e) => {
            if (e.target === audioDropZone || audioDropZone.contains(e.target)) {
                audioFileInput.click();
            }
        });

        // Acessibilidade - teclado
        audioDropZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                audioFileInput.click();
            }
        });
    }

    // === SUBMISSÃO DO FORMULÁRIO ===

    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            // Validações antes da submissão
            if (isSubmitting) {
                e.preventDefault();
                console.log('Submissão já em andamento, ignorando...');
                return;
            }

            if (!fileIsValid) {
                e.preventDefault();
                showFileError('Por favor, selecione um arquivo de áudio válido.');
                return;
            }

            if (submitPredictionBtn && submitPredictionBtn.dataset.modelExists !== 'true') {
                e.preventDefault();
                showFileError('Modelo de detecção não disponível. Treine um modelo primeiro.');
                return;
            }

            // Marcar como enviando
            isSubmitting = true;

            // Atualizar UI
            if (submitPredictionBtn) {
                submitPredictionBtn.disabled = true;
                const btnContent = submitPredictionBtn.querySelector('.btn-content');
                if (btnContent) {
                    btnContent.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analisando...';
                }
            }

            // Mostrar barra de progresso
            showProgressBar();

            // Mostrar overlay de loading se disponível
            if (window.showLoadingOverlay) {
                window.showLoadingOverlay('Analisando Áudio...', 'Este processo pode levar alguns instantes.');
            }

            // Simular progresso durante o upload
            simulateUploadProgress();

            // Permitir submissão normal do formulário
            // A página será recarregada com o resultado
        });
    }

    // === SIMULAÇÃO DE PROGRESSO ===

    function simulateUploadProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 95) {
                progress = 95; // Parar em 95% para aguardar o processamento real
                clearInterval(interval);
            }
            updateProgress(progress);
        }, 200);
    }

    // === ROLAGEM PARA RESULTADO ===

    if (analysisResultSection) {
        // Se existe seção de resultado, rolar para ela após um pequeno delay
        setTimeout(() => {
            analysisResultSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

            // Ocultar overlay e resetar estado
            if (window.hideLoadingOverlay) {
                window.hideLoadingOverlay();
            }
            hideProgressBar();
            resetSubmitButton();
        }, 800);
    } else {
        // Se não há resultado, garantir que a UI seja resetada
        if (window.hideLoadingOverlay) {
            window.hideLoadingOverlay();
        }
        hideProgressBar();
        resetSubmitButton();
    }

    // === FORMULÁRIOS AVANÇADOS (PLACEHOLDER) ===

    const advancedResearchForm = document.querySelector('.advanced-research-form');
    if (advancedResearchForm) {
        advancedResearchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log("Formulário de pesquisa avançada submetido.");

            // Coletar dados do formulário
            const personalityName = document.getElementById('personalityName')?.value;
            const analyzeMetadata = document.getElementById('analyzeMetadata')?.checked;
            const analyzeContext = document.getElementById('analyzeContext')?.checked;
            const advancedResearch = document.getElementById('advancedResearch')?.checked;

            console.log('Dados coletados:', {
                personalityName,
                analyzeMetadata,
                analyzeContext,
                advancedResearch
            });

            // Placeholder para implementação futura
            alert('Funcionalidade de pesquisa avançada será implementada em breve!');
        });
    }

    // === DRAG ZONES ADICIONAIS ===

    const additionalDropZones = [
        document.getElementById('advancedAnalysisDropZone'),
        document.getElementById('audioQualityDropZone')
    ];

    additionalDropZones.forEach(dropZone => {
        if (dropZone) {
            dropZone.addEventListener('click', () => {
                if (audioFileInput) {
                    audioFileInput.click();
                }
            });

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });

            dropZone.addEventListener('dragleave', (e) => {
                if (!dropZone.contains(e.relatedTarget)) {
                    dropZone.classList.remove('drag-over');
                }
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');

                const files = e.dataTransfer.files;
                if (files.length > 0 && audioFileInput) {
                    const dt = new DataTransfer();
                    dt.items.add(files[0]);
                    audioFileInput.files = dt.files;
                    audioFileInput.dispatchEvent(new Event('change'));

                    // Rolar para o formulário principal
                    document.getElementById('analysis-section')?.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            });
        }
    });

    // === TRATAMENTO DE ERROS GLOBAIS ===

    window.addEventListener('error', function(e) {
        console.error('Erro JavaScript:', e.error);
        if (window.hideLoadingOverlay) {
            window.hideLoadingOverlay();
        }
        hideProgressBar();
        resetSubmitButton();
    });

    // === INICIALIZAÇÃO ===

    // Estado inicial do botão
    updateSubmitButtonState();

    console.log("fake_detector.js: Inicialização completa.");
});

// === FUNÇÕES GLOBAIS (se necessário) ===

// Função para resetar o formulário externamente
window.resetDetectorForm = function() {
    const audioFileInput = document.getElementById('audio_file');
    const selectedFileNameDisplay = document.getElementById('selectedFileName');
    const fileErrorMessages = document.getElementById('fileErrorMessages');

    if (audioFileInput) audioFileInput.value = '';
    if (selectedFileNameDisplay) {
        selectedFileNameDisplay.textContent = '';
        selectedFileNameDisplay.style.display = 'none';
    }
    if (fileErrorMessages) {
        fileErrorMessages.style.display = 'none';
    }

    fileIsValid = false;
    isSubmitting = false;
    updateSubmitButtonState();
};