// train.js - JavaScript specific to the Model Training page

document.addEventListener('DOMContentLoaded', function() {
    console.log("train.js: Model training scripts loaded.");

    // Seletores ajustados para o train.html
    const trainForm = document.getElementById('trainForm'); // O formulário no train.html
    const trainingSpinner = document.getElementById('training-spinner'); // O spinner inicial do HTML
    const customLoadingOverlay = document.getElementById('customLoadingOverlay'); // O overlay global
    const loadingText = customLoadingOverlay ? customLoadingOverlay.querySelector('.loading-text') : null;
    const loadingSubtitle = customLoadingOverlay ? customLoadingOverlay.querySelector('.loading-subtitle') : null;

    // Novos elementos para feedback de progresso no HTML
    const trainingFeedbackArea = document.getElementById('trainingFeedbackArea');
    const trainingProgressBar = document.getElementById('trainingProgressBar');
    const trainingStatusDisplay = document.getElementById('trainingStatus');

    // --- Training form submission logic ---
    if (trainForm) {
        trainForm.addEventListener('submit', function(e) {
            e.preventDefault(); // Previne o envio padrão do formulário

            const submitButton = this.querySelector('button[type="submit"]');

            // 1. Esconder o botão de submissão e mostrar o spinner inicial
            submitButton.classList.add('d-none');
            trainingSpinner.classList.remove('d-none');

            // 2. Mostrar o overlay personalizado com mensagem inicial
            if (customLoadingOverlay) {
                customLoadingOverlay.style.display = 'flex';
                if (loadingText) loadingText.textContent = 'Iniciando Treinamento do Modelo...';
                if (loadingSubtitle) loadingSubtitle.textContent = 'Processando dados, otimizando parâmetros e preparando o ambiente. Por favor, aguarde.';
            }

            // Ação do formulário para o endpoint de treinamento
            const trainUrl = this.action;

            fetch(trainUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Assumindo que o backend espera JSON
                    'Accept': 'application/json'
                },
                body: JSON.stringify({}) // Envia um corpo vazio ou dados de configuração, se houver
            })
            .then(response => {
                const clonedResponse = response.clone();
                if (!response.ok) {
                    return clonedResponse.json().then(errorData => {
                        throw new Error(errorData.message || 'Ocorreu um erro no servidor.');
                    }).catch(() => {
                        return clonedResponse.text().then(text => {
                            throw new Error(text || 'Erro de rede ou resposta inesperada do servidor.');
                        });
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Training request sent:', data);
                if (data.status === 'success') {
                    // O treinamento foi iniciado. Agora vamos mostrar a barra de progresso.
                    // E iniciar o polling para atualizações de status.

                    // Esconde o overlay inicial após o sucesso da requisição de início
                    if (customLoadingOverlay) customLoadingOverlay.style.display = 'none';

                    // Mostra a área de feedback de progresso
                    if (trainingFeedbackArea) trainingFeedbackArea.classList.remove('d-none');

                    // Mensagem inicial de status
                    if (trainingStatusDisplay) trainingStatusDisplay.textContent = data.message || 'Treinamento iniciado com sucesso!';

                    // Inicia o polling de status (se houver um training_id)
                    if (data.training_id) {
                        pollTrainingStatus(data.training_id);
                    } else {
                        // Se não houver ID de treinamento para polling, apenas informa sucesso
                        window.showToast(data.message || 'Treinamento iniciado com sucesso!', 'success');
                        // Garante que os elementos de carregamento sejam resetados
                        submitButton.classList.remove('d-none');
                        trainingSpinner.classList.add('d-none');
                    }

                } else {
                    // Se o status não for sucesso, mostra um erro
                    window.showToast(data.message || 'Falha ao iniciar o treinamento. Tente novamente mais tarde.', 'error');
                    // Garante que os elementos de carregamento sejam resetados
                    submitButton.classList.remove('d-none');
                    trainingSpinner.classList.add('d-none');
                    if (customLoadingOverlay) customLoadingOverlay.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Erro ao enviar formulário de treinamento:', error);
                window.showToast(`Erro: ${error.message}` || 'Erro de conexão ao tentar iniciar o treinamento. Verifique sua rede.', 'error');
                // Em caso de erro na requisição inicial, reseta os elementos visuais
                submitButton.classList.remove('d-none');
                trainingSpinner.classList.add('d-none');
                if (customLoadingOverlay) customLoadingOverlay.style.display = 'none';
            });
        });
    }

    // --- Real-time training status polling ---
    function pollTrainingStatus(trainingId) {
        let pollInterval = setInterval(() => {
            fetch(`/api/training_status/${trainingId}`) // Endpoint real do seu backend para status
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Atualiza a barra de progresso
                    const progress = data.progress || 0;
                    if (trainingProgressBar) {
                        trainingProgressBar.style.width = `${progress}%`;
                        trainingProgressBar.setAttribute('aria-valuenow', progress);
                        trainingProgressBar.textContent = `${progress}%`;
                    }
                    // Atualiza o texto de status
                    const statusMessage = data.status_message || `Progresso: ${progress}% - Status: ${data.status || 'Processando...'}`;
                    if (trainingStatusDisplay) trainingStatusDisplay.textContent = statusMessage;

                    // Verifica se o treinamento foi concluído ou falhou
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollInterval); // Para o polling
                        console.log('Training finished:', data.status);
                        window.showToast(`Treinamento ${data.status === 'completed' ? 'concluído' : 'falhou'}!`, data.status === 'completed' ? 'success' : 'error');

                        // Reseta os elementos visuais para o estado original
                        const submitButton = trainForm.querySelector('button[type="submit"]');
                        if (submitButton) {
                            submitButton.classList.remove('d-none');
                        }
                        if (trainingSpinner) {
                            trainingSpinner.classList.add('d-none');
                        }
                        if (trainingFeedbackArea) {
                            trainingFeedbackArea.classList.add('d-none'); // Oculta a área de progresso
                        }
                        if (trainingProgressBar) {
                            trainingProgressBar.style.width = '0%'; // Reseta a barra
                            trainingProgressBar.setAttribute('aria-valuenow', 0);
                            trainingProgressBar.textContent = '0%';
                        }
                        if (trainingStatusDisplay) {
                            trainingStatusDisplay.textContent = ''; // Limpa o status
                        }
                    }
                })
                .catch(error => {
                    console.error('Error polling status:', error);
                    clearInterval(pollInterval); // Para o polling em caso de erro
                    window.showToast(`Erro ao verificar status: ${error.message}`, 'error');

                    // Em caso de erro no polling, reseta os elementos visuais
                    const submitButton = trainForm.querySelector('button[type="submit"]');
                    if (submitButton) {
                        submitButton.classList.remove('d-none');
                    }
                    if (trainingSpinner) {
                        trainingSpinner.classList.add('d-none');
                    }
                    if (trainingFeedbackArea) {
                        trainingFeedbackArea.classList.add('d-none'); // Oculta a área de progresso
                    }
                    if (trainingProgressBar) {
                        trainingProgressBar.style.width = '0%';
                        trainingProgressBar.setAttribute('aria-valuenow', 0);
                        trainingProgressBar.textContent = '0%';
                    }
                    if (trainingStatusDisplay) {
                        trainingStatusDisplay.textContent = '';
                    }
                });
        }, 3000); // Poll a cada 3 segundos (ajuste conforme a necessidade do seu backend)
    }
});