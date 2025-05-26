// main.js - Global JavaScript for the DeepAudio application

document.addEventListener('DOMContentLoaded', function() {
    console.log("main.js: Global scripts loaded and DOM content parsed.");

    // --- Global UI/UX elements and Utilities ---

    // Example: Smooth scrolling for anchor links (Applicable across multiple pages)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId && targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // --- Universal Loading Overlay Management ---
    // The loading overlay is defined in base.html with id="customLoadingOverlay".
    const customLoadingOverlay = document.getElementById('customLoadingOverlay');

    // Expose global functions to show/hide loading overlay
    window.showLoadingOverlay = function(message = 'Carregando...', subtitle = 'Por favor, aguarde.') {
        if (customLoadingOverlay) {
            const loadingText = customLoadingOverlay.querySelector('.loading-text');
            const loadingSubtitle = customLoadingOverlay.querySelector('.loading-subtitle');
            if (loadingText) loadingText.textContent = message;
            if (loadingSubtitle) loadingSubtitle.textContent = subtitle;
            customLoadingOverlay.style.display = 'flex';
            customLoadingOverlay.setAttribute('aria-hidden', 'false');
        }
    };

    window.hideLoadingOverlay = function() {
        if (customLoadingOverlay) {
            customLoadingOverlay.style.display = 'none';
            customLoadingOverlay.setAttribute('aria-hidden', 'true');
        }
    };

    // Hide loading overlay on window load if it's acting as a preloader
    window.addEventListener('load', () => {
        hideLoadingOverlay();
    });

    // --- Conditional Execution for Page-Specific Logic within main.js ---
    // For simpler pages like index.html and about.html, their specific JS can live here
    // rather than in separate files, conditionally executed.
    const currentPath = window.location.pathname;

    // Logic for Index Page (index.html) - assuming index page is at '/' or '/index.html'
    if (currentPath === '/' || currentPath === '/index.html') {
        console.log("main.js: Executing scripts specific to index.html.");
        // Add specific JavaScript for index.html here, e.g., carousel initializations
        const testimonialCarousel = document.getElementById('testimonialCarousel'); // Assuming an ID for a carousel
        if (testimonialCarousel) {
            // new bootstrap.Carousel(testimonialCarousel, { interval: 5000 });
            console.log("Testimonial carousel (if any) initialized for index.html.");
        }
    }

    // Logic for About Page (about.html) - assuming about page is at '/about' or '/about.html'
    if (currentPath === '/about' || currentPath === '/about.html') {
        console.log("main.js: Executing scripts specific to about.html.");
        // Example: Animated statistics counters on scroll
        const statisticNumbers = document.querySelectorAll('.statistic-number');
        if (statisticNumbers.length > 0) {
            const observerOptions = {
                root: null, // relative to the viewport
                rootMargin: '0px',
                threshold: 0.5 // Trigger when 50% of the element is visible
            };

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const target = entry.target;
                        const finalValue = parseInt(target.dataset.target);
                        let currentValue = 0;
                        const increment = Math.ceil(finalValue / 100);
                        const duration = 2000; // milliseconds
                        const stepTime = Math.abs(Math.floor(duration / (finalValue / increment)));

                        const timer = setInterval(() => {
                            currentValue += increment;
                            if (currentValue >= finalValue) {
                                currentValue = finalValue;
                                clearInterval(timer);
                            }
                            target.textContent = currentValue.toLocaleString('pt-BR');
                            if (target.dataset.target.includes('%')) {
                                target.textContent += '%';
                            }
                        }, stepTime);
                        observer.unobserve(target); // Stop observing once animated
                    }
                });
            }, observerOptions);

            statisticNumbers.forEach(number => {
                observer.observe(number);
            });
            console.log("Statistic counters initialized for about.html.");
        }
    }
});