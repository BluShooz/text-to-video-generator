/**
 * Text-to-Video Generator Frontend
 * Handles form submission, progress tracking, and result display.
 */

class VideoGenerator {
    constructor() {
        this.form = document.getElementById('generateForm');
        this.promptInput = document.getElementById('prompt');
        this.charCount = document.getElementById('charCount');
        this.generateBtn = document.getElementById('generateBtn');

        this.progressSection = document.getElementById('progressSection');
        this.progressBar = document.getElementById('progressBar');
        this.progressStatus = document.getElementById('progressStatus');

        this.resultSection = document.getElementById('resultSection');
        this.resultVideo = document.getElementById('resultVideo');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.newVideoBtn = document.getElementById('newVideoBtn');

        this.currentJobId = null;
        this.pollInterval = null;

        this.init();
    }

    init() {
        // Character counter
        this.promptInput.addEventListener('input', () => {
            this.charCount.textContent = this.promptInput.value.length;
        });

        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.generateVideo();
        });

        // New video button
        this.newVideoBtn.addEventListener('click', () => {
            this.reset();
        });
    }

    async generateVideo() {
        const formData = new FormData(this.form);

        const payload = {
            prompt: formData.get('prompt'),
            duration: parseInt(formData.get('duration')),
            language: formData.get('language'),
            upscale: document.getElementById('upscale').checked,
            pro: document.getElementById('pro').checked,
        };

        // Disable form
        this.setLoading(true);

        try {
            // Submit generation request
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            }

            const data = await response.json();
            this.currentJobId = data.job_id;

            // Show progress section
            this.showProgress();

            // Start polling for status
            this.startPolling();

        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            this.setLoading(false);
        }
    }

    async checkStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/status/${this.currentJobId}`);
            const data = await response.json();
            const job = data.job;

            // Update progress bar
            this.progressBar.style.width = `${job.progress}%`;
            this.progressStatus.textContent = job.current_step;

            // Update step indicators
            this.updateSteps(job.status, job.progress);

            // Check if complete
            if (job.status === 'completed') {
                this.stopPolling();
                this.showResult(job.output_url);
            } else if (job.status === 'failed') {
                this.stopPolling();
                alert(`Generation failed: ${job.error}`);
                this.reset();
            }

        } catch (error) {
            console.error('Status check error:', error);
        }
    }

    updateSteps(status, progress) {
        const steps = ['step1', 'step2', 'step3', 'step4'];
        const thresholds = [0, 25, 50, 75];

        steps.forEach((stepId, index) => {
            const step = document.getElementById(stepId);
            step.classList.remove('active', 'completed');

            if (progress >= thresholds[index]) {
                if (progress >= (thresholds[index + 1] || 100)) {
                    step.classList.add('completed');
                } else {
                    step.classList.add('active');
                }
            }
        });
    }

    startPolling() {
        this.pollInterval = setInterval(() => {
            this.checkStatus();
        }, 2000); // Poll every 2 seconds
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    setLoading(loading) {
        this.generateBtn.disabled = loading;
        if (loading) {
            this.generateBtn.querySelector('.btn-text').textContent = 'Processing...';
            this.generateBtn.classList.add('loading');
        } else {
            this.generateBtn.querySelector('.btn-text').textContent = 'Generate Video';
            this.generateBtn.classList.remove('loading');
        }
    }

    showProgress() {
        this.progressSection.classList.remove('hidden');
        this.resultSection.classList.add('hidden');
        this.progressBar.style.width = '0%';
        this.progressStatus.textContent = 'Starting generation...';

        // Reset steps
        ['step1', 'step2', 'step3', 'step4'].forEach(id => {
            document.getElementById(id).classList.remove('active', 'completed');
        });
    }

    showResult(videoUrl) {
        this.progressSection.classList.add('hidden');
        this.resultSection.classList.remove('hidden');
        this.setLoading(false);

        // Set video source
        this.resultVideo.src = videoUrl;
        this.resultVideo.load();

        // Set download link
        this.downloadBtn.href = videoUrl;
    }

    reset() {
        this.stopPolling();
        this.currentJobId = null;

        this.progressSection.classList.add('hidden');
        this.resultSection.classList.add('hidden');

        this.setLoading(false);
        this.form.reset();
        this.charCount.textContent = '0';

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VideoGenerator();
});
