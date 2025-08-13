// TripoSR Application JavaScript

// Global variables
let currentImages = [null, null, null, null];
let processedImage = null;
let generatedModels = null;
let currentQRSession = null;
let qrCheckInterval = null;

// DOM elements
const imageSlots = document.querySelectorAll('.image-upload-slot');
const imageInputs = document.querySelectorAll('.image-input');
const processedPreview = document.getElementById('processed-preview');
const processedImageEl = document.getElementById('processed-image');
const generateBtn = document.getElementById('generate-btn');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const modelPlaceholder = document.getElementById('model-placeholder');
const loadingState = document.getElementById('loading-state');
const modelDisplay = document.getElementById('model-display');
const downloadSection = document.getElementById('download-section');
const removeBgCheckbox = document.getElementById('remove-bg');
const foregroundRatioSlider = document.getElementById('foreground-ratio');
const mcResolutionSlider = document.getElementById('mc-resolution');
const ratioValue = document.getElementById('ratio-value');
const resolutionValue = document.getElementById('resolution-value');

// QR Code elements
const generateQRBtn = document.getElementById('generate-qr-btn');
const qrCodeContainer = document.getElementById('qr-code-container');
const qrCodeCanvas = document.getElementById('qr-code-canvas');
const qrStatus = document.getElementById('qr-status');

// Download buttons
const downloadObjBtn = document.getElementById('download-obj');
const downloadGlbBtn = document.getElementById('download-glb');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSliders();
    setupEnhancedAnimations();
});

function initializeEventListeners() {
    // Image upload handlers
    imageSlots.forEach((slot, index) => {
        const uploadArea = slot.querySelector('.upload-area');
        const input = slot.querySelector('.image-input');
        const preview = slot.querySelector('.preview-image');
        const content = slot.querySelector('.upload-content');

        uploadArea.addEventListener('click', () => input.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('border-blue-400');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('border-blue-400');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-blue-400');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleImageUpload(files[0], index);
            }
        });

        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleImageUpload(e.target.files[0], index);
            }
        });
    });

    // Generate button
    generateBtn.addEventListener('click', generateModel);

    // QR Code button
    generateQRBtn.addEventListener('click', generateQRCode);

    // Download buttons
    downloadObjBtn.addEventListener('click', () => downloadModel('obj'));
    downloadGlbBtn.addEventListener('click', () => downloadModel('glb'));
}

function initializeSliders() {
    // Foreground ratio slider
    foregroundRatioSlider.addEventListener('input', (e) => {
        ratioValue.textContent = e.target.value;
        if (currentImages[0]) {
            processImage();
        }
    });

    // MC resolution slider
    mcResolutionSlider.addEventListener('input', (e) => {
        resolutionValue.textContent = e.target.value;
    });

    // Remove background checkbox
    removeBgCheckbox.addEventListener('change', () => {
        if (currentImages[0]) {
            processImage();
        }
    });
}

function handleImageUpload(file, slotIndex) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        currentImages[slotIndex] = e.target.result;
        
        const slot = imageSlots[slotIndex];
        const preview = slot.querySelector('.preview-image');
        const content = slot.querySelector('.upload-content');
        
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        content.classList.add('hidden');
        
        // Only process the first image for 3D generation
        if (slotIndex === 0) {
            processImage();
        }
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!currentImages[0]) return;

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImages[0],
                removeBackground: removeBgCheckbox.checked,
                foregroundRatio: parseFloat(foregroundRatioSlider.value)
            })
        });

        const result = await response.json();
        
        if (result.success) {
            processedImage = result.processedImage;
            processedImageEl.src = processedImage;
            processedPreview.classList.remove('hidden');
            generateBtn.disabled = false;
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error processing image:', error);
        alert('Error processing image: ' + error.message);
    }
}

async function generateModel() {
    if (!processedImage) {
        alert('Please upload and process an image first');
        return;
    }

    try {
        generateBtn.disabled = true;
        showModelLoading();
        showProgress('Initializing...', 10);

        setTimeout(() => showProgress('Preprocessing image...', 30), 500);
        setTimeout(() => showProgress('Generating 3D model...', 60), 2000);
        setTimeout(() => showProgress('Finalizing...', 90), 8000);

        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                processedImage: processedImage,
                mcResolution: parseInt(mcResolutionSlider.value)
            })
        });

        const result = await response.json();
        
        if (result.success) {
            generatedModels = result.models;
            displayModel();
            downloadSection.classList.remove('hidden');
            hideProgress();
            showSuccessMessage();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error generating model:', error);
        alert('Error generating model: ' + error.message);
        hideProgress();
        hideModelLoading();
    } finally {
        generateBtn.disabled = false;
    }
}

function displayModel() {
    hideModelLoading();
    modelDisplay.classList.remove('hidden');
    
    // Create a simple 3D preview placeholder
    const canvas = document.getElementById('model-canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Draw a simple 3D cube wireframe as placeholder
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#764ba2';
    ctx.lineWidth = 2;
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const size = 100;
    
    // Draw wireframe cube
    ctx.beginPath();
    // Front face
    ctx.rect(centerX - size/2, centerY - size/2, size, size);
    // Back face (offset)
    ctx.rect(centerX - size/2 + 20, centerY - size/2 - 20, size, size);
    // Connect corners
    ctx.moveTo(centerX - size/2, centerY - size/2);
    ctx.lineTo(centerX - size/2 + 20, centerY - size/2 - 20);
    ctx.moveTo(centerX + size/2, centerY - size/2);
    ctx.lineTo(centerX + size/2 + 20, centerY - size/2 - 20);
    ctx.moveTo(centerX - size/2, centerY + size/2);
    ctx.lineTo(centerX - size/2 + 20, centerY + size/2 - 20);
    ctx.moveTo(centerX + size/2, centerY + size/2);
    ctx.lineTo(centerX + size/2 + 20, centerY + size/2 - 20);
    ctx.stroke();
    
    // Add success text
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('3D Model Generated Successfully!', centerX, centerY + size + 30);
    ctx.fillText('Click download buttons to save', centerX, centerY + size + 50);
}

function downloadModel(format) {
    if (!generatedModels || !generatedModels[format]) {
        alert('Model not available for download');
        return;
    }

    // Create download link
    const modelData = generatedModels[format];
    const byteCharacters = atob(modelData);
    const byteNumbers = new Array(byteCharacters.length);
    
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'application/octet-stream' });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `triposr_model.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showProgress(text, progress) {
    progressContainer.classList.remove('hidden');
    progressText.textContent = text;
    progressFill.style.width = progress + '%';
}

function hideProgress() {
    progressContainer.classList.add('hidden');
}

function showModelLoading() {
    modelPlaceholder.classList.add('hidden');
    loadingState.classList.remove('hidden');
    modelDisplay.classList.add('hidden');
}

function hideModelLoading() {
    loadingState.classList.add('hidden');
}

function showSuccessMessage() {
    // Create and show success notification
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
    notification.textContent = '3D Model generated successfully!';
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// QR Code functionality
async function generateQRCode() {
    try {
        generateQRBtn.disabled = true;
        generateQRBtn.textContent = 'Generating...';
        
        // Create upload session
        const response = await fetch('/api/qr-upload-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentQRSession = result.session_id;
            
            // Try to generate QR code with library, fallback to API if needed
            try {
                if (typeof QRCode !== 'undefined' && window.QRCodeLoaded) {
                    await QRCode.toCanvas(qrCodeCanvas, result.upload_url, {
                        width: 200,
                        margin: 2,
                        color: {
                            dark: '#000000',
                            light: '#FFFFFF'
                        }
                    });
                } else {
                    // Use fallback method
                    generateQRCodeFallback(result.upload_url, qrCodeCanvas);
                }
            } catch (qrError) {
                console.log('Primary QR method failed, using fallback:', qrError);
                generateQRCodeFallback(result.upload_url, qrCodeCanvas);
            }
            
            qrCodeContainer.classList.remove('hidden');
            generateQRBtn.textContent = 'Regenerate QR';
            generateQRBtn.disabled = false;
            
            // Start checking for uploads
            startQRSessionCheck();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error generating QR code:', error);
        alert('Error generating QR code: ' + error.message);
        generateQRBtn.disabled = false;
        generateQRBtn.textContent = 'Scan QR Code';
    }
}

// Fallback QR code generator using Google Charts API
function generateQRCodeFallback(text, canvas) {
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.onerror = function() {
        // If even the fallback fails, draw a simple message
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#000000';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('QR Code Generation Failed', canvas.width/2, canvas.height/2 - 10);
        ctx.fillText('Please refresh and try again', canvas.width/2, canvas.height/2 + 10);
    };
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(text)}`;
    img.src = qrUrl;
}

function startQRSessionCheck() {
    if (qrCheckInterval) {
        clearInterval(qrCheckInterval);
    }
    
    qrCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/check-qr-session/${currentQRSession}`);
            const result = await response.json();
            
            if (result.success) {
                if (result.status === 'completed' && result.images.length > 0) {
                    // Images uploaded successfully
                    qrStatus.innerHTML = `
                        <div class="inline-flex items-center gap-2 text-green-400">
                            <div class="w-2 h-2 bg-green-400 rounded-full"></div>
                            <span class="text-sm">Successfully uploaded ${result.image_count} images!</span>
                        </div>
                    `;
                    
                    // Load the first image
                    if (result.images[0]) {
                        currentImages[0] = result.images[0];
                        const slot = imageSlots[0];
                        const preview = slot.querySelector('.preview-image');
                        const content = slot.querySelector('.upload-content');
                        
                        preview.src = result.images[0];
                        preview.classList.remove('hidden');
                        content.classList.add('hidden');
                        
                        processImage();
                    }
                    
                    clearInterval(qrCheckInterval);
                    
                    // Hide QR code after successful upload
                    setTimeout(() => {
                        qrCodeContainer.classList.add('hidden');
                        generateQRBtn.textContent = 'Scan QR Code';
                    }, 3000);
                }
            } else if (response.status === 404 || response.status === 410) {
                // Session expired or not found
                qrStatus.innerHTML = `
                    <div class="inline-flex items-center gap-2 text-red-400">
                        <div class="w-2 h-2 bg-red-400 rounded-full"></div>
                        <span class="text-sm">Session expired. Please generate a new QR code.</span>
                    </div>
                `;
                clearInterval(qrCheckInterval);
            }
        } catch (error) {
            console.error('Error checking QR session:', error);
        }
    }, 2000);
}

function setupEnhancedAnimations() {
    // Add glow effect to interactive elements
    document.querySelectorAll('.enhanced-glass').forEach(el => {
        el.addEventListener('mouseenter', () => {
            el.style.boxShadow = `
                0 20px 60px rgba(0, 0, 0, 0.6),
                0 0 60px rgba(147, 51, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2)
            `;
        });
        el.addEventListener('mouseleave', () => {
            el.style.boxShadow = `
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.05)
            `;
        });
    });
}
