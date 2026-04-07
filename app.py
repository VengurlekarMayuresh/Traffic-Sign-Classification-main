from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import cv2
import numpy as np
import json
import codecs
from model import TrafficSignCNN
import torchvision.transforms as transforms
import os
from pathlib import Path

app = Flask(__name__)

# Load model and labels once at startup
device = torch.device('cpu')
model = TrafficSignCNN(43)
model.load_state_dict(torch.load('serialized_data/model.pt', map_location=device))
model.to(device)
model.eval()

# Load label names
label_json = codecs.open("DataProfiling/label_names.json", 'r', encoding='utf-8').read()
label_names = json.loads(label_json)

def predict_image(image_bytes):
    """Predict traffic sign from image bytes"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None, "Could not read image"

    # Preprocess
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    class_name = label_names.get(str(prediction), "Unknown")
    return {
        'class_id': int(prediction),
        'label': class_name,
        'confidence': float(confidence)
    }, None

@app.route('/')
def index():
    """Main page with upload functionality and class gallery"""
    return render_template('index.html', label_names=label_names)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image_bytes = file.read()
    result, error = predict_image(image_bytes)
    if error:
        return jsonify({'error': error}), 400

    return jsonify(result)

@app.route('/sample/<int:class_id>')
def sample_image(class_id):
    """Serve a pre-generated sample image for a given class"""
    samples_dir = 'static/samples'
    # Look for any image file with the class_id as base name
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        filename = f"{class_id}{ext}"
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            return send_from_directory(samples_dir, filename)
    return send_from_directory('static', 'placeholder.png')

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Generate HTML template with enhanced UI
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚦 Traffic Sign Classifier - GTSRB</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }

        .class-card {
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .class-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .upload-zone {
            transition: all 0.3s ease;
            border: 3px dashed #3b82f6;
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: #10b981;
            background: #f0fdf4;
        }

        .confidence-bar {
            background: linear-gradient(90deg, #3b82f6, #10b981);
        }

        .prediction-card {
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .sample-img {
            object-fit: cover;
            aspect-ratio: 1;
        }

        .nav-tab {
            transition: all 0.2s ease;
        }

        .nav-tab.active {
            background: #3b82f6;
            color: white;
        }

        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <i class="fas fa-traffic-light text-3xl text-blue-600"></i>
                <h1 class="text-xl font-bold text-gray-800">Traffic Sign Classifier</h1>
            </div>
            <div class="flex space-x-2">
                <button onclick="showSection('upload')" class="nav-tab active px-4 py-2 rounded-lg hover:bg-blue-50">
                    <i class="fas fa-upload mr-2"></i>Upload
                </button>
                <button onclick="showSection('gallery')" class="nav-tab px-4 py-2 rounded-lg hover:bg-blue-50">
                    <i class="fas fa-th mr-2"></i>Gallery
                </button>
            </div>
        </div>
    </nav>

    <!-- Upload Section -->
    <section id="upload-section" class="max-w-7xl mx-auto px-4 py-8">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Upload Area -->
            <div class="bg-white rounded-2xl shadow-lg p-8">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">
                    <i class="fas fa-cloud-upload-alt text-blue-600 mr-2"></i>
                    Upload Traffic Sign
                </h2>
                <div class="upload-zone rounded-xl p-12 text-center cursor-pointer" id="uploadZone">
                    <input type="file" id="fileInput" accept="image/*" class="hidden">
                    <i class="fas fa-camera text-6xl text-blue-400 mb-4"></i>
                    <p class="text-lg font-medium text-gray-700 mb-2">Drop image here or click to browse</p>
                    <p class="text-sm text-gray-500">Supports PNG, JPG, JPEG (max 5MB)</p>
                </div>

                <!-- Preview -->
                <div id="previewContainer" class="mt-6 hidden">
                    <h3 class="text-sm font-semibold text-gray-600 mb-2">Preview</h3>
                    <div class="flex justify-center">
                        <img id="previewImg" class="max-w-xs max-h-64 rounded-lg shadow-md border-2 border-blue-200">
                    </div>
                </div>

                <!-- Loading -->
                <div id="loading" class="hidden mt-6 text-center">
                    <div class="loading-spinner mx-auto mb-3"></div>
                    <p class="text-gray-600">Analyzing traffic sign...</p>
                </div>

                <!-- Error -->
                <div id="error" class="hidden mt-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
                </div>

                <!-- Result -->
                <div id="result" class="hidden mt-6 prediction-card">
                    <div class="bg-gradient-to-r from-blue-500 to-green-500 rounded-xl p-6 text-white">
                        <h3 class="text-lg font-semibold mb-4">
                            <i class="fas fa-check-circle mr-2"></i>Prediction Result
                        </h3>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span>Class ID:</span>
                                <span class="font-mono font-bold" id="classId">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Label:</span>
                                <span class="font-semibold" id="label">-</span>
                            </div>
                            <div class="mt-4">
                                <div class="flex justify-between text-sm mb-1">
                                    <span>Confidence:</span>
                                    <span id="confidenceText">-</span>
                                </div>
                                <div class="bg-white/30 rounded-full h-3 overflow-hidden">
                                    <div id="confidenceBar" class="confidence-bar h-full rounded-full" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Info Panel -->
            <div class="bg-white rounded-2xl shadow-lg p-8">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">
                    <i class="fas fa-info-circle text-blue-600 mr-2"></i>
                    About
                </h2>
                <div class="space-y-4 text-gray-700">
                    <p>
                        This classifier uses a <strong>Convolutional Neural Network (CNN)</strong> trained on the
                        <strong>GTSRB</strong> (German Traffic Sign Recognition Benchmark) dataset.
                    </p>
                    <div class="bg-blue-50 rounded-lg p-4">
                        <h3 class="font-semibold mb-2">Dataset Stats:</h>
                        <ul class="space-y-1 text-sm">
                            <li><i class="fas fa-check text-green-500 mr-2"></i>43 classes of traffic signs</li>
                            <li><i class="fas fa-check text-green-500 mr-2"></i>~39,209 training images</li>
                            <li><i class="fas fa-check text-green-500 mr-2"></i>~12,630 test images</li>
                            <li><i class="fas fa-check text-green-500 mr-2"></i>Model: TrafficSignCNN</li>
                        </ul>
                    </div>
                    <p class="text-sm">
                        <i class="fas fa-lightbulb text-yellow-500 mr-2"></i>
                        <strong>Tip:</strong> For best results, upload clear, front-facing images of traffic signs with good lighting.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Gallery Section -->
    <section id="gallery-section" class="max-w-7xl mx-auto px-4 py-8 hidden">
        <div class="bg-white rounded-2xl shadow-lg p-8">
            <h2 class="text-2xl font-bold mb-6 text-gray-800">
                <i class="fas fa-th text-blue-600 mr-2"></i>
                All Traffic Sign Classes
            </h2>
            <p class="text-gray-600 mb-6">Explore all 43 classes with sample images from the GTSRB dataset.</p>

            <!-- Search -->
            <div class="mb-6">
                <input type="text" id="searchInput" placeholder="Search signs..."
                       class="w-full max-w-md px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                       onkeyup="filterGallery()">
            </div>

            <!-- Grid -->
            <div id="galleryGrid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                <!-- Cards will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <script>
        const labelNames = ''' + json.dumps(label_names) + ''';
        const sampleImages = {};

        // Fetch sample images for each class
        async function loadSampleImages() {
            for (let i = 0; i < 43; i++) {
                try {
                    const response = await fetch(`/sample/${i}`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = URL.createObjectURL(blob);
                        sampleImages[i] = url;
                    }
                } catch (e) {
                    console.log(`No sample for class ${i}`);
                }
            }
            renderGallery();
        }

        function renderGallery(filter = '') {
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';

            Object.entries(labelNames).forEach(([id, name]) => {
                const numId = parseInt(id);
                if (filter && !name.toLowerCase().includes(filter.toLowerCase()) && !id.includes(filter)) {
                    return;
                }

                const card = document.createElement('div');
                card.className = 'class-card bg-gray-50 rounded-xl overflow-hidden shadow hover:shadow-lg';
                card.innerHTML = `
                    <div class="h-32 bg-gray-200 flex items-center justify-center">
                        <img src="${sampleImages[numId] || '/static/placeholder.png'}"
                             class="sample-img w-full h-full"
                             onerror="this.src='/static/placeholder.png'"
                             alt="${name}">
                    </div>
                    <div class="p-3">
                        <div class="text-xs font-semibold text-gray-500 mb-1">Class ${id}</div>
                        <div class="text-sm font-medium text-gray-800 line-clamp-2">${name}</div>
                    </div>
                `;
                card.onclick = () => {
                    // Scroll to upload section and show class info
                    showSection('upload');
                    document.getElementById('label').textContent = name;
                    document.getElementById('classId').textContent = id;
                };
                grid.appendChild(card);
            });
        }

        function filterGallery() {
            const query = document.getElementById('searchInput').value;
            renderGallery(query);
        }

        function showSection(section) {
            document.getElementById('upload-section').classList.add('hidden');
            document.getElementById('gallery-section').classList.add('hidden');
            document.getElementById(`${section}-section`).classList.remove('hidden');

            document.querySelectorAll('.nav-tab').forEach(btn => {
                btn.classList.remove('active');
                btn.classList.add('bg-gray-100');
            });
            event?.target.classList.add('active');
            event?.target.classList.remove('bg-gray-100');

            if (section === 'gallery' && Object.keys(sampleImages).length === 0) {
                loadSampleImages();
            }
        }

        // Upload functionality
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const result = document.getElementById('result');

        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        async function handleFile(file) {
            if (!file.type.match('image.*')) {
                showError('Please upload an image file');
                return;
            }

            // Preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewContainer.classList.remove('hidden');
                uploadImage(file);
            };
            reader.readAsDataURL(file);
        }

        async function uploadImage(file) {
            result.classList.add('hidden');
            error.classList.add('hidden');
            loading.classList.remove('hidden');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    showError(data.error || 'Prediction failed');
                } else {
                    showResult(data);
                }
            } catch (err) {
                showError('Network error. Please try again.');
            } finally {
                loading.classList.add('hidden');
            }
        }

        function showResult(data) {
            document.getElementById('classId').textContent = data.class_id;
            document.getElementById('label').textContent = data.label;
            const confidencePercent = (data.confidence * 100).toFixed(2) + '%';
            document.getElementById('confidenceText').textContent = confidencePercent;
            document.getElementById('confidenceBar').style.width = confidencePercent;
            result.classList.remove('hidden');
        }

        function showError(msg) {
            error.textContent = msg;
            error.classList.remove('hidden');
            result.classList.add('hidden');
        }

        // Initialize
        showSection('upload');
        loadSampleImages();
    </script>
</body>
</html>
    '''

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Create a simple placeholder image
    from PIL import Image, ImageDraw
    placeholder = Image.new('RGB', (200, 200), color='#e5e7eb')
    draw = ImageDraw.Draw(placeholder)
    draw.text((100, 100), 'No Image', fill='#9ca3af', anchor='mm', font=None)
    placeholder.save('static/placeholder.png')

    print("Starting Enhanced Traffic Sign Classifier Web App...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
