// DOM elements (ensure these IDs exist in your HTML)
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const previewContainer = document.getElementById('preview-container');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingSpinner = document.getElementById('loading-spinner');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress');
const progressText = document.getElementById('progress-text');
const analysisResults = document.getElementById('analysis-results');
const diagnosisContainer = document.getElementById('diagnosis-container');
// const processingCanvas = document.getElementById('processing-canvas'); // Keep if needed for other client-side processing

// --- Global variable to store the selected file ---
let currentFile = null;
// --- Correct API Endpoint URL ---
// Use 127.0.0.1 or localhost, ensure your backend is running on port 8000
const API_ENDPOINT = 'http://127.0.0.1:8000/predict/';

// --- Skin conditions database (for detailed info) ---
// IMPORTANT: Make sure the 'name' here EXACTLY matches
// the 'predicted_class' string returned by your API for each condition.
const skinConditions = [
    {
        id: 1,
        name: "Acne and Rosacea",
        description: "Common inflammatory skin conditions often presenting with pimples, redness, and irritation.",
        modernTreatments: [
            "Topical retinoids and antibiotics",
            "Oral isotretinoin or antibiotics",
            "Laser and light therapy",
            "Hormonal therapy for severe cases"
        ],
        ayurvedicTreatments: [
            "Application of neem and turmeric paste",
            "Use of Manjistha for blood purification",
            "Diet rich in cooling and non-spicy foods",
            "Triphala and aloe vera for detoxification"
        ]
    },
    {
        id: 2,
        name: "Actinic Keratosis and Basal Cell Carcinoma",
        description: "Precancerous and cancerous skin lesions caused by prolonged sun exposure.",
        modernTreatments: [
            "Cryotherapy",
            "Topical chemotherapy (5-FU, Imiquimod)",
            "Surgical excision",
            "Photodynamic therapy"
        ],
        ayurvedicTreatments: [
            "Neem and turmeric applications",
            "Internal detox with Triphala and Guduchi",
            "Avoiding sun exposure and spicy food",
            "Use of blood-purifying herbs like Manjistha"
        ]
    },
    {
        id: 3,
        name: "Atopic Dermatitis",
        description: "A chronic skin condition causing inflammation, itching, and dryness.",
        modernTreatments: [
            "Topical corticosteroids and moisturizers",
            "Immunosuppressants (e.g., Dupilumab)",
            "Antihistamines",
            "Phototherapy"
        ],
        ayurvedicTreatments: [
            "Use of cooling herbs like neem and sandalwood",
            "Internal detox with Panchakarma",
            "Dietary changes avoiding allergens",
            "Application of coconut oil and aloe vera"
        ]
    },
    {
        id: 4,
        name: "Bullous Disease",
        description: "Autoimmune disorders characterized by large fluid-filled blisters.",
        modernTreatments: [
            "Systemic corticosteroids",
            "Immunosuppressive agents",
            "Antibiotics for secondary infection",
            "Wound care and supportive therapy"
        ],
        ayurvedicTreatments: [
            "Herbal cooling treatments like sandalwood paste",
            "Triphala and neem decoctions",
            "Use of medicated ghee",
            "Avoiding spicy and fermented foods"
        ]
    },
    {
        id: 5,
        name: "Cellulitis and Impetigo",
        description: "Bacterial skin infections often caused by Staphylococcus or Streptococcus.",
        modernTreatments: [
            "Oral and topical antibiotics",
            "Wound care",
            "Drainage of abscesses (if any)",
            "Pain management"
        ],
        ayurvedicTreatments: [
            "Neem and turmeric paste for antibacterial action",
            "Triphala and Guduchi for immunity",
            "Proper hygiene and diet control",
            "Topical use of Jatyadi oil"
        ]
    },
    {
        id: 6,
        name: "Eczema",
        description: "Inflammatory skin disorder causing itchy, red, and dry skin patches.",
        modernTreatments: [
            "Moisturizers and topical steroids",
            "Calcineurin inhibitors",
            "Phototherapy",
            "Oral antihistamines"
        ],
        ayurvedicTreatments: [
            "Application of coconut and neem oil",
            "Panchakarma detox",
            "Use of herbs like Khadira and Manjistha",
            "Avoidance of triggering foods"
        ]
    },
    {
        id: 7,
        name: "Exanthems and Drug Eruptions",
        description: "Skin rashes triggered by infections or medications.",
        modernTreatments: [
            "Discontinuation of causative drug",
            "Antihistamines and corticosteroids",
            "Supportive care and monitoring",
            "Hospitalization if severe"
        ],
        ayurvedicTreatments: [
            "Triphala and Guduchi for detox",
            "Topical turmeric paste",
            "Cooling diet and lifestyle",
            "Neem bath or paste for skin soothing"
        ]
    },
    {
        id: 8,
        name: "Herpes, HPV, and other STDs",
        description: "Sexually transmitted infections that often present with skin manifestations.",
        modernTreatments: [
            "Antiviral medications (Acyclovir, Valacyclovir)",
            "HPV vaccination",
            "Safe sex practices and counseling",
            "Topical treatments for lesions"
        ],
        ayurvedicTreatments: [
            "Neem and aloe vera for topical soothing",
            "Ashwagandha and Guduchi for immunity",
            "Diet rich in antioxidants",
            "Avoiding aggravating factors like heat and stress"
        ]
    },
    {
        id: 9,
        name: "Lupus and Connective Tissue Diseases",
        description: "Autoimmune disorders that affect the skin and internal organs.",
        modernTreatments: [
            "Corticosteroids and immunosuppressants",
            "Antimalarial drugs (Hydroxychloroquine)",
            "Sun protection and supportive care"
        ],
        ayurvedicTreatments: [
            "Use of Ashwagandha and Giloy for immunity",
            "Sandalwood and aloe vera for skin",
            "Diet free of inflammatory triggers",
            "Panchakarma for systemic detox"
        ]
    },
    {
        id: 10,
        name: "Melanoma, Nevi, and Moles",
        description: "Pigmented skin lesions that can range from benign to malignant.",
        modernTreatments: [
            "Surgical excision and biopsy",
            "Sentinel node analysis (for melanoma)",
            "Immunotherapy or targeted therapy (advanced melanoma)"
        ],
        ayurvedicTreatments: [
            "Turmeric and neem paste for skin health",
            "Monitoring for changes in moles",
            "Manjistha for blood purification",
            "Avoid sun exposure and toxins"
        ]
    }
    // ... you can continue similarly for remaining folders as needed
];


// --- Event Listeners Setup ---
document.addEventListener('DOMContentLoaded', () => {
    // Check if elements exist before adding listeners
    if (!fileInput || !uploadArea || !analyzeBtn || !previewContainer || !previewImg || !loadingSpinner || !progressContainer || !progressBar || !progressText || !analysisResults || !diagnosisContainer) {
        console.error("One or more required DOM elements are missing. Please check HTML IDs.");
        return; // Stop script execution if essential elements are missing
    }

    // Make upload area clickable
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection via input
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Analyze button click handler
    analyzeBtn.addEventListener('click', analyzeImage);
    analyzeBtn.disabled = true; // Disable initially
    analyzeBtn.style.display = 'none'; // Hide initially

    // Optional: Make the "Browse Files" button also trigger the file input
    const browseButton = uploadArea.querySelector('.upload-btn');
    if (browseButton) {
         browseButton.addEventListener('click', (e) => {
             e.stopPropagation(); // Prevent triggering uploadArea's click
             fileInput.click();
         });
    }

    console.log('SkinCureX Interface Ready (' + new Date().toISOString() + ')');
});

// --- File Handling Functions ---

// Handle file selection (from input change or drop)
function handleFileSelect(e) {
    // Determine the source of the file
    const file = e.target.files ? e.target.files[0] : e.dataTransfer.files[0];

    if (file && isValidImageFile(file)) {
        currentFile = file; // Store the valid file
        displayImagePreview(file); // Show preview

        // Update UI - Enable and show Analyze button
        analyzeBtn.disabled = false;
        analyzeBtn.style.display = 'inline-block';
        analyzeBtn.textContent = 'Analyze Skin';
        analysisResults.style.display = 'none'; // Hide previous results

    } else {
        currentFile = null; // Reset if invalid
        if (file) { // Only alert if a file was provided but was invalid
           alert('Please select or drop a valid image file (JPEG, PNG, GIF, BMP, TIFF).');
        }
        // Reset button state
        analyzeBtn.disabled = true;
        analyzeBtn.style.display = 'none';
        // Optionally reset preview
        // previewImg.src = '/api/placeholder/400/320'; // Reset to placeholder
        // previewContainer.style.display = 'none';
        // uploadArea.style.display = 'block'; // Or 'flex' depending on CSS
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault(); // Necessary to allow drop
    e.stopPropagation();
    uploadArea.style.borderColor = '#FF6B6B'; // Highlight effect
    uploadArea.style.backgroundColor = 'rgba(255, 107, 107, 0.05)';
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#dee2e6'; // Reset highlight
    uploadArea.style.backgroundColor = 'transparent';
}

// Handle drop - Calls handleFileSelect
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#dee2e6'; // Reset highlight
    uploadArea.style.backgroundColor = 'transparent';
    // Pass the event directly to handleFileSelect to get the file
    handleFileSelect(e);
}

// Check if file is a valid image
function isValidImageFile(file) {
    const acceptedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
    return file && acceptedTypes.includes(file.type);
}

// Display image preview
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result; // Set preview image source
        uploadArea.style.display = 'none'; // Hide upload prompt
        previewContainer.style.display = 'flex'; // Show preview container
        // Ensure container styles are appropriate (set in CSS or here)
        previewContainer.style.flexDirection = 'column';
        previewContainer.style.alignItems = 'center';
    };
    reader.onerror = function(e) {
        console.error("FileReader error:", e);
        alert("Error reading file for preview.");
    };
    reader.readAsDataURL(file); // Read file content for preview
}

// --- Analysis Function (API Call) ---

async function analyzeImage() {
    if (!currentFile) {
        alert("Cannot analyze: No valid image file is selected.");
        return;
    }

    // --- UI Update: Start Loading ---
    analyzeBtn.disabled = true;
    analyzeBtn.style.display = 'none';
    loadingSpinner.style.display = 'block'; // Show spinner
    progressContainer.style.display = 'block'; // Show progress bar area
    progressBar.style.width = '0%'; // Reset progress bar
    progressText.textContent = 'Uploading & Analyzing...';
    analysisResults.style.display = 'none'; // Hide previous results

    // --- Prepare Data ---
    const formData = new FormData();
    formData.append('file', currentFile, currentFile.name); // Key 'file' must match backend

    // --- API Call ---
    console.log(`[${new Date().toISOString()}] Sending image "${currentFile.name}" to ${API_ENDPOINT}`);
    try {
        progressBar.style.width = '30%'; // Simulate some progress

        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
            // No Content-Type header needed for FormData
        });

        progressBar.style.width = '70%';
        progressText.textContent = 'Processing Results...';

        if (!response.ok) {
            // Attempt to parse error details from the server response
            let errorMsg = `API Error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                // Use the 'detail' field if provided by FastAPI, otherwise stringify
                errorMsg = `API Error (${response.status}): ${errorData.detail || JSON.stringify(errorData)}`;
            } catch (e) {
                console.warn("Could not parse error response body as JSON.");
            }
            throw new Error(errorMsg); // Throw error to be caught below
        }

        // Parse successful JSON response
        const predictionData = await response.json();
        console.log(`[${new Date().toISOString()}] API Success:`, predictionData);

        progressBar.style.width = '100%';
        progressText.textContent = 'Analysis Complete!';

        // Show results after a brief delay
        setTimeout(() => {
            showResults(predictionData); // Pass API data to display function
        }, 500);

    } catch (error) {
        console.error(`[${new Date().toISOString()}] Analysis Failed:`, error);
        // Display error message clearly to the user
        diagnosisContainer.innerHTML = `<p class="error" style="color: red; font-weight: bold;">Analysis Failed: ${error.message}. Please check network connection, ensure the server is running, and try again.</p>`;
        analysisResults.style.display = 'block'; // Show the error area

        // --- UI Update: Reset on Error ---
        loadingSpinner.style.display = 'none'; // Hide spinner
        // Keep progress container hidden or hide after delay in finally block
        analyzeBtn.disabled = false; // Re-enable button
        analyzeBtn.style.display = 'inline-block';
        analyzeBtn.textContent = 'Try Again';

    } finally {
        // This block runs regardless of success or failure
        // Hide progress bar after results/error are shown (with a delay)
        setTimeout(() => {
           progressContainer.style.display = 'none';
        }, 1500); // Give user time to see "Complete!" or error
    }
}

// --- Result Display Function ---

function showResults(predictionData) {
    const predictedClass = predictionData.predicted_class;
    const apiConfidence = predictionData.confidence;

    // Find matching condition details locally using case-insensitive comparison
    const conditionDetails = skinConditions.find(
        condition => condition.name.toLowerCase() === predictedClass.toLowerCase()
    );

    let diagnosisHTML = '';

    if (conditionDetails) {
        // Found details - use them
        const confidencePercentage = (apiConfidence * 100).toFixed(2);
        console.log(`Displaying details for "${conditionDetails.name}" with ${confidencePercentage}% confidence.`);

        // Helper function to format list items
        const formatList = (items, icon) => items.map(item =>
            `<li class="treatment-item"><span class="treatment-icon">${icon}</span>${item}</li>`
        ).join('');

        diagnosisHTML = `
            <div class="diagnosis-name">
                ${conditionDetails.name} <!-- Use consistent name from local DB -->
                <span class="confidence">${confidencePercentage}% Match</span> <!-- Use confidence from API -->
            </div>
            <p class="diagnosis-description">${conditionDetails.description || 'No description available.'}</p>

            <h4 class="treatment-title">Modern Treatment Recommendations:</h4>
            <ul class="treatment-list">
                ${formatList(conditionDetails.modernTreatments || ['Consult a dermatologist.'], '💊')}
            </ul>

            <h4 class="treatment-title">Ayurvedic Treatment Recommendations:</h4>
            <ul class="treatment-list">
                ${formatList(conditionDetails.ayurvedicTreatments || ['Consult an Ayurvedic practitioner.'], '🌿')}
            </ul>
            <p><small>Analyzed File: ${predictionData.filename || 'N/A'}</small></p>
        `;
    } else {
        // Fallback if prediction is not found in the local 'skinConditions' array
        const confidencePercentage = (apiConfidence * 100).toFixed(2);
        console.warn(`Predicted class "${predictedClass}" not found in local skinConditions database.`);
        diagnosisHTML = `
             <div class="diagnosis-name">
                ${predictedClass} <!-- Show the raw predicted class name from API -->
                <span class="confidence">${confidencePercentage}% Match</span>
            </div>
            <p class="diagnosis-description error" style="color: orange;">
                Detailed treatment information for "${predictedClass}" is not available in our current database.
                The analysis suggests this condition with ${confidencePercentage}% confidence.
                Please consult a qualified healthcare professional for accurate diagnosis and treatment options.
            </p>
             <p><small>Analyzed File: ${predictionData.filename || 'N/A'}</small></p>
        `;
    }

    // Update the UI
    diagnosisContainer.innerHTML = diagnosisHTML;
    analysisResults.style.display = 'block'; // Show results section
    loadingSpinner.style.display = 'none'; // Hide spinner
    analyzeBtn.disabled = false; // Re-enable button
    analyzeBtn.style.display = 'inline-block';
    analyzeBtn.textContent = 'Analyze Another Image'; // Update button text
}

// --- Smooth Scrolling for Navigation Links ---
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        try {
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust offset for fixed header if needed
                    behavior: 'smooth'
                });
            } else {
                console.warn(`Smooth scroll target not found: ${targetId}`);
            }
        } catch (error) {
             console.error(`Error finding smooth scroll target "${targetId}":`, error);
        }
    });
});

// --- Mobile Menu Toggle (Basic Example) ---
const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navMenu = document.querySelector('.nav-menu');

if (mobileMenuBtn && navMenu) {
    mobileMenuBtn.addEventListener('click', () => {
        navMenu.classList.toggle('active'); // You'll need CSS for the .active class
    });
} else {
    console.warn("Mobile menu button or nav menu not found.");
}