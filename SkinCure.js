// DOM elements
const fileInput = document.getElementById("file-input");
const uploadArea = document.getElementById("upload-area");
const previewContainer = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const analyzeBtn = document.getElementById("analyze-btn");
const loadingSpinner = document.getElementById("loading-spinner");
const progressContainer = document.getElementById("progress-container");
const progressBar = document.getElementById("progress");
const progressText = document.getElementById("progress-text");
const analysisResults = document.getElementById("analysis-results");
const diagnosisContainer = document.getElementById("diagnosis-container");

// Global variable to store the selected file
let currentFile = null;

// API Endpoint - Update this to match your backend
const API_ENDPOINT = "http://127.0.0.1:8000/predict/";

// Skin conditions database with treatments
const skinConditions = [
  {
    id: 1,
    name: "Acne and Rosacea",
    description: "Common inflammatory skin conditions often presenting with pimples, redness, and irritation.",
    severity: "Moderate",
    modernTreatments: [
      "Topical retinoids and antibiotics",
      "Benzoyl peroxide for bacterial control",
      "Oral isotretinoin for severe cases",
      "Laser therapy for redness reduction"
    ],
    modernTreatmentsAll: [
      "Topical retinoids and antibiotics",
      "Benzoyl peroxide for bacterial control",
      "Oral isotretinoin for severe cases",
      "Laser therapy for redness reduction",
      "Azelaic acid for inflammation",
      "Light therapy and chemical peels",
      "Hormonal therapy for severe cases",
      "Dietary modifications and stress management"
    ],
    ayurvedicTreatments: [
      "Neem and turmeric paste",
      "Manjistha for blood purification",
      "Cooling diet, avoid spicy foods",
      "Triphala for detoxification"
    ],
    ayurvedicTreatmentsAll: [
      "Neem and turmeric paste",
      "Manjistha for blood purification",
      "Cooling diet, avoid spicy foods",
      "Triphala for detoxification",
      "Aloe vera gel for soothing",
      "Sandalwood paste as face mask",
      "Khadira decoction for skin health",
      "Yoga and meditation for stress relief"
    ]
  },
  {
    id: 2,
    name: "Actinic Keratosis and Basal Cell Carcinoma",
    description: "Precancerous and cancerous skin lesions caused by prolonged sun exposure.",
    severity: "High",
    modernTreatments: [
      "Cryotherapy (freezing treatment)",
      "Topical chemotherapy creams",
      "Surgical excision",
      "Photodynamic therapy"
    ],
    modernTreatmentsAll: [
      "Cryotherapy (freezing treatment)",
      "Topical chemotherapy creams",
      "Surgical excision",
      "Photodynamic therapy",
      "Mohs micrographic surgery",
      "Electrodesiccation and curettage",
      "Radiation therapy for non-surgical candidates",
      "Regular monitoring and skin checks"
    ],
    ayurvedicTreatments: [
      "Neem and turmeric applications",
      "Guduchi for immunity boost",
      "Triphala for detoxification",
      "Avoid sun exposure"
    ],
    ayurvedicTreatmentsAll: [
      "Neem and turmeric applications",
      "Guduchi for immunity boost",
      "Triphala for detoxification",
      "Avoid sun exposure",
      "Manjistha for blood purification",
      "Anti-inflammatory diet",
      "Natural sun protection herbs",
      "Rasayana therapy for skin rejuvenation"
    ]
  },
  {
    id: 3,
    name: "Atopic Dermatitis",
    description: "A chronic skin condition causing inflammation, itching, and dryness.",
    severity: "Moderate",
    modernTreatments: [
      "Moisturizers and emollients",
      "Topical corticosteroids",
      "Antihistamines for itching",
      "Phototherapy (UVB treatment)"
    ],
    modernTreatmentsAll: [
      "Moisturizers and emollients",
      "Topical corticosteroids",
      "Antihistamines for itching",
      "Phototherapy (UVB treatment)",
      "Calcineurin inhibitors",
      "Dupilumab injections for severe cases",
      "Wet wrap therapy",
      "Trigger avoidance strategies"
    ],
    ayurvedicTreatments: [
      "Neem oil and coconut oil",
      "Sandalwood paste for cooling",
      "Panchakarma detoxification",
      "Avoid trigger foods"
    ],
    ayurvedicTreatmentsAll: [
      "Neem oil and coconut oil",
      "Sandalwood paste for cooling",
      "Panchakarma detoxification",
      "Avoid trigger foods",
      "Khadira decoction",
      "Manjistha for skin purification",
      "Aloe vera gel application",
      "Stress reduction through yoga"
    ]
  },
  {
    id: 4,
    name: "Bullous Disease",
    description: "Autoimmune disorders characterized by large fluid-filled blisters.",
    severity: "High",
    modernTreatments: [
      "Systemic corticosteroids",
      "Immunosuppressive agents",
      "Antibiotics for infections",
      "Wound care and blister management"
    ],
    modernTreatmentsAll: [
      "Systemic corticosteroids",
      "Immunosuppressive agents",
      "Antibiotics for infections",
      "Wound care and blister management",
      "Rituximab for severe cases",
      "IVIG therapy",
      "Nutritional support",
      "Regular monitoring"
    ],
    ayurvedicTreatments: [
      "Sandalwood paste for cooling",
      "Triphala for detoxification",
      "Neem decoction",
      "Avoid spicy and fermented foods"
    ],
    ayurvedicTreatmentsAll: [
      "Sandalwood paste for cooling",
      "Triphala for detoxification",
      "Neem decoction",
      "Avoid spicy and fermented foods",
      "Medicated ghee (Tikta ghrita)",
      "Guduchi for immune modulation",
      "Turmeric milk for healing",
      "Cooling diet and lifestyle"
    ]
  },
  {
    id: 5,
    name: "Cellulitis and Impetigo",
    description: "Bacterial skin infections often caused by Staphylococcus or Streptococcus.",
    severity: "Moderate to High",
    modernTreatments: [
      "Oral antibiotics",
      "Topical antibiotics for impetigo",
      "Drainage of abscesses if present",
      "Pain management"
    ],
    modernTreatmentsAll: [
      "Oral antibiotics",
      "Topical antibiotics for impetigo",
      "Drainage of abscesses if present",
      "Pain management",
      "IV antibiotics for severe cases",
      "Elevation of affected limb",
      "Wound care and dressing changes",
      "Prevention of recurrence"
    ],
    ayurvedicTreatments: [
      "Neem and turmeric paste",
      "Jatyadi oil for wounds",
      "Triphala for immunity",
      "Proper hygiene maintenance"
    ],
    ayurvedicTreatmentsAll: [
      "Neem and turmeric paste",
      "Jatyadi oil for wounds",
      "Triphala for immunity",
      "Proper hygiene maintenance",
      "Guduchi for immunity boost",
      "Manjistha for blood purification",
      "Vitamin C rich diet",
      "Avoid contaminated water contact"
    ]
  },
  {
    id: 6,
    name: "Eczema",
    description: "Inflammatory skin disorder causing itchy, red, and dry skin patches.",
    severity: "Moderate",
    modernTreatments: [
      "Moisturizers and emollients",
      "Topical corticosteroids",
      "Oral antihistamines",
      "Phototherapy"
    ],
    modernTreatmentsAll: [
      "Moisturizers and emollients",
      "Topical corticosteroids",
      "Oral antihistamines",
      "Phototherapy",
      "Calcineurin inhibitors",
      "Dupilumab for chronic cases",
      "Avoid triggers and allergens",
      "Stress management"
    ],
    ayurvedicTreatments: [
      "Coconut oil and neem oil",
      "Panchakarma detoxification",
      "Aloe vera gel",
      "Stress management through yoga"
    ],
    ayurvedicTreatmentsAll: [
      "Coconut oil and neem oil",
      "Panchakarma detoxification",
      "Aloe vera gel",
      "Stress management through yoga",
      "Khadira and Manjistha decoctions",
      "Avoid trigger foods",
      "Triphala for gut health",
      "Meditation for stress relief"
    ]
  },
  {
    id: 7,
    name: "Exanthems and Drug Eruptions",
    description: "Skin rashes triggered by infections or medications.",
    severity: "Moderate to High",
    modernTreatments: [
      "Stop causative drug immediately",
      "Oral antihistamines",
      "Systemic corticosteroids",
      "Supportive care"
    ],
    modernTreatmentsAll: [
      "Stop causative drug immediately",
      "Oral antihistamines",
      "Systemic corticosteroids",
      "Supportive care",
      "Epinephrine for anaphylaxis",
      "Hospitalization for severe cases",
      "Topical emollients",
      "Monitoring and follow-up"
    ],
    ayurvedicTreatments: [
      "Triphala for detoxification",
      "Turmeric paste",
      "Cooling diet",
      "Neem bath for soothing"
    ],
    ayurvedicTreatmentsAll: [
      "Triphala for detoxification",
      "Turmeric paste",
      "Cooling diet",
      "Neem bath for soothing",
      "Guduchi for detoxification",
      "Manjistha for blood purification",
      "Avoid hot and spicy foods",
      "Sandalwood paste for cooling"
    ]
  },
  {
    id: 8,
    name: "Herpes, HPV, and other STDs",
    description: "Sexually transmitted infections that often present with skin manifestations.",
    severity: "Moderate to High",
    modernTreatments: [
      "Antiviral medications",
      "HPV vaccination",
      "Safe sex practices",
      "Regular STD screening"
    ],
    modernTreatmentsAll: [
      "Antiviral medications",
      "HPV vaccination",
      "Safe sex practices",
      "Regular STD screening",
      "Cryotherapy for warts",
      "Partner notification",
      "Suppressive therapy",
      "Counseling and education"
    ],
    ayurvedicTreatments: [
      "Neem and aloe vera topically",
      "Ashwagandha for immunity",
      "Guduchi for immune support",
      "Antioxidant-rich diet"
    ],
    ayurvedicTreatmentsAll: [
      "Neem and aloe vera topically",
      "Ashwagandha for immunity",
      "Guduchi for immune support",
      "Antioxidant-rich diet",
      "Turmeric for anti-inflammatory",
      "Triphala for detoxification",
      "Avoid stress and heat",
      "Boost immunity naturally"
    ]
  },
  {
    id: 9,
    name: "Lupus and Connective Tissue Diseases",
    description: "Autoimmune disorders that affect the skin and internal organs.",
    severity: "High",
    modernTreatments: [
      "Systemic corticosteroids",
      "Immunosuppressants",
      "Antimalarial drugs",
      "Strict sun protection"
    ],
    modernTreatmentsAll: [
      "Systemic corticosteroids",
      "Immunosuppressants",
      "Antimalarial drugs",
      "Strict sun protection",
      "Biologics for severe cases",
      "NSAIDs for joint pain",
      "Regular organ monitoring",
      "Lifestyle modifications"
    ],
    ayurvedicTreatments: [
      "Ashwagandha for immune modulation",
      "Giloy for systemic support",
      "Sandalwood and aloe vera",
      "Anti-inflammatory diet"
    ],
    ayurvedicTreatmentsAll: [
      "Ashwagandha for immune modulation",
      "Giloy for systemic support",
      "Sandalwood and aloe vera",
      "Anti-inflammatory diet",
      "Panchakarma detoxification",
      "Turmeric for inflammation",
      "Stress management through meditation",
      "Adequate rest and sleep"
    ]
  },
  {
    id: 10,
    name: "Melanoma, Nevi, and Moles",
    description: "Pigmented skin lesions that can range from benign to malignant.",
    severity: "High (for melanoma)",
    modernTreatments: [
      "Surgical excision",
      "Sentinel lymph node biopsy",
      "Immunotherapy",
      "Regular dermoscopic monitoring"
    ],
    modernTreatmentsAll: [
      "Surgical excision",
      "Sentinel lymph node biopsy",
      "Immunotherapy",
      "Regular dermoscopic monitoring",
      "Targeted therapy (BRAF inhibitors)",
      "Radiation therapy",
      "Adjuvant therapy",
      "Monthly self-examinations"
    ],
    ayurvedicTreatments: [
      "Turmeric and neem paste",
      "Regular self-monitoring",
      "Manjistha for blood purification",
      "Avoid excessive sun exposure"
    ],
    ayurvedicTreatmentsAll: [
      "Turmeric and neem paste",
      "Regular self-monitoring",
      "Manjistha for blood purification",
      "Avoid excessive sun exposure",
      "Antioxidant-rich diet",
      "Guduchi for immune support",
      "Triphala for detoxification",
      "Natural sun protection"
    ]
  }
];

// Header and navigation
const header = document.getElementById("header");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const navMenu = document.getElementById("navMenu");

if (header && window) {
  window.addEventListener("scroll", () => {
    if (window.scrollY > 50) {
      header.classList.add("scrolled");
    } else {
      header.classList.remove("scrolled");
    }
  });
}

if (mobileMenuBtn && navMenu) {
  mobileMenuBtn.addEventListener("click", () => {
    navMenu.classList.toggle("active");
  });
}

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    if (navMenu) navMenu.classList.remove("active");
  });
});

// Scroll animations
const observerOptions = {
  threshold: 0.1,
  rootMargin: "0px 0px -100px 0px",
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
    }
  });
}, observerOptions);

document.querySelectorAll(".fade-in").forEach((el) => observer.observe(el));

// File upload functionality
if (uploadArea && fileInput) {
  uploadArea.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelect);
  uploadArea.addEventListener("dragover", handleDragOver);
  uploadArea.addEventListener("dragleave", handleDragLeave);
  uploadArea.addEventListener("drop", handleDrop);
}

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", analyzeImage);
}

const uploadBtn = uploadArea?.querySelector(".upload-btn");
if (uploadBtn) {
  uploadBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
}

function handleFileSelect(e) {
  const file = e.target.files ? e.target.files[0] : e.dataTransfer.files[0];
  if (file && isValidImageFile(file)) {
    currentFile = file;
    displayImagePreview(file);
    if (analyzeBtn) {
      analyzeBtn.style.display = "inline-flex";
      analyzeBtn.disabled = false;
    }
    if (analysisResults) {
      analysisResults.style.display = "none";
    }
  } else {
    alert("Please select a valid image file (JPEG, PNG, GIF, BMP, or TIFF)");
  }
}

function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  if (uploadArea) {
    uploadArea.style.borderColor = "var(--primary, #6366f1)";
    uploadArea.style.background = "rgba(99, 102, 241, 0.05)";
  }
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  if (uploadArea) {
    uploadArea.style.borderColor = "var(--gray-300, #ddd)";
    uploadArea.style.background = "white";
  }
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  if (uploadArea) {
    uploadArea.style.borderColor = "var(--gray-300, #ddd)";
    uploadArea.style.background = "white";
  }
  handleFileSelect(e);
}

function isValidImageFile(file) {
  const acceptedTypes = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff"
  ];
  return file && acceptedTypes.includes(file.type.toLowerCase());
}

function displayImagePreview(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    if (previewImg) previewImg.src = e.target.result;
    if (uploadArea) uploadArea.style.display = "none";
    if (previewContainer) previewContainer.style.display = "block";
  };
  reader.readAsDataURL(file);
}

async function analyzeImage() {
  if (!currentFile) {
    alert("Please select an image first");
    return;
  }

  // UI updates
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.style.display = "none";
  }
  if (loadingSpinner) loadingSpinner.style.display = "block";
  if (progressContainer) progressContainer.style.display = "block";
  if (progressBar) progressBar.style.width = "0%";
  if (progressText) progressText.textContent = "Uploading & Analyzing...";
  if (analysisResults) analysisResults.style.display = "none";

  const formData = new FormData();
  formData.append("file", currentFile, currentFile.name);

  try {
    // Progress update
    if (progressBar) progressBar.style.width = "30%";

    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    // Progress update
    if (progressBar) progressBar.style.width = "70%";
    if (progressText) progressText.textContent = "Processing Results...";

    if (!response.ok) {
      let errorMsg = `API Error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        errorMsg = `API Error (${response.status}): ${errorData.detail || JSON.stringify(errorData)}`;
      } catch (e) {
        console.warn("Could not parse error response");
      }
      throw new Error(errorMsg);
    }

    const predictionData = await response.json();
    console.log("Prediction data received:", predictionData);

    // Progress complete
    if (progressBar) progressBar.style.width = "100%";
    if (progressText) progressText.textContent = "Analysis Complete!";

    setTimeout(() => {
      showResults(predictionData);
    }, 500);

  } catch (error) {
    console.error("Analysis Failed:", error);
    
    if (diagnosisContainer) {
      diagnosisContainer.innerHTML = `
        <div style="padding: 20px; background: #fee; border-radius: 8px; border: 2px solid #fcc;">
          <h3 style="color: #c33; margin: 0 0 10px 0;">⚠️ Analysis Failed</h3>
          <p style="color: #600; margin: 0;"><strong>Error:</strong> ${error.message}</p>
          <p style="color: #600; margin: 10px 0 0 0; font-size: 14px;">
            Please check:
            <ul style="margin: 5px 0; padding-left: 20px;">
              <li>Backend server is running on ${API_ENDPOINT}</li>
              <li>Network connection is stable</li>
              <li>Image file is valid</li>
            </ul>
          </p>
        </div>
      `;
    }
    
    if (analysisResults) analysisResults.style.display = "block";
    if (loadingSpinner) loadingSpinner.style.display = "none";
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.style.display = "inline-flex";
      analyzeBtn.textContent = "Try Again";
    }
  } finally {
    setTimeout(() => {
      if (progressContainer) progressContainer.style.display = "none";
    }, 1500);
  }
}

function showResults(predictionData) {
  const predictedClass = predictionData.predicted_class;
  const apiConfidence = predictionData.confidence;
  
  console.log("Looking for condition:", predictedClass);
  
  // Try to find condition with flexible matching
  let conditionDetails = skinConditions.find(
    (condition) => condition.name.toLowerCase() === predictedClass.toLowerCase()
  );
  
  // If not found, try partial matching
  if (!conditionDetails) {
    conditionDetails = skinConditions.find(
      (condition) => 
        condition.name.toLowerCase().includes(predictedClass.toLowerCase()) ||
        predictedClass.toLowerCase().includes(condition.name.toLowerCase())
    );
  }

  let diagnosisHTML = "";

  if (conditionDetails) {
    const confidencePercentage = (apiConfidence * 100).toFixed(2);
    
    const formatList = (items, icon) =>
      items.map(item => 
        `<li class="treatment-item">
          <span class="treatment-icon">${icon}</span>
          <span>${item}</span>
        </li>`
      ).join("");

    diagnosisHTML = `
      <div class="diagnosis-card">
        <div class="diagnosis-header">
          <div class="diagnosis-name">${conditionDetails.name}</div>
          <div class="confidence-badge ${apiConfidence > 0.8 ? 'high' : apiConfidence > 0.6 ? 'medium' : 'low'}">
            ${confidencePercentage}% Match
          </div>
        </div>
        
        <div class="severity-badge severity-${conditionDetails.severity?.toLowerCase().replace(/\s/g, '-') || 'moderate'}">
          Severity: ${conditionDetails.severity || 'Moderate'}
        </div>
        
        <p class="diagnosis-description">${conditionDetails.description}</p>

        <div class="treatment-section">
          <h4 class="treatment-title">
            <span class="title-icon">💊</span>
            Modern Treatment
          </h4>
          <ul class="treatment-list" id="modern-treatment-list">
            ${formatList(conditionDetails.modernTreatments, "💊")}
          </ul>
          ${conditionDetails.modernTreatmentsAll ? `
          <button class="view-all-btn" onclick="toggleTreatments('modern', ${conditionDetails.id})">
            <span class="btn-icon">👁️</span> View All Treatments
          </button>
          ` : ''}
        </div>

        <div class="treatment-section">
          <h4 class="treatment-title">
            <span class="title-icon">🌿</span>
            Ayurvedic Treatment
          </h4>
          <ul class="treatment-list" id="ayurvedic-treatment-list">
            ${formatList(conditionDetails.ayurvedicTreatments, "🌿")}
          </ul>
          ${conditionDetails.ayurvedicTreatmentsAll ? `
          <button class="view-all-btn" onclick="toggleTreatments('ayurvedic', ${conditionDetails.id})">
            <span class="btn-icon">👁️</span> View All Treatments
          </button>
          ` : ''}
        </div>

        <div class="important-notice">
          <strong>⚠️ Important:</strong> This is an AI-assisted analysis and should not replace professional medical advice. 
          Please consult a qualified dermatologist for accurate diagnosis and personalized treatment.
        </div>

        <div class="file-info">
          <small>📁 Analyzed File: ${predictionData.filename || currentFile?.name || "N/A"}</small>
        </div>

        <div class="action-buttons">
          <button class="action-btn primary-btn" onclick="analyzeNewImage()">
            <span class="btn-icon">📷</span> Analyze New Image
          </button>
          <button class="action-btn secondary-btn" onclick="resetAnalysis()">
            <span class="btn-icon">🔄</span> Try Again
          </button>
        </div>
      </div>
    `;
  } else {
    const confidencePercentage = (apiConfidence * 100).toFixed(2);
    diagnosisHTML = `
      <div class="diagnosis-card">
        <div class="diagnosis-header">
          <div class="diagnosis-name">${predictedClass}</div>
          <div class="confidence-badge ${apiConfidence > 0.8 ? 'high' : apiConfidence > 0.6 ? 'medium' : 'low'}">
            ${confidencePercentage}% Match
          </div>
        </div>
        
        <div class="warning-message">
          <p style="color: #f59e0b; font-weight: 600;">
            ⚠️ Condition Detected: "${predictedClass}"
          </p>
          <p style="color: #78716c;">
            Detailed treatment information for this condition is not available in our current database.
            However, we detected this condition with ${confidencePercentage}% confidence.
          </p>
        </div>

        <div class="important-notice">
          <strong>🏥 Next Steps:</strong>
          <ul style="margin: 10px 0; padding-left: 20px;">
            <li>Consult a qualified dermatologist immediately for accurate diagnosis</li>
            <li>Bring this analysis result to your appointment</li>
            <li>Do not attempt self-treatment without professional guidance</li>
            <li>Monitor the condition for any changes</li>
          </ul>
        </div>

        <div class="file-info">
          <small>📁 Analyzed File: ${predictionData.filename || currentFile?.name || "N/A"}</small>
        </div>

        <div class="action-buttons">
          <button class="action-btn primary-btn" onclick="analyzeNewImage()">
            <span class="btn-icon">📷</span> Analyze New Image
          </button>
          <button class="action-btn secondary-btn" onclick="resetAnalysis()">
            <span class="btn-icon">🔄</span> Try Again
          </button>
        </div>
      </div>
    `;
  }

  if (diagnosisContainer) {
    diagnosisContainer.innerHTML = diagnosisHTML;
  }
  if (analysisResults) {
    analysisResults.style.display = "block";
    analysisResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  if (loadingSpinner) loadingSpinner.style.display = "none";
  if (analyzeBtn) analyzeBtn.style.display = "none";
}

// Toggle treatment visibility
function toggleTreatments(type, conditionId) {
  const condition = skinConditions.find(c => c.id === conditionId);
  if (!condition) return;

  const listId = type === 'modern' ? 'modern-treatment-list' : 'ayurvedic-treatment-list';
  const list = document.getElementById(listId);
  const btn = event.target.closest('.view-all-btn');
  
  const treatments = type === 'modern' ? condition.modernTreatments : condition.ayurvedicTreatments;
  const allTreatments = type === 'modern' ? condition.modernTreatmentsAll : condition.ayurvedicTreatmentsAll;
  
  const icon = type === 'modern' ? '💊' : '🌿';
  const isExpanded = btn.classList.contains('expanded');
  
  if (isExpanded) {
    // Show fewer treatments
    list.innerHTML = treatments.map(item => 
      `<li class="treatment-item">
        <span class="treatment-icon">${icon}</span>
        <span>${item}</span>
      </li>`
    ).join("");
    btn.innerHTML = '<span class="btn-icon">👁️</span> View All Treatments';
    btn.classList.remove('expanded');
  } else {
    // Show all treatments
    list.innerHTML = allTreatments.map(item => 
      `<li class="treatment-item">
        <span class="treatment-icon">${icon}</span>
        <span>${item}</span>
      </li>`
    ).join("");
    btn.innerHTML = '<span class="btn-icon">👁️</span> Show Less';
    btn.classList.add('expanded');
  }
}

// Analyze new image function
function analyzeNewImage() {
  // Reset the interface for new image
  currentFile = null;
  if (fileInput) fileInput.value = "";
  if (previewImg) previewImg.src = "";
  if (previewContainer) previewContainer.style.display = "none";
  if (uploadArea) uploadArea.style.display = "block";
  if (analysisResults) analysisResults.style.display = "none";
  if (analyzeBtn) {
    analyzeBtn.style.display = "none";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
  }
  
  // Scroll to upload area
  if (uploadArea) {
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// Reset current analysis (try again with same image)
function resetAnalysis() {
  if (analysisResults) analysisResults.style.display = "none";
  if (analyzeBtn) {
    analyzeBtn.style.display = "inline-flex";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
  }
  
  // Scroll to image preview
  if (previewContainer) {
    previewContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const targetId = this.getAttribute("href");
    const targetElement = document.querySelector(targetId);
    if (targetElement) {
      const offsetTop = targetElement.offsetTop - 80;
      window.scrollTo({
        top: offsetTop,
        behavior: "smooth",
      });
    }
  });
});

// Form submission
const contactForm = document.getElementById("contactForm");
if (contactForm) {
  contactForm.addEventListener("submit", function (e) {
    e.preventDefault();
    alert("Thank you for your message! We will get back to you soon.");
    this.reset();
  });
}