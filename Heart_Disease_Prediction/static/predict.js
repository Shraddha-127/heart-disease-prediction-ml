// =====================================================================
// predict.js — Prediction page interactions
// (loading overlay sequence, circular progress fill, SHAP bar animation)
// =====================================================================

document.addEventListener("DOMContentLoaded", function () {

    // ---------------- Loading Overlay on Submit ----------------
    const form = document.getElementById("predictForm");
    const overlay = document.getElementById("loadingOverlay");
    const loadingText = document.getElementById("loadingText");

    const loadingSteps = [
        "Analyzing patient data...",
        "Applying Machine Learning...",
        "Generating SHAP explanation...",
        "Preparing recommendations..."
    ];

    if (form) {
        form.addEventListener("submit", function (e) {

            const inputs = form.querySelectorAll("input[required], select[required]");
            for (let input of inputs) {
                if (input.value.trim() === "") {
                    alert("Please fill all required fields.");
                    input.focus();
                    e.preventDefault();
                    return;
                }
            }

            if (overlay) {
                overlay.classList.add("active");
                let step = 0;
                loadingText.textContent = loadingSteps[0];
                const interval = setInterval(() => {
                    step++;
                    if (step < loadingSteps.length) {
                        loadingText.textContent = loadingSteps[step];
                    } else {
                        clearInterval(interval);
                    }
                }, 700);
            }
        });
    }

    // ---------------- Animate Circular Confidence Progress ----------------
    const circleFill = document.querySelector(".circle-fill");
    const progressValueEl = document.querySelector(".progress-value");

    if (circleFill && progressValueEl) {
        const target = parseFloat(progressValueEl.getAttribute("data-target")) || 0;
        const radius = 52;
        const circumference = 2 * Math.PI * radius;

        circleFill.style.strokeDasharray = circumference;
        circleFill.style.strokeDashoffset = circumference;

        setTimeout(() => {
            const offset = circumference - (target / 100) * circumference;
            circleFill.style.strokeDashoffset = offset;

            // Animate the number counting up alongside the ring
            let current = 0;
            const step = target / 40;
            const counterInterval = setInterval(() => {
                current += step;
                if (current >= target) {
                    progressValueEl.textContent = target;
                    clearInterval(counterInterval);
                } else {
                    progressValueEl.textContent = current.toFixed(0);
                }
            }, 25);
        }, 300);
    }

    // ---------------- Animate SHAP Bars ----------------
    const shapBars = document.querySelectorAll(".shap-bar-fill");
    shapBars.forEach((bar, index) => {
        const pct = parseFloat(bar.getAttribute("data-pct")) || 0;
        setTimeout(() => {
            bar.style.width = pct + "%";
        }, 400 + index * 150);
    });

    // ---------------- Scroll to Results ----------------
    const resultsWrap = document.querySelector(".results-wrap");
    if (resultsWrap) {
        setTimeout(() => {
            resultsWrap.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 200);
    }

});
