// =====================================================================
// auth.js — Login & Registration page interactions
// =====================================================================

document.addEventListener("DOMContentLoaded", function () {

    // ---------------- Show / Hide Password ----------------
    document.querySelectorAll(".toggle-password").forEach((btn) => {
        btn.addEventListener("click", function () {
            const targetId = btn.getAttribute("data-target");
            const input = document.getElementById(targetId);
            const icon = btn.querySelector("i");

            if (!input) return;

            if (input.type === "password") {
                input.type = "text";
                icon.classList.replace("fa-eye", "fa-eye-slash");
            } else {
                input.type = "password";
                icon.classList.replace("fa-eye-slash", "fa-eye");
            }
        });
    });

    // ---------------- Password Strength Meter (Register page) ----------------
    const regPassword = document.getElementById("reg_password");
    const strengthFill = document.getElementById("strengthFill");
    const strengthText = document.getElementById("strengthText");

    function evaluateStrength(password) {
        let score = 0;
        if (password.length >= 6) score++;
        if (password.length >= 10) score++;
        if (/[A-Z]/.test(password)) score++;
        if (/[0-9]/.test(password)) score++;
        if (/[^A-Za-z0-9]/.test(password)) score++;
        return score;
    }

    if (regPassword && strengthFill) {
        regPassword.addEventListener("input", function () {
            const score = evaluateStrength(regPassword.value);
            const percentages = [10, 25, 45, 65, 85, 100];
            const colorsList = ["#ef4444", "#ef4444", "#f59e0b", "#f59e0b", "#22c55e", "#16a34a"];
            const labels = ["Very Weak", "Very Weak", "Weak", "Fair", "Strong", "Very Strong"];

            strengthFill.style.width = percentages[score] + "%";
            strengthFill.style.background = colorsList[score];
            strengthText.textContent = regPassword.value ? labels[score] : "Password strength";
        });
    }

    // ---------------- Confirm Password Match (Register page) ----------------
    const confirmPassword = document.getElementById("confirm_password");
    const matchError = document.getElementById("matchError");
    const registerForm = document.getElementById("registerForm");

    function checkMatch() {
        if (!regPassword || !confirmPassword || !matchError) return true;

        if (confirmPassword.value && regPassword.value !== confirmPassword.value) {
            matchError.textContent = "Passwords do not match.";
            return false;
        }
        matchError.textContent = "";
        return true;
    }

    if (confirmPassword) {
        confirmPassword.addEventListener("input", checkMatch);
    }
    if (regPassword) {
        regPassword.addEventListener("input", checkMatch);
    }

    // ---------------- Client-side validation on submit ----------------
    if (registerForm) {
        registerForm.addEventListener("submit", function (e) {
            const email = document.getElementById("reg_email");
            const mobile = document.getElementById("mobile");

            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            const mobilePattern = /^[6-9]\d{9}$/;

            let valid = true;

            if (email && !emailPattern.test(email.value.trim())) {
                alert("Please enter a valid email address.");
                valid = false;
            } else if (mobile && !mobilePattern.test(mobile.value.trim())) {
                alert("Please enter a valid 10-digit mobile number.");
                valid = false;
            } else if (regPassword && regPassword.value.length < 6) {
                alert("Password must be at least 6 characters long.");
                valid = false;
            } else if (!checkMatch()) {
                alert("Passwords do not match.");
                valid = false;
            }

            if (!valid) e.preventDefault();
        });
    }

    // Restrict mobile field to digits only
    const mobileField = document.getElementById("mobile");
    if (mobileField) {
        mobileField.addEventListener("input", function () {
            mobileField.value = mobileField.value.replace(/\D/g, "").slice(0, 10);
        });
    }

});
