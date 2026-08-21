// =====================================================================
// script.js — Global behaviors shared across authenticated pages
// (navbar toggle, animated counters, ripple buttons, flash auto-dismiss)
// =====================================================================

document.addEventListener("DOMContentLoaded", function () {

    // ---------------- Mobile Nav Toggle ----------------
    const navToggle = document.getElementById("navToggle");
    const navLinks = document.getElementById("navLinks");

    if (navToggle && navLinks) {
        navToggle.addEventListener("click", function () {
            navLinks.classList.toggle("open");
        });
    }

    // ---------------- Animated Counters (dashboard stats) ----------------
    const counters = document.querySelectorAll(".counter");

    counters.forEach((counter) => {
        const target = parseFloat(counter.getAttribute("data-target")) || 0;
        const isDecimal = counter.getAttribute("data-decimal") === "true";
        let current = 0;
        const steps = 40;
        const increment = target / steps;

        const update = () => {
            current += increment;
            if (current >= target) {
                counter.textContent = isDecimal ? target.toFixed(2) : Math.round(target);
                return;
            }
            counter.textContent = isDecimal ? current.toFixed(2) : Math.round(current);
            requestAnimationFrame(update);
        };

        setTimeout(update, 150);
    });

    // ---------------- Ripple Effect on Buttons ----------------
    document.querySelectorAll(".ripple").forEach((btn) => {
        btn.addEventListener("click", function (e) {
            const circle = document.createElement("span");
            circle.classList.add("ripple-effect");

            const rect = btn.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);

            circle.style.width = circle.style.height = size + "px";
            circle.style.left = (e.clientX - rect.left - size / 2) + "px";
            circle.style.top = (e.clientY - rect.top - size / 2) + "px";

            btn.appendChild(circle);

            setTimeout(() => circle.remove(), 600);
        });
    });

    // ---------------- Auto-dismiss Flash Messages ----------------
    document.querySelectorAll(".flash-msg").forEach((msg) => {
        setTimeout(() => {
            msg.style.transition = "opacity .4s ease";
            msg.style.opacity = "0";
            setTimeout(() => msg.remove(), 400);
        }, 5000);
    });

});

// =====================================================================
// AI Health Chatbot Widget (Ollama / Llama 3)
// =====================================================================
// Talks to the Flask backend at /api/chat and /api/chat/status
// (see app.py + chatbot.py). Conversation history is kept only in
// this in-memory array on the client - nothing chat-related is
// stored server-side, so it resets on page reload/navigation.
// =====================================================================

document.addEventListener("DOMContentLoaded", function () {

    const widget = document.getElementById("chatbotWidget");
    const toggleBtn = document.getElementById("chatbotToggleBtn");
    const closeBtn = document.getElementById("chatbotCloseBtn");
    const messagesBox = document.getElementById("chatbotMessages");
    const statusBox = document.getElementById("chatbotStatus");
    const form = document.getElementById("chatbotForm");
    const input = document.getElementById("chatbotInput");
    const sendBtn = document.getElementById("chatbotSendBtn");
    const explainBtn = document.getElementById("chatbotExplainBtn");

    if (!widget || !form) {
        // Chat widget isn't on this page (e.g. login/register) - nothing to wire up.
        return;
    }

    // Full conversation history for this page session, sent with every
    // request so the assistant has conversational context.
    let chatHistory = [];
    let isSending = false;

    // -----------------------------------------------------
    // Open / close the panel
    // -----------------------------------------------------

    toggleBtn.addEventListener("click", function () {
        widget.classList.toggle("open");

        if (widget.classList.contains("open")) {
            checkOllamaStatus();
            setTimeout(() => input.focus(), 150);
        }
    });

    if (closeBtn) {
        closeBtn.addEventListener("click", function () {
            widget.classList.remove("open");
        });
    }

    // -----------------------------------------------------
    // Check whether Ollama is reachable, show a banner if not
    // -----------------------------------------------------

    function checkOllamaStatus() {

        if (!statusBox) return;

        fetch("/api/chat/status")
            .then(res => res.json())
            .then(data => {

                if (data.available) {
                    statusBox.className = "chatbot-status";
                    statusBox.textContent = "";
                } else {
                    statusBox.className = "chatbot-status chatbot-status-offline";
                    statusBox.textContent =
                        "AI assistant offline - start Ollama locally (ollama serve) to chat.";
                }

            })
            .catch(() => {

                statusBox.className = "chatbot-status chatbot-status-offline";
                statusBox.textContent = "Could not check AI assistant status.";

            });

    }

    // -----------------------------------------------------
    // Render a message bubble
    // -----------------------------------------------------

    function addMessage(text, sender) {

        const bubble = document.createElement("div");
        bubble.className = "chatbot-msg " + (sender === "user" ? "chatbot-msg-user" : "chatbot-msg-bot");
        bubble.textContent = text;

        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;

        return bubble;

    }

    function addTypingIndicator() {

        const bubble = document.createElement("div");
        bubble.className = "chatbot-msg chatbot-msg-typing";
        bubble.id = "chatbotTypingIndicator";
        bubble.textContent = "CardioAI is typing...";

        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;

    }

    function removeTypingIndicator() {

        const el = document.getElementById("chatbotTypingIndicator");
        if (el) el.remove();

    }

    // -----------------------------------------------------
    // Send a message to the backend (STREAMED response - the
    // reply is rendered token-by-token as Ollama generates it,
    // instead of waiting for the entire message to finish)
    // -----------------------------------------------------

    function sendMessage(text) {

        const trimmed = (text || "").trim();
        if (!trimmed || isSending) return;

        addMessage(trimmed, "user");
        chatHistory.push({ role: "user", content: trimmed });

        input.value = "";
        isSending = true;
        sendBtn.disabled = true;
        addTypingIndicator();

        fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: trimmed,
                // Send history WITHOUT the message we just added locally,
                // since the backend appends the latest message itself.
                history: chatHistory.slice(0, -1)
            })
        })
            .then(res => {

                const contentType = res.headers.get("content-type") || "";

                // The backend returns JSON only for the "couldn't even
                // start talking to Ollama" case (offline, model missing).
                // A successful reply comes back as a streamed text body.
                if (contentType.includes("application/json")) {
                    return res.json().then(data => {
                        removeTypingIndicator();
                        addMessage(data.error || "Something went wrong. Please try again.", "bot");
                    });
                }

                removeTypingIndicator();

                const botBubble = addMessage("", "bot");
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";

                function readNextChunk() {
                    return reader.read().then(({ done, value }) => {

                        if (done) {
                            if (fullText) {
                                chatHistory.push({ role: "assistant", content: fullText });
                            }
                            return;
                        }

                        fullText += decoder.decode(value, { stream: true });
                        botBubble.textContent = fullText;
                        messagesBox.scrollTop = messagesBox.scrollHeight;

                        return readNextChunk();

                    });
                }

                return readNextChunk();

            })
            .catch(() => {

                removeTypingIndicator();
                addMessage("Could not reach the server. Please check your connection and try again.", "bot");

            })
            .finally(() => {

                isSending = false;
                sendBtn.disabled = false;

            });

    }

    // -----------------------------------------------------
    // Form submit (send button / Enter key)
    // -----------------------------------------------------

    form.addEventListener("submit", function (e) {
        e.preventDefault();
        sendMessage(input.value);
    });

    // -----------------------------------------------------
    // "Explain my latest result" quick action - only present
    // when the user has at least one saved prediction
    // -----------------------------------------------------

    if (explainBtn) {

        explainBtn.addEventListener("click", function () {

            widget.classList.add("open");
            checkOllamaStatus();
            sendMessage("Can you explain my heart disease prediction result in simple terms?");

        });

    }

});
