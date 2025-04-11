// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    console.log("PayPal AI FX Optimizer Initialized - Enhanced AI/Savings Version");

    // --- DOM Elements ---
    const amountInput = document.getElementById('amount');
    const fromCurrencySelect = document.getElementById('from-currency');
    const toCurrencySelect = document.getElementById('to-currency');
    const currentRateValueElement = document.getElementById('current-rate-value');
    const recipientAmountDisplayElement = document.getElementById('recipient-amount-display');
    const loadingIndicator = document.getElementById('loading-indicator');
    const aiInfoButton = document.getElementById('ai-info-button');
    const recommendationBox = document.getElementById('recommendation-box');
    const savingsInfoElement = document.getElementById('savings-info'); // ADDED
    const recIconElement = document.getElementById('rec-icon');
    const recMessageElement = document.getElementById('recommendation-message');
    const recommendationDetailsDiv = document.getElementById('recommendation-details');
    const recDetailTrend = document.getElementById('rec-detail-trend');
    const recDetailFund = document.getElementById('rec-detail-fund');
    const recDetailNews = document.getElementById('rec-detail-news');
    const nextButton = document.getElementById('next-button');
    const cancelButton = document.getElementById('cancel-button');

    let currentRate = null; // Store the precise rate fetched from API
    let currentPair = null; // Store the pair for which rate/prediction was fetched
    let lastPredictionData = null; // Store the full prediction object
    let fetchTimeoutId = null; // For debouncing API calls
    const DEBOUNCE_DELAY = 700; // Slightly longer delay

    // --- Event Listeners ---
    amountInput.addEventListener('input', handleInputChange);
    fromCurrencySelect.addEventListener('change', handleInputChange);
    toCurrencySelect.addEventListener('change', handleInputChange);
    aiInfoButton.addEventListener('click', toggleRecommendation);
    nextButton.addEventListener('click', handleSendNow);
    cancelButton.addEventListener('click', handleMaybeLater);

    // --- Helper Functions ---
    function debounce(func, delay) { /* ... (debounce function as before) ... */
        return function(...args) {
            clearTimeout(fetchTimeoutId);
            fetchTimeoutId = setTimeout(() => { func.apply(this, args); }, delay);
        };
    }
    function showLoading(isLoading) { /* ... (showLoading function as before) ... */
        loadingIndicator.style.display = isLoading ? 'inline-block' : 'none';
    }
    function formatRateText(rate, pair) { /* ... (formatRateText function as before) ... */
        if (!rate || !pair || pair.length !== 6) {
            const fromCcy = fromCurrencySelect.value; const toCcy = toCurrencySelect.value;
            return (fromCcy && toCcy && fromCcy !== toCcy) ? `1 ${fromCcy} = --.-- ${toCcy}` : "Select currencies"; }
        const baseCcy = pair.substring(0,3); const quoteCcy = pair.substring(3);
        let precision = 4; if (quoteCcy === "JPY") precision = 2; if (pair in ["EURUSD", "GBPUSD", "EURGBP"]) precision = 5;
        return `1 ${baseCcy} = ${rate.toFixed(precision)} ${quoteCcy}`;
    }
    function formatRecipientAmount(amount, currency) { /* ... (formatRecipientAmount function as before) ... */
        if (amount === null || isNaN(amount)) return "--.--";
        let options = { minimumFractionDigits: 2, maximumFractionDigits: 2 }; if (currency === "JPY") options = { minimumFractionDigits: 0, maximumFractionDigits: 0 };
        try { return amount.toLocaleString('en-US', options); } catch(e) { return amount.toFixed(options.minimumFractionDigits); }
    }
    function resetRateAndRecommendation() { /* ... (resetRateAndRecommendation function as before) ... */
        recipientAmountDisplayElement.textContent = '--.--';
        currentRateValueElement.textContent = formatRateText(null, null);
        recommendationBox.style.display = 'none';
        savingsInfoElement.style.display = 'none'; // Hide savings too
        aiInfoButton.disabled = true;
        currentRate = null; currentPair = null; lastPredictionData = null;
    }

    // Main function to fetch data and update UI (debounced)
    const debouncedFetch = debounce(() => {
        const amountStr = amountInput.value.trim(); // Get amount as string
        const amount = parseFloat(amountStr); // Parse amount
        const fromCurrency = fromCurrencySelect.value;
        const toCurrency = toCurrencySelect.value;

        if (fromCurrency === toCurrency || !fromCurrency || !toCurrency) {
             resetRateAndRecommendation(); return;
        }
        // Basic check if amount is valid number for API call (can be 0)
        if (amountStr === "" || isNaN(amount)) {
             // We can still fetch the rate even if amount is 0 or invalid, but won't calculate savings
             // Modify reset function? Or handle in API response? Let's reset savings display
             resetRateAndRecommendation(); // Reset fully if amount invalid for now
             currentRateValueElement.textContent = formatRateText(null, null); // Ensure placeholder shown
             return;
             // Alternative: Fetch rate anyway, but savings calc needs amount > 0
        }

        const requestedPair = `${fromCurrency}${toCurrency}`;
        currentPair = requestedPair; // Store the pair we are requesting

        showLoading(true);
        aiInfoButton.disabled = true; // Disable while fetching
        recommendationBox.style.display = 'none'; // Hide old recommendation immediately
        savingsInfoElement.style.display = 'none'; // Hide old savings

        // **MODIFIED API CALL to include amount**
        fetch(`/api/fx_prediction/${requestedPair}/${amount}`)
            .then(response => { /* ... (error handling as before) ... */
                if (!response.ok) { return response.json().then(err => { throw new Error(err.error || `Server error: ${response.status}`); }).catch(() => { throw new Error(`Server error: ${response.status} ${response.statusText}`); }); }
                return response.json();
            })
            .then(prediction => { /* ... (error check as before) ... */
                if (prediction.error) { throw new Error(prediction.error); }

                lastPredictionData = prediction; // Store the full response
                currentRate = prediction.current_rate; // Store the rate (display rate from API)

                // Update UI immediately with rate and recipient amount
                updateRateDisplay(prediction);
                updateRecommendationContent(prediction); // Populate hidden recommendation box + savings

                aiInfoButton.disabled = false; // Enable info button
            })
            .catch(error => { /* ... (error handling as before) ... */
                console.error("Error fetching prediction:", error);
                resetRateAndRecommendation();
                currentRateValueElement.textContent = "Error loading rate";
            })
            .finally(() => { showLoading(false); });
    }, DEBOUNCE_DELAY);

    // Called after successful fetch to update rate and recipient amount
    function updateRateDisplay(prediction) { /* ... (updateRateDisplay function as before) ... */
         const senderAmount = parseFloat(amountInput.value) || 0;
         const rate = prediction.current_rate; // Use display rate from API
         const pair = prediction.pair;
         const toCurrency = pair.substring(3);
         const recipientAmount = (rate && senderAmount > 0) ? senderAmount * rate : null;
         recipientAmountDisplayElement.textContent = formatRecipientAmount(recipientAmount, toCurrency);
         currentRateValueElement.textContent = formatRateText(rate, pair);
    }

    // Called after successful fetch to populate the (hidden) recommendation box content AND Savings
    function updateRecommendationContent(prediction) {
        recMessageElement.textContent = prediction.message || "Recommendation not available.";

        // Update Savings Info display
        if (prediction.savings && prediction.savings.message) {
             savingsInfoElement.textContent = prediction.savings.message;
             savingsInfoElement.style.display = 'block'; // Show savings
        } else {
             savingsInfoElement.style.display = 'none'; // Hide savings
             savingsInfoElement.textContent = ''; // Clear content
        }

        // Update recommendation details and icon/style
        recommendationBox.className = 'recommendation-box'; // Reset classes before adding new one
        let iconClass = 'fas fa-info-circle';

        if (prediction.details && Object.keys(prediction.details).length > 0) {
             recDetailTrend.textContent = prediction.details.trend_summary || "";
             recDetailFund.textContent = prediction.details.fundamental_summary || "";
             recDetailNews.textContent = prediction.details.news_summary || "";
             const hasDetails = recDetailTrend.textContent || recDetailFund.textContent || recDetailNews.textContent;
             recommendationDetailsDiv.style.display = hasDetails ? 'block' : 'none';
        } else {
             recommendationDetailsDiv.style.display = 'none';
        }

        switch (prediction.recommendation) {
            case 'wait': iconClass = 'fas fa-hourglass-half'; recommendationBox.classList.add('wait'); break;
            case 'send_now': iconClass = 'fas fa-rocket'; recommendationBox.classList.add('send_now'); break; // Add class for savings style
            case 'stable': iconClass = 'fas fa-check-circle'; recommendationBox.classList.add('stable'); break;
            case 'uncertain': default: iconClass = 'fas fa-compass'; recommendationBox.classList.add('uncertain'); break;
        }
        recIconElement.className = iconClass;
        // Recommendation box itself is still hidden here, toggled by info icon
    }

    // Triggered by amount/currency changes
    function handleInputChange() { /* ... (handleInputChange function mostly as before, calls debouncedFetch) ... */
        const fromCurrency = fromCurrencySelect.value; const toCurrency = toCurrencySelect.value;
        if (fromCurrency === toCurrency || !fromCurrency || !toCurrency) { resetRateAndRecommendation(); return; }
        // Basic amount validation moved to debouncedFetch
        debouncedFetch();
    }

    // Toggles the visibility of the recommendation box
    function toggleRecommendation() { /* ... (toggleRecommendation function as before) ... */
        if (lastPredictionData) {
            const isHidden = recommendationBox.style.display === 'none' || recommendationBox.style.display === '';
            recommendationBox.style.display = isHidden ? 'block' : 'none';
        } else { console.log("No prediction data available to display."); }
    }

    // Mapped to the "Next" button
    function handleSendNow() { /* ... (handleSendNow function mostly as before, uses currentRate/currentPair) ... */
        const amount = parseFloat(amountInput.value); const fromCcy = fromCurrencySelect.value; const toCcy = toCurrencySelect.value;
        if (isNaN(amount) || amount <= 0) { alert("Please enter a valid amount to send."); return; }
        const selectedPair = `${fromCcy}${toCcy}`;
        if (currentRate === null || currentPair !== selectedPair) { alert("Please wait for the rate to load for the selected amount and currencies."); return; }
        const recipientAmount = amount * currentRate; // Use stored rate
        const recipientAmountFormatted = formatRecipientAmount(recipientAmount, toCcy);
        const rateFormatted = formatRateText(currentRate, currentPair);
        let aiMessage = ""; let savingsMessage = "";
        if (recommendationBox.style.display === 'block' && lastPredictionData) { // Check if box is visible
            aiMessage = `\nAI Advice: ${lastPredictionData.recommendation}`; // Shorter message
            if (lastPredictionData.savings && lastPredictionData.savings.message) {
                 savingsMessage = `\nPotential Savings: ${lastPredictionData.savings.message}`;
            }
        }
        alert( `Confirm Transfer (Simulated):\n` + `--------------------------\n` + `Sending: ${amount.toFixed(2)} ${fromCcy}\n` + `Recipient Gets (Approx.): ${recipientAmountFormatted} ${toCcy}\n` + `Using Rate: ${rateFormatted}` + `${aiMessage}${savingsMessage}\n` + `--------------------------\n` + `(This is a prototype - no real money will be sent)` );
    }

    // Mapped to the "Cancel" button
    function handleMaybeLater() { /* ... (handleMaybeLater function as before) ... */
        console.log("Cancel clicked"); resetRateAndRecommendation(); amountInput.value = "100.00";
    }

    // --- Initial Setup ---
    handleInputChange(); // Trigger initial fetch on page load

}); // End DOMContentLoaded