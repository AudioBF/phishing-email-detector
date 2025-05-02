// Theme toggle functionality
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const themeToggle = document.getElementById('theme-toggle');
    const isDark = document.body.classList.contains('dark-theme');
    themeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Load saved theme preference
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        document.getElementById('theme-toggle').innerHTML = '<i class="fas fa-sun"></i>';
    }
    loadEmailExamples();
    
    // Add event listeners after DOM is loaded
    const themeToggle = document.getElementById('theme-toggle');
    const emailText = document.getElementById('email-text');
    const checkButton = document.getElementById('check-button');
    
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    if (emailText) emailText.addEventListener('input', updateWordCount);
    if (checkButton) checkButton.addEventListener('click', checkEmail);
});

// Load email examples
async function loadEmailExamples() {
    try {
        const response = await fetch('/samples');
        if (!response.ok) {
            throw new Error('Failed to load examples');
        }
        const data = await response.json();
        const selector = document.getElementById('email-examples');
        
        if (!selector) {
            console.error('Email examples selector not found');
            return;
        }
        
        // Clear existing options except the first one
        while (selector.options.length > 1) {
            selector.remove(1);
        }
        
        // Add examples
        data.emails.forEach((email, index) => {
            const option = document.createElement('option');
            option.value = `${email.type}_${index}`;
            option.textContent = `${email.type.toUpperCase()} Example ${index + 1}`;
            selector.appendChild(option);
        });

        selector.addEventListener('change', (e) => {
            if (e.target.value !== '') {
                const [type, index] = e.target.value.split('_');
                const emailData = data.emails[parseInt(index)];
                document.getElementById('email-text').value = emailData.email;
                updateWordCount();
            }
        });
    } catch (error) {
        console.error('Error loading email examples:', error);
        showNotification('Error loading email examples', 'error');
    }
}

// Update word count
function updateWordCount() {
    const text = document.getElementById('email-text').value;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    document.getElementById('word-count').textContent = `${wordCount} words`;
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    if (!notification) {
        console.error('Notification element not found');
        return;
    }
    
    notification.textContent = message;
    notification.className = `notification show ${type}`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Check email
async function checkEmail() {
    const emailText = document.getElementById('email-text');
    if (!emailText) {
        console.error('Email text input not found');
        return;
    }
    
    const text = emailText.value.trim();
    if (!text) {
        showNotification('Please enter an email to check', 'error');
        return;
    }
    
    const resultBox = document.getElementById('result-box');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const probability = document.getElementById('probability');
    
    if (!resultBox || !classification || !confidence || !probability) {
        console.error('Result elements not found');
        return;
    }
    
    try {
        // Show loading state
        resultBox.style.display = 'block';
        classification.textContent = 'Analyzing...';
        confidence.textContent = '-';
        probability.textContent = '-';
        
        // Make API call
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });
        
        if (!response.ok) throw new Error('Failed to analyze email');
        
        const result = await response.json();
        
        // Update UI with results
        classification.textContent = result.is_spam ? 'SPAM' : 'HAM';
        confidence.textContent = `${result.confidence}%`;
        probability.textContent = `${result.probability}%`;
        
        // Update result box class
        resultBox.className = `result-box ${result.is_spam ? 'spam' : 'ham'}`;
        
        showNotification('Analysis completed successfully', 'success');
        
    } catch (error) {
        console.error('Error analyzing email:', error);
        classification.textContent = 'ERROR';
        confidence.textContent = '-';
        probability.textContent = '-';
        showNotification('Failed to analyze email', 'error');
    }
}
