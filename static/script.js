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
    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });
    
    function updateThemeIcon(theme) {
        const icon = themeToggle.querySelector('i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
    
    // Email examples functionality
    const emailExamples = document.getElementById('email-examples');
    const emailText = document.getElementById('email-text');
    const wordCount = document.getElementById('word-count');
    
    // Load email examples
    fetch('/samples')
        .then(response => response.json())
        .then(data => {
            data.emails.forEach(email => {
                const option = document.createElement('option');
                option.value = email.text;
                option.textContent = email.title;
                emailExamples.appendChild(option);
            });
        })
        .catch(error => console.error('Error loading email examples:', error));
    
    // Handle email example selection
    emailExamples.addEventListener('change', () => {
        if (emailExamples.value) {
            emailText.value = emailExamples.value;
            updateWordCount();
        }
    });
    
    // Word count functionality
    emailText.addEventListener('input', updateWordCount);
    
    function updateWordCount() {
        const text = emailText.value.trim();
        const count = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = `${count} words`;
    }
    
    // Email analysis functionality
    const checkButton = document.getElementById('check-button');
    const resultBox = document.getElementById('result-box');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const probability = document.getElementById('probability');
    const notification = document.getElementById('notification');
    
    checkButton.addEventListener('click', async () => {
        const text = emailText.value.trim();
        
        if (!text) {
            showNotification('Please enter an email to analyze', 'error');
            return;
        }
        
        try {
            checkButton.disabled = true;
            checkButton.textContent = 'Analyzing...';
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_text: text })
            });
            
            const data = await response.json();
            
            // Update result display
            resultBox.style.display = 'block';
            resultBox.className = `result-box ${data.is_spam ? 'spam' : 'ham'}`;
            
            classification.textContent = data.is_spam ? 'Spam' : 'Ham';
            classification.className = `value ${data.is_spam ? 'spam' : 'ham'}`;
            
            confidence.textContent = `${Math.round(data.confidence * 100)}%`;
            probability.textContent = `${Math.round(data.probability * 100)}%`;
            
            showNotification('Analysis complete!', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            showNotification('An error occurred while analyzing the email', 'error');
        } finally {
            checkButton.disabled = false;
            checkButton.textContent = 'Check Email';
        }
    });
    
    function showNotification(message, type) {
        notification.textContent = message;
        notification.className = `notification ${type}`;
        notification.style.display = 'block';
        
        setTimeout(() => {
            notification.style.display = 'none';
        }, 3000);
    }
});
