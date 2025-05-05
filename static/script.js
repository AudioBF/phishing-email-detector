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
            console.log('Loaded email samples:', data);
            
            // Limpar opções existentes
            emailExamples.innerHTML = '<option value="">Choose an example...</option>';
            
            // Adicionar novas opções
            data.emails.forEach(email => {
                const option = document.createElement('option');
                option.dataset.email = email.email;  // Armazenar o email como um atributo de dados
                option.dataset.type = email.type;    // Armazenar o tipo como um atributo de dados
                option.value = email.title;  // Usar o título como valor visível
                option.textContent = `${email.title} (${email.type.toUpperCase()})`;  // Mostrar título e tipo
                emailExamples.appendChild(option);
            });
            
            // Adicionar evento de mudança após carregar as opções
            emailExamples.addEventListener('change', () => {
                const selectedOption = emailExamples.options[emailExamples.selectedIndex];
                const emailContent = selectedOption.dataset.email || '';
                const emailType = selectedOption.dataset.type || '';
                console.log('Selected email content:', emailContent);
                console.log('Selected email type:', emailType);
                
                emailText.value = emailContent;
                updateWordCount();
            });
        })
        .catch(error => {
            console.error('Error loading email examples:', error);
            showNotification('Error loading email examples', 'error');
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
    const expectedType = document.getElementById('expected-type');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const probability = document.getElementById('probability');
    const notification = document.getElementById('notification');
    
    checkButton.addEventListener('click', async () => {
        const text = emailText.value.trim();
        const selectedOption = emailExamples.options[emailExamples.selectedIndex];
        const emailType = selectedOption ? selectedOption.dataset.type : '';
        
        if (!text) {
            showNotification('Please enter an email to analyze', 'error');
            return;
        }
        
        try {
            checkButton.disabled = true;
            checkButton.textContent = 'Analyzing...';
            
            // Atualizar o tipo esperado
            expectedType.textContent = emailType ? emailType.toUpperCase() : '-';
            expectedType.className = `value ${emailType}`;
            
            console.log('Sending request with text:', text);
            const response = await fetch('/email/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_text: text })
            });
            
            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Response data:', data);
            
            // Update result display
            resultBox.style.display = 'block';
            resultBox.className = `result-box ${data.is_spam ? 'spam' : 'ham'}`;
            
            classification.textContent = data.is_spam ? 'Spam' : 'Ham';
            classification.className = `value ${data.is_spam ? 'spam' : 'ham'}`;
            
            confidence.textContent = `${Math.min(Math.round(data.confidence * 100), 99)}%`;
            probability.textContent = `${Math.min(Math.round(data.probability * 100), 99)}%`;
            
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
