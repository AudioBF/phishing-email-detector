import pandas as pd
import numpy as np
from faker import Faker
import random
import re
from typing import List, Dict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailGenerator:
    def __init__(self):
        self.faker = Faker()
        self.spam_templates = [
            "Subject: Urgent Security Alert - Account Verification Required\n\nDear Valued Customer,\n\nWe have detected unusual activity on your {account_type} account. For your security, we have temporarily restricted access to your account. To restore full access, please verify your identity by clicking the link below:\n\n{url}\n\nThis is a mandatory security check. Failure to verify within 24 hours will result in permanent account suspension.\n\nBest regards,\nSecurity Team\n{company}",
            
            "Subject: Your Tax Refund is Ready for Processing\n\nDear Taxpayer,\n\nWe are pleased to inform you that your tax refund of ${amount} has been approved by the IRS. However, we need additional verification to process your payment.\n\nPlease complete the attached W-9 form and provide your banking information through our secure portal:\n{url}\n\nNote: This is a time-sensitive matter. Your refund will be processed within 24-48 hours after verification.\n\nSincerely,\nIRS Tax Processing Department",
            
            "Subject: Important: Your Subscription Renewal\n\nDear {service} Member,\n\nWe noticed that your payment method for your {service} {plan_type} subscription has expired. To avoid service interruption, please update your payment information immediately.\n\nYour current subscription includes:\n- {feature1}\n- {feature2}\n- {feature3}\n\nClick here to update your payment method:\n{url}\n\nIf you do not update your payment information within 24 hours, your account will be downgraded to the Basic plan.\n\nThank you for being a valued {service} member.\n\n{service} Customer Service",
            
            "Subject: Security Alert: Unusual Login Detected\n\nDear {name},\n\nWe detected a login attempt from {location} at {time}. If this was you, you can ignore this message. If not, please secure your account immediately.\n\nClick here to verify your account:\n{url}\n\nFor your security, we recommend:\n1. Changing your password\n2. Enabling two-factor authentication\n3. Reviewing recent account activity\n\nBest regards,\nSecurity Team\n{company}",
            
            "Subject: Your Payment Failed - Action Required\n\nDear Customer,\n\nWe were unable to process your recent payment of ${amount} for your {service} subscription. Please update your payment information to avoid service interruption.\n\nCurrent Payment Method: {payment_method}\nLast Attempt: {date}\n\nClick here to update your payment information:\n{url}\n\nIf you have any questions, please contact our support team.\n\nSincerely,\nBilling Department\n{company}"
        ]
        
        self.ham_templates = [
            "Subject: Project Status Update - {quarter}\n\nHi Team,\n\nI wanted to provide an update on our current project status and upcoming milestones.\n\nKey Achievements:\n- {achievement1}\n- {achievement2}\n- {achievement3}\n\nNext Steps:\n1. {next_step1}\n2. {next_step2}\n3. {next_step3}\n\nPlease review the attached project timeline and let me know if you have any questions or concerns.\n\nBest regards,\n{name}\nProject Manager",
            
            "Subject: Monthly Financial Report - {month} {year}\n\nDear Finance Team,\n\nPlease find attached the monthly financial report for {month} {year}. The report includes:\n\n- Revenue analysis by department\n- Expense breakdown\n- Budget vs. actual comparison\n- Key performance indicators\n\nKey highlights:\n- Total revenue {revenue_change}\n- {department} expenses were {expense_change}\n- New customer acquisition {acquisition_status}\n\nPlease review the report and prepare any questions for our monthly finance meeting scheduled for {meeting_date}.\n\nRegards,\n{name}\nFinance Director",
            
            "Subject: Security Policy Update - Required Action\n\nHello Everyone,\n\nAs part of our ongoing commitment to security, we are implementing new password requirements effective {date}. Here's what you need to know:\n\nNew Requirements:\n- Minimum {min_length} characters\n- Must include uppercase, lowercase, numbers, and special characters\n- Cannot contain common words or patterns\n- Must be changed every {days} days\n\nAction Required:\n1. Update your password by {deadline}\n2. Enable two-factor authentication if not already done\n3. Review the attached security guidelines\n\nIT Support will be available for any questions or assistance.\n\nBest regards,\nIT Security Team",
            
            "Subject: Weekly Team Meeting Agenda\n\nHi Team,\n\nHere's the agenda for our weekly team meeting scheduled for {date} at {time}:\n\n1. Project Updates\n   - {project1} status\n   - {project2} status\n   - {project3} status\n\n2. Key Discussion Points\n   - {topic1}\n   - {topic2}\n   - {topic3}\n\n3. Action Items\n   - {action1}\n   - {action2}\n   - {action3}\n\nPlease come prepared with your updates and any questions.\n\nBest regards,\n{name}\nTeam Lead",
            
            "Subject: Client Meeting Follow-up\n\nDear {client_name},\n\nThank you for taking the time to meet with us yesterday. As discussed, here's a summary of our conversation and next steps:\n\nKey Points Discussed:\n- {point1}\n- {point2}\n- {point3}\n\nNext Steps:\n1. {step1}\n2. {step2}\n3. {step3}\n\nTimeline:\n- {milestone1}: {date1}\n- {milestone2}: {date2}\n- {milestone3}: {date3}\n\nPlease review and let me know if you'd like to make any adjustments.\n\nBest regards,\n{name}\n{position}"
        ]
        
        self.spam_keywords = {
            'account_type': ['bank', 'email', 'social media', 'shopping', 'gaming', 'investment', 'trading', 'cloud storage'],
            'company': ['Bank of America', 'Chase', 'Wells Fargo', 'PayPal', 'Amazon', 'Microsoft', 'Apple', 'Google'],
            'amount': ['2,850.00', '1,500.00', '3,200.00', '4,750.00', '5,000.00', '1,250.00', '2,500.00', '3,750.00'],
            'service': ['Netflix', 'Spotify', 'Amazon Prime', 'Microsoft 365', 'Adobe Creative Cloud', 'Dropbox', 'Slack', 'Zoom'],
            'plan_type': ['Premium', 'Professional', 'Business', 'Enterprise', 'Ultimate', 'Family', 'Student'],
            'feature1': ['4K Ultra HD streaming', 'Unlimited music streaming', 'Free shipping', 'Full Office suite', 'Creative apps'],
            'feature2': ['4 simultaneous screens', 'Ad-free experience', 'Prime Video', '1TB cloud storage', 'Premium templates'],
            'feature3': ['Download for offline', 'Exclusive content', 'Prime Music', 'Advanced security', 'Priority support'],
            'location': ['New York, NY', 'London, UK', 'Tokyo, Japan', 'Sydney, Australia', 'Dubai, UAE', 'Paris, France', 'Berlin, Germany'],
            'time': ['2:35 AM', '3:15 PM', '11:20 PM', '5:45 AM', '1:30 PM', '9:10 PM', '4:25 AM'],
            'payment_method': ['Visa ending in 1234', 'Mastercard ending in 5678', 'American Express ending in 9012', 'PayPal account'],
            'date': ['March 15, 2024', 'April 1, 2024', 'May 10, 2024', 'June 5, 2024', 'July 20, 2024']
        }
        
        self.ham_keywords = {
            'quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            'achievement1': ['Completed backend API integration', 'Implemented new authentication system', 'Deployed staging environment'],
            'achievement2': ['Optimized database performance', 'Enhanced security measures', 'Improved user interface'],
            'achievement3': ['Reduced system latency', 'Increased test coverage', 'Streamlined deployment process'],
            'next_step1': ['Conduct user testing', 'Address critical bugs', 'Prepare for production deployment'],
            'next_step2': ['Update documentation', 'Train support team', 'Monitor system performance'],
            'next_step3': ['Gather user feedback', 'Plan next sprint', 'Review analytics data'],
            'month': ['January', 'February', 'March', 'April', 'May', 'June'],
            'year': ['2024'],
            'revenue_change': ['increased by 15%', 'decreased by 5%', 'remained stable', 'exceeded projections'],
            'department': ['Marketing', 'Sales', 'Operations', 'IT', 'HR'],
            'expense_change': ['5% under budget', '10% over budget', 'on target', 'significantly reduced'],
            'acquisition_status': ['exceeded targets by 20%', 'met expectations', 'needs improvement', 'showing positive trend'],
            'meeting_date': ['next Tuesday at 10 AM', 'this Friday at 2 PM', 'Monday at 11 AM'],
            'min_length': ['12', '14', '16', '18'],
            'days': ['90', '60', '45', '30'],
            'deadline': ['next Monday', 'by the end of this week', 'within the next 7 days'],
            'project1': ['Website Redesign', 'Mobile App Development', 'CRM Implementation'],
            'project2': ['Data Migration', 'Security Upgrade', 'API Integration'],
            'project3': ['Content Strategy', 'User Testing', 'Performance Optimization'],
            'topic1': ['Q2 Goals', 'Resource Allocation', 'Team Structure'],
            'topic2': ['Client Feedback', 'Market Analysis', 'Competitor Research'],
            'topic3': ['Training Schedule', 'Process Improvement', 'Quality Assurance'],
            'action1': ['Update project documentation', 'Schedule client meeting', 'Review test results'],
            'action2': ['Prepare presentation', 'Analyze metrics', 'Update status report'],
            'action3': ['Coordinate with other teams', 'Set up monitoring', 'Plan next phase'],
            'client_name': lambda: self.faker.name(),
            'point1': ['Project timeline and milestones', 'Budget and resource allocation', 'Key deliverables'],
            'point2': ['Technical requirements and specifications', 'User experience improvements', 'Integration points'],
            'point3': ['Quality assurance process', 'Support and maintenance', 'Training and documentation'],
            'step1': ['Review and approve requirements', 'Provide access credentials', 'Schedule kickoff meeting'],
            'step2': ['Assign project team members', 'Set up development environment', 'Create project timeline'],
            'step3': ['Conduct initial training', 'Begin development phase', 'Start testing process'],
            'milestone1': ['Requirements Review', 'Design Approval', 'Development Start'],
            'milestone2': ['First Prototype', 'User Testing', 'Security Review'],
            'milestone3': ['Final Delivery', 'Training Completion', 'Go-Live'],
            'date': lambda: self.faker.date_between(start_date='today', end_date='+30d').strftime('%B %d, %Y'),
            'date1': lambda: self.faker.date_between(start_date='today', end_date='+30d').strftime('%B %d, %Y'),
            'date2': lambda: self.faker.date_between(start_date='+31d', end_date='+60d').strftime('%B %d, %Y'),
            'date3': lambda: self.faker.date_between(start_date='+61d', end_date='+90d').strftime('%B %d, %Y'),
            'position': ['Project Manager', 'Technical Lead', 'Solutions Architect', 'Product Owner'],
            'time': ['10:00 AM', '2:00 PM', '3:30 PM', '11:00 AM', '4:00 PM']
        }

    def generate_spam_email(self) -> str:
        template = random.choice(self.spam_templates)
        placeholders = re.findall(r'\{(\w+)\}', template)
        replacements = {}
        
        for placeholder in placeholders:
            if placeholder in self.spam_keywords:
                replacements[placeholder] = random.choice(self.spam_keywords[placeholder])
            elif placeholder == 'url':
                domain = random.choice(['secure', 'verify', 'update', 'claim', 'renew'])
                tld = random.choice(['com', 'net', 'org', 'info', 'biz'])
                replacements[placeholder] = f'https://{domain}-{self.faker.word()}.{tld}/{self.faker.uuid4()}'
            elif placeholder == 'name':
                replacements[placeholder] = self.faker.name()
        
        return template.format(**replacements)

    def generate_ham_email(self) -> str:
        template = random.choice(self.ham_templates)
        placeholders = re.findall(r'\{(\w+)\}', template)
        replacements = {}
        
        for placeholder in placeholders:
            if placeholder in self.ham_keywords:
                if callable(self.ham_keywords[placeholder]):
                    replacements[placeholder] = self.ham_keywords[placeholder]()
                else:
                    replacements[placeholder] = random.choice(self.ham_keywords[placeholder])
            elif placeholder == 'name':
                replacements[placeholder] = self.faker.name()
        
        return template.format(**replacements)

def expand_dataset(input_file: str, output_file: str, expansion_factor: int = 3):
    """Expand the dataset with synthetic examples"""
    logger.info(f"Loading original dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    # Calculate target size (3x original)
    target_size = len(df) * expansion_factor
    
    # Calculate how many new examples we need
    current_size = len(df)
    new_examples_needed = target_size - current_size
    
    if new_examples_needed <= 0:
        logger.info("Dataset already has enough examples")
        return
    
    logger.info(f"Generating {new_examples_needed} new examples")
    
    # Calculate number of spam and ham examples needed (50/50 split)
    new_spam = new_ham = new_examples_needed // 2
    
    # Generate new examples
    generator = EmailGenerator()
    new_emails = []
    new_labels = []
    
    # Generate spam emails
    for _ in range(new_spam):
        new_emails.append(generator.generate_spam_email())
        new_labels.append('spam')
    
    # Generate ham emails
    for _ in range(new_ham):
        new_emails.append(generator.generate_ham_email())
        new_labels.append('ham')
    
    # Create new dataframe
    new_df = pd.DataFrame({
        'text': new_emails,
        'label': new_labels
    })
    
    # Combine with original dataset
    expanded_df = pd.concat([df, new_df], ignore_index=True)
    
    # Shuffle the dataset
    expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save expanded dataset
    expanded_df.to_csv(output_file, index=False)
    logger.info(f"Expanded dataset saved to {output_file}")
    logger.info(f"Total examples: {len(expanded_df)}")
    logger.info(f"Spam examples: {len(expanded_df[expanded_df['label'] == 'spam'])}")
    logger.info(f"Ham examples: {len(expanded_df[expanded_df['label'] == 'ham'])}")

if __name__ == "__main__":
    input_file = "data/spam_ham_dataset.csv"
    output_file = "data/expanded_spam_ham_dataset.csv"
    expand_dataset(input_file, output_file, expansion_factor=3) 