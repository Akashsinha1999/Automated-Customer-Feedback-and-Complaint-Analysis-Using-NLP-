
# Automated Customer Feedback and Complaint Analysis Using NLP 

This project aims to automate the analysis of a large dataset of more than 10 lakh text complaints submitted to a bank. Analyzing such a vast volume of complaints manually is impractical. To address this challenge, I have used a NLP for sentiment analysis and classification.


## EXAMPLE
sentence =["""your debt deaprtment is group of croocks ,
They never deal a person with any diginity"""]

new=text_tf.transform(sentence)
new1=pd.DataFrame(new.toarray(), columns=text_tf.get_feature_names())
ppt=dt_tf.predict(new1)

if ppt[0]==0:
    print("Credit_Reporting")
elif ppt[0]==1:
    print("Debt_collection")
elif ppt[0]==2:
    print("Mortgage")
elif ppt[0]==3:
    print("Loan")
elif ppt[0]==4:
    print("Cards")
elif ppt[0]==5:
    print("Bank_Accounts")
elif ppt[0]==6:
    print("Money_Service_or_Currency_Service")


Output is 
The complain is registered for "Dept Collection" department.
## PROBLEM STATEMENT
The primary goal of this project is to determine the department or category to which each complaint is relevant. The following departments were considered for categorization:

Credit reporting, credit repair services, or other personal

consumer reports

Debt collection

Mortgage

Bank account or service

Credit card

Credit card or prepaid card

Student loan

Checking or savings account

Consumer Loan

Vehicle loan or lease

Money transfer, virtual currency, or money service

Payday loan, title loan, or personal loan

Payday loan

Money transfers

Prepaid card

Other financial service

Virtual currency

I convert these 18 department into 7 most import department.
## Techniques/Method Used
I utilized natural language processing to perform sentiment analysis on the text complaints.

Also I have used libraries like Numpy, pandas for data manipulation.

Imported warnings warnings.filterwarnings("ignore"), imported re (regex) , imported BeautifulSoupas , as i have used WordNetLametizer to perform lemmatization on text complaints.

I have also used train_test_split to split data into train and test set , then i have used countvectorizer and TF-IDF both, which convert text into tokens (means assigning number according to the frequesncy of the number like word with highest frequency will be assigned as 1 ).

After generating the tokens,I used DecisionTreeClassifier to classify the complains and feedback of the customer. Got 69 percent of accuracy in predicting feedbacks and complaints that which complain and feedback is registered for which department.

This model can understand the sentiment expressed in each complaint and categorize it into one of the relevant departments.
## Data  
This dataset consists of more than 10 lakh text complaints, which were collected from customers. The complaints are diverse and contain varying degrees of sentiment, making it challenging for manual analysis.

dataset link - https://drive.google.com/file/d/1mVeJmR-hJVP2xlu8b4me2b3W9RQKL76K/view?usp=drive_link
## Result
The nlp model successfully analyzed and categorized the complaints into the respective departments with 69% of accuracy. This automation has significantly reduced the manual workload and improved the efficiency of addressing customer complaints & feedback
## Usage
To use this project, you can follow the instructions provided in the codebase to train the nlp model on your own dataset of bank customer feedback & complaints. The trained model can then be used to automatically categorize complaints and feedback into their relevant departments.
## Conslusion
Automating the analysis of a vast number of bank complaints & feedback using the NLP algorithm not only saves time and resources but also enhances the efficiency of the complaint handling process. This approach can be applied to similar text classification tasks in various domains, providing valuable insights and automation capabilities.