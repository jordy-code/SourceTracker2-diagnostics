# Random Forest and Confusion Matrix Script
This script output a random forest analysis and confusion matrix for the input data. 

## Input Data Format
The input data table should have the taxa (species, genus...) as column headers and the environment as row names, and counts as the data. Your table should be saved as a .csv file. 

## Environment
Before running this script, the package pandas needs to be downgraded to pandas version 0.25, since the current pandas is not compatible with pandas_ml, which is used to create the confusion matrices. If you do not want to produce the confustion matrix but just run the random forest, you can comment these three lines out in the script and avoid downgrading pandas:
    cm = ConfusionMatrix(test["Env"].tolist(), preds)
    cm.print_stats()
    cm.plot()

## Implementation
If not using the test data, the name of the csv to be read in line 16 needs to be changed to the name of your csv. Your file should be in the same directory as the script. This script can be run as an executable. If you are using other data, values in line 29 will need to be changed. 
Depending on whether you are running the random forest for all environments to produce confusion matrices or if you want to know important features for individual environments, imput needs to be changed. The 'df' variable uses all the environments to either classify an individual sample or produce confusion matrices. The 'domain_df' variable labels  chosen environments as 'other' to determine important features in classifying individual envrionments from all other environments. 
The domain_df is used to determine important features for each environment. For each environment to be tested, remove that environment name from the regex list on line 29, all remaining environments in the list will be considered 'other'.
For instance, if I want to know important features for gut in the example table, I would remove 'gut' from the regex list and change the domain variable to the domain and the environent I want to test. 

