# Kaiju Output to SourceTracker2 Input
This script will take the text file that kaiju outputs for each sample and combine the counts by taxa into a file to be used in SourceTracker2.   

## Input Data Format
The input data should be the exact output from the kaiju2table program. The text files in this folder are examples. 

## Implementation
This script can be run as a command line executable. The script and the kaiju tables need to be in the same directory, with no other .tsv files. The output will be a separate OTU table for each domain and for the metagenome as a whole. 

## Before using these tables in SourceTracker2
Delete the first column and all taxa columns but one from each file, leaving only the taxa level that you want SourceTracker2 to use to distinguish environments, like "species". 
Add "#OTU ID" as the column header for the first column. Save as a tab-delimited .txt file. 
