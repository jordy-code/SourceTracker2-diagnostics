# SourceTracker2-diagnostics
Add-on diagnostic tool used with metagenomic data for SourceTracker2 as described in (paper link). 
## Documentation
This additional feature has been added to the [sourcetracker](https://github.com/biota/sourcetracker2) program describe in [Knights et al., 2011](http://www.ncbi.nlm.nih.gov/pubmed/21765408). Please refer to and cite the original work if you use this package and for description of the sourcetracker program.
## Diagnostic function for SourceTracker2
The diagnostic function provides output information on the precision of predictons made by sourcetracker. This is essentialy done by running multiple draws and comparing the difference between the maximum and minimum draws for each sink sample across each environment. 
# Implementation
The diagnsotic feature is called by using ```--diagnostics``` as an option in the command line when running soucetracker. Because this feature uses multiple draws to calculate the sourcetracker performance, the ```--draws_per_restart``` option in sourcetracker must also be called and set to a minimum of ```2``` when using this diagnostic feature.  When the sourcetracker program is run with these options, a '/diagnostics' folder will be found in the user-defined output folder with graphs and tables produced by the diagnostic feature. Graphs display the predicted proportions for all draws and will only be found in the diagnostics folder if the maximum and minimum draws exceed a diffference greater than the set limit (default 5%). The limit value can be changed by using the option ```--limit``` in the command line followed by the desired percentage value. A limit value of ```0``` will produce graphs for all sink samples across each environment.
## Installation
Refer to the [sourcetracker](https://github.com/biota/sourcetracker2) page and follow the installation instructions. Once the installation is complete, replace the ```/sourcetracker``` folder with the ```/sourcetracker``` folder from this build. 
## Examples
#### Running sourcetracker without the diagnostic feature 
```
sourcetracker2 gibbs -i OTU.txt -m map.txt -o output/ 
```
#### Sourcetracker using diagnostic feature, (draws_per_restart must be >1)
```
sourcetracker2 gibbs -i OTU.txt -m map.txt -o output/ --diagnostics --draws_per_restart 2
```
#### Sourcetracker using diagnostic feature with limit set at 10%
```
sourcetracker2 gibbs -i OTU.txt -m map.txt -o output/ --diagnostics --draws_per_restart 2 --limit 0.10
```
