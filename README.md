# multiFLEX-LF

In high-throughput LC-MS/MS-based proteomics information about the presence and stoichiometry of post-translational are normally not readily available. Here we introduce multiFLEX-LF, a computational tool that overcomes this problem. The tool build upon the tool FLEXIQuant-LF (https://github.com/SteenOmicsLab/FLEXIQuantLF). FLEXIQuant-LF was developed to analyze a single protein to identify differentially modified peptides and quantify their modification extent. multiFLEX-LF analyzes all proteins of given dataset consecutively (on 1 core) or in parallel employing multiple cores/threads. multiFLEX-LF requires label-free quantification of unmodified peptides and a within study reference (e.g., a reference time point or control group). multiFLEX-LF employs random sample consensus (RANSAC)-based robust linear regression to compare a sample to a reference and compute a relative modification (RM) score. A low RM score corresponds to a high differential modification extent. The peptides are hierarchically clustered based on the RM scores. The clustering which is saved in an interactive dendrogram and heatmap can be investigated for groups of peptides with similar modification dynamics. multiFLEX-LF is unbiased towards the type of modification. Hence, multiFLEX-LF drives large-scale investigation of the modification extent and the modification stoichiometries of peptides in time series and case-control studies. multiFLEX-LF was implemented in python and comes with a CLI and a GUI version.

## Download and Installation of the GUI
- Requirements: Windows 10 and 800MB of free disk space. 
- multiFLEX-LF executables for Windows 10 systems available here: https://gitlab.com/SteenOmicsLab/multiflex-lf/-/releases#v1.0.0
- No installation needed. Just move the zip file to your location of choice and extract it. To start the program, double-click multiFLEX_LF_GUI.exe. The program might take a while to start.

For a detailed description of the input file, the input parameters and the output files see: multiFLEX-LF_documentation.pdf



