@echo off
mkdir ".\simlog\OneWellPotential"
mkdir ".\simlog\InitializeTrainData_OneWellPotential"
python InitializeTrainData_OneWellPotential.py
xcopy ".\simlog\InitializeTrainData_OneWellPotential" ".\simlog\OneWellPotential" /Y
python OneWellPotential.py
