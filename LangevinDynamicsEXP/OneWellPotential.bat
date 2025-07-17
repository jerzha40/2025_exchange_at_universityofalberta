@echo off
python InitializeTrainData_OneWellPotential.py
xcopy ".\simlog\InitializeTrainData_OneWellPotential" ".\simlog\OneWellPotential" /Y
python OneWellPotential.py
