@echo off
title Re-organize Thesis Structure
cls

echo Dang tao cac thu muc tieu chuan...
mkdir 01_Documents 02_Literature 03_Data\raw 03_Data\processed 04_Source_Code\models 05_Results\charts 06_Thesis_Draft

echo Dang di chuyen file vao dung vi tri...

:: Gom file Document & Mau LHU
move "Mẫu LHU" 01_Documents >nul 2>&1
move "Biên bản chấm lv.pdf" 01_Documents >nul 2>&1

:: Gom file tham khao
move "Sach-thamkhao" 02_Literature >nul 2>&1
move "Tài liệu tham Khảo" 02_Literature >nul 2>&1

:: Gom du lieu
move *.xlsx 03_Data\raw >nul 2>&1
move *.csv 03_Data\processed >nul 2>&1

:: Gom Code va Models
move *.py 04_Source_Code >nul 2>&1
move *.pkl 04_Source_Code\models >nul 2>&1
move models\* 04_Source_Code\models >nul 2>&1

:: Gom hinh anh va ket qua
move *.png 05_Results\charts >nul 2>&1
move *.jpg 05_Results\charts >nul 2>&1
move *.webp 05_Results\charts >nul 2>&1

:: Gom file Luan van
move "LV_Finally*" 06_Thesis_Draft >nul 2>&1
move "De_cuong*" 06_Thesis_Draft >nul 2>&1

echo [OK] Da sap xep xong! Hay kiem tra lai thu muc F20022026.
pause