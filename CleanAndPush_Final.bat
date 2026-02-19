@echo off
title Final Git Push - F20022026
cls
echo Dang chuan bi day ban Luan van moi nhat len GitHub...

:: Thêm các file mới tạo
git add requirements.txt README.md
git add .

:: Commit voi thong tin chi tiet
git commit -m "Update Final Thesis Structure - MLP 8 Layers - Standard Format"

:: Push len server
git push origin main

echo ======================================================
echo   DA CAP NHAT THANH CONG!
echo ======================================================
pause