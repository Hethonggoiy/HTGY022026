@echo off
title Git New Folder Setup - Master Thesis 2026
cls

echo ======================================================
echo   1. KHOI TAO LAI GIT CHO THU MUC MOI...
echo ======================================================

:: Khởi tạo lại Git từ đầu
git init

:: Thêm Remote (Kết nối lại với Repository của bạn)
git remote add origin https://github.com/Hethonggoiy/LuanVan_MLP.git

echo [OK] Da ket noi voi GitHub.

echo.
echo ======================================================
echo   2. TAO FILE .GITIGNORE CHUAN (TRANH FILE RAC)
echo ======================================================

(
echo # Office & OS files
echo ~$*.doc*
echo ~$*.xls*
echo Thumbs.db
echo Desktop.ini
echo.
echo # Research Data & Large files
echo *.zip
echo *.7z
echo *.rar
echo *.mp4
echo *.wav
echo /data/
echo /backup/
) > .gitignore

echo [OK] Da toi uu file .gitignore.

echo.
echo ======================================================
echo   3. DONG BO VA DAY DU LIEU MOI
echo ======================================================

:: Cấu hình buffer
git config --global http.postBuffer 1048576000

:: Thêm tất cả file hiện có trong thư mục mới
git add .

:: Commit đầu tiên cho thư mục mới
git commit -m "Khoi tao thu muc moi gon nhe F20022026"

:: Buộc GitHub chấp nhận bản mới này (Force push) 
:: Lưu ý: Lệnh này sẽ thay thế bản cũ 4GB trên GitHub bằng bản mới nhẹ hơn này.
echo Dang day du lieu moi len GitHub (Force Push)...
git push -u origin main --force

echo.
echo ======================================================
echo   HOAN THANH! THU MUC MOI DA DUOC DONG BO SACH SE.
echo ======================================================
pause