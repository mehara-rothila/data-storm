@echo off
echo ========================================================================
echo                     IMPROVED KAGGLE WINNER SOLUTION
echo                     FIXED DATA LEAKAGE ISSUES
echo ========================================================================
echo.

echo Backing up old solution...
if exist "D:\DATA STORM\kaggle_winner.py.bak" (
    del "D:\DATA STORM\kaggle_winner.py.bak"
)
if exist "D:\DATA STORM\kaggle_winner.py" (
    rename "D:\DATA STORM\kaggle_winner.py" "kaggle_winner.py.bak"
)

echo Copying improved solution...
copy "D:\DATA STORM\improved_kaggle_winner.py" "D:\DATA STORM\kaggle_winner.py"

echo Running improved Kaggle solution...
python "D:\DATA STORM\kaggle_winner.py"

echo.
echo ========================================================================
echo                   KAGGLE SUBMISSIONS GENERATED
echo ========================================================================
echo.

echo Opening submission files...
if exist "D:\DATA STORM\outputs\submission.csv" (
    start notepad "D:\DATA STORM\outputs\submission.csv"
)

echo.
echo The optimal threshold was automatically selected based on training data distribution.
echo.
echo Available submission files:
echo - D:\DATA STORM\outputs\submission.csv (OPTIMAL - USE THIS ONE)
dir /b "D:\DATA STORM\outputs\submission_threshold_*.csv"
echo.

echo IMPORTANT: Submit the optimal submission.csv file to Kaggle for best results.
echo.

echo Press any key to exit...
pause > nul