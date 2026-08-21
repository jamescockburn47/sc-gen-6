@echo off
echo ==========================================
echo Timeline Extraction - Overnight Run
echo ==========================================
echo.
echo Processing ~4500 chunks with date patterns
echo Estimated time: 6-10 hours at ~20 sec/chunk
echo.
echo Started at: %date% %time%
echo.

cd /d "%~dp0"
.venv\Scripts\python.exe -c "from src.graph.chunk_timeline_extractor import run_timeline_extraction; import json; stats = run_timeline_extraction(); print('\\n=== COMPLETED ==='); print(json.dumps(stats, indent=2, default=str)); input('Press Enter to exit...')"

echo.
echo Finished at: %date% %time%
pause
