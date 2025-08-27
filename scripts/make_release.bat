@echo off
set OUT=release_qwen25_qlora.zip
del %OUT% 2>nul
powershell -command "Compress-Archive -Path README.md, LICENSE, requirements.txt, src*, portable_infer*, kb*, checkpoints\qwen25-3b-dolly-qlora-steps150*, outputs\EVAL_SUMMARY.json -DestinationPath %OUT%"
echo Wrote %OUT%