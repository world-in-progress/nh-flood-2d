@echo off
cd /d D:\codespace\wzp\nh-flood-2d
if not exist .github mkdir .github
python create_instructions.py
pause
