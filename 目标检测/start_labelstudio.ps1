# start_labelstudio.ps1 - start Label Studio for this project.
# ASCII-ONLY ON PURPOSE (like ../ui/start_manual_annotate.ps1): Windows PowerShell 5.1
# parses BOM-less .ps1 files as ANSI/GBK. UTF-8 Chinese comments get mis-decoded and
# their byte sequences can swallow the NEXT code line. That is exactly how the
# $env:LOCAL_FILES_DOCUMENT_ROOT assignment below silently vanished (found 2026-09-22):
# LS fell back to the drive root D:\, every /data/local-files URL returned 404 and the
# labeling page showed "There was an issue loading URL from $image value".
# Rule: keep this file ASCII-only, or save it as UTF-8 WITH BOM.
$ErrorActionPreference = "Stop"
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
# Image library moved under <project>\dataset\ (2026-09-20): serve root = PROJECT ROOT
# so both the images (<project>\dataset\...) and the ls_view tif proxies
# (<project>\results\detection\ls_view) resolve under it. ls_common.local_file_url
# generates URLs relative to this same root.
# $PSScriptRoot = the top-level detection dir (this script moved there on 2026-10-09),
# so PROJECT ROOT = one level up. That depth is pinned by a landing assertion in
# test_import_chain.py (a missed edit here would silently push the LS serve root to D:\dataset).
$env:LOCAL_FILES_DOCUMENT_ROOT = (Resolve-Path "$PSScriptRoot\..").Path
# Launch the venv python directly on label_studio.server:main (bypasses the
# label-studio.exe wrapper chain exe -> venv python -> system python; fewer moving
# parts, and the server process is a direct child that inherits the env above).
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -c "import sys; from label_studio.server import main; sys.exit(main())" start --port 8080 --host 127.0.0.1
