[app]
# App basic info
title = Dental Detection
package.name = orai
package.domain = org.example
source.dir = .
source.include_exts = py,tflite,png,jpg,kv
version = 0.1

# Dependencies (keep only what’s needed)
requirements = python3,kivy,tensorflow,pillow,opencv-python

# App behavior
orientation = portrait
fullscreen = 1

# Entry point (main Python file)
android.entrypoint = .

# Permissions
android.permissions = INTERNET,CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (optional) icon and presplash
# icon.filename = %(source.dir)s/data/icon.png
# presplash.filename = %(source.dir)s/data/presplash.png

# Exclude unnecessary files
[app:source.exclude_patterns]
*.pyc
*.pyo
*.pyd
__pycache__/
.git*
docs/
tests/
images/
nltk_data/

[buildozer]
log_level = 2
warn_on_root = 1

# Android build configuration
[app.android]
android.api = 34
android.minapi = 21
android.ndk = 25b
android.arch = arm64-v8a
android.allow_backup = True
android.debug = True
