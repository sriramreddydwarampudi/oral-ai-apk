[app]
# App basic info
title = Dental Detection
package.name = dentaldetection
package.domain = org.orai
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,tflite
version = 0.1

# Requirements - simplified for stability
requirements = python3,kivy,numpy,pillow,opencv,android

# App behavior
orientation = portrait
fullscreen = 0

# Permissions
android.permissions = INTERNET,CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# Android configuration
android.api = 32
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.allow_backup = True
android.accept_sdk_license = True

# Logging
android.logcat_filters = *:S python:D

# Exclude unnecessary files
source.exclude_patterns = *.pyc,*.pyo,__pycache__/*,.git/*,.github/*,*.md

[buildozer]
log_level = 2
warn_on_root = 1
