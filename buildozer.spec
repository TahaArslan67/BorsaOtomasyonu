[app]

# (str) Title of your application
title = BorsaBot - Halka Arz & GMSTR

# (str) Package name
package.name = borsabot

# (str) Package domain (needed for android/ios packaging)
package.domain = com.example.borsabot

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (let empty to not exclude anything)
#source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning (method 1)
version = 0.1

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,numpy,pandas,requests

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (list) Supported orientations
# Valid options are: landscape, portrait, reverse-landscape, reverse-portrait
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

#
# Android specific
#

# (list) Permissions
android.permissions = INTERNET, VIBRATE

# (int) Target Android API, should be as high as possible.
#android.api = 31

# (int) Minimum API your APK will support.
#android.minapi = 21

# (int) Android NDK API to use. Default is 0.
#android.ndk = 23

# (bool) If True, then skip trying to update the Android SDK
# This can be useful to avoid excess Internet downloads or save time
# if an update is due and you just want to test/build your package
#android.skip_update = False

# (bool) If True, then automatically accept SDK license
# agreements. This is intended for automated build environments.
#android.accept_sdk_license = False

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.kivy.android.PythonActivity

# (list) Pattern to whitelist for the whole project
#android.whitelist = src/*,images/*.png

# (str) Path to a custom whitelist file
#android.whitelist_src =

# (str) Path to a custom blacklist file
#android.blacklist_src =

# (list) List of Java .jar files to add to the libs so that pyjnius can access
# their classes. Don't add any files that you do not need, since extra jars can
# slow down the build process. Allows wildcards matching, for example:
#android.add_jars = my_lib.jar,my_lib2.jar,libs/*.jar

# (list) List of Java .jar files to add to the libs so that pyjnius can access
# their classes. Don't add any files that you do not need, since extra jars can
# slow down the build process. Allows wildcards matching, for example:
#android.add_jars = my_lib.jar,my_lib2.jar,libs/*.jar

# (list) List of Python files to package from the python-for-android root
#android.add_src =

# (list) List of files to build APK from
#android.sources =

# (list) List of files to build APK from (will be compiled first)
#android.sources =

# (list) Android AAR archives to add (currently works only with sdl2_gradle
# bootstrap)
#android.add_aars =

# (list) Gradle dependencies to add (currently works only with sdl2_gradle
# bootstrap)
#android.gradle_dependencies =

# (list) Java classes to add as activities to the manifest.
#android.add_activities = com.example.ExampleActivity

# (list) Android extra libraries to copy into libs/armeabi-v7a
#android.add_libs_armeabi_v7a = libsqlite3.so

# (list) Android extra libraries to copy into libs/arm64-v8a
#android.add_libs_arm64_v8a = libsqlite3.so

# (list) Android extra libraries to copy into libs/x86
#android.add_libs_x86 = libsqlite3.so

# (list) Android extra libraries to copy into libs/x86_64
#android.add_libs_x86_64 = libsqlite3.so

# (bool) Indicate whether the screen should stay on
# Don't set to True on Android. This can cause issues. You will need to use a
# wakelock in Python instead.
#android.wakelock = False

# (list) Android application meta-data to set (key=value format)
#android.meta_data =

# (list) Android library project to add (will be added in the
# project.properties automatically.)
#android.library_references =

# (list) Android shared libraries which will be added to AndroidManifest.xml using <uses-library> tag
#android.uses_lib =

# (str) Android logcat filters to use
#android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a libpymodules.so
#android.copy_libs = 1

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = arm64-v8a

#
# Python for android (p4a) specific
#

# (str) python-for-android branch to use, defaults to master
#p4a.branch = master

# (str) python-for-android specific commit to use, defaults to HEAD, must be within p4a.branch
#p4a.commit = HEAD

# (str) python-for-android git clone directory
#p4a.source_dir =

# (str) The directory in which python-for-android should look for your own build recipes (if any)
#p4a.local_recipes =

# (str) Filename to the hook for p4a
#p4a.hook =

# (str) Bootstrap to use for android builds
# p4a.bootstrap = sdl2

# (int) port number to specify an explicit --port= p4a argument (eg for bootstrap flask)
#p4a.port =

#
# iOS specific
#

# (str) Path to a custom kivy-ios directory
#ios.kivy_ios_dir = ../kivy-ios
# Alternately, specify the full path
#ios.kivy_ios_dir = /path/to/my/kivy-ios

# (bool) If True, then automatically try to find a working iOS SDK on your
# system
#ios.ios_sdk_path = 

# (bool) If True, then automatically try to find a working Xcode on your
# system
#ios.xcode_path = 

# (str) Xcode project name
#ios.xcode_project =

#
# macOS specific
#

# (str) Application identifier
#macos.bundle_identifier =

# (str) Application name
#macos.application_name =

#
# Windows specific
#

# (str) WIndows icon path
#windows.icon_src = images/icon.ico

# (str) Windows icon file
#windows.icon = assets/windows_icon.ico

# (list) Windows requirements
#windows.requirements = kivy, numpy, pandas

# (str) Windows command to run before running the application
#windows.cmd_prebuild = 

# (str) Windows command to run after building the application
#windows.cmd_postbuild = 

#
# Web specific
#

# (list) Web requirements
#web.requirements = kivy, numpy, pandas

# (str) HTML file that should be rendered as the application entry point
#web.index = index.html

#
# Test specific
#

# (str) Tests directory
#tests.dir = tests

#
# Kivy 2.0 specific
#

# (list) Patterns to exclude from build
#exclude_patterns =

# (str) Command to run when building with `python setup.py build`
#setup.build_command = 

#
# Other sections
#

# (list) Source files to include (let empty to include all the files)
#source.include_exts = py,png,jpg,kv,atlas

# (list) Source files to exclude (let empty to not exclude anything)
#source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning
#version.regex = __version__ = ['"](.*)['"]
#version.filename = %(source.dir)s/main.py

# (str) Application description
#description = Your application description

# (list) Application requirements
#requirements = kivy,numpy,pandas,requests

# (list) Garden requirements
#garden_requirements =

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (list) Supported orientations
#orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
#fullscreen = 0
