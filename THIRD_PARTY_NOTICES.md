# Third-party notices

## Qt, PySide6 and Shiboken6

Touchstone Editor uses the unmodified Qt 6.11.0 libraries, PySide6 6.11.0 and
Shiboken6 6.11.0 under the GNU Lesser General Public License, version 3.
Copyright The Qt Company Ltd. and the respective contributors.

The installed `licenses` directory includes the LGPL v3 and GPL v3 texts.
`licenses/Qt-6.11.0-third-party.txt` collects component copyright notices and
license texts from the Qt 6.11.0 documentation archives, plus Qt's attribution
for the bundled Mesa software OpenGL renderer.
The Qt DLLs and Python binding extensions reside in the `_internal` directory
beside `TouchstoneEditor.exe` and can be replaced with compatible builds.
You may modify the libraries and reverse engineer the application as needed
to debug your modifications to them.

Corresponding source is available from Qt at no charge.

- [Qt 6.11.0 source archives](https://download.qt.io/archive/qt/6.11/6.11.0/single/)
- [PySide6 and Shiboken6 6.11.0 source archives](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.0-src/)
- [Qt 6.11 third-party component notices](https://doc.qt.io/qt-6.11/licenses-used-in-qt.html)

## NumPy

Touchstone Editor uses NumPy 2.5.3 under the BSD 3-Clause License.
Copyright 2005–2025 NumPy Developers. The Windows wheel also bundles OpenBLAS
(BSD 3-Clause), the LAPACK routines inside it (BSD-3-Clause-Open-MPI) and the GCC
runtime library (GPL 3 or later with the GCC Runtime Library Exception 3.1).
`licenses/NumPy/LICENSE.txt` contains these notices and license texts, and the
other files under `licenses/NumPy` cover the components vendored in NumPy.
[NumPy 2.5.3 source](https://github.com/numpy/numpy/tree/v2.5.3)

## CPython

The Windows build includes the CPython runtime. Its license and copyright
notices are in `licenses/CPython.txt`.

## OpenSSL

The Windows CPython runtime uses OpenSSL 3.0.16. Copyright 1998–2025 The
OpenSSL Authors. The Apache License 2.0 text is in `licenses/OpenSSL-3.0.16.txt`.
[OpenSSL 3.0.16 license](https://github.com/openssl/openssl/blob/openssl-3.0.16/LICENSE.txt)

## PyInstaller

The executable uses the PyInstaller 6.20.0 bootloader. PyInstaller's GPL
distribution exception permits distribution of generated executables under
the application's license. Some runtime files use the Apache License 2.0.
The license texts and exception are in `licenses/PyInstaller.txt`.
[PyInstaller license and exception](https://github.com/pyinstaller/pyinstaller/blob/v6.20.0/COPYING.txt)

## Inno Setup

The installer was built with Inno Setup by Jordan Russell and Martijn Laan.
[Inno Setup](https://jrsoftware.org/isinfo.php)
