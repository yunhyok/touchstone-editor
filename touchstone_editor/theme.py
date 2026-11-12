from PySide6.QtGui import QColor, QPalette


def apply_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.Window: "#202226", QPalette.WindowText: "#eceef1",
        QPalette.Base: "#1d2025", QPalette.AlternateBase: "#25282e",
        QPalette.Text: "#eceef1", QPalette.Button: "#363b43",
        QPalette.ButtonText: "#eceef1", QPalette.Highlight: "#345d8c",
        QPalette.HighlightedText: "#ffffff", QPalette.PlaceholderText: "#a4aebb",
        QPalette.ToolTipBase: "#2b2e34", QPalette.ToolTipText: "#eceef1",
        QPalette.Link: "#8fc1ff",
    }.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor("#929ba8"))
    app.setPalette(palette)
    font = app.font()
    font.setFamily("Malgun Gothic")
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet("""
        QMainWindow, QScrollArea { background: #202226; }
        QGroupBox { background: #2b2e34; border: 1px solid #505761;
                    border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px;
                          color: #dce3ed; font-weight: bold; }
        QLabel { background: transparent; }
        QLabel#title { font-size: 24px; font-weight: 700; }
        QLabel#subtitle, QLabel#hint { color: #b9c1cc; }
        QLineEdit, QComboBox, QSpinBox, QTableView, QPlainTextEdit {
            background: #1d2025; color: #eceef1; border: 1px solid #626c7a;
            border-radius: 3px; selection-background-color: #345d8c;
            selection-color: white;
        }
        QLineEdit, QComboBox, QSpinBox { padding: 5px; min-height: 20px; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTableView:focus {
            border: 1px solid #8fc1ff;
        }
        QTableView { alternate-background-color: #25282e; gridline-color: #414750; }
        QTableView::item { padding: 5px; }
        QHeaderView::section { background: #353a42; color: #eceef1; padding: 7px;
            border: none; border-right: 1px solid #505761; font-weight: bold; }
        QTableCornerButton::section { background: #353a42; border: none; }
        QPushButton { background: #3b414b; color: #eceef1; border: 1px solid #717c8d;
            border-radius: 4px; padding: 6px 10px; min-height: 20px; }
        QPushButton:hover { background: #48515e; border-color: #a0adbf; }
        QPushButton:pressed { background: #252d39; }
        QPushButton:focus { border: 1px solid #8fc1ff; }
        QPushButton[primary="true"] { background: #386cb0; border-color: #7fa9e0; }
        QPushButton[primary="true"]:hover { background: #437dc7; }
        QPushButton:disabled { background: #30353e; color: #929ba8; border-color: #4f5865; }
        QProgressBar { background: #15181d; border: 1px solid #505761; text-align: center; }
        QProgressBar::chunk { background: #386cb0; }
        QPlainTextEdit { background: #15181d; padding: 5px; }
        QSplitter::handle { background: #343940; }
        QCheckBox { spacing: 7px; padding: 3px 0; }
        QTabWidget::pane { background: #202226; border: none; border-top: 1px solid #505761; }
        QTabBar::tab { background: #2b2e34; color: #b9c1cc; border: 1px solid #454b55;
            border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
            padding: 7px 18px; margin-right: 3px; }
        QTabBar::tab:hover { background: #363b43; color: #eceef1; }
        QTabBar::tab:selected { background: #202226; color: #ffffff; font-weight: bold;
            border-color: #8fc1ff; border-top: 2px solid #8fc1ff; }
        QTabBar::tab:disabled { color: #808894; }
        QTableView QComboBox { background: transparent; border: none; border-radius: 0;
            padding: 2px 6px; min-height: 0; }
        QTableView QComboBox:focus { border: 1px solid #8fc1ff; }
    """)
