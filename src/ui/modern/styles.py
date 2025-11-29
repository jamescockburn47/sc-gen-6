
DARK_THEME = """
/* Main Window */
QMainWindow {
    background-color: #0f0f12;
    color: #f1f5f9;
}

QWidget {
    background-color: #0f0f12;
    color: #f1f5f9;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}

/* Sidebar */
QFrame#sidebar {
    background-color: #18181b;
    border-right: 1px solid #27272a;
}

QPushButton#sidebar_btn {
    background-color: transparent;
    color: #a1a1aa;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-weight: 500;
}

QPushButton#sidebar_btn:hover {
    background-color: #27272a;
    color: #f1f5f9;
}

QPushButton#sidebar_btn:checked {
    background-color: #27272a;
    color: #f1f5f9;
    border-left: 3px solid #3b82f6;
}

QPushButton#new_chat_btn {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px;
    font-weight: 600;
}

QPushButton#new_chat_btn:hover {
    background-color: #2563eb;
}

/* Chat Area */
QScrollArea {
    border: none;
    background-color: #0f0f12;
}

QScrollBar:vertical {
    border: none;
    background: #18181b;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3f3f46;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Input Area */
QTextEdit#input_box {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 8px;
    color: #f1f5f9;
    padding: 10px;
    font-size: 14px;
}

QTextEdit#input_box:focus {
    border: 1px solid #3b82f6;
}

QPushButton#send_btn {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px;
}

QPushButton#send_btn:disabled {
    background-color: #27272a;
    color: #71717a;
}
"""
