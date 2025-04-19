import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QFormLayout, QMessageBox, \
    QMainWindow, QPushButton
from PyQt5.QtGui import QIntValidator, QDoubleValidator


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        # 创建QLabel控件和QLineEdit控件
        self.label = QLabel('Enter a number:', self)
        self.lineedit = QLineEdit(self)

        # 设置默认值和浮点数校验器
        self.lineedit.setText('0.0')
        validator = QDoubleValidator()
        self.lineedit.setValidator(validator)

        # 创建QPushButton控件
        self.button = QPushButton('Calculate', self)
        self.button.setToolTip('Click to calculate the square of the input value')
        self.button.clicked.connect(self.calculate)

        # 设置控件布局
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.lineedit)
        vbox.addWidget(self.button)

        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        # 设置窗口属性
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('QLineEdit Example')
        self.show()

    def calculate(self):

        # 获取QLineEdit控件中的值
        value = self.lineedit.text()

        # 如果值为空，则弹出警告框
        if not value:
            QMessageBox.warning(self, 'Warning', 'Please enter a number.')
            return

        # 进行计算，并显示结果
        try:
            result = float(value) ** 2
            QMessageBox.information(self, 'Result', f'The square of {value} is {result}.')
        except ValueError:
            QMessageBox.warning(self, 'Warning', 'Please enter a valid number.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
