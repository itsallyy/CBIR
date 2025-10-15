import sys
from PyQt5.QtWidgets import QApplication
from features import FeatureExtractor, build_database
from cbir_app import CBIRApp

def main():
    image_folder = "data"
    feature_extractor = FeatureExtractor()
    database = build_database(image_folder, feature_extractor)
    app = QApplication(sys.argv)
    window = CBIRApp(feature_extractor, database)
    window.show()
    sys.exit(app.exec_())


main()