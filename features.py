import os
import glob
import io
import pickle
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


FEATURE_WEIGHTS = {
    "shape": 1.0,
    "color": 1.0,
    "color_hist": 1.0,
    "sift": 1.0,
    "harris": 1.0,
    "hog": 1.0,
    "glcm": 1.0,
    "lbp": 1.0,
    "orb": 1.0,
    "cnn": 1.0
}


class FeatureExtractor:
    
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.cnn_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    def remove_background(self, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        output = remove(img_bytes)
        result = Image.open(io.BytesIO(output)).convert("RGBA")
        return result

    def get_mask_from_alpha(self, image_rgba):
        image_np = np.array(image_rgba)
        alpha = image_np[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        return mask

    def extract_shape_features(self, mask):
        # Always return 10 values
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or hierarchy is None:
            return [0] * 10
        outer_contours = []
        outer_indices = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:
                outer_contours.append(contours[i])
                outer_indices.append(i)
        if len(outer_contours) == 0:
            return [0] * 10
        largest_contour = max(outer_contours, key=cv2.contourArea)
        index_in_outer = None
        for idx, cnt in enumerate(outer_contours):
            if np.array_equal(cnt, largest_contour):
                index_in_outer = idx
                break
        if index_in_outer is None:
            index_in_outer = 0
        largest_index = outer_indices[index_in_outer]
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        hole_area = 0
        for i, h in enumerate(hierarchy[0]):
            if h[3] == largest_index:
                hole_area += cv2.contourArea(contours[i])
        har = hole_area / area if area > 0 else 0
        x, y, w, h_rect = cv2.boundingRect(largest_contour)
        rectangularity = area / (w * h_rect) if w * h_rect > 0 else 0
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
        else:
            eccentricity = 0
        # Fourier descriptors
        contour_array = largest_contour.reshape(-1, 2)
        complex_contour = contour_array[:, 0] + 1j * contour_array[:, 1]
        fourier_result = np.fft.fft(complex_contour)
        num_fd = 5
        fd = []
        for i in range(1, num_fd + 1):
            if i < len(fourier_result):
                fd.append(np.abs(fourier_result[i]))
            else:
                fd.append(0)
        # Return 5 basic shape features + 5 Fourier descriptors = 10 values
        return [len(outer_contours), har, eccentricity, rectangularity, circularity] + fd

    def extract_color_features(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        means, stds = cv2.meanStdDev(hsv)
        return np.concatenate([means.flatten(), stds.flatten()])

    def extract_color_histogram_features(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0,256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        return np.array(hist_features)

    def extract_sift_features(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        num_keypoints = len(keypoints)
        if descriptors is not None and len(descriptors) > 0:
            avg_descriptor = np.mean(descriptors, axis=0)
        else:
            avg_descriptor = np.zeros(128)
        return np.concatenate([[num_keypoints], avg_descriptor])

    def extract_harris_features(self, gray):
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        threshold = 0.01 * dst.max()
        corners = (dst > threshold).sum()
        return np.array([corners])

    def extract_hog_features(self, gray):
        gray_resized = cv2.resize(gray, (128, 128))
        hog_descriptor, _ = hog(
            gray_resized, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm='L2-Hys',
            visualize=True, feature_vector=True
        )
        expected_dim = 8100
        current_dim = hog_descriptor.shape[0]
        if current_dim < expected_dim:
            hog_descriptor = np.pad(hog_descriptor, (0, expected_dim - current_dim), 'constant')
        elif current_dim > expected_dim:
            hog_descriptor = hog_descriptor[:expected_dim]
        return hog_descriptor

    def extract_glcm_features(self, gray):
        gray_quantized = np.uint8((gray / 256.0) * 8)
        glcm = graycomatrix(gray_quantized, distances=[1], angles=[0],
                            levels=8, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0,0]
        energy = graycoprops(glcm, 'energy')[0,0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
        glcm_probs = glcm[:,:,0,0]
        glcm_probs = glcm_probs[glcm_probs>0]
        entropy = -np.sum(glcm_probs * np.log2(glcm_probs))
        return np.array([contrast, entropy, energy, homogeneity, dissimilarity])

    def extract_lbp_features(self, gray):
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        expected_dim = n_points + 2
        current_dim = hist.shape[0]
        if current_dim < expected_dim:
            hist = np.pad(hist, (0, expected_dim - current_dim), 'constant')
        elif current_dim > expected_dim:
            hist = hist[:expected_dim]
        return hist

    def extract_orb_features(self, gray):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        num_keypoints = len(keypoints)
        if descriptors is not None and len(descriptors) > 0:
            avg_descriptor = np.mean(descriptors, axis=0)
        else:
            avg_descriptor = np.zeros(32)
        return np.concatenate([[num_keypoints], avg_descriptor])

    def extract_cnn_features(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224,224))
        image_array = img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = self.cnn_model.predict(image_array)
        return features.flatten()

    def extract_features(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print("Error reading image:", image_path)
            return None
        return self.extract_features_from_image(image_bgr)

    def extract_features_from_image(self, image_bgr):
        if image_bgr is None:
            return None
        image_for_color = image_bgr.copy()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_rgba = self.remove_background(pil_image)
        mask = self.get_mask_from_alpha(image_rgba)
        shape_features = self.extract_shape_features(mask)
        color_features = self.extract_color_features(image_for_color)
        color_hist_features = self.extract_color_histogram_features(image_for_color)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sift_features = self.extract_sift_features(gray)
        harris_features = self.extract_harris_features(gray)
        hog_features = self.extract_hog_features(gray)
        glcm_features = self.extract_glcm_features(gray)
        lbp_features = self.extract_lbp_features(gray)
        orb_features = self.extract_orb_features(gray)
        cnn_features = self.extract_cnn_features(image_bgr)
        features = {
            "shape": np.array(shape_features).flatten(),
            "color": color_features.flatten(),
            "color_hist": color_hist_features.flatten(),
            "sift": sift_features.flatten(),
            "harris": harris_features.flatten(),
            "hog": hog_features.flatten(),
            "glcm": glcm_features.flatten(),
            "lbp": lbp_features.flatten(),
            "orb": orb_features.flatten(),
            "cnn": cnn_features.flatten()
        }

        EXPECTED_DIMS = {
            "shape": 10,
            "color": 6,
            "color_hist": 48,
            "sift": 129,
            "harris": 1,
            "hog": 8100,
            "glcm": 5,
            "lbp": 26,
            "orb": 33,
            "cnn": 512
        }
        for key in features:
            current = features[key].shape[0]
            expected = EXPECTED_DIMS[key]
            if current < expected:
                features[key] = np.pad(features[key], (0, expected - current), 'constant')
            elif current > expected:
                features[key] = features[key][:expected]
        return features


def build_database(image_folder, feature_extractor, database_path="features_database.pkl"):
    if os.path.exists(database_path):
        with open(database_path, "rb") as f:
            database = pickle.load(f)
        print("Loaded feature database from", database_path)
        return database
    database = {}
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))
    for img_path in image_paths:
        print("Processing", img_path)
        features = feature_extractor.extract_features(img_path)
        if features is not None:
            database[img_path] = features
    with open(database_path, "wb") as f:
        pickle.dump(database, f)
    print("Feature database built and saved to", database_path)
    return database