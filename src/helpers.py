import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import ultralytics as u

def thermal_image_to_array(image_path):
    with Image.open(image_path) as img:
        img_array = np.array(img)
    return img_array

def low_pass_filter_fft(image, cutoff):
    """
    Apply a low-pass filter to the image using FFT.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input image (could be noisy).
    cutoff : int or float
        Radius of the circular low-pass filter (in pixels).
        
    Returns:
    --------
    filtered_image : 2D numpy array
        The reconstructed image after filtering.
    """
    # Compute the 2D FFT and shift the zero frequency component to the center
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    
    # Create a circular low-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
    mask = np.zeros_like(image)
    mask[mask_area] = 1
    
    # Apply the mask to the shifted FFT
    Fshift_filtered = Fshift * mask
    
    # Shift back (inverse shift) and compute the inverse FFT
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    filtered_image = np.fft.ifft2(F_ishift)
    filtered_image = np.abs(filtered_image)
    
    return filtered_image


def display_thermal_data(image: np.ndarray, auto_scale: bool = True) -> None:
    """
    Displays thermal image data using a thermal colormap.
    
    Parameters
    ----------
    image : np.ndarray
        A 2D NumPy array representing the thermal image data.
    auto_scale : bool, optional
        If True, the display scales to the image's min and max values.
        If False, it uses the full 16-bit range (0 to 65535). Default is True.
    """
    plt.figure(figsize=(8, 6))
    
    # Determine display range: use auto scaling or assume full 16-bit range.
    if auto_scale:
        vmin, vmax = image.min(), image.max()
    else:
        vmin, vmax = 0, 65535
    
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title("Thermal Image Data")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label="Intensity")
    plt.axis("off")
    plt.show()

def get_exif_metadata(file_path: str) -> dict:
    """
    Extracts EXIF metadata from an image file.

    Parameters
    ----------
    file_path : str
        Path to the image file.

    Returns
    -------
    dict
        A dictionary of EXIF metadata with human-readable tag names.
    """
    metadata = {}
    with Image.open(file_path) as img:
        exif_data = img.getexif()
        if exif_data:
            metadata = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    return metadata

def radial_profile(data, center=None):
    """
    Compute the radial profile of a 2D array.
    """
    y, x = np.indices(data.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    # Compute distances from the center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.flatten()
    data = data.flatten()
    
    # Bin by integer value of the radius
    r_int = r.astype(np.int32)
    tbin = np.bincount(r_int, weights=data)
    nr = np.bincount(r_int)
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

def plot_fft_radial_profile(image):
    # Compute FFT and shift it
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    magnitude_spectrum = np.abs(Fshift)
    # Use logarithmic scale for better visualization
    magnitude_log = np.log1p(magnitude_spectrum)
    
    # Compute the radial profile
    profile = radial_profile(magnitude_log)
    
    plt.figure(figsize=(8, 5))
    plt.plot(profile, 'b-', marker='o')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Average log magnitude')
    plt.title('Radial Profile of FFT Magnitude Spectrum')
    plt.grid(True)
    plt.show()

def low_pass_filter_fft(image, cutoff):
    """
    Apply a circular low-pass filter using FFT.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input image.
    cutoff : int or float
        Radius (in pixels) for the low-pass filter.
        
    Returns:
    --------
    filtered_image : 2D numpy array
        Reconstructed image after low-pass filtering.
    """
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
    Fshift_filtered = Fshift * mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    filtered = np.fft.ifft2(F_ishift)
    return np.abs(filtered)

def high_pass_filter_fft(image, cutoff):
    """
    Apply a circular high-pass filter using FFT.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input image.
    cutoff : int or float
        Radius (in pixels) for the high-pass filter.
        
    Returns:
    --------
    filtered_image : 2D numpy array
        Reconstructed image after high-pass filtering.
    """
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 > cutoff**2
    Fshift_filtered = Fshift * mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    filtered = np.fft.ifft2(F_ishift)
    return np.abs(filtered)



def load_and_scale_images_from_directory(directory):
    """
    Load all images from a directory, apply linear scaling to convert non-8-bit images
    to 8-bit, and return a list of tuples (file_path, PIL Image).

    Parameters:
        directory (str): Path to the directory containing image files.

    Returns:
        list: A list of tuples, where each tuple is (file_path, PIL Image object).
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')
    images = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(directory, filename)
            try:
                img = Image.open(file_path)
                img_np = np.array(img)
                
                # If the image is not already 8-bit, apply linear scaling
                if img_np.dtype != np.uint8:
                    scaled = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
                    img_np_8bit = scaled.astype(np.uint8)
                    img = Image.fromarray(img_np_8bit)
                
                images.append((file_path, img))
            except Exception as e:
                print(f"Failed to load image {file_path}: {e}")
    
    return images

# Example usage:
# image_list = load_and_scale_images_from_directory("path/to/your/images")
# print(f"Loaded {len(image_list)} images.")

def yolo_to_bbox(yolo_bbox, image_width=512, image_height=640):
    # yolo_bbox is [x_center, y_center, width, height] with values between 0 and 1
    x_center, y_center, w, h = yolo_bbox
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    x2 = (x_center + w / 2) * image_width
    y2 = (y_center + h / 2) * image_height
    return x1, y1, x2, y2

# Example usage:
if __name__ == "__main__":
    file_path = "test_images_16_bit/image_2.tiff"
    image = thermal_image_to_array(file_path)
    display_thermal_data(image)