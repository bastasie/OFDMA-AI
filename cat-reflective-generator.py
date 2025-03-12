import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import requests
from io import BytesIO
import cv2
from tqdm import tqdm
import random
from numba import jit, prange

class CatReflectiveImageGenerator:
    """
    Image generator that applies reflective algebra transformations to cat images
    """
    
    def __init__(self, output_size=(1024, 1024), use_parallel=True):
        """
        Initialize the generator
        
        Parameters:
        - output_size: Size of output images (width, height)
        - use_parallel: Whether to use parallel processing
        """
        self.output_size = output_size
        self.use_parallel = use_parallel
        self.num_cpus = os.cpu_count()
        
        # Create coordinate grids for transformations
        self.x = np.linspace(-1, 1, output_size[0])
        self.y = np.linspace(-1, 1, output_size[1])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        print(f"Initialized generator with output size {output_size}")
        print(f"Number of CPU cores available: {self.num_cpus}")
    
    def download_cat_images(self, num_images=10, tags=None):
        """
        Download cat images from Cataas (Cat as a Service) which requires no API key
        
        Parameters:
        - num_images: Number of images to download
        - tags: Optional list of tags to filter cats (e.g., ['cute', 'funny'])
        
        Returns:
        - List of image arrays
        """
        cat_images = []
        
        # Base URL for Cataas
        base_url = 'https://cataas.com/cat'
        
        # Add tags if provided
        if tags:
            tag_str = '/'.join(tags)
            url = f"{base_url}/{tag_str}"
        else:
            url = base_url
        
        print(f"Downloading {num_images} cat images from Cataas...")
        
        # Download images
        for i in tqdm(range(num_images)):
            try:
                # Add a random parameter to avoid caching
                random_param = f"?random={random.randint(1, 100000)}"
                response = requests.get(url + random_param)
                
                # Check if the request was successful
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    img = img.resize(self.output_size)
                    cat_images.append(np.array(img) / 255.0)
                else:
                    print(f"Failed to download image: HTTP {response.status_code}")
                    
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error downloading image: {e}")
        
        print(f"Successfully downloaded {len(cat_images)} cat images")
        return cat_images
    
    def load_cat_images_from_folder(self, folder_path, num_images=None):
        """
        Load cat images from a local folder
        
        Parameters:
        - folder_path: Path to folder containing cat images
        - num_images: Maximum number of images to load (None=all)
        
        Returns:
        - List of image arrays
        """
        cat_images = []
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return cat_images
        
        # Get list of image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in valid_extensions]
        
        # Limit number of images if specified
        if num_images is not None and num_images < len(image_files):
            image_files = image_files[:num_images]
        
        print(f"Loading {len(image_files)} cat images from {folder_path}")
        
        # Load images
        for filename in tqdm(image_files):
            try:
                filepath = os.path.join(folder_path, filename)
                img = Image.open(filepath)
                img = img.convert('RGB')
                img = img.resize(self.output_size)
                cat_images.append(np.array(img) / 255.0)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        
        return cat_images
    
    @staticmethod
    @jit(nopython=True)
    def _extract_triplet_from_image(image):
        """
        Extract triplet components from an image using its RGB channels
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Triplet field as a 3D array (height, width, 3)
        """
        # Get image dimensions
        height, width, channels = image.shape
        
        # Create triplet field
        triplet_field = np.zeros((height, width, 3))
        
        # Extract RGB channels and normalize
        for i in range(height):
            for j in range(width):
                r = image[i, j, 0]
                g = image[i, j, 1]
                b = image[i, j, 2]
                
                # Create triplet components a, b, c
                a = r
                b = g
                c = b
                
                triplet_field[i, j, 0] = a
                triplet_field[i, j, 1] = b
                triplet_field[i, j, 2] = c
        
        return triplet_field
    
    def mirror_transform(self, triplet_field):
        """
        Apply the mirror principle from the paper (section 2)
        
        Parameters:
        - triplet_field: Input field of triplets [a, b, c]
        
        Returns:
        - Mirrored field
        """
        # Extract components
        a = triplet_field[:,:,0]
        b = triplet_field[:,:,1]
        c = triplet_field[:,:,2]
        
        # Apply mirror transformation ρ([A, B, C]) = [C, B, A] from section 2.2.2
        mirrored = np.zeros_like(triplet_field)
        mirrored[:,:,0] = c  # a becomes c
        mirrored[:,:,1] = b  # b stays the same
        mirrored[:,:,2] = a  # c becomes a
        
        return mirrored
    
    def apply_reflective_operator(self, triplet_field, alpha=2.0, iterations=1):
        """
        Apply the reflective operator # from the paper
        
        Parameters:
        - triplet_field: Input field of triplets
        - alpha: Parameter controlling the transformation intensity
        - iterations: Number of iterations to apply the operator
        
        Returns:
        - Transformed field
        """
        result = triplet_field.copy()
        
        for _ in range(iterations):
            # Extract components
            a = result[:,:,0]
            b = result[:,:,1]
            c = result[:,:,2]
            
            # Apply a transformation inspired by sections 1.4.1-1.4.3
            # Component-wise action with operators f, g, h
            new_a = 0.5 * (np.sin(np.pi * a) * np.cos(alpha * b * c) + a)
            new_b = 0.5 * (np.cos(np.pi * b) * np.sin(alpha * a * c) + b)
            new_c = 0.5 * (np.sin(np.pi * c) * np.cos(alpha * a * b) + c)
            
            # Update result
            result[:,:,0] = new_a
            result[:,:,1] = new_b
            result[:,:,2] = new_c
        
        return result
    
    def apply_baz_transform(self, image, alpha=2.0, t=0.5):
        """
        Apply a transform inspired by the BAZ wave function concept
        
        Parameters:
        - image: Input RGB image
        - alpha: BAZ parameter α controlling complexity
        - t: Time parameter
        
        Returns:
        - Transformed image
        """
        # Convert to frequency domain
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Perform FFT on each channel
        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        # Shift to center
        r_fft_shift = np.fft.fftshift(r_fft)
        g_fft_shift = np.fft.fftshift(g_fft)
        b_fft_shift = np.fft.fftshift(b_fft)
        
        # Get dimensions
        rows, cols = r.shape
        
        # Create the BAZ operator in frequency domain
        # Using equations from section 7.15
        # e^(-s²t/(1+4αt)) / (1+4αt)^(3/2)
        
        # Create meshgrid for frequencies
        u = np.fft.fftfreq(rows)
        v = np.fft.fftfreq(cols)
        u, v = np.meshgrid(u, v)
        s_squared = u**2 + v**2
        
        # BAZ operator
        baz_operator = np.exp(-s_squared * t / (1 + 4 * alpha * t)) / (1 + 4 * alpha * t)**(3/2)
        
        # Apply operator to each channel
        r_transformed = r_fft_shift * baz_operator
        g_transformed = g_fft_shift * baz_operator
        b_transformed = b_fft_shift * baz_operator
        
        # Inverse FFT
        r_back = np.fft.ifft2(np.fft.ifftshift(r_transformed))
        g_back = np.fft.ifft2(np.fft.ifftshift(g_transformed))
        b_back = np.fft.ifft2(np.fft.ifftshift(b_transformed))
        
        # Take real part and normalize
        r_back = np.real(r_back)
        g_back = np.real(g_back)
        b_back = np.real(b_back)
        
        # Create output image
        result = np.zeros_like(image)
        
        # Normalize each channel
        r_back = (r_back - np.min(r_back)) / (np.max(r_back) - np.min(r_back))
        g_back = (g_back - np.min(g_back)) / (np.max(g_back) - np.min(g_back))
        b_back = (b_back - np.min(b_back)) / (np.max(b_back) - np.min(b_back))
        
        # Combine channels
        result[:,:,0] = r_back
        result[:,:,1] = g_back
        result[:,:,2] = b_back
        
        return result
    
    def generate_reflective_cat_image(self, cat_image, alpha=2.5, iterations=3):
        """
        Generate a reflective transformation of a cat image
        
        Parameters:
        - cat_image: Input cat image
        - alpha: Parameter controlling transformation intensity
        - iterations: Number of iterations for reflective operator
        
        Returns:
        - Transformed image
        """
        # Start timer
        start_time = time.time()
        
        # Extract triplet field from cat image
        print("Extracting triplet field from cat image...")
        triplet_field = self._extract_triplet_from_image(cat_image)
        
        # Apply mirror transformation
        print("Applying mirror transformation...")
        mirrored = self.mirror_transform(triplet_field)
        
        # Apply reflective operator
        print("Applying reflective operator...")
        reflected = self.apply_reflective_operator(mirrored, alpha=alpha, iterations=iterations)
        
        # Apply BAZ transform
        print("Applying BAZ transform...")
        transformed = self.apply_baz_transform(cat_image, alpha=alpha, t=0.3)
        
        # Combine the original and transformed representations
        print("Creating final image...")
        reflected_rgb = np.zeros_like(cat_image)
        reflected_rgb[:,:,0] = reflected[:,:,0]
        reflected_rgb[:,:,1] = reflected[:,:,1]
        reflected_rgb[:,:,2] = reflected[:,:,2]
        
        # Blend original, reflected and transformed images
        final = 0.4 * reflected_rgb + 0.6 * transformed
        
        # Normalize
        final = np.clip(final, 0, 1)
        
        # Apply some final smoothing
        final = gaussian_filter(final, sigma=[0.5, 0.5, 0])
        
        print(f"Image generation completed in {time.time() - start_time:.2f} seconds")
        
        return final
    
    def apply_prime_count_segmentation(self, image, levels=10):
        """
        Apply a segmentation filter inspired by the prime count concept
        
        Parameters:
        - image: Input RGB image
        - levels: Number of quantization levels
        
        Returns:
        - Segmented image
        """
        # Convert to grayscale for segmentation
        if len(image.shape) == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image
        
        # Quantize to discrete levels
        quantized = np.floor(gray * levels) / levels
        
        # Define "prime" levels within our quantization range
        # Use actual prime number positions scaled to our range
        prime_positions = [2/levels, 3/levels, 5/levels, 7/levels]
        
        # Create output
        result = np.zeros_like(quantized)
        
        # Enhance regions near prime levels
        for prime in prime_positions:
            mask = np.abs(quantized - prime) < 0.5/levels
            result[mask] = 1.0
        
        # Create RGB result if input was RGB
        if len(image.shape) == 3:
            rgb_result = np.zeros_like(image)
            
            # Create custom colormap
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if result[i, j] > 0:
                        # Use yellow for "prime" regions
                        rgb_result[i, j, 0] = 1.0  # Red
                        rgb_result[i, j, 1] = 1.0  # Green
                        rgb_result[i, j, 2] = 0.0  # Blue
                    else:
                        # Use input colors with blue-green emphasis for non-prime
                        rgb_result[i, j, 0] = 0.1 * image[i, j, 0]  # Reduce red
                        rgb_result[i, j, 1] = 0.8 * image[i, j, 1]  # Emphasize green
                        rgb_result[i, j, 2] = 1.0 * image[i, j, 2]  # Emphasize blue
            
            return rgb_result
        
        return result
    
    def generate_bhlm_style_image(self, cat_image, alpha=2.5, apply_segmentation=True):
        """
        Generate an image with BHLM style (as mentioned in the paper)
        
        Parameters:
        - cat_image: Input cat image
        - alpha: Parameter controlling transformation
        - apply_segmentation: Whether to apply prime count segmentation
        
        Returns:
        - Styled image
        """
        # Apply BAZ transform
        transformed = self.apply_baz_transform(cat_image, alpha=alpha, t=0.5)
        
        # Apply reflective transformations
        triplet_field = self._extract_triplet_from_image(cat_image)
        mirrored = self.mirror_transform(triplet_field)
        reflected = self.apply_reflective_operator(mirrored, alpha=alpha, iterations=4)
        
        # Convert reflected triplet to RGB
        reflected_rgb = np.zeros_like(cat_image)
        reflected_rgb[:,:,0] = reflected[:,:,0]
        reflected_rgb[:,:,1] = reflected[:,:,1]
        reflected_rgb[:,:,2] = reflected[:,:,2]
        
        # Blend transformations for BHLM effect
        blended = 0.5 * transformed + 0.5 * reflected_rgb
        
        # Apply segmentation if requested
        if apply_segmentation:
            segmented = self.apply_prime_count_segmentation(blended)
            # Blend with original for better cat visibility
            final = 0.7 * segmented + 0.3 * blended
        else:
            final = blended
        
        # Normalize
        final = np.clip(final, 0, 1)
        
        return final
    
    def _process_single_image(self, input_data):
        """
        Helper function for parallel processing that processes a single image
        
        Parameters:
        - input_data: Tuple of (image, alpha, iterations, style)
        
        Returns:
        - Processed image
        """
        img, alpha, iterations, style = input_data
        
        if style == 'reflective':
            return self.generate_reflective_cat_image(img, alpha=alpha, iterations=iterations)
        else:  # bhlm style
            return self.generate_bhlm_style_image(img, alpha=alpha)
    
    def process_cat_image_batch(self, cat_images, alpha=2.5, iterations=3, style='reflective'):
        """
        Process a batch of cat images
        
        Parameters:
        - cat_images: List of cat image arrays
        - alpha: Parameter controlling transformation
        - iterations: Number of iterations for operator
        - style: 'reflective' or 'bhlm'
        
        Returns:
        - List of processed images
        """
        if not cat_images:
            print("No cat images provided!")
            return []
        
        start_time = time.time()
        num_images = len(cat_images)
        
        print(f"Processing {num_images} cat images...")
        
        if self.use_parallel and self.num_cpus > 1 and num_images > 1:
            print(f"Using parallel processing with {min(self.num_cpus, num_images)} workers")
            
            # Prepare input data tuples
            input_data = [(img, alpha, iterations, style) for img in cat_images]
            
            with ProcessPoolExecutor(max_workers=min(self.num_cpus, num_images)) as executor:
                results = list(executor.map(self._process_single_image, input_data))
        else:
            print("Processing sequentially...")
            results = []
            for img in tqdm(cat_images):
                if style == 'reflective':
                    processed = self.generate_reflective_cat_image(img, alpha=alpha, iterations=iterations)
                else:  # bhlm style
                    processed = self.generate_bhlm_style_image(img, alpha=alpha)
                results.append(processed)
        
        print(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
        
        return results
    
    def save_image(self, image, filename='reflective_cat.png'):
        """Save the generated image to a file"""
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Save using PIL
        img = Image.fromarray(img_uint8)
        img.save(filename)
        print(f"Image saved as {filename}")
    
    def save_image_batch(self, images, base_filename='reflective_cat', start_index=1):
        """Save a batch of images"""
        for i, img in enumerate(images):
            filename = f"{base_filename}_{i+start_index}.png"
            self.save_image(img, filename)
    
    def display_image(self, image):
        """Display the generated image"""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    def create_video_from_transformations(self, cat_image, frames=60, alpha_range=(1.5, 3.5), 
                                         style='reflective', output_file='cat_transformation.mp4'):
        """
        Create a video showing gradual transformation of a cat image
        
        Parameters:
        - cat_image: Input cat image
        - frames: Number of frames in video
        - alpha_range: Range of alpha values
        - style: 'reflective' or 'bhlm'
        - output_file: Output video filename
        """
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, 30, 
                               (self.output_size[1], self.output_size[0]))
        
        print(f"Creating {frames}-frame video...")
        
        # Generate frames
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], frames)
        for i, alpha in enumerate(tqdm(alpha_values)):
            # Generate frame
            if style == 'reflective':
                frame = self.generate_reflective_cat_image(cat_image, alpha=alpha, iterations=3)
            else:  # bhlm style
                frame = self.generate_bhlm_style_image(cat_image, alpha=alpha)
            
            # Convert to BGR for OpenCV
            frame_bgr = (frame[:, :, ::-1] * 255).astype(np.uint8)
            
            # Write frame
            video.write(frame_bgr)
        
        # Release video writer
        video.release()
        print(f"Video saved as {output_file}")

# Example usage
if __name__ == "__main__":
    # Create generator with parallel processing set to False by default for better compatibility
    generator = CatReflectiveImageGenerator(output_size=(1024, 1024), use_parallel=False)
    
    # Get cat images from Cataas (no API key required)
    print("Downloading cat images from Cataas...")
    cat_images = generator.download_cat_images(num_images=5)
    
    # Alternative: specify tags for the cats
    # cat_images = generator.download_cat_images(num_images=5, tags=['cute'])
    
    # Alternative: load cat images from folder if available
    if not cat_images:
        print("Trying to load cat images from local folder...")
        cat_images = generator.load_cat_images_from_folder("cat_images", num_images=5)
    
    # If no images were loaded, use a placeholder
    if not cat_images:
        print("No cat images available, generating a placeholder...")
        # Create a solid color image as placeholder
        placeholder = np.ones((1024, 1024, 3))
        placeholder[:,:,0] = 0.8  # R
        placeholder[:,:,1] = 0.7  # G
        placeholder[:,:,2] = 0.6  # B
        cat_images = [placeholder]
    
    # Process in reflective style
    print("\nGenerating reflective style images...")
    reflective_images = generator.process_cat_image_batch(cat_images, style='reflective')
    generator.save_image_batch(reflective_images, base_filename='reflective_cat')
    
    # Process in BHLM style (this style creates images similar to the shared example)
    print("\nGenerating BHLM style images (blue-green-yellow patterns)...")
    bhlm_images = generator.process_cat_image_batch(cat_images, style='bhlm')
    generator.save_image_batch(bhlm_images, base_filename='bhlm_cat')
    
    # Create video transformation for the first cat
    if cat_images:
        print("\nCreating transformation video...")
        generator.create_video_from_transformations(
            cat_images[0], 
            frames=90, 
            style='bhlm',
            output_file='cat_bhlm_transformation.mp4'
        )
    
    print("\nAll processing completed successfully.")
    print("The BHLM style images (bhlm_cat_*.png) should have similar blue-green-yellow patterns to your example image.")
