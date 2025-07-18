import cv2
import numpy as np
import sys

# --- Global variables for mouse callback ---
flag_destination_points = []
current_image_display = None
click_count = 0


def mouse_callback_flag_dst(event, x, y, flags, param):
    """Mouse callback function to select destination points on the flag image."""
    global flag_destination_points, click_count, current_image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count < 4:
            flag_destination_points.append((x, y))
            print(
                f"Flag Destination Point {len(flag_destination_points)}: ({x}, {y})")
            # Draw a circle to visualize the selected point
            cv2.circle(current_image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(
                "Select 4 points on Flag (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", current_image_display)
            click_count += 1
            if click_count == 4:
                print("All 4 flag destination points selected.")
                cv2.destroyWindow(
                    "Select 4 points on Flag (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")


def create_flag_mask(flag_image, dest_points):
    """
    Creates a robust mask for the flag using GrabCut and user-provided points.

    Args:
        flag_image (np.array): The image of the flag.
        dest_points (list): A list of 4 (x, y) tuples selected by the user.

    Returns:
        np.array: refined flag mask.
    """
    print("Generating robust flag mask using GrabCut...")

    # Define a rough ROI based on user-selected points
    dest_points_np = np.array(dest_points)
    x_min, y_min = np.min(dest_points_np, axis=0)
    x_max, y_max = np.max(dest_points_np, axis=0)

    # Add a small buffer to the ROI to ensure the entire flag is included
    buffer = 20
    roi = (max(0, x_min - buffer), max(0, y_min - buffer),
           min(flag_image.shape[1], x_max - x_min + 2 * buffer),
           min(flag_image.shape[0], y_max - y_min + 2 * buffer))

    # Ensure ROI is valid
    if roi[2] <= 0 or roi[3] <= 0:
        print("Error: Invalid ROI for GrabCut. Exiting.")
        return np.zeros(flag_image.shape[:2], dtype=np.uint8)

    # Initialize GrabCut
    mask_grabcut = np.zeros(flag_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Use the ROI to initialize GrabCut
    try:
        cv2.grabCut(flag_image, mask_grabcut, roi, bgdModel,
                    fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"GrabCut Error: {e}")
        return np.zeros(flag_image.shape[:2], dtype=np.uint8)

    # Create a binary mask from the GrabCut result
    # The mask contains GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD.
    # We consider GC_FGD and GC_PR_FGD as our foreground.
    mask_final = np.where((mask_grabcut == cv2.GC_FGD) | (
        mask_grabcut == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Refine the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_final = cv2.morphologyEx(
        mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_final = cv2.morphologyEx(
        mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the largest contour to isolate the main flag body (removes noise)
    contours, _ = cv2.findContours(
        mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask_cleaned = np.zeros_like(mask_final)
        cv2.drawContours(final_mask_cleaned, [
                         largest_contour], -1, 255, cv2.FILLED)
        mask_final = final_mask_cleaned
    else:
        print("Warning: No prominent contours found after GrabCut. Using original GrabCut output.")

    # Apply a slight Gaussian blur to the mask edges for smoother blending (feathering).
    mask_final = cv2.GaussianBlur(mask_final, (15, 15), 0)

    return mask_final


def main():
    global flag_destination_points, current_image_display, click_count

    # Input/Output File Paths
    pattern_path = "Pattern.png"
    flag_path = "Flag.png"
    output_path = "Output.jpg"

    # Load Images
    pattern_img = cv2.imread(pattern_path)
    flag_img = cv2.imread(flag_path)

    # Basic Error Handling
    if pattern_img is None:
        print(f"Error: Could not load pattern image at {pattern_path}")
        sys.exit(1)
    if flag_img is None:
        print(f"Error: Could not load flag image at {flag_path}")
        sys.exit(1)

    # Define Source Points from Pattern (Automated)
    h_pattern, w_pattern = pattern_img.shape[:2]
    src_points = np.float32([
        [0, 0],
        [w_pattern - 1, 0],
        [w_pattern - 1, h_pattern - 1],
        [0, h_pattern - 1]
    ])
    print("\nPattern source points (full rectangle) defined automatically.")

    # Interactive Point Selection for Destination (User picks on Flag)
    print("\n--- Point Selection for Flag (Destination) ---")
    print("Click on the FOUR corners of the WHITE FLAG fabric where you want the pattern to be mapped.")
    print("Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
    print("Try to follow the natural curves/corners of the flag for best results.")

    current_image_display = flag_img.copy()
    cv2.imshow("Select 4 points on Flag (Top-Left, Top-Right, Bottom-Right, Bottom-Left)",
               current_image_display)
    cv2.setMouseCallback(
        "Select 4 points on Flag (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", mouse_callback_flag_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(flag_destination_points) != 4:
        print("Error: 4 destination points not selected on the flag. Exiting.")
        sys.exit(1)

    dst_points = np.float32(flag_destination_points)

    # Generate a precise mask for the flag using the new function
    print("\nGenerating flag mask...")
    flag_mask = create_flag_mask(flag_img, flag_destination_points)

    cv2.imshow("Generated Flag Mask (with feathering)", flag_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Warp Pattern to align with the perspective and curvature of Flag
    print("Warping pattern image...")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_pattern = cv2.warpPerspective(
        pattern_img, M, (flag_img.shape[1], flag_img.shape[0]))

    # Blend the warped pattern with Flag for realism (Advanced Luminosity Blending)
    print("Blending warped pattern with flag folds (applying flag's luminosity and color bias)...")

    # Convert flag image to grayscale to extract its luminosity for shading transfer
    flag_gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)

    # Normalize flag_gray to 0-255 range to ensure it covers the full spectrum
    flag_detail = cv2.normalize(flag_gray, None, 0, 255, cv2.NORM_MINMAX)
    flag_detail_float = flag_detail.astype(np.float32) / 255.0

    # Convert warped_pattern to HSV to manipulate its V (Value/Luminosity) channel
    warped_pattern_hsv = cv2.cvtColor(warped_pattern, cv2.COLOR_BGR2HSV)

    # Replace the Value channel of the warped pattern with the normalized luminosity from the flag.
    # This makes the pattern take on the shadows and highlights of the flag.
    # Apply normalized flag luminosity to V channel
    warped_pattern_hsv[:, :, 2] = (flag_detail_float * 255).astype(np.uint8)

    blended_pattern = cv2.cvtColor(warped_pattern_hsv, cv2.COLOR_HSV2BGR)

    # Final Compositing with Feathered Mask
    print("Compositing final image...")

    # Create a mask for the warped pattern itself (where it's not black due to warp)
    warped_pattern_existence_mask = cv2.cvtColor(
        warped_pattern, cv2.COLOR_BGR2GRAY)
    _, warped_pattern_existence_mask = cv2.threshold(
        warped_pattern_existence_mask, 1, 255, cv2.THRESH_BINARY)

    # Combine the precise flag_mask (which is already feathered) with the warped_pattern_existence_mask
    # This new mask ensures smooth edges and only applies where the pattern exists and the flag is present
    final_blend_mask = cv2.bitwise_and(
        flag_mask, warped_pattern_existence_mask)

    # Normalize the mask to 0-1 for alpha blending
    final_blend_mask_float = final_blend_mask.astype(np.float32) / 255.0
    final_blend_mask_3ch_float = cv2.cvtColor(
        (final_blend_mask_float * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0

    final_output = (blended_pattern.astype(np.float32) * final_blend_mask_3ch_float) + \
                   (flag_img.astype(np.float32) * (1 - final_blend_mask_3ch_float))

    final_output = np.clip(final_output, 0, 255).astype(np.uint8)

    # Save the final composited image
    cv2.imwrite(output_path, final_output)
    print(f"\nFinal composited image saved as {output_path}")

    # Display final result
    cv2.imshow("Final Composited Image", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
