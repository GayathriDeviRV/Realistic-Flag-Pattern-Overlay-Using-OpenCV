import gradio as gr
import cv2
import numpy as np
import sys
import os


def create_flag_mask(flag_image, dest_points):
    """
    Creates a robust mask for the flag using GrabCut and user-provided points.

    Args:
        flag_image (np.array): The image of the flag.
        dest_points (list): A list of 4 (x, y) tuples selected by the user.

    Returns:
        np.array: refined flag mask.
    """
    if flag_image is None or not dest_points:
        return None

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
        print("Error: Invalid ROI for GrabCut.")
        return np.zeros(flag_image.shape[:2], dtype=np.uint8)

    # Initialize GrabCut
    mask_grabcut = np.zeros(flag_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(flag_image, mask_grabcut, roi, bgdModel,
                    fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"GrabCut Error: {e}")
        return np.zeros(flag_image.shape[:2], dtype=np.uint8)

    mask_final = np.where((mask_grabcut == cv2.GC_FGD) | (
        mask_grabcut == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Refine the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_final = cv2.morphologyEx(
        mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_final = cv2.morphologyEx(
        mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the largest contour to isolate the main flag body
    contours, _ = cv2.findContours(
        mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask_cleaned = np.zeros_like(mask_final)
        cv2.drawContours(final_mask_cleaned, [
                         largest_contour], -1, 255, cv2.FILLED)
        mask_final = final_mask_cleaned

    # Apply a slight Gaussian blur to the mask edges for smoother blending
    mask_final = cv2.GaussianBlur(mask_final, (15, 15), 0)

    return mask_final


def process_image(flag_img, pattern_img, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y):
    """
    Performs the full perspective warp and blending.

    This function will be called when the user clicks the 'Process' button.
    """
    if flag_img is None or pattern_img is None:
        return None, "Please upload both images."

    # Define Source Points from Pattern (Automated)
    h_pattern, w_pattern = pattern_img.shape[:2]
    src_points = np.float32([
        [0, 0],
        [w_pattern - 1, 0],
        [w_pattern - 1, h_pattern - 1],
        [0, h_pattern - 1]
    ])

    # Define Destination Points from Sliders
    dst_points = np.float32([
        [tl_x, tl_y],
        [tr_x, tr_y],
        [br_x, br_y],
        [bl_x, bl_y]
    ])

    # Generate a precise mask for the flag
    flag_destination_points = dst_points.astype(int).tolist()
    flag_mask = create_flag_mask(flag_img, flag_destination_points)
    if flag_mask is None:
        return None, "Failed to create flag mask. Please check inputs."

    # Warp Pattern to align with the flag
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_pattern = cv2.warpPerspective(
        pattern_img, M, (flag_img.shape[1], flag_img.shape[0]))

    # Blend with Luminosity
    flag_gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    flag_detail = cv2.normalize(flag_gray, None, 0, 255, cv2.NORM_MINMAX)
    flag_detail_float = flag_detail.astype(np.float32) / 255.0

    warped_pattern_hsv = cv2.cvtColor(warped_pattern, cv2.COLOR_BGR2HSV)
    warped_pattern_hsv[:, :, 2] = (flag_detail_float * 255).astype(np.uint8)
    blended_pattern = cv2.cvtColor(warped_pattern_hsv, cv2.COLOR_HSV2BGR)

    # Final Compositing with Feathered Mask
    warped_pattern_existence_mask = cv2.cvtColor(
        warped_pattern, cv2.COLOR_BGR2GRAY)
    _, warped_pattern_existence_mask = cv2.threshold(
        warped_pattern_existence_mask, 1, 255, cv2.THRESH_BINARY)

    final_blend_mask = cv2.bitwise_and(
        flag_mask, warped_pattern_existence_mask)

    final_blend_mask_float = final_blend_mask.astype(np.float32) / 255.0
    final_blend_mask_3ch_float = cv2.cvtColor(
        (final_blend_mask_float * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0

    final_output = (blended_pattern.astype(np.float32) * final_blend_mask_3ch_float) + \
        (flag_img.astype(np.float32) * (1 - final_blend_mask_3ch_float))

    final_output = np.clip(final_output, 0, 255).astype(np.uint8)

    return final_output, "Success"


def live_preview(flag_img, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y):
    """
    Draws the corner points and connecting lines on the flag image for a live preview.

    This function will be triggered by slider changes.
    """
    # Add a check to ensure all inputs are valid before proceeding
    if flag_img is None or any(x is None for x in [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]):
        return None

    display_img = flag_img.copy()

    # Now that we've checked for None, we can safely convert to int
    points = [
        (int(tl_x), int(tl_y)),
        (int(tr_x), int(tr_y)),
        (int(br_x), int(br_y)),
        (int(bl_x), int(bl_y))
    ]

    # Draw points
    for i, p in enumerate(points):
        # Green circle for the point
        cv2.circle(display_img, p, 7, (0, 255, 0), -1)
        cv2.putText(display_img, str(
            i), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw lines connecting the points
    cv2.polylines(display_img, [np.array(points, np.int32).reshape(
        (-1, 1, 2))], True, (0, 0, 255), 2)  # Red line

    return display_img


# Gradio Interface Setup
# Create a demo to hold all components
with gr.Blocks() as demo:
    gr.Markdown("# Realistic Flag Pattern Overlay ")
    gr.Markdown(
        "Upload a pattern and a flag image, then use the sliders to select the four corners of the flag to apply the pattern.")

    with gr.Row():
        pattern_input = gr.Image(label="Pattern to Apply", type="numpy")
        flag_input = gr.Image(label="Flag Image", type="numpy")

    gr.Markdown("## Adjust Corner Coordinates")

    with gr.Row():
        with gr.Column():
            tl_x = gr.Slider(minimum=0, maximum=1000,
                             value=162, label="Top-Left X")
            tl_y = gr.Slider(minimum=0, maximum=1000,
                             value=197, label="Top-Left Y")
        with gr.Column():
            tr_x = gr.Slider(minimum=0, maximum=1000,
                             value=655, label="Top-Right X")
            tr_y = gr.Slider(minimum=0, maximum=1000,
                             value=229, label="Top-Right Y")
        with gr.Column():
            br_x = gr.Slider(minimum=0, maximum=1000,
                             value=737, label="Bottom-Right X")
            br_y = gr.Slider(minimum=0, maximum=1000,
                             value=527, label="Bottom-Right Y")
        with gr.Column():
            bl_x = gr.Slider(minimum=0, maximum=1000,
                             value=236, label="Bottom-Left X")
            bl_y = gr.Slider(minimum=0, maximum=1000,
                             value=573, label="Bottom-Left Y")

    # ADDED: Examples section for quick value loading
    gr.Examples(
        examples=[
            [75, 49, 263, 123, 240, 274, 72, 177],
            [160, 200, 650, 220, 730, 510, 240, 580]
        ],
        inputs=[tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y],
        label="Quickly Load Example Corner Coordinates"
    )

    # Live preview section
    with gr.Row():
        preview_image = gr.Image(
            label="Flag Image with Live Point Preview", interactive=False)

    with gr.Row():
        process_btn = gr.Button("Apply Pattern to Flag", variant="primary")

    # Final output section
    with gr.Row():
        output_image = gr.Image(
            label="Final Composited Image", interactive=False)
        output_message = gr.Textbox(label="Status")

    # Gradio Interactions ---

    # Get initial image dimensions and update slider ranges
    def update_ranges(flag_img):
        if flag_img is not None:
            h, w = flag_img.shape[:2]
            # Set a more appropriate range based on image size
            return gr.Slider(minimum=0, maximum=w, value=w//4), \
                gr.Slider(minimum=0, maximum=h, value=h//4), \
                gr.Slider(minimum=0, maximum=w, value=w*3//4), \
                gr.Slider(minimum=0, maximum=h, value=h//4), \
                gr.Slider(minimum=0, maximum=w, value=w*3//4), \
                gr.Slider(minimum=0, maximum=h, value=h*3//4), \
                gr.Slider(minimum=0, maximum=w, value=w//4), \
                gr.Slider(minimum=0, maximum=h, value=h*3//4)
        return gr.Slider(), gr.Slider(), gr.Slider(), gr.Slider(), gr.Slider(), gr.Slider(), gr.Slider(), gr.Slider()

    # When a new flag image is uploaded, update slider ranges and show initial preview
    flag_input.upload(
        update_ranges,
        inputs=[flag_input],
        outputs=[tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y],
        queue=False
    )

    # Call live_preview whenever any slider changes or a new image is uploaded
    slider_inputs = [flag_input, tl_x, tl_y,
                     tr_x, tr_y, br_x, br_y, bl_x, bl_y]

    for slider in [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]:
        slider.change(
            live_preview,
            inputs=slider_inputs,
            outputs=preview_image,
            queue=False  # Essential for responsive live preview
        )

    flag_input.change(
        live_preview,
        inputs=slider_inputs,
        outputs=preview_image,
        queue=False
    )

    # Process button click event
    process_btn.click(
        process_image,
        inputs=[flag_input, pattern_input, tl_x,
                tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y],
        outputs=[output_image, output_message]
    )

demo.launch(debug=True)
