# Quantify fluorescence intensity in the two uploaded eye images.
# Steps:
# 1) Load images
# 2) Extract red channel (fluorescence)
# 3) Segment eyes automatically with Otsu threshold + morphology
# 4) Compute background as median intensity outside the mask
# 5) Measure per-eye and per-image metrics: area, mean, median, integrated density, background-corrected versions
# 6) Save overlay images and a CSV; display a results table and a simple bar chart for means

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
import os
import glob

# 240229_hyP_G1_raisingフォルダ内のすべてのJPGファイルを取得
image_folder = "/Users/ohde/Documents/KULIP/projects/Gbim_transgenesis/hypbase/RAISING/240229_hyP_G1_raising"
paths = glob.glob(os.path.join(image_folder, "*.jpg"))
paths.sort()  # ファイル名順にソート

print(f"処理対象ファイル数: {len(paths)}")
for path in paths:
    print(f"  - {os.path.basename(path)}")

results = []

def analyze_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # take red channel (0-255)
    red = arr[..., 0].astype(np.float32)
    
    # Smooth a bit to stabilize thresholding
    from scipy.ndimage import gaussian_filter
    from skimage.exposure import equalize_adapthist
    red_blur = gaussian_filter(red, sigma=1.2)
    
    # Try multiple thresholding approaches for robust segmentation
    def try_thresholding_methods(image):
        methods = []
        
        # Method 1: Standard Otsu
        try:
            thresh_otsu = filters.threshold_otsu(image)
            mask_otsu = image > thresh_otsu
            methods.append(("Otsu", mask_otsu, thresh_otsu))
        except:
            pass
        
        # Method 2: Local Otsu (adaptive) - use larger disk to reduce over-segmentation
        try:
            from skimage.filters import rank
            from skimage.morphology import disk
            # Use larger disk for local thresholding to reduce over-segmentation
            local_otsu = rank.otsu(image.astype(np.uint8), disk(30))
            mask_local = image > local_otsu
            methods.append(("Local_Otsu", mask_local, np.mean(local_otsu)))
        except:
            pass
        
        # Method 3: Triangle method (good for bimodal distributions with unequal peaks)
        try:
            thresh_triangle = filters.threshold_triangle(image)
            mask_triangle = image > thresh_triangle
            methods.append(("Triangle", mask_triangle, thresh_triangle))
        except:
            pass
        
        # Method 4: Percentile-based threshold (for very weak signals)
        thresh_percentile = np.percentile(image, 85)  # Top 15% of pixels
        mask_percentile = image > thresh_percentile
        methods.append(("Percentile_85", mask_percentile, thresh_percentile))
        
        # Method 5: Enhanced contrast + Otsu
        try:
            enhanced = equalize_adapthist(image / 255.0, clip_limit=0.02) * 255
            thresh_enhanced = filters.threshold_otsu(enhanced)
            mask_enhanced = enhanced > thresh_enhanced
            methods.append(("Enhanced_Otsu", mask_enhanced, thresh_enhanced))
        except:
            pass
        
        return methods
    
    # Try all methods
    threshold_methods = try_thresholding_methods(red_blur)
    
    # Evaluate each method and select the best one
    best_mask = None
    best_method = None
    best_score = 0
    
    filename = os.path.basename(path)
    print(f"\n処理中: {filename}")
    
    for method_name, mask, thresh_val in threshold_methods:
        # Apply morphology
        mask_cleaned = morphology.remove_small_objects(mask, min_size=50)
        mask_cleaned = morphology.binary_closing(mask_cleaned, morphology.disk(3))
        mask_cleaned = morphology.binary_opening(mask_cleaned, morphology.disk(2))
        
        # Get connected components
        labeled = measure.label(mask_cleaned)
        props = sorted(measure.regionprops(labeled), key=lambda p: p.area, reverse=True)
        
        if len(props) == 0:
            score = 0
        else:
            # Score based on: reasonable total area, presence of two components, area ratio
            total_area = sum(p.area for p in props)
            num_components = len(props)
            
            # Prefer methods that find 2 eye-like regions with reasonable size
            score = total_area
            if num_components == 2:
                score *= 1.5  # Bonus for finding two eyes
                area_ratio = min(props[0].area, props[1].area) / max(props[0].area, props[1].area)
                if area_ratio > 0.3:  # Eyes should be somewhat similar in size
                    score *= 1.2
            
            # Penalize extremely small or large areas
            if total_area < 1000:  # Too small
                score *= 0.1
            elif total_area < 5000:  # Small but acceptable for weak signals
                score *= 0.8
            elif total_area > 300000:  # Way too large (over-segmentation)
                score *= 0.1
            elif total_area > 150000:  # Too large
                score *= 0.3
            
            # Prefer areas in reasonable range for eye segmentation (10k-150k pixels)
            if 10000 <= total_area <= 150000:
                score *= 1.5
            
            # Give bonus to Triangle and Enhanced_Otsu methods for better performance
            if method_name in ["Triangle", "Enhanced_Otsu"]:
                score *= 1.3
            
            # Penalize Local_Otsu if it finds too large areas (over-segmentation)
            if method_name == "Local_Otsu" and total_area > 200000:
                score *= 0.05
        
        print(f"  {method_name}: 閾値={thresh_val:.1f}, 面積={sum(p.area for p in props) if props else 0:.0f}, スコア={score:.0f}")
        
        if score > best_score:
            best_score = score
            best_mask = mask_cleaned
            best_method = method_name
    
    if best_mask is None:
        print(f"  警告: {filename} で適切な閾値処理が見つかりませんでした。デフォルトのOtsuを使用します。")
        thresh = filters.threshold_otsu(red_blur)
        best_mask = red_blur > thresh
        best_mask = morphology.remove_small_objects(best_mask, min_size=50)
        best_method = "Default_Otsu"
    
    print(f"  選択された手法: {best_method}")
    
    # Keep the largest two components (the eyes)
    labeled = measure.label(best_mask)
    props = sorted(measure.regionprops(labeled), key=lambda p: p.area, reverse=True)
    keep_labels = [p.label for p in props[:2]] if len(props) >= 2 else [p.label for p in props]
    mask2 = np.isin(labeled, keep_labels)
    
    # Background as median of outside-mask pixels
    bg = np.median(red[~mask2]) if np.any(~mask2) else 0.0
    
    # Connected components for per-eye
    labeled2 = measure.label(mask2)
    props2 = sorted(measure.regionprops(labeled2, intensity_image=red), key=lambda p: p.centroid[1])
    eyes = ["left", "right"] if len(props2) == 2 else [f"eye{i+1}" for i in range(len(props2))]
    
    # Prepare overlay for visualization
    overlay = arr.copy()
    # draw mask as boosted red
    overlay_mask = np.zeros_like(arr)
    overlay_mask[..., 0] = (mask2 * 255).astype(np.uint8)
    # Blend: original + mask outline
    # We'll outline by dilating mask and subtracting
    from scipy.ndimage import binary_dilation
    outline = binary_dilation(mask2, iterations=2) & (~mask2)
    overlay[outline, 1:] = 0  # make outline pure red
    overlay[outline, 0] = 255
    
    # Save overlay
    base = os.path.splitext(os.path.basename(path))[0]
    output_dir = "/Users/ohde/Documents/KULIP/projects/Gbim_transgenesis/hypbase/RAISING"
    overlay_path = os.path.join(output_dir, f"{base}_overlay_{best_method}.png")
    Image.fromarray(overlay).save(overlay_path)
    
    # Per-image metrics
    pix_area = float(mask2.sum())
    mean_int = float(red[mask2].mean()) if pix_area > 0 else 0.0
    median_int = float(np.median(red[mask2])) if pix_area > 0 else 0.0
    integrated = float(red[mask2].sum())
    corrected_integrated = float((red[mask2] - bg).sum())
    corrected_mean = float(mean_int - bg)
    
    per_image_row = dict(
        image=os.path.basename(path),
        scope="image_total",
        threshold_method=best_method,
        area_px=pix_area,
        mean_intensity=mean_int,
        median_intensity=median_int,
        integrated_density=integrated,
        background=bg,
        corrected_integrated_density=corrected_integrated,
        corrected_mean_intensity=corrected_mean,
        overlay_path=overlay_path
    )
    
    rows = [per_image_row]
    
    # Per-eye metrics
    for label, p in zip(eyes, props2):
        eye_mask = labeled2 == p.label
        area = float(eye_mask.sum())
        mean_i = float(red[eye_mask].mean()) if area > 0 else 0.0
        median_i = float(np.median(red[eye_mask])) if area > 0 else 0.0
        integ = float(red[eye_mask].sum())
        corr_integ = float((red[eye_mask] - bg).sum())
        corr_mean = float(mean_i - bg)
        rows.append(dict(
            image=os.path.basename(path),
            scope=f"{label}_eye",
            threshold_method=best_method,
            area_px=area,
            mean_intensity=mean_i,
            median_intensity=median_i,
            integrated_density=integ,
            background=bg,
            corrected_integrated_density=corr_integ,
            corrected_mean_intensity=corr_mean,
            overlay_path=overlay_path
        ))
    
    return rows, overlay_path

all_rows = []
overlay_paths = []
for p in paths:
    rows, op = analyze_image(p)
    all_rows.extend(rows)
    overlay_paths.append(op)

df = pd.DataFrame(all_rows)

# Save CSV
output_dir = "/Users/ohde/Documents/KULIP/projects/Gbim_transgenesis/hypbase/RAISING"
csv_path = os.path.join(output_dir, "fluorescence_eye_quantification.csv")
df.to_csv(csv_path, index=False)

# Display the table for quick inspection
print("Fluorescence quantification (per image and per eye):")
print(df.to_string(index=False))

# Plot a simple bar chart comparing corrected mean intensity per image (total mask)
totals = df[df["scope"] == "image_total"]
# ファイル数に応じて図のサイズを調整
fig_width = max(10, len(totals) * 0.8)
plt.figure(figsize=(fig_width, 6))
plt.bar(totals["image"], totals["corrected_mean_intensity"])
plt.ylabel("Corrected mean intensity (a.u.)")
plt.title("Eye fluorescence (per image)")
plt.xticks(rotation=45, ha='right')  # ファイル名を斜めに表示
plt.tight_layout()
plot_path = os.path.join(output_dir, "fluorescence_barplot.png")
plt.savefig(plot_path, dpi=200)
plt.close()

{"csv_path": csv_path, "overlays": overlay_paths, "plot_path": plot_path}
