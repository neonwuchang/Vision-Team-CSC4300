# =============================================================================
# run_validation.py
# AI assisted validation analysis
# Validation Code for LV Segmentation Project
# 
# INSTRUCTIONS:
# 1. Upload this file to: Google Drive/Colab Notebooks/Segmentation Project/
# 2. In your Colab notebook, after training completes, add a new cell
# 3. Copy and paste the code below into that cell
# 4. Run the cell
# =============================================================================

# COPY EVERYTHING BELOW THIS LINE INTO YOUR COLAB NOTEBOOK
# ============================================================

print("running validation evaluation")

# Evaluate the model on validation set
results = val.evaluate(model, val_loader, device)

# Print summary report
val.print_report(results)
print("generating visuals")

# Visualize predictions (6 random samples)
val.plot_predictions(model, val_dataset, device, n_samples=6)

# Show metric distributions
val.plot_metric_distribution(results)
print("saving results")

# Save metrics to CSV for your report
val.save_metrics_csv(
    results, 
    "/content/drive/MyDrive/Colab Notebooks/Segmentation Project/validation_results.csv"
)

print("\n✓ Validation complete!")
print("✓ Results saved to validation_results.csv")
