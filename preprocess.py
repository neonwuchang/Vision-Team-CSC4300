#import numpy as np
#import matplotlib.pyplot as plt



# Define where the images are, even better if you can make a train/validation split
# Maybe make a train v Validation split with python referencing cine sax

#path = r"C:\Users\user\OneDrive - Louisiana State University\PhD Research\Scripts\segmentation\automation\Cardiac Atlas Project\SCD_IMAGES_01\SCD0000101\CINESAX_300\IM-0003-0001.dcm"
# ensure the file is being ran out of the correct directory

#### Whole path version for debugging purposes
#patient_map_df = pd.read_csv(r"C:\Users\user\OneDrive - Louisiana State University\PhD Research\Scripts\Cardiac Atlas Project\scd_patientdata.csv")
patient_map_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Segmentation Project/scd_patientdata.csv')

## AI suggests changing the column names to more easily source of reference ID
patient_map_df = patient_map_df.rename(columns={
    "PatientID": "dicom_pid",
    "OriginalID": "contour_pid"
})
assert patient_map_df["dicom_pid"].is_unique
assert patient_map_df["contour_pid"].is_unique

#### Whole path version for debugging purposes
#contour_root = Path(r"C:\Users\user\OneDrive - Louisiana State University\PhD Research\Scripts\segmentation\automation\Cardiac Atlas Project\scd_manualcontours\SCD_ManualContours")
contour_root = Path('/content/drive/MyDrive/Colab Notebooks/Segmentation Project/Cardiac Atlas Project/scd_manualcontours/SCD_ManualContours')

contour_files = list(contour_root.rglob("*contour-manual.txt"))

icontour_files = list(contour_root.rglob("*-icontour-manual.txt"))
ocontour_files = list(contour_root.rglob("*-ocontour-manual.txt"))
print(f"Found {len(icontour_files)} icontour files")
print(f"Found {len(ocontour_files)} ocontour files")

# AI assisted data frame builder for contours
##
rows = []

for path in contour_files:
    contour_type = ppf.extract_contour_type(path)
    if contour_type is None:
        continue

    rows.append({
        "patient_id": ppf.extract_patient_id(path),
        "slice_id": ppf.extract_slice_id(path),
        "contour_type": contour_type,
        "contour_path": path
    })

contours_df = pd.DataFrame(rows)
##
# AI generated merger for patient ID clarification
contours_df = contours_df.merge(
    patient_map_df,
    left_on="patient_id",
    right_on="contour_pid",
    how="inner"
)
# ensure we have a number for each slice, easier to cross reference with
contours_df["slice"] = contours_df["slice_id"].str.extract(r"(\d+)(?=\D*$)").astype(int)
##

# Remove unnecessary columns
contours_df = contours_df.drop(["Pathology", "Age", "Gender", "patient_id"], axis=1) ## patient_id is covered by dicom_pid and contour_pid

############ Onto Image_df construction #############

# full path for debugging purposes
#image_root = Path(r"C:\Users\user\OneDrive - Louisiana State University\PhD Research\Scripts\segmentation\automation\Cardiac Atlas Project")
image_root = Path('/content/drive/MyDrive/Colab Notebooks/Segmentation Project/Cardiac Atlas Project')

dicom_files = list(image_root.rglob("*CINESAX*/*.dcm"))
print(f"Found {len(dicom_files)} DICOM files")


im_rows = []

for path in dicom_files:
    has_contour = ppf.check_for_contour(path, contours_df)
    if has_contour:
        im_rows.append({
            "image_path": path,
            "slice_id_dcm": ppf.extract_slice_id(path),
            "dicom_pid": ppf.extract_dicom_pid(path),
            "slice": ppf.extract_slice_number(path)
            })

image_df = pd.DataFrame(im_rows)

df = contours_df.merge(
    image_df,
    on=["dicom_pid", "slice"],
    how="inner"
    )
'''
df is now the map dataframe we will use going forward to
 set up the training and validation data splits
'''
