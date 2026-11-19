# ============================================================
# Extract Hazus depth-damage curves without installing hazus
# ============================================================

# Packages we still need for filtering/export
if (!requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("readr", quietly = TRUE)) {
  stop('Install dplyr and readr before running this optional extraction script.')
}

library(dplyr)
library(readr)

# ------------------------------------------------------------
# 1. Set project paths
# ------------------------------------------------------------

project_root <- normalizePath(".", mustWork = TRUE) # Run from the repository root.

output_folder <- file.path(
  project_root,
  "data",
  "processed",
  "simulation"
)

dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

download_url <- "https://cran.r-project.org/src/contrib/Archive/hazus/hazus_0.1.tar.gz"

download_path <- file.path(
  output_folder,
  "hazus_0.1.tar.gz"
)

extract_folder <- file.path(
  output_folder,
  "hazus_package_extract"
)

dir.create(extract_folder, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Download archived hazus package source
# ------------------------------------------------------------

download.file(
  url = download_url,
  destfile = download_path,
  mode = "wb"
)

print("Downloaded archived hazus package.")

# ------------------------------------------------------------
# 3. Extract package contents
# ------------------------------------------------------------

untar(
  tarfile = download_path,
  exdir = extract_folder
)

print("Extracted archived hazus package.")

# ------------------------------------------------------------
# 4. Find available data files
# ------------------------------------------------------------

all_files <- list.files(
  extract_folder,
  recursive = TRUE,
  full.names = TRUE
)

data_files <- all_files[
  grepl("\\.rda$|\\.RData$|\\.rds$", all_files, ignore.case = TRUE)
]

print("Data files found:")
print(data_files)

# ------------------------------------------------------------
# 5. Load haz_fl_dept data file
# ------------------------------------------------------------

haz_fl_dept_file <- data_files[
  grepl("haz_fl_dept", data_files, ignore.case = TRUE)
]

if (length(haz_fl_dept_file) == 0) {
  stop("Could not find haz_fl_dept data file in extracted package.")
}

load(haz_fl_dept_file[1])

if (!exists("haz_fl_dept")) {
  stop("haz_fl_dept object was not created after loading the data file.")
}

print("haz_fl_dept loaded successfully.")
print(names(haz_fl_dept))
print(dim(haz_fl_dept))
print(head(haz_fl_dept))

# ------------------------------------------------------------
# 6. Export full Hazus depth-damage table
# ------------------------------------------------------------

full_output_path <- file.path(
  output_folder,
  "hazus_full_depth_damage_curves.csv"
)

write_csv(
  haz_fl_dept,
  full_output_path
)

print("Full Hazus depth-damage table exported:")
print(full_output_path)

# ------------------------------------------------------------
# 7. Select function IDs we care about first
# ------------------------------------------------------------

selected_ids <- c(
  # Structure damage functions
  129, 658, 217, 341, 545, 559, 616, 631, 643,

  # Contents damage functions
  45, 488, 90, 195, 358, 384, 460, 472, 480
)

# Check ID column exists
if (!"DmgFnId" %in% names(haz_fl_dept)) {
  stop(
    paste(
      "Column DmgFnId not found. Available columns are:",
      paste(names(haz_fl_dept), collapse = ", ")
    )
  )
}

selected_curves <- haz_fl_dept %>%
  filter(DmgFnId %in% selected_ids)

# ------------------------------------------------------------
# 8. Export selected curves
# ------------------------------------------------------------

selected_output_path <- file.path(
  output_folder,
  "hazus_selected_depth_damage_curves.csv"
)

write_csv(
  selected_curves,
  selected_output_path
)

print("Selected Hazus depth-damage curves exported:")
print(selected_output_path)

print("Selected curve dimensions:")
print(dim(selected_curves))

print("Selected curves:")
display(selected_curves)
