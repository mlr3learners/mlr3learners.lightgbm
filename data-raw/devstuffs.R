packagename = "mlr3learners.lightgbm"

# remove existing description object
unlink("DESCRIPTION")

# Create a new description object
my_desc = desc::description$new("!new")

# Set your package name
my_desc$set("Package", packagename)

# Set author names 2
my_desc$set_authors(c(
  person("Lorenz A.", "Kapsner",
    email = "lorenz.kapsner@gmail.com",
    role = c("cre", "aut"),
    comment = c(ORCID = "0000-0003-1866-860X")),
  person(
    given = "Patrick",
    family = "Schratz",
    role = "ctb",
    email = "patrick.schratz@gmail.com",
    comment = c(ORCID = "0000-0003-0748-6624")),
  person(
    given = "Andrey",
    family = "Ogurtsov",
    role = "ctb",
    email = "ogurtsov.a.b@gmail.com")
))

# Remove some author fields
my_desc$del("Maintainer")

# Set the version
my_desc$set_version("0.0.5.9011")

# The title of your package
my_desc$set(Title = "mlr3: LightGBM Learner")

# The description of your package
my_desc$set(Description = paste0("Adds `lgb.train()` from the lightgbm package to mlr3."))

# The date when description file has been created
my_desc$set("Date" = as.character(Sys.Date()))

# The urls
my_desc$set("URL", "https://github.com/mlr3learners/mlr3learners.lightgbm")
my_desc$set("BugReports", "https://github.com/mlr3learners/mlr3learners.lightgbm/issues")

# Vignette Builder
my_desc$set("VignetteBuilder" = "knitr")

# License
my_desc$set("License", "LGPL-3")

# Save everyting
my_desc$write(file = "DESCRIPTION")

# License
# usethis::use_lgpl_license(name = "Lorenz A. Kapsner")


# Depends
usethis::use_package("R", min_version = "2.10", type = "Depends")

# Imports
usethis::use_package("data.table", type = "Imports")
usethis::use_package("R6", type = "Imports")
usethis::use_package("paradox", type = "Imports")
usethis::use_package("mlr3misc", type = "Imports")
usethis::use_package("mlr3", type = "Imports")
usethis::use_package("plyr", type = "Imports")
usethis::use_package("lightgbm", type = "Imports")


# Suggests
usethis::use_package("testthat", type = "Suggests")
usethis::use_package("ggplot2", type = "Suggests")
usethis::use_package("stats", type = "Suggests")
usethis::use_package("lintr", type = "Suggests")
usethis::use_package("checkmate", type = "Suggests")
usethis::use_package("MLmetrics", type = "Suggests") # for custom metrics

# for vignettes
usethis::use_package("rmarkdown", type = "Suggests")
usethis::use_package("knitr", type = "Suggests")
usethis::use_package("future", type = "Suggests")
usethis::use_package("mlr3tuning", type = "Suggests")
usethis::use_package("mlbench", type = "Suggests")

# Remotes
desc::desc_set_remotes(c(paste0(
  # see https://github.com/microsoft/LightGBM/tree/master/R-package
  "url::https://github.com/microsoft/LightGBM/releases/download/v3.0.0/lightgbm-3.0.0-r-cran.tar.gz")
),
file = usethis::proj_get())

# buildignore
# usethis::use_build_ignore("LICENSE.md")
# usethis::use_build_ignore(".gitlab-ci.yml")
# usethis::use_build_ignore("data-raw")
# usethis::use_build_ignore("*.Rproj")
# usethis::use_build_ignore("data-raw")

# gitignore
# usethis::use_git_ignore("/*")
# usethis::use_git_ignore("/*/")
# usethis::use_git_ignore("*.log")
# usethis::use_git_ignore("!/.gitignore")
# usethis::use_git_ignore("!/.gitlab-ci.yml")
# usethis::use_git_ignore("!/data-raw/")
# usethis::use_git_ignore("!/DESCRIPTION")
# usethis::use_git_ignore("!/inst/")
# usethis::use_git_ignore("!/LICENSE.md")
# usethis::use_git_ignore("!/man/")
# usethis::use_git_ignore("!NAMESPACE")
# usethis::use_git_ignore("!/R/")
# usethis::use_git_ignore("!/tests/")
# usethis::use_git_ignore("!/vignettes/")
# usethis::use_git_ignore("!/README.md")
# usethis::use_git_ignore("!/tests/")
# usethis::use_git_ignore("/.Rhistory")
# usethis::use_git_ignore("!/*.Rproj")
# usethis::use_git_ignore("/.Rproj*")
# usethis::use_git_ignore("/.RData")
# usethis::use_git_ignore("!/man-roxygen/")

# code coverage
# covr::package_coverage()

# lint package
# lintr::lint_package()

usethis::use_tidy_description()
